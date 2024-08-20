import json
import time
from pathlib import Path
from string import Template

import chromadb
import ollama
import pdfplumber
from dynaconf import settings
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pdfplumber.utils.pdfinternals import resolve_and_decode, resolve
from tqdm import tqdm

from aiassistant.database import get_all_qa, get_parse_stats, get_audit_record, upsert_audit_record, tb_upload_stats, \
    get_qa_doc_id_from_question_id
from aiassistant.log import get_logger

logger = get_logger(__name__)

client = OpenAI(
    base_url=settings.OPENAI_BASE_URL,
    api_key='ollama',  # required, but unused
)

model_parse = ChatOpenAI(
    model=settings.MODEL_PARSE,
    temperature=0,
    base_url=settings.OPENAI_BASE_URL,
    api_key='ollama'
)

vector_db_client = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
collection = vector_db_client.get_or_create_collection(name="docs", metadata={"hnsw:space": "cosine"})


def read_pdf_content(path):
    document = ""
    count = 0
    logger.info(f'Reading document at: {path}')
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            count += 1
            text = page.extract_text(x_tolerance=1)
            # print(f'{count} -> {text}')
            document += text
    logger.info(f'Document parsed. Number of pages: {count}')
    return document


def save_llm_io_to_disk(label, response):
    tmp_dir = Path(settings.TMP_DIR_PATH)
    tmp_dir.mkdir(exist_ok=True)
    path = tmp_dir / f'{label}.txt'
    with open(path, 'w+') as response_file:
        response_file.write(response)
        logger.info(f'LLM response {label} saved at {path}')


def get_response_from_llm(prompt, model="mistral"):
    logger.info(f'Requesting LLM({model}) response. Prompt length: {len(prompt)}')
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    response_content = response.choices[0].message.content
    logger.info(f'Response received from LLM ({model}). Response length: {len(response_content)}')
    return response_content


def parse_with_llm(form_id: int):
    upload_stat = tb_upload_stats.get(doc_id=form_id)
    file_name = upload_stat.get('file_name')
    file_path = Path(settings.UPLOADS_DIR) / file_name
    logger.info(f'Started reading pdf at {file_path}')
    doc = read_pdf_content(path=file_path)
    user_prompt = settings.FORM_PARSE_PROMT_PREFIX + '\n' + doc
    parser = CommaSeparatedListOutputParser()
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    model = model_parse
    chain = prompt | model | parser
    logger.info(f'Requesting LLM ({settings.MODEL_PARSE}) response for parsing pdf')
    response = chain.invoke({"query": user_prompt})
    logger.info(f'Response received from LLM ({settings.MODEL_PARSE}). List of form fields: {response}')
    logger.info(f'completed parsing pdf')
    return response


def parse_acro_form(form_id):
    upload_stat = tb_upload_stats.get(doc_id=form_id)
    file_name = upload_stat.get('file_name')
    file_path = Path(settings.UPLOADS_DIR) / file_name
    logger.info(f'Started reading pdf at {file_path}')
    pdf = pdfplumber.open(file_path)
    form_data = []

    def parse_field_helper(form_data, field, prefix=None):
        """ appends any PDF AcroForm field/value pairs in `field` to provided `form_data` list

            if `field` has child fields, those will be parsed recursively.
        """
        resolved_field = field.resolve()
        field_name = '.'.join(filter(lambda x: x, [prefix, resolve_and_decode(resolved_field.get("T"))]))
        if "Kids" in resolved_field:
            for kid_field in resolved_field["Kids"]:
                parse_field_helper(form_data, kid_field, prefix=field_name)
        if "T" in resolved_field or "TU" in resolved_field:
            # "T" is a field-name, but it's sometimes absent.
            # "TU" is the "alternate field name" and is often more human-readable
            # your PDF may have one, the other, or both.
            alternate_field_name = resolve_and_decode(resolved_field.get("TU")) if resolved_field.get("TU") else None
            field_value = resolve_and_decode(resolved_field["V"]) if 'V' in resolved_field else None
            form_data.append([field_name, alternate_field_name, field_value])

    fields = resolve(resolve(pdf.doc.catalog["AcroForm"])["Fields"])
    for field in fields:
        parse_field_helper(form_data, field)
    form_fields = [f'{alternate_field_name} ({field_name})' for field_name, alternate_field_name, field_value in
                   form_data]
    return form_fields


def qa_with_llm(form_id: int):
    parse_stats = get_parse_stats(form_id=form_id)
    qa_stats = []
    for parse_stat in tqdm(parse_stats, desc=f'Auto QA [{form_id}]'):
        question_id = parse_stat.doc_id
        form_field = parse_stat['field']
        question_embedding = get_embeddings(form_field)
        result_docs = query_embeddings(embedding=question_embedding)
        qa_display_list = []
        for qa_str in result_docs['documents'][0]:
            qa_dict = json.loads(qa_str)
            qa_display = f'Question: {qa_dict["question"]}\nAnswer:{qa_dict["answer"]}\n'
            qa_display_list.append(qa_display)
        prompt_template = Template(settings.FORM_QA_DOC_PROMT_PREFIX)
        qa_prompt = prompt_template.substitute(dict(qa_docs='\n'.join(qa_display_list)))
        qa_prompt = qa_prompt + '\n' + f'Question: {form_field} ?'
        logger.info(f'{qa_prompt=}')
        response = get_response_from_llm(prompt=qa_prompt, model=settings.MODEL_QA)
        answer = ' '.join([word.strip() for word in response.split()])
        logger.info(f'Q: {form_field} | A: {answer}')
        qa_stats.append(dict(question_id=question_id, answer_auto=answer))
    return qa_stats


def create_docs_embeddings():
    logger.info(f'Initializing QA embeddings')
    task_id = str(int(time.time()))
    update_audit(event='embeddings_init', task=task_id)
    all_qa_docs = get_all_qa()
    for qa_doc in all_qa_docs:
        doc_id = qa_doc.doc_id
        form_id = qa_doc['form_id']
        question_id = qa_doc['question_id']
        question = get_parse_stats(form_id, parse_id=question_id)['field']
        answer = qa_doc.get('answer_user')
        if not answer:
            continue
        doc = dict(
            question=question,
            answer=answer
        )
        doc_str = json.dumps(doc)
        embedding = get_embeddings(question)
        collection.add(
            ids=[str(doc_id)],
            embeddings=[embedding],
            documents=[doc_str]
        )
        logger.info(f'Added QA doc to embeddings for [{doc_id:3}] {question}')
    logger.info(f'Finished QA embeddings')
    update_audit(event='embeddings_init', task=task_id)


def get_embeddings(doc):
    response = ollama.embeddings(model="mxbai-embed-large", prompt=doc)
    embedding = response["embedding"]
    return embedding


def update_embedding(form_id: int, question_id: int, answer: str):
    question = get_parse_stats(form_id=form_id, parse_id=question_id)['field']
    doc_id = get_qa_doc_id_from_question_id(question_id=question_id)
    doc = dict(
        question=question,
        answer=answer
    )
    doc_str = json.dumps(doc)
    embedding = get_embeddings(question)
    collection.upsert(
        ids=[str(doc_id)],
        embeddings=[embedding],
        documents=[doc_str]
    )
    logger.info(f'Updated embedding for question_id: {question_id}')


def query_embeddings(embedding, similarity_threshold: float = 0.25):
    results = collection.query(
        query_embeddings=[embedding],
        n_results=2
    )
    documents = results['documents'][0]
    distances = results['distances'][0]
    filtered_documents = []
    for index, distance in enumerate(distances):
        if not distance <= similarity_threshold:
            continue
        filtered_documents.append(documents[index])
    result_dict = dict(documents=[filtered_documents])
    return result_dict


def update_audit(event='form_process', form_id=None, task=None, sub_task=None, code='SUCCESS', error_msg=None, tags=''):
    audit_record = get_audit_record(event=event, form_id=form_id, task=task, sub_task=sub_task)
    if audit_record:
        doc_id = audit_record.doc_id
        rec = dict(
            code=code,
            error_msg=error_msg,
            end_time=time.time()
        )
    else:
        doc_id = None
        rec = dict(
            event=event,
            form_id=form_id,
            task=task,
            sub_task=sub_task,
            start_time=time.time(),
            code='STARTED',
            tags=tags
        )
    upsert_audit_record(delta=rec, doc_id=doc_id)
