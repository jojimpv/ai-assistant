import asyncio
from collections import namedtuple
from pathlib import Path
from typing import List

from dynaconf import settings
from faker import Faker
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from aiassistant.backend import process_uploads, process_parse_form, process_qa_form
from aiassistant.core import read_pdf_content, update_audit, update_embedding, vector_db_client
from aiassistant.core import collection as vectordb_collection
from aiassistant.database import add_upload_stats, tb_qa_stats, tb_upload_stats, tb_parse_stats, db
from aiassistant.log import get_logger

# GOV_FORM_PATH = '../data/Change_of_Address_Form_25.04.16.pdf'
GOV_FORM_PATH = '../data/sample-form-1.pdf'
TEST_FORM_ID = 1
model = ChatOpenAI(
    model='mistral',
    temperature=0,
    base_url=settings.OPENAI_BASE_URL,
    api_key='ollama'
)
logger = get_logger(__name__)


class SingleQA(BaseModel):
    question: str = Field(description="question to user")
    answer: str = Field(description="answer given by user")


class UserQA(BaseModel):
    qa: List[SingleQA] = []


prompt_template_2 = """
You are an interviewer asking questions by looking at a government form.
The questions shouldn't be exactly same as what is given in the government form, but similar to it.
Please create the questions and plausible answers for each question. 
The length of answers need to be limited to 100 words. 

Assume the questions are being asked to a user with demographics given below.
Ask minimum questions.

Demographics:
=============
{demographics}


Government form content
=======================
{gov_form}

"""


def get_basic_user_profile():
    faker = Faker()
    name = faker.name()
    address = faker.address()
    job = faker.job()
    user_demographics = f"""
    Name: {name}

    Address: {address}

    Job: {job} 
    """
    return user_demographics


def get_prompt_for_llm(user_demographics, gov_form_content):
    full_prompt = prompt_template_2.format(gov_form=gov_form_content, demographics=user_demographics)
    return full_prompt


def get_evaluation_dataset():
    logger.info(f'Started evaluation dataset LLM call')
    user_demographics = get_basic_user_profile()
    gov_form_content = read_pdf_content(GOV_FORM_PATH)
    user_prompt = get_prompt_for_llm(user_demographics=user_demographics, gov_form_content=gov_form_content)
    parser = JsonOutputParser(pydantic_object=UserQA)
    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | model | parser
    result = chain.invoke({"query": user_prompt})
    logger.info(f'Finished evaluation dataset LLM call')
    return result


def populate_evaluation_dataset(form_id):
    eval_ds = get_evaluation_dataset()
    parse_form_docs = []
    qa_form_docs = []
    vector_db_docs = []
    VecDbRec = namedtuple('VecDbRec', ['form_id', 'question_id', 'answer'])
    for index, qa in enumerate(eval_ds['qa'], start=1):
        question_id = index
        question, answer = qa['question'], qa['answer']
        parse_form_docs.append(dict(form_id=form_id, field=question))
        qa_form_docs.append(dict(form_id=form_id, question_id=question_id, answer_user=answer))
        vector_db_docs.append(VecDbRec(form_id=form_id, question_id=question_id, answer=answer))
    tb_parse_stats.insert_multiple(parse_form_docs)
    tb_qa_stats.insert_multiple(qa_form_docs)
    for doc in vector_db_docs:
        update_embedding(form_id=doc.form_id, question_id=doc.question_id, answer=doc.answer)

    tb_upload_stats.update(dict(status=settings.FORM_STATUS_QA), doc_ids=[form_id])


def add_reference_form_stats():
    form_id = TEST_FORM_ID
    name = 'Reference form'
    file_name = 'reference-form'
    update_audit(form_id=form_id, task='UPLOAD')
    upload_stat = dict(
        name=name,
        file_name=file_name,
        file_size=0,
        status=settings.FORM_STATUS_UPLOADED
    )
    add_upload_stats(upload_stat=upload_stat, file_id=form_id)
    update_audit(form_id=form_id, task='UPLOAD')
    return form_id


def read_form_from_file(file_path):
    name = Path(file_path).stem
    logger.info(f'Reading form at {file_path}')
    with open(file_path, 'rb') as f:
        file_content = f.read()
    return name, file_content


def setup():
    db.drop_tables()
    for doc_id in vectordb_collection.get()['ids']:
        vectordb_collection.delete(ids=[doc_id])
    form_id = add_reference_form_stats()
    return form_id


async def test_evaluate():
    logger.info('Started evaluation')
    logger.info(f'=== Stage 0: Clear DB entries and init reference form ===')
    ref_form_id = setup()
    logger.info(f'=== Stage 1: Create DB entries for reference form ===')
    populate_evaluation_dataset(form_id=ref_form_id)
    logger.info(f'=== Stage 2.1: Read test form and make form_id ===')
    name, file_content = read_form_from_file(file_path=GOV_FORM_PATH)
    form_id = process_uploads(name=name, file_content=file_content)
    logger.info(f'{form_id = }')
    logger.info(f'=== Stage 2.2: Parse test form ===')
    await process_parse_form(form_id=form_id)
    logger.info(f'=== Stage 2.3: QA test form ===')
    await process_qa_form(form_id=form_id)
    logger.info('Completed evaluation')


if __name__ == '__main__':
    asyncio.run(test_evaluate())
