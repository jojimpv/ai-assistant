import asyncio
import json
import shutil
from glob import glob
from pathlib import Path

from box import Box
from dynaconf import settings
from langchain.schema import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from slugify import slugify
from tqdm import tqdm

from aiassistant.database import get_combined_qa
from aiassistant.log import get_logger

eval_chat_model = ChatOpenAI(
    model=settings.MODEL_EVAL,
    temperature=0,
    base_url=settings.OPENAI_BASE_URL,
    api_key='ollama'
)
logger = get_logger(__name__)


def check_score(ref_form_id, form_id):
    qa_ref = get_combined_qa(form_id=ref_form_id)
    qa_test = get_combined_qa(form_id=form_id)

    qa_display_list = []
    for qa_dict in qa_ref:
        qa_display = f'Question: {qa_dict["question"]}\nAnswer:{qa_dict["answer"]}\n'
        qa_display_list.append(qa_display)
    qa_display_str = '\n'.join(qa_display_list)

    evaluation_prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are a fair evaluator language model."),
            HumanMessagePromptTemplate.from_template(settings.EVALUATION_PROMPT),
        ]
    )

    scores = []
    for qa_dict in tqdm(qa_test, desc='Processing QA'):
        question = qa_dict["question"]
        answer = qa_dict["answer"]
        eval_prompt = evaluation_prompt_template.format_messages(
            instruction=question,
            response=answer,
            reference_answer=qa_display_str,
        )
        eval_result = eval_chat_model.invoke(eval_prompt)
        logger.info(f'{eval_result.content = }')
        feedback, score = [item.strip() for item in eval_result.content.split("[RESULT]")]
        logger.info(f'{question = }, {answer = }, {score = }, {feedback = }')
        scores.append(dict(
            question=question,
            answer=answer,
            score=score,
            feedback=feedback,
        ))
    return scores


def get_results_db_backups():
    return glob(f'{settings.TEST_RESULTS_PATH}/*.json')


def get_form_meta(path):
    with open(path, 'r') as db_backup_file:
        d = json.load(db_backup_file)
        b = Box(d)
        form_ids = list(b.upload_stats.keys())
        form_ids.remove(str(settings.REFERENCE_FORM_ID))
        form_id = int(form_ids[0])
        form_name = b.upload_stats[str(form_id)]['name']
    return Box(dict(form_id=form_id, form_name=form_name))


def replace_test_db_file(path):
    shutil.copy(src=path, dst=settings.DB_PATH)
    logger.info(f'Replaced test DB file at {settings.DB_PATH} by {path}')


def setup():
    Path(settings.TEST_EVALUATIONS_PATH).mkdir(exist_ok=True)


def save_score(form_meta, score_info):
    form_id = form_meta.form_id
    evaluation_score_path = Path(settings.TEST_EVALUATIONS_PATH) / f'eval_{form_id}.json'
    with open(evaluation_score_path, 'w+') as evaluation_file:
        score_key = f'score_{slugify(settings.MODEL_EVAL)}'

        json.dump({
            **form_meta,
            score_key: score_info
        }, fp=evaluation_file)
        logger.info(f'Saved evaluation scores at {evaluation_score_path}')


def check_eval_key_exists(form_id):
    score_key = f'score_{slugify(settings.MODEL_EVAL)}'
    evaluation_score_path = Path(settings.TEST_EVALUATIONS_PATH) / f'eval_{form_id}.json'
    if evaluation_score_path.exists() and evaluation_score_path.is_file():
        with open(evaluation_score_path) as evaluation_score_file:
            evaluation_scores = [x for x in Box(json.load(evaluation_score_file)) if x.startswith('score_')]
            if score_key in evaluation_scores:
                return True


async def main():
    logger.info(f'Started evaluation using model: {settings.MODEL_EVAL}')
    setup()
    results_db_backups = get_results_db_backups()
    logger.info(f'Number of results db backups: {len(results_db_backups)}')
    for db_backup in tqdm(results_db_backups, desc='Processing DB backup'):
        try:
            logger.info(f'Processing result backup file at: {db_backup}')
            form_meta = get_form_meta(path=db_backup)
            if check_eval_key_exists(form_id=form_meta.form_id):
                logger.info(f'Evaluation using model: {settings.MODEL_EVAL} already exists '
                            f'for for form_id {form_meta.form_id}')
                continue
            replace_test_db_file(path=db_backup)
            score_info = check_score(ref_form_id=1, form_id=form_meta.form_id)
            save_score(form_meta, score_info)
        except Exception as error:
            logger.exception(f'Evaluation error while processing {db_backup} file. Details: {str(error)}')
    logger.info('Completed evaluation')


if __name__ == '__main__':
    asyncio.run(main())
