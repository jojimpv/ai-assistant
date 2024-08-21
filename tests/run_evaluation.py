import asyncio
import json
import shutil
import sys
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

model_eval_judge = ChatOpenAI(
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
    for qa_dict in tqdm(qa_test, desc=f'Processing QA [{form_id}]'):
        question = qa_dict["question"]
        answer = qa_dict["answer"]
        eval_prompt = evaluation_prompt_template.format_messages(
            question=question,
            answer=answer,
            reference_qa=qa_display_str,
        )
        eval_result = model_eval_judge.invoke(eval_prompt)
        # logger.info(f'{eval_result.content = }')
        feedback, score = [item.strip() for item in eval_result.content.split("[RESULT]")]
        logger.info(f'{question = }, {answer = }, {score = }, {feedback = }')
        scores.append(dict(
            question=question,
            answer=answer,
            score=score,
            feedback=feedback,
        ))
    return scores


def get_results_db_backups(result_form_id=None):
    if result_form_id:
        return glob(f'{settings.TEST_RESULTS_PATH}/*{result_form_id}.json'), 0
    all_result_backups = glob(f'{settings.TEST_RESULTS_PATH}/*.json')
    filtered_result_backups = []
    eval_key_exists_count = 0
    for db_backup in all_result_backups:
        form_meta = get_form_meta(path=db_backup)
        if not check_eval_key_exists(form_id=form_meta.form_id):
            filtered_result_backups.append(db_backup)
        else:
            eval_key_exists_count += 1
            logger.debug(f'Evaluation using model: {settings.MODEL_EVAL} already exists '
                         f'for for form_id {form_meta.form_id}')
    return filtered_result_backups, eval_key_exists_count


def get_form_meta(path):
    with open(path, 'r') as db_backup_file:
        d = json.load(db_backup_file)
        b = Box(d)
        form_ids = list(b.upload_stats.keys())
        form_ids.remove(str(settings.REFERENCE_FORM_ID))
        form_id = int(form_ids[0])
        form_name = b.upload_stats[str(form_id)]['name']
        model_parse_tag = [x for x in b.audit.values() if x.task == 'PARSE'][0].tags
        model_parse = slugify(model_parse_tag.replace('model_parse=', ''))
        model_qa_tag = [x for x in b.audit.values() if x.task == 'AUTO_QA'][0].tags
        model_qa = slugify(model_qa_tag.replace('model_qa=', ''))
    form_meta = Box(dict(
        form_id=form_id,
        form_name=form_name,
        model_qa=model_qa,
        model_parse=model_parse
    ))
    return form_meta


def replace_test_db_file(path):
    shutil.copy(src=path, dst=settings.DB_PATH)
    logger.info(f'Replaced test DB file at {settings.DB_PATH} by {path}')


def setup():
    Path(settings.TEST_EVALUATIONS_PATH).mkdir(exist_ok=True)


def save_score(form_meta, score_info):
    form_id = form_meta.form_id
    evaluation_score_path = Path(settings.TEST_EVALUATIONS_PATH) / f'eval_{form_id}.json'
    scores = dict()
    if evaluation_score_path.exists() and evaluation_score_path.is_file():
        with open(evaluation_score_path, 'r') as evaluation_file:
            scores = json.load(evaluation_file)
    with open(evaluation_score_path, 'w+') as evaluation_file:
        score_key = f'score_{slugify(settings.MODEL_EVAL)}'
        new_scores = {
            score_key: score_info
        }
        scores.update(new_scores)
        json.dump({
            **form_meta,
            **scores
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


async def main(result_form_id=None):
    logger.info(f'Started evaluation using model: {settings.MODEL_EVAL}')
    setup()
    results_db_backups, skip_count = get_results_db_backups(result_form_id=result_form_id)
    logger.info(f'Number of results db backups to be processed: {len(results_db_backups)}')
    new_eval_key_count = len(results_db_backups)
    for db_backup in tqdm(results_db_backups, desc='Processing DB backup'):
        try:
            form_meta = get_form_meta(path=db_backup)
            logger.info(f'Processing score calculation for result backup file at: {db_backup} . '
                        f'Form name: {form_meta.form_name}')
            replace_test_db_file(path=db_backup)
            score_info = check_score(ref_form_id=1, form_id=form_meta.form_id)
            save_score(form_meta, score_info)
        except Exception as error:
            logger.exception(f'Evaluation error while processing {db_backup} file. Details: {str(error)}')
    logger.info(f'Completed evaluation. Counts: {new_eval_key_count}(new), {skip_count}(skipped).')


if __name__ == '__main__':
    args = sys.argv[1:]
    asyncio.run(main(*args))
