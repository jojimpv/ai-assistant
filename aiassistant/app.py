"""
This script contains a console application which demonstrate form parsing and QA using LLM.
It can be executed to check response from LLM for different forms within a terminal window.
"""
from pathlib import Path

from dynaconf import settings
from openai import OpenAI

from aiassistant.log import get_logger
from core import get_response_from_llm, parse_with_llm

logger = get_logger(__name__)

client = OpenAI(
    base_url=settings.OPENAI_BASE_URL,
    api_key='ollama',  # required, but unused
)


def main():
    logger.info(f'Started AI Assistant')
    form_path = Path(settings.FORM_PATH)
    response = parse_with_llm(form_path=form_path)
    form_fields = response
    logger.info(f'List of form fields: {form_fields}')
    logger.info('Started QA stage')
    for form_field in form_fields:
        qa_prompt = settings.FORM_QA_PROMT_PREFIX + '\n' + f'Question: {form_field}'
        response = get_response_from_llm(prompt=qa_prompt)
        logger.info(f'Q: {form_field} | A: {response}')
    logger.info(f'Finished AI Assistant')


if __name__ == '__main__':
    main()
