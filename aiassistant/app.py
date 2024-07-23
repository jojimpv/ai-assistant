import ast

from dynaconf import settings
from openai import OpenAI

from aiassistant.log import get_logger
from core import read_pdf_content, get_response_from_llm

logger = get_logger(__name__)

client = OpenAI(
    base_url=settings.OPENAI_BASE_URL,
    api_key='ollama',  # required, but unused
)


def main():
    logger.info(f'Started AI Assistant')
    doc = read_pdf_content(path=settings.FORM_PATH)
    parse_prompt = settings.FORM_PARSE_PROMT_PREFIX + '\n' + doc
    response = get_response_from_llm(prompt=parse_prompt)
    form_fields = ast.literal_eval(response)
    logger.info(f'List of form fields: {form_fields}')
    logger.info('Started QA stage')
    for form_field in form_fields:
        qa_prompt = settings.FORM_QA_PROMT_PREFIX + '\n' + f'Question: {form_field}'
        response = get_response_from_llm(prompt=qa_prompt)
        logger.info(f'Q: {form_field} | A: {response}')
    logger.info(f'Finished AI Assistant')


if __name__ == '__main__':
    main()
