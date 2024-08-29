"""
This script holds the URL routing for API service started along with GUI portal.
The API is served via Python package FastAPI.
"""
from nicegui import app

from aiassistant.core import parse_with_llm, qa_with_llm, parse_acro_form
from aiassistant.log import get_logger

logger = get_logger(__name__)


@app.get('/api/parse_form/{form_id}')
def parse_form(form_id: int):
    response = []
    try:
        response = parse_acro_form(form_id=form_id)
        logger.info(f'Form parsed using Acro form parser')
    except KeyError:
        pass
    if not response:
        logger.info(f'Parsing pdf with LLM')
        response = parse_with_llm(form_id=form_id)

    return dict(form_id=form_id, form_fields=response)


@app.get('/api/qa_form/{form_id}')
def qa_form(form_id: int):
    response = qa_with_llm(form_id=form_id)
    return dict(form_id=form_id, qa_stats=response)
