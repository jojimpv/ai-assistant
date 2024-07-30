import datetime
import time

from nicegui import ui

from aiassistant.database import get_all_audit

audit_columns_form = [
    dict(name='task', label='TASK', field='task', required=False, align='left'),
    dict(name='sub_task', label='SUB TASK', field='sub_task', required=False, align='left'),
    dict(name='start', label='START TIME', field='start', required=True, align='left'),
    dict(name='end', label='END TIME', field='end', required=False, align='left'),
    dict(name='duration', label='DURATION', field='duration', required=False, align='center'),
    dict(name='code', label='CODE', field='code', required=True, align='left'),
    dict(name='tags', label='TAGS', field='tags', required=False, align='left'),
    dict(name='error_msg', label='DETAILS', field='error_msg', required=False, align='left'),
]
audit_columns = [
    dict(name='event', label='EVENT', field='event', required=True, align='left'),
    dict(name='task', label='TASK', field='task', required=False, align='left'),
    dict(name='sub_task', label='SUB TASK', field='sub_task', required=False, align='left'),
    dict(name='form_id', label='FORM ID', field='form_id', required=False, align='left'),
    dict(name='start', label='START TIME', field='start', required=True, align='left'),
    dict(name='end', label='END TIME', field='end', required=False, align='left'),
    dict(name='duration', label='DURATION', field='duration', required=False, align='center'),
    dict(name='code', label='CODE', field='code', required=True, align='left'),
    dict(name='tags', label='TAGS', field='tags', required=False, align='left'),
    dict(name='error_msg', label='ERROR MESSAGE', field='error_msg', required=False, align='left'),
]


def get_formatted_audit_row(x):
    rec_form_id = x.get('form_id')
    start_time = x.get('start_time')
    end_time = x.get('end_time')
    start = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S') if start_time else ''
    end = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S') if end_time else ''
    time_diff = end_time - start_time if (start_time and end_time) else None
    duration = str(datetime.timedelta(seconds=int(time_diff))) if time_diff else ''
    audit_row = dict(
        event=x.get('event'),
        task=x.get('task'),
        sub_task=x.get('sub_task'),
        code=x.get('code'),
        form_id=rec_form_id,
        tags=x.get('tags'),
        error_msg=x.get('error_msg'),
        start=start,
        end=end,
        duration=duration,
    )
    return audit_row


@ui.page('/audit')
def load_audit_page():
    audit_records = get_all_audit()
    current_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    audit_rows = []
    for x in audit_records:
        audit_row = get_formatted_audit_row(x=x)
        audit_rows.append(audit_row)
    # Tabs for All and 'Form process only'
    with ui.tabs().props('align=left').classes('w-full bg-blue text-white') as tabs:
        all_tab = ui.tab(f'Audit Records (as of {current_time})')
        fpo_tab = ui.tab('Form processes')
    with ui.tab_panels(tabs, value=fpo_tab).classes('w-full'):
        with ui.tab_panel(all_tab):
            ui.table(
                columns=audit_columns,
                rows=audit_rows,
                row_key='event',
                pagination=15
            ).props('dense table-header-class="text-blue"')
        with ui.tab_panel(fpo_tab):
            ui.table(
                columns=audit_columns,
                rows=[x for x in audit_rows if x['event'] == 'form_process'],
                row_key='event',
                pagination=15
            ).props('dense table-header-class="text-blue" title-class="text-green"')
