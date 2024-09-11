"""
This script creates the score graphs and metrics which shows performance of AI assistant tool
on different forms and when different LLM are used in the Auto QA stage.
"""
import json
import time
from glob import glob
from pathlib import Path

import matplotlib.colors as mcolors
import numpy as np
import pandas
import pandas as pd
from box import Box
from dynaconf import settings
from matplotlib import pyplot as plt
from tqdm import tqdm

from aiassistant.log import get_logger

logger = get_logger(__name__)
colours = list(mcolors.TABLEAU_COLORS.keys())


def save_scores(df, report_dir, by='model_qa') -> list[dict]:
    """Save score-model graph for judge model or QA model and calculate mean and standard deviation
    for each form-model combination.

    Args:
        df: Dataframe with question, answer, score and form metadata
        report_dir: Report directory
        by: model_qa or model_judge

    Returns:
        list of dicts with form, model, mean score and standard deviation of score for each form-model combination.

    """
    df12 = df[[by, 'form_name', 'form_id', 'question', 'score_int']]
    df31 = df12[[by, 'form_name', 'score_int']]
    df31['form_model'] = df31['form_name'] + ':' + df31[by]
    df32 = df31.drop([by, 'form_name'], axis=1)
    form_model_list = df32['form_model'].unique()
    fig, ax = plt.subplots()
    mean_sd_data = []
    max_x = 5
    for index, form_model in enumerate(form_model_list):
        df33 = df32.query(f'form_model == "{form_model}"')
        scores = df33['score_int'].to_list()
        mean_score = np.mean(scores)
        sd_score = np.std(scores)
        df40 = df33['score_int'].expanding().mean().reset_index()
        Y = df40['score_int'].to_list()
        X = list(range(1, len(Y) + 1))
        ax.plot(X, Y, marker='.',
                color=colours[index - 1],
                label=form_model
                )
        ax.annotate(f'{round(Y[-1], 2)}',
                    xy=(X[-1], Y[-1]),
                    xytext=(X[-1] + 1, Y[-1]),
                    ),
        mean_sd_data.append(dict(
            form_model=form_model,
            mean=mean_score,
            sd=sd_score
        ))
        max_x = max(max_x, max(X))
    # ax.set_xticks(range(max_x+5))
    ax.set_xlabel('Number of questions')
    ax.set_ylabel('Mean score')
    ax.legend()
    # plt.xticks(rotation=90)
    img_path = report_dir / f'score_{by}.png'
    plt.savefig(img_path, bbox_inches='tight')
    logger.info(f'Saved report part at {img_path}')
    # plt.show()
    return mean_sd_data


def get_score_df() -> pd.DataFrame:
    """Create a single pandas dataframe by collecting evaluation result json files

    Returns:
         Pandas dataframe with question, answer, score and form metadata info

    """
    df = pd.DataFrame()
    # iterate over the evaluations dir for json
    for eval_json in tqdm(glob(f'{settings.TEST_EVALUATIONS_PATH}/*.json'), desc='Processing evaluation jsons'):
        logger.info(f'Processing {eval_json}')
        eval_dict = json.load(open(eval_json, 'r'))
        eval_box = Box(eval_dict)
        form_id = eval_box.form_id
        form_name = eval_box.form_name
        model_parse = eval_box.model_parse
        model_qa = eval_box.model_qa
        model_keys = [x for x in eval_box.keys() if x.startswith('score_')]
        for model_key in model_keys:
            model_scores = eval_box[model_key]
            model_judge = model_key.replace('score_', '')
            model_scores_with_meta = [{**dict(model_judge=model_judge,
                                              model_parse=model_parse,
                                              model_qa=model_qa,
                                              form_name=form_name,
                                              form_id=form_id), **x}
                                      for x in model_scores]
            if df.empty:
                df = pd.DataFrame.from_records(model_scores_with_meta)
            else:
                df2 = pd.DataFrame.from_records(model_scores_with_meta)
                df = pd.concat([df, df2])
    df.reset_index(drop=True, inplace=True)
    return df


def setup():
    """Create report directory for the current run and return the directory path.

    Returns:
        Return the report directory path

    """
    Path(settings.TEST_REPORTS_PATH).mkdir(exist_ok=True)
    file_id = int(time.strftime('%Y%m%d%H%M%S', time.gmtime()))
    report_dir = Path(settings.TEST_REPORTS_PATH) / f'report_{file_id}'
    report_dir.mkdir()
    logger.info(f'Created report directory at: {report_dir}')
    return report_dir


def create_index_html(report_dir, model_qa_mean_sd_data: list[dict], model_judge_mean_sd_data: list[dict]) -> None:
    """Create HTML report to display scores for different forms when different judge and QA models used.

    Args:
        report_dir: Report directory
        model_qa_mean_sd_data: Mean and standard deviation data when different QA models used
        model_judge_mean_sd_data: Mean and standard deviation data when different judge models used

    Returns:
        None

    """
    index_html_path = report_dir / 'index.html'
    model_qa_df = pandas.DataFrame.from_records(model_qa_mean_sd_data)
    model_qa_df.index = model_qa_df.index + 1
    model_judge_df = pandas.DataFrame.from_records(model_judge_mean_sd_data)
    model_judge_df.index = model_judge_df.index + 1
    timestamp = report_dir.stem.replace('report_','')
    report_html = settings.REPORT_INDEX_TEMPLATE.format(
        timestamp=str(timestamp),
        model_qa_df=model_qa_df.to_html(),
        model_judge_df=model_judge_df.to_html()
    )
    index_html_path.write_text(report_html)
    logger.info(f'Saved report index file at: {index_html_path}')


def main():
    report_dir = setup()
    df3 = get_score_df()
    # print(df3.to_markdown())
    df_csv_path = report_dir / f'score_df.csv'
    df3.to_csv(df_csv_path)
    logger.info(f'Saved source dataframe for the report at: {df_csv_path}')
    df4 = df3.query('score != "0"')
    df4['score_int'] = pd.to_numeric(df4['score'], errors='coerce')
    df5 = df4.dropna()
    filtered_df_csv_path = report_dir / f'filtered_score_df.csv'
    df5.to_csv(filtered_df_csv_path)
    logger.info(f'Saved filtered source dataframe for the report at: {filtered_df_csv_path}')
    model_qa_mean_sd_data = save_scores(df=df5, report_dir=report_dir, by='model_qa')
    model_judge_mean_sd_data = save_scores(df=df5, report_dir=report_dir, by='model_judge')
    create_index_html(report_dir=report_dir,
                      model_qa_mean_sd_data=model_qa_mean_sd_data,
                      model_judge_mean_sd_data=model_judge_mean_sd_data
                      )


if __name__ == '__main__':
    main()
