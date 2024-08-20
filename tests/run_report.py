import json
from glob import glob

import matplotlib.colors as mcolors
import pandas as pd
from box import Box
from dynaconf import settings
from matplotlib import pyplot as plt
from tqdm import tqdm

from aiassistant.log import get_logger

logger = get_logger(__name__)
colours = list(mcolors.TABLEAU_COLORS.keys())


def draw_scores(df):
    df10 = df.query('score != "0"')
    df10['score_int'] = pd.to_numeric(df10['score'], errors='coerce')
    df11 = df10.dropna()
    df12 = df11[['model_name', 'form_name', 'form_id', 'question', 'score_int']]
    df31 = df12[['model_name', 'form_name', 'score_int']]
    df31['form_model'] = df31['form_name'] + ':' + df31['model_name']
    df32 = df31.drop(['model_name', 'form_name'], axis=1)
    form_model_list = df32['form_model'].unique()
    fig, ax = plt.subplots()
    for index, form_model in enumerate(form_model_list):
        df = df32.query(f'form_model == "{form_model}"')
        df40 = df['score_int'].expanding().mean().reset_index()
        Y = df40['score_int'].to_list()
        X = list(range(1, len(Y) + 1))
        ax.plot(X, Y, marker='.',
                color=colours[index - 1],
                label=form_model
                )
        ax.annotate(f'{round(Y[-1], 2)}',
                    xy=(X[-5], Y[-1]),
                    xytext=(X[-1]+1, Y[-1]),
                    ),
    ax.set_xlabel('Number of questions')
    ax.set_ylabel('Mean score')
    ax.legend()
    plt.show()


def main():
    df = pd.DataFrame()
    # iterate over the evaluations dir for json
    for eval_json in tqdm(glob(f'{settings.TEST_EVALUATIONS_PATH}/*.json'), desc='Processing evaluation jsons'):
        logger.info(f'Processing {eval_json}')
        eval_dict = json.load(open(eval_json, 'r'))
        eval_box = Box(eval_dict)
        form_id = eval_box.form_id
        form_name = eval_box.form_name
        model_keys = [x for x in eval_box.keys() if x.startswith('score_')]
        for model_key in model_keys:
            model_scores = eval_box[model_key]
            model_name = model_key.replace('score_', '')
            model_scores_with_meta = [{**dict(model_name=model_name, form_name=form_name, form_id=form_id), **x}
                                      for x in model_scores]
            if df.empty:
                df = pd.DataFrame.from_records(model_scores_with_meta)
            else:
                df2 = pd.DataFrame.from_records(model_scores_with_meta)
                df = pd.concat([df, df2])
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == '__main__':
    df3 = main()
    draw_scores(df=df3)
    # print(df3.to_markdown())
