import json
from glob import glob

import pandas as pd
from box import Box
from dynaconf import settings
from tqdm import tqdm

from aiassistant.log import get_logger

logger = get_logger(__name__)


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
    print(df3.to_markdown())
