import pandas as pd
import langdetect
from tqdm import tqdm
import seaborn as sns
from whatthelang import WhatTheLang
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import swifter
import itertools
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def main(dump_file: Path, output_file, limit=None, sample_size=None, random_state=None):
    logger.info(f'Reading file {dump_file}')
    df_github = pd.read_csv(dump_file, nrows=limit)

    if sample_size is not None:
        df_github = df_github.sample(sample_size, )

    wtl = WhatTheLang()
    df_github['lang'] = wtl.predict_lang(df_github['body'].tolist())
    df_english = df_github[df_github['lang'] == 'en']
    df_english['body'] = df_english['body'].str.lower()

    df_lm = pd.DataFrame({'issue_body': df_english['body']})

    df_lm.to_json(output_file, orient='records', lines=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(
        Path('data/raw/github/github_issues.csv'),
        output_file=Path('data/interim/github.jsonl'),
        limit= 1_000
    )





