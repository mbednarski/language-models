import itertools
import logging
from pathlib import Path

import cleantext
import langdetect
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import swifter
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from whatthelang import WhatTheLang

logger = logging.getLogger(__name__)


def clean_text(text: str):
    return cleantext.clean(
        text,
        fix_unicode=True,  # fix various unicode errors
        to_ascii=True,  # transliterate to closest ASCII representation
        lower=True,  # lowercase text
        no_line_breaks=False,  # fully strip line breaks as opposed to only normalizing them
        no_urls=True,  # replace all URLs with a special token
        no_emails=True,  # replace all email addresses with a special token
        no_phone_numbers=True,  # replace all phone numbers with a special token
        no_numbers=False,  # replace all numbers with a special token
        no_digits=False,  # replace all digits with a special token
        no_currency_symbols=False,  # replace all currency symbols with a special token
        no_punct=False,  # fully remove punctuation
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en",  # set to 'de' for German special handling
    )


def main(
    dump_file: Path,
    output_file,
    corpus_file: Path,
    limit=None,
    sample_size=None,
    random_state=None,
):
    logger.info(f'Reading file {dump_file}')
    df_github = pd.read_csv(dump_file, nrows=limit)

    if sample_size is not None:
        logger.info(f'Selecting {sample_size} random samples...')
        df_github = df_github.sample(sample_size)

    wtl = WhatTheLang()
    bodies = df_github['body'].tolist()
    batch_size = 512
    list_df = [bodies[i : i + batch_size] for i in range(0, len(bodies), batch_size)]
    langs = []

    logger.info('Recognizing language...')
    for df in tqdm(list_df):
        langs += wtl.predict_lang(df)

    df_github['lang'] = langs

    df_english = df_github[df_github['lang'] == 'en']
    df_english['body'] = df_english['body'].swifter.apply(clean_text)

    df_lm = pd.DataFrame({'issue_body': df_english['body']})

    logger.info(f'Saving {len(df_lm)} rows')

    df_lm.to_json(output_file, orient='records', lines=True)
    corpus_file.write_text(
        '\n'.join(df_english['body']), encoding='utf8', errors='strict'
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(
        Path('data/raw/github/github_issues.csv'),
        output_file=Path('data/interim/github.jsonl'),
        corpus_file=Path('data/interim/github.txt'),
        # limit=500_000,
        sample_size=100_000,
    )
