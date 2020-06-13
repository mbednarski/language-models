from pathlib import Path

import tokenizers


def train_tokenizer(corpus_file: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = tokenizers.CharBPETokenizer(lowercase=True)
    tokenizer.train(files=str(corpus_file), special_tokens=['<pad>', '<s>', '</s>'])

    tokenizer.save(str(output_dir))


def get_tokenizer(saved_dir: Path):
    tokenizer = tokenizers.CharBPETokenizer(
        lowercase=True,
        vocab_file=str(saved_dir / 'vocab.json'),
        merges_file=str(saved_dir / 'merges.txt'),
    )
    tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>")),
    )

    tokenizer.enable_padding(pad_token='<pad>')
    return tokenizer


if __name__ == '__main__':
    train_tokenizer(Path('data/interim/github.txt'), output_dir=Path('bpe_tokenizer'))
