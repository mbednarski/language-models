from hypothesis import given, assume
from hypothesis.strategies import text

from language_models.tokenizer import FixedTokenizer


def test_defaults():
    tok = FixedTokenizer()

    assert 30 == tok.get_vocab_size()
    assert 0 == tok.tok2idx[tok.PAD_TOKEN]
    assert 1 == tok.tok2idx[tok.SOS_TOKEN]
    assert 2 == tok.tok2idx[tok.EOS_TOKEN]

@given(text(alphabet=FixedTokenizer.DEFAULT_ALLOWED_TOKENS))
def test_encode_decode(s:str):
    assume(len(s) > 0)
    tok = FixedTokenizer()

    encoded = tok.encode(s, include_special_tokens=False)
    decoded = tok.decode(encoded)

    assert decoded == s

@given(text(alphabet=FixedTokenizer.DEFAULT_ALLOWED_TOKENS))
def test_encode_decode_with_special(s:str):
    tok = FixedTokenizer()

    encoded = tok.encode(s, include_special_tokens=True)
    decoded = tok.decode(encoded)

    assert decoded == tok.SOS_TOKEN + s + tok.EOS_TOKEN
