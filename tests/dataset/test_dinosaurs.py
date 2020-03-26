from language_models.dataset.dinosaurs import CharacterVocabulary

import torch
from torch.testing import assert_allclose

def test_encode():
    vocab = CharacterVocabulary()

    text = 'trex'
    encoded = vocab.encode(text)

    assert encoded.shape == (4,)
    assert_allclose(encoded, torch.LongTensor([21,19,6,25]))

def test_encode_special():
    vocab = CharacterVocabulary()

    text = 'trex'
    encoded = vocab.encode(text, include_special_tokens=True)

    assert encoded.shape == (6,)
    assert_allclose(torch.LongTensor([0,21,19,6,25,1]), encoded)

def test_decode():
    vocab = CharacterVocabulary()

    encoded = torch.LongTensor([0,21,19,6,25,1])
    decoded = vocab.decode(encoded)

    assert vocab.SOS_TOKEN + "trex" + vocab.EOS_TOKEN == decoded