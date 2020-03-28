from language_models.lm_sampler import LanguageModelSampler
from language_models.character_language_model import CharacterLanguageModel
from language_models.vocabulary import CharacterVocabulary

if __name__ == '__main__':

    model = CharacterLanguageModel.load_from_checkpoint('models/epoch=100.ckpt')
    model.eval()
    model.freeze()

    vocab = CharacterVocabulary()
    sampler = LanguageModelSampler(model, vocab, strategy='greedy')

    while True:
        seed = input('Provide seed: ')
        print(sampler.sample_sentence(seed=seed))
