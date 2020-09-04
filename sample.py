from language_models.lm_sampler import LanguageModelSampler
from language_models.tokenizer import FixedTokenizer
from language_models.character_language_model import CharacterLanguageModel

if __name__ == '__main__':

    model = CharacterLanguageModel.load_from_checkpoint('models/epoch=38.ckpt')
    model.eval()
    model.freeze()

    tokenizer = FixedTokenizer(
        allowed_tokens=list(
            'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZĄąĆćĘęŁłŃńÓóŚśŹźŻż -."'
        )
    )
    sampler = LanguageModelSampler(model, tokenizer, strategy='proba')

    while True:
        seed = input('Provide seed: ')
        print(''.join(sampler.sample_sentence(strategy='proba', seed=seed)))
