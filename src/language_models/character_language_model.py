import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.nn.utils.rnn as ru
from language_models.dataset.word import WordDataset
from torch.utils.data import random_split, DataLoader

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from argparse import Namespace
from language_models.tokenizer import FixedTokenizer


class CharacterLanguageModel(pl.LightningModule):
    def __init__(self, hparams, padding_token):
        super(CharacterLanguageModel, self).__init__()

        self.hparams = hparams
        vocab_size = hparams.vocab_size

        self.embed = nn.Embedding.from_pretrained(torch.eye(vocab_size, vocab_size))
        self.rnn = nn.GRU(input_size=vocab_size, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, vocab_size)

        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=padding_token)

    def forward(self, x, hidden_state=None):
        x.rename_(None)
        e = self.embed(x)
        out, hidden = self.rnn(e, hidden_state)
        out.rename_('L', 'B', 'H')
        y = self.fc(out)
        y.rename_('L', 'B', 'V')

        return y, hidden

    def training_step(self, batch, batch_idx):
        x, y = batch

        y = y.t()
        y.rename_(None)
        # y: BxL

        output, _ = self.forward(x)
        output.rename_(None)
        output = output.permute(1,2,0)
        # output ('B', 'V', 'L')

        # softmax along V dimension
        probas = output.softmax(dim=1)
        probas.rename_('B', 'V', 'L')

        loss = self.criterion(output, y)

        probas.rename_(None)
        log_perp = 0
        for seq_id in range(output.shape[2]):
            y_squeezed = y.squeeze()
            out_squeezed = y.squeeze()

            char_prob = probas[0, y_squeezed[seq_id], seq_id]

            # word_pp = torch.log(probas[0,y[0][idx],idx])
            log_perp += torch.log(char_prob)

        perp_from_manual = torch.exp(log_perp)
        perp_from_manual = torch.pow(perp_from_manual, -1/output.shape[2])
        perp_from_loss = torch.exp(loss)

        logs = {'train_loss': loss,
        'loss_perp': perp_from_loss,
        'manual_perp': perp_from_manual
            }
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        output, _ = self.forward(x)
        output.rename_(None)
        output = output.permute(1,2,0)
        # output ('B', 'V', 'L')

        y = y.t()
        y.rename_(None)

        loss = self.criterion(output, y)

        logs = {'val_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}

        return {'loss': avg_loss, 'log': logs}

    def configure_optimizer(self):
        return optim.Adam(self.parameters())

    def on_save_checkpoint(self, checkpoint):
        return super().on_save_checkpoint(checkpoint)



if __name__ == '__main__':
    tokenizer = FixedTokenizer()
    dset = WordDataset(tokenizer=tokenizer)

    train_size = int(0.8 * len(dset))
    train_dset, val_dset = random_split(dset, [train_size, len(dset) - train_size])

    train_loader = DataLoader(train_dset, batch_size=16, shuffle=True, collate_fn=tokenizer.collate_sequences)
    val_loader = DataLoader(val_dset, batch_size=16, shuffle=False, collate_fn=tokenizer.collate_sequences)

    n = Namespace()
    n.vocab_size = tokenizer.get_vocab_size()

    model = CharacterLanguageModel(n, padding_token=tokenizer.tok2idx[tokenizer.PAD_TOKEN])

    trainer = pl.Trainer(
        gradient_clip=1.0,
        accumulate_grad_batches=16,
        overfit_pct=0.1,
        max_epochs=101,
        early_stop_callback=EarlyStopping(
            monitor='val_loss', strict=True, verbose=True, patience=10
        ),
        checkpoint_callback=ModelCheckpoint('models/{epoch}', verbose=True),
    )

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
