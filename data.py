import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random


def spacy_models():
    en = spacy.load('en')
    de = spacy.load('de')
    return en,de


def tokenize_en(text):
    en,de = spacy_models()
    return [tok.text for tok in de.tokenizer(text)][::-1]


def tokenize_de(text):
    en, de = spacy_models()
    return [tok.text for tok in en.tokenizer(text)]


def get_data():

    SEED = 1234

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    SRC = Field(tokenize=tokenize_de,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 128
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    PAD_IDX = TRG.vocab.stoi['<pad>']

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)
    return train_iterator, valid_iterator, test_iterator, INPUT_DIM, OUTPUT_DIM, PAD_IDX
