import torch
from torchtext import data 
import random
import config
seed = config.seed
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def load_data(file_path):
    '''
    load and prepare dataset to training and validation iterator
    '''
    TEXT = data.Field(tokenize="spacy", batch_first=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float, batch_first=True)

    fields = [(None,None), ('text',TEXT), ('label',LABEL)]

    total_data = data.TabularDataset(path=file_path,
                                    format="csv", 
                                    fields=fields,
                                    skip_header=True)
    # split data
    train_data, valid_data = total_data.split(split_ratio=0.7, random_state = random.seed(seed))
    # initialize glove embeddings
    TEXT.build_vocab(train_data, min_freq=3, vectors="glove.6B.100d")
    LABEL.build_vocab(train_data)

    #No. of unique tokens in text
    print("Size of TEXT vocabulary:",len(TEXT.vocab))

    #No. of unique tokens in label
    # print("Size of LABEL vocabulary:",len(LABEL.vocab))

    #Commonly used words
    # print(TEXT.vocab.freqs.most_common(10))

    #Word dictionary
    # print(TEXT.vocab.stoi)
    batch_size = config.BATCH_SIZE
    device = config.device
    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data),
        batch_size = batch_size,
        sort_key = lambda x: len(x.text),
        sort_within_batch = True,
        device=device
    )
    
    return train_iterator, valid_iterator, TEXT, LABEL

# define metric
def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc