# import libraries
import pandas as pd
import numpy as np
import nltk
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# read the file
df = pd.read_csv('df_short.csv')
print(df.info())

df['label'] = df['reviews.doRecommend'].apply(lambda x: 1 if x == True else 0)
print(df.label.value_counts())
df_train = df[:3500]
df_test = df[3500:]


# ---- hyper parameters ---
class Args:
    def __init__(self):
        self.seq_len = "get_max_from_data"
        self.embedding_dim = 50
        self.n_neurons = (100, 200, 100)
        self.n_epochs = 10
        self.lr = 1e-2
        self.batch_size = 512
        self.dropout = 0.2
        self.freeze_embeddings = True
        self.train = True
        self.save_model = True

args = Args()


def extract_vocab_dict_and_msl(train, test): # {token: token_id}
    """ Tokenizes all the sentences and gets a dictionary of unique tokens and also the maximum sequence length """
    tokens, ms_len = [], 0
    for sentence in list(train) + list(test):
        tokens_in_sentence = nltk.word_tokenize(sentence)
        if ms_len < len(tokens_in_sentence):
            ms_len = len(tokens_in_sentence)
        tokens += tokens_in_sentence
    token_vocab = {key: i for key, i in zip(set(tokens), range(1, len(set(tokens))+1))}
    if len(np.unique(list(token_vocab.values()))) != len(token_vocab):
        "There are some rep words..."
    return token_vocab, ms_len


token_ids, msl = extract_vocab_dict_and_msl(df_train['clean_text'].str.lower(), df_test['clean_text'].str.lower() )
#print(token_ids)
#print(msl)

def get_glove_embeddings(vocab_dict): # {token index: glove_embedding}
    with open("glove.6B.50d.txt", "r") as s:
        glove = s.read()
    embeddings_dict = {}
    for line in glove.split("\n")[:-1]:
        text = line.split()
        if text[0] in vocab_dict:
            embeddings_dict[vocab_dict[text[0]]] = torch.from_numpy(np.array(text[1:], dtype="float32"))
    return embeddings_dict

def get_glove_table(vocab_dict, glove_dict):
    lookup_table = torch.empty((len(vocab_dict)+2, 50))
    for token_id in sorted(vocab_dict.values()):
        if token_id in glove_dict:
            lookup_table[token_id] = glove_dict[token_id]
        else:
            lookup_table[token_id] = torch.zeros((1, 50))  # For unknown tokens
    lookup_table[0] = torch.zeros((1, 50))
    return lookup_table

glove_embeddings = get_glove_embeddings(token_ids)
#print(glove_embeddings[1])

def convert_to_ids(raw_sentences, vocab_dict, pad_to):
    x = np.empty((len(raw_sentences), pad_to))
    for idx, sentence in enumerate(raw_sentences):
        word_ids = []
        for token in nltk.word_tokenize(sentence):
            try:
                word_ids.append(vocab_dict[token])
            except:
                word_ids.append(vocab_dict[token])
        if pad_to < len(word_ids):
            x[idx] = word_ids[:pad_to]
        else:
            x[idx] = word_ids + [0] * (pad_to - len(word_ids))
    return x


# converting all the sentences to sequences of token ids
x_train_raw = df_train['clean_text'].str.lower().values
x_train = convert_to_ids(x_train_raw, token_ids, msl)
y_train = torch.LongTensor(df_train['label'].values)

x_test_raw = df_test['clean_text'].str.lower().values
x_test = convert_to_ids(x_test_raw, token_ids, msl)
y_test = torch.LongTensor(df_test['label'].values)

x_train, x_test = torch.LongTensor(x_train), torch.LongTensor(x_test)

# %% -------------------------------------- MLP Class ------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, vocab_size, neurons_per_layer):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size+2, args.embedding_dim)
        dims = (msl*args.embedding_dim, *neurons_per_layer)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dims[i], dims[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(dims[i+1]),
                nn.Dropout(args.dropout)
            ) for i in range(len(dims)-1)
        ])
        self.layers.extend(nn.ModuleList([nn.Linear(neurons_per_layer[-1], 2)]))

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return x

model = MLP(len(token_ids), args.n_neurons)
look_up_table = get_glove_table(token_ids, glove_embeddings)

# replace weights with pre-trained glove embeddings
model.embedding.weight.data.copy_(look_up_table)

# option to freeze embeddings
if args.freeze_embeddings:
    model.embedding.weight.requires_grad = False


optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss()

# --- training MLP -----
def acc(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*accuracy_score(y.cpu().numpy(), pred_labels)

def f1(x, y, return_labels=False):
    with torch.no_grad():
        logits = model(x)
        pred_labels = np.argmax(logits.cpu().numpy(), axis=1)
    if return_labels:
        return pred_labels
    else:
        return 100*f1_score(y.cpu().numpy(), pred_labels)

if args.train:
    acc_test_best = 0
    print("Starting training loop...")
    for epoch in range(args.n_epochs):

        loss_train = 0
        model.train()
        for batch in range(len(x_train)//args.batch_size + 1):
            inds = slice(batch*args.batch_size, (batch+1)*args.batch_size)
            optimizer.zero_grad()
            logits = model(x_train[inds])
            loss = criterion(logits, y_train[inds])
            loss.backward()
            optimizer.step()
            loss_train += loss.item()

        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test)
            loss = criterion(y_test_pred, y_test)
            loss_test = loss.item()

        acc_test = acc(x_test, y_test)
        print("Epoch {} | Train Loss {:.5f}, Train Acc {:.2f} - Test Loss {:.5f}, Test Acc {:.2f}".format(
            epoch, loss_train/args.batch_size, acc(x_train, y_train), loss_test, acc_test))

        if acc_test > acc_test_best and args.save_model:
            torch.save(model.state_dict(), "mlp_sentiment_glove.pt")
            print("The model has been saved!")
            acc_test_best = acc_test

# %% ------------------------------------------ Final test -------------------------------------------------------------
model.load_state_dict(torch.load("mlp_sentiment_glove.pt"))
model.eval()
y_test_pred = acc(x_test, y_test, return_labels=True)
print("The accuracy on the test set is {:.2f}".format(100*accuracy_score(y_test.cpu().numpy(), y_test_pred), "%"))
print("The confusion matrix is")
print(confusion_matrix(y_test.cpu().numpy(), y_test_pred))

