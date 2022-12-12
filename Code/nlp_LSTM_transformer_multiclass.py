import datasets
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoConfig
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from datasets import load_metric
import evaluate
import pandas as pd
import numpy as np
import torch.nn as nn



# read the file
df = pd.read_csv('df_short.csv')
print(df.info())


# ---- Data pre-processing ------
class2idx = {
    1:0,
    2:1,
    3:2,
    4:3,
    5:4
}

idx2class = {v: k for k, v in class2idx.items()}

df['labels'] = df['reviews.rating'].replace(class2idx, inplace=False)
# print(df.labels.value_counts())


df = df[['reviews.text','labels']]
#df_train = df[:3500]
#df_test = df[3500:]

df_train = df[:100]
df_test = df[100:200]

# creating dataset objects for transformers
dataset_train = datasets.Dataset.from_pandas(df_train)
dataset_test = datasets.Dataset.from_pandas(df_test)
data_dict = datasets.DatasetDict({"train":dataset_train, "test":dataset_test})

# define the transformer base model, tokenization, and data collator
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["reviews.text"], truncation=True)

tokenized_datasets = data_dict.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tokenized_datasets = tokenized_datasets.remove_columns(["reviews.text"])
tokenized_datasets.set_format("torch")
print("Columns in train data:",tokenized_datasets["train"].column_names)


# data loader
train_dataloader = DataLoader(tokenized_datasets["train"],
                              shuffle=True, batch_size=8, collate_fn=data_collator)
eval_dataloader = DataLoader(tokenized_datasets["test"],
                             batch_size=8, collate_fn=data_collator)

# helper function to convert tuple of tensors to tensor
def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return  torch.stack(list(tuple_of_tensors), dim=0)

# ---- Build LSTM Model ----
class LSTM_Model(nn.Module):
    def __init__(self, checkpoint, num_labels, input_dim, hidden_dim):
        super(LSTM_Model, self).__init__()
        self.num_labels = num_labels

        # Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(checkpoint,
                                               config=AutoConfig.from_pretrained(checkpoint,
                                                                                 output_attentions=True,
                                                                                 output_hidden_states=True))
        self.rnn = nn.LSTM(input_dim,
                           hidden_dim,
                           batch_first=True
                           )
        self.fc = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        # Add custom layers
        sequence_output = outputs[0] # outputs[0]=last hidden state

        logits, (hidden, cell) = self.rnn(sequence_output)
        hidden = tuple_of_tensors_to_tensor(hidden)

        return self.fc(hidden.squeeze(0))

# --- Pre-training setup ---
num_epochs = 5
num_training_steps = num_epochs * len(train_dataloader)
# metric = load_metric("glue",'mnli') # accuracy
metric = evaluate.load('accuracy')
# metric = evaluate.load('f1')
model = LSTM_Model(checkpoint, num_labels=5, input_dim=768, hidden_dim=128)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler( "linear", optimizer=optimizer,
                              num_warmup_steps=0, num_training_steps=num_training_steps)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
criterion = nn.CrossEntropyLoss()
model.to(device)

# --- Training & Evaluation ---
progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epochs * len(eval_dataloader)))

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch).squeeze(1)
        loss = criterion(outputs, batch['labels'])
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar_train.update(1)

    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch).squeeze(1)

        predictions = torch.argmax(outputs, dim=-1) # return the index of the max logits
        metric.add_batch(predictions=predictions, references=batch["labels"])
        progress_bar_eval.update(1)

    # print(metric.compute(average="weighted"))
    print(metric.compute())

