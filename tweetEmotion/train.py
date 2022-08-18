import pandas as pd
import datasets as ds
import torch.nn as nn
import numpy as np
from datasets import load_metric
import math
import os
import pickle
import datasets
from sklearn.metrics import f1_score
from transformers import DistilBertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import DistilBertModel
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel
from sklearn.model_selection import StratifiedKFold
import logging
import torch
import random
import gc
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)



# hyper-parameters
gradient_accumulation_steps = 1
max_length = 200
# train
num_train_epochs = 3
train_batch_size = 32
eval_batch_size = 64
batch_size=64
# optimizer
learning_rate = 5e-5
weight_decay = 1e-2
epsilon=1e-8
# scheduler
scheduler_type= 'linear'
warmup_ratio = 0.1
num_warmup_steps=10
# evaluate
nfold=4
no_classes=6
#output_dir="output"
model_ckpt = "distilbert-base-uncased"
data_path="/content/tweet_emotion.csv"
seed=2022

data=pd.read_csv(data_path)

def create_folds(df):
  df['fold']=-99
  skf=StratifiedKFold(n_splits=nfold,random_state=seed,shuffle=True)
  for i,(train_index, test_index) in enumerate(skf.split(df, df['label'])):
      df.loc[test_index,'fold']=i
create_folds(data)


def setting_seed(seed):
  np.random.seed(seed)
  set_seed(seed)
  torch.manual_seed(seed)
  random.seed(seed)
setting_seed(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

distilbert_config = AutoConfig.from_pretrained(model_ckpt,num_labels=no_classes)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt, use_fast=True)

with open("tokenizer_distilbert.pickle","wb") as file:
  pickle.dump(tokenizer,file)

class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.init_weights()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.
        """

        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,**kwargs):

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,**kwargs)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
              loss_fct =nn.CrossEntropyLoss()
              loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )

def prepare_data(examples):
  tokenized_input=tokenizer(examples['text'],truncation=True,padding="max_length",max_length=max_length,)
  tokenized_input['labels']=examples['label']
  return tokenized_input

def compute_metrics(y_true,y_preds):
  return f1_score(y_true,y_preds,average='weighted')


def main():
  logger=get_logger(__name__)
  for i in range(nfold):
    accelerator=Accelerator()
    model=DistilBertForSequenceClassification.from_pretrained(model_ckpt, config=distilbert_config).to(device)
    data_train=data[data['fold']!=i]
    data_val=data[data['fold']==i]
    data_train.drop(['label_name','fold'],axis=1,inplace=True)
    data_val.drop(['label_name','fold'],axis=1,inplace=True)


    data_train=datasets.Dataset.from_pandas(data_train)
    data_train=data_train.map(prepare_data,batch_size=batch_size,remove_columns=['text','label'])

    data_val=datasets.Dataset.from_pandas(data_val)
    data_val=data_val.map(prepare_data,batch_size=batch_size,remove_columns=['text','label'])

    train_dataloader = DataLoader(
        data_train, shuffle=True, collate_fn=default_data_collator, batch_size=train_batch_size)
    eval_dataloader = DataLoader(data_val, collate_fn=default_data_collator, batch_size=eval_batch_size)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) /gradient_accumulation_steps)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch


    lr_scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Prepare everything with the `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(data_train)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    print(f"Fold:{i}")
    for epoch in range(num_train_epochs):
        model.train()
        total_train_loss=0
        val_loss=0
        for step, batch in enumerate(train_dataloader):
            for key,value in batch.items():
                batch[key]=value.to(device)

            outputs = model(batch['input_ids'],batch['attention_mask'],batch['labels'])
            loss = outputs.loss
            # We keep track of the loss at each epoch
            total_train_loss += loss.detach().float()

            loss = loss /gradient_accumulation_steps
            accelerator.backward(loss)
            if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        preds=[]
        y_true=[]
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(batch['input_ids'],batch['attention_mask'],batch['labels'])
            predictions = outputs.logits.argmax(dim=-1)
            predictions=predictions.cpu().numpy().squeeze().tolist()
            preds.extend(predictions)
            target=batch['labels'].cpu().numpy().squeeze().tolist()
            y_true.extend(target)
            val_loss+=outputs.loss.detach().float()

        print(f"epoch:{epoch} train_loss:{total_train_loss},validation_loss:{val_loss}")
        print(f"Validation f1 score:{compute_metrics(y_true,preds)}")
    # save the models to the directory
    #output_dir=os.path.join(output_dir,f"checkpoint_fold-{i}")
    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    #torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
    #distilbert_config.save_pretrained(output_dir)
    #tokenizer.save_pretrained(output_dir)
    with open(f'model_{i}.pickel', 'wb') as file:
      pickle.dump(model, file)
    del model
    del train_dataloader
    del eval_dataloader
    del data_train
    del data_val
    del optimizer
    gc.collect()

if __name__ == "__main__":
    main()
