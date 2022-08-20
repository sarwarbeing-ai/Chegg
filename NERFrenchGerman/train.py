import pickle
import os
import torch
import random
import torch.nn as nn
from collections import defaultdict
from datasets import DatasetDict,concatenate_datasets,load_dataset
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer,
    XLMRobertaConfig,
    AutoConfig,
    TrainingArguments,
    Trainer,
    set_seed)
from sklearn.metrics import f1_score

from transformers import DataCollatorForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel,RobertaPreTrainedModel
from seqeval.metrics import classification_report,f1_score
from transformers import DataCollatorForTokenClassification


# hyper-parameters
gradient_accumulation_steps = 1
# train
num_train_epochs = 3
train_batch_size = 24
eval_batch_size = 24
# optimizer
learning_rate = 5e-5
weight_decay = 1e-2
epsilon=1e-8
# scheduler
scheduler_type= 'linear'
warmup_ratio = 0.1
num_warmup_steps=10
#output_dir="output"
output_dir="xlm-roberta-ner"
output_dir_model="xlm-roberta-ner"
seed=2022
no_classes=7
model_ckpt = "xlm-roberta-base"
langs = ["de", "fr", "it", "en"]
fracs = [0.629, 0.229, 0.084, 0.059] # fraction of samples from each dataset



with open("/content/index2tag.pickle","rb") as file:
  index2tag=pickle.load(file)
with open("/content/tag2index.pickle","rb") as file:
  tag2index=pickle.load(file)

def setting_seed(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  random.seed(seed)
  set_seed(seed)
setting_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

xlmr_config = AutoConfig.from_pretrained(model_ckpt,num_labels=no_classes,id2label=index2tag,label2id=tag2index)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt,use_fast=True)

data_collator = DataCollatorForTokenClassification(tokenizer)

with open("tokenizer-xlmroberta.pickle","wb") as file:
  pickle.dump(tokenizer,file)
with open("config-xlmr.pickle","wb") as file:
  pickle.dump(xlmr_config,file)


panx_ch = defaultdict(DatasetDict)
for lang, frac in zip(langs, fracs):
  # Load monolingual corpus
  ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
  # Shuffle and downsample each split according to spoken proportion
  for split in ds:
    panx_ch[lang][split] = ds[split].shuffle(seed=2022).select(range(int(frac * ds[split].num_rows)))

def tokenize_and_align_labels(examples):
  tokenized_inputs = tokenizer(examples["tokens"],truncation=True,is_split_into_words=True)
  labels = []
  for idx, label in enumerate(examples["ner_tags"]):
    word_ids = tokenized_inputs.word_ids(batch_index=idx)
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
      if word_idx is None or word_idx == previous_word_idx:
        label_ids.append(-100)
      else:
        label_ids.append(label[word_idx])
      previous_word_idx = word_idx
    labels.append(label_ids)
  tokenized_inputs["labels"] = labels
  return tokenized_inputs



def encode_panx_dataset(corpus):
  return corpus.map(tokenize_and_align_labels, batched=True,remove_columns=['langs', 'ner_tags', 'tokens'])

panx_de_encoded = encode_panx_dataset(panx_ch["de"])
panx_en_encoded = encode_panx_dataset(panx_ch["en"])
panx_fr_encoded = encode_panx_dataset(panx_ch["fr"])
panx_it_encoded = encode_panx_dataset(panx_ch["it"])

def concatenate_splits(corpora):
  multi_corpus = DatasetDict()
  corpus=corpora[0]
  keys=corpus.keys()
  for split in keys:
      multi_corpus[split] = concatenate_datasets([corpus[split] for corpus in corpora]).shuffle(seed=seed)
  return multi_corpus


panx_de_fr_en_it_encoded = concatenate_splits([panx_de_encoded,panx_fr_encoded,panx_en_encoded,panx_it_encoded])


class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
  config_class = XLMRobertaConfig
  def __init__(self, config):
    super().__init__(config)
    self.num_labels = config.num_labels
    # Load model body
    self.roberta = RobertaModel(config, add_pooling_layer=False)
    # Set up token classification head
    self.dropout = nn.Dropout(config.hidden_dropout_prob)
    self.classifier = nn.Linear(config.hidden_size,config.num_labels)
    # Load and initialize weights
    self.init_weights()
  def forward(self, input_ids=None, attention_mask=None,
              token_type_ids=None,
              labels=None, **kwargs):
    # Use model body to get encoder representations
    outputs = self.roberta(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, **kwargs)# Apply classifier to encoder representation
    sequence_output = self.dropout(outputs[0])
    logits = self.classifier(sequence_output)
    # Calculate losses
    loss = None
    if labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      loss = loss_fct(logits.view(-1, self.num_labels),labels.view(-1))
    # Return model output object
    return TokenClassifierOutput(loss=loss, logits=logits,
                                hidden_states=outputs.hidden_states,
                                attentions=outputs.attentions)


def align_predictions(predictions, label_ids):
  preds = np.argmax(predictions, axis=2)
  batch_size, seq_len = preds.shape
  labels_list, preds_list = [], []
  for batch_idx in range(batch_size):
    example_labels, example_preds = [], []
    for seq_idx in range(seq_len):
      # Ignore label IDs = -100
      if label_ids[batch_idx, seq_idx] != -100:
        example_labels.append(index2tag[label_ids[batch_idx][seq_idx]])
        example_preds.append(index2tag[preds[batch_idx][seq_idx]])
    labels_list.append(example_labels)
    preds_list.append(example_preds)
  return preds_list, labels_list


def compute_metrics(eval_pred):
  y_pred, y_true = align_predictions(eval_pred.predictions,eval_pred.label_ids)
  return {"f1": f1_score(y_true, y_pred)}

def model_init():
  return (XLMRobertaForTokenClassification.from_pretrained(model_ckpt, config=xlmr_config).to(device))

def main():
  training_args=TrainingArguments( output_dir=output_dir,
                                  evaluation_strategy="epoch",
                                  per_device_train_batch_size=train_batch_size,
                                  per_device_eval_batch_size=eval_batch_size,
                                  gradient_accumulation_steps=gradient_accumulation_steps,
                                  learning_rate=learning_rate,
                                  weight_decay=weight_decay,
                                  adam_epsilon=epsilon,
                                  num_train_epochs=num_train_epochs,
                                  lr_scheduler_type=scheduler_type,
                                  warmup_ratio=warmup_ratio,
                                  warmup_steps=num_warmup_steps,
                                  log_level="error",
                                  save_strategy="no",
                                  save_steps=1e6,
                                  seed=seed,
                                  disable_tqdm=False)
  trainer=Trainer(model_init=model_init,args=training_args,
                  data_collator=data_collator,
                  compute_metrics=compute_metrics,
                  train_dataset=panx_de_fr_en_it_encoded["train"],
                  eval_dataset=panx_de_fr_en_it_encoded["validation"],
                  tokenizer=tokenizer)
  return trainer


if __name__=="__main__":
  trainer=main()
  trainer.train()
  trainer.save_model(output_dir_model)
