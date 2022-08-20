from datasets import(
    get_dataset_config_names,
    load_dataset,
    DatasetDict,
    concatenate_datasets)
import pickle
import os
import torch
import random
from collections import defaultdict
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


seed=2022
model_ckpt="xlm-roberta-base"
langs = ["de", "fr", "it", "en"]
fracs = [0.629, 0.229, 0.084, 0.059] # fraction of samples from each dataset


panx_ch = defaultdict(DatasetDict)
for lang, frac in zip(langs, fracs):
  # Load monolingual corpus
  ds = load_dataset("xtreme", name=f"PAN-X.{lang}")
  # Shuffle and downsample each split according to spoken proportion
  for split in ds:
    panx_ch[lang][split] = ds[split].shuffle(seed=2022).select(range(int(frac * ds[split].num_rows)))


tags = panx_ch["de"]["train"].features["ner_tags"].feature
def create_tag_names(batch):
  return {"ner_tags_str": [tags.int2str(idx) for idx in batch["ner_tags"]]}

index2tag={idx:tag for idx,tag in enumerate(tags.names)}
tag2index={tag:idx for idx,tag in enumerate(tags.names)}

tokenizer=AutoTokenizer.from_pretrained(model_ckpt)

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

panx_de_fr_encoded = concatenate_splits([panx_de_encoded,panx_fr_encoded])

# pickle files
with open("panx-de.pickle","wb") as file:
  pickle.dump(panx_de_encoded,file)

with open("panx-en.pickle","wb") as file:
  pickle.dump(panx_en_encoded,file)

with open("panx-fr.pickle","wb") as file:
  pickle.dump(panx_fr_encoded,file)
with open("panx-it.pickle","wb") as file:
  pickle.dump(panx_it_encoded,file)


# pickel index2tag
with open("index2tag.pickle","wb") as file:
  pickle.dump(index2tag,file)

with open("tag2index.pickle","wb") as file:
  pickle.dump(tag2index,file)
