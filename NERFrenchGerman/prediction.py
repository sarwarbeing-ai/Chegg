import numpy as np
import pandas as pd
import pickle
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel,RobertaPreTrainedModel
from transformers import XLMRobertaConfig
import torch


with open("/content/tokenizer-xlmroberta.pickle","rb") as file:
  tokenizer=pickle.load(file)

with open("/content/config-xlmr.pickle","rb") as file:
  config=pickle.load(file)

with open("/content/index2tag.pickle","rb") as file:
  index2tag=pickle.load(file)

model_path="/content/xlm-roberta-ner"


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



model=XLMRobertaForTokenClassification.from_pretrained(model_path,config=config)

def prediction(text):
  tokenized_input=tokenizer(text,truncation=True,return_tensors="pt")
  input_ids = tokenized_input["input_ids"]
  attention_mask = tokenized_input["attention_mask"]
  with torch.no_grad():
    # Pass data through model
    output = model(input_ids, attention_mask)
  predictions = np.argmax(output.logits,axis=-1)
  predictions=predictions.squeeze()[1:-1] # don't consider the start and end of the sentence token
  predicted_label=[index2tag[label.item()] for label in predictions]
  return tokenized_input.tokens()[1:-1],predicted_label


if __name__=="__main__":
  text="Jeff Dean works at Google!"
  tokens,predicted_label=prediction(text)
  print("Tokens:",tokens)
  print()
  print("NER tags:",predicted_label)
