import torch
import transformers
import numpy as np
import pickle

model1_file="/content/model_0.pickel"
model2_file="/content/model_1.pickel"
model3_file="/content/model_2.pickel"
model4_file="/content/model_4.pickel"
tokenizer_file="/content/tokenizer_distilbert.pickle"

with open(model1_file,"rb") as file:
  model1=pickle.load(file)

with open(model2_file,"rb") as file:
  model2=pickle.load(file)

with open(model3_file),"rb") as file:
  model3=pickle.load(file)

with open(model4_file,"rb") as file:
  model4=pickle.load(file)
with open(tokenizer_file ,"rb") as file:
  tokenizer=pickle.load(file)

labels=['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_raw_text_and_prediction(text):
  tokenized_text=tokenizer(text,return_tensors='pt',truncation=True,padding="max_length",max_length=200)
  input_ids=tokenized_text['input_ids'].to(device)
  attention_mask=tokenized_text['attention_mask'].to(device)
  logits1=model1(input_ids,attention_mask,None).logits.detach().cpu().numpy()
  logits2=model2(input_ids,attention_mask,None).logits.detach().cpu().numpy()
  logits3=model3(input_ids,attention_mask,None).logits.detach().cpu().numpy()
  logits4=model4(input_ids,attention_mask,None).logits.detach().cpu().numpy()
  logits=np.mean(np.stack([logits1,logits2,logits3,logits4]),axis=0)
  preds=logits.argmax(-1)[0]
  prediction=labels[preds]
  print(f"Sentiment:{prediction},Confidence Score:{(np.exp(logits)/np.sum(np.exp(logits))).max()}")
 
if __name__=="__main__":
    text="Tesla car is awesome!!"
    process_raw_text_and_prediction(text) # Sentiment:joy,Confidence Score:0.9761346578598022
