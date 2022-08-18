import datasets as ds
import pandas as pd

emotions=ds.load_dataset("emotion")

emotions.set_format("pandas")
df_train=emotions['train'][:]
df_val=emotions['validation'][:]
df_test=emotions['test'][:]
# merge the train as well as validation data
df=pd.concat([df_train,df_val,df_test],ignore_index=True)

# label names
label_names=emotions['train'].features['label'].names

# index to label
d={}
for i,name in enumerate(label_names):
  d[i]=name

# create a new column called label_name
df['label_name']=df['label'].map(d)
df.to_csv("tweet_emotion.csv",index=False)
