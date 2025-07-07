#this script was used in order to transform the asin and user_id from string to int 
import pandas as pd
import json

 
data1 = []
with open('amazon/amazon.jsonl', 'r') as f:
    for line in f:
        data1.append(json.loads(line.strip()))

df = pd.DataFrame(data1)
df = df.dropna(subset=['asin', 'user_id', 'title', 'category']).fillna('')


asin_to_aid = {asin: i for i, asin in enumerate(df['asin'].unique())}
user_to_uid = {uid: i for i, uid in enumerate(df['user_id'].unique())}

df['asin'] = df['asin'].map(asin_to_aid).astype(int)
df['user_id'] = df['user_id'].map(user_to_uid).astype(int)

with open('amazon.jsonl', 'w') as fout:
    for _, row in df.iterrows():
        json_record = row.to_dict()
        fout.write(json.dumps(json_record) + '\n')

print("Saved amazon.jsonl with integer asin and user_id")