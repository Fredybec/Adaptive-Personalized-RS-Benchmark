import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import os
import json

data = []
with open('../data/amazon/amazon.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

amazon_df = pd.DataFrame(data)

amazon_df = amazon_df.dropna(subset=['asin', 'title', 'category'])

amazon_df.fillna('', inplace=True)

amazon_df['metadata'] = amazon_df[['title', 'category']].apply(lambda x: ' '.join(x), axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(amazon_df['metadata'])

svd = TruncatedSVD(n_components=25, random_state=42)
latent_matrix = svd.fit_transform(tfidf_matrix)

explained = svd.explained_variance_ratio_.cumsum()
plt.plot(explained, '.-', ms=10, color='red')
plt.xlabel('Singular value components')
plt.ylabel('Cumulative variance explained')
plt.title('Explained Variance by SVD Components')
plt.show()

latent_matrix = preprocessing.scale(latent_matrix)

os.makedirs('Dataset/processed_data', exist_ok=True)
with open('Dataset/processed_data/Amazon_FeatureVectors.dat', 'w') as f:
    f.write('asin\tFeatureVector\n')
    for i, asin in enumerate(amazon_df['asin']):
        featureVector = ';'.join([f"{x:.6f}" for x in latent_matrix[i]])
        f.write(f"{asin}\t{featureVector}\n")
