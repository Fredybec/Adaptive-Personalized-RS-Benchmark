import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import os

movies = pd.read_csv('../data/ml-1m/movies.dat', sep='::', engine='python', 
                     names=['movieId', 'title', 'genres'], encoding='latin-1' )

movies['genres'] = movies['genres'].str.replace('|', ' ')

genome_scores = pd.read_csv('../data/tag-genome/tag_relevance.dat', sep='\t', 
                            names=['movieId', 'tagId', 'relevance'])

genome_scores = genome_scores[genome_scores.relevance >= 0.5]

genome_tags = pd.read_csv('../data/tag-genome/tags.dat', sep='\t', 
                          names=['tagId', 'tag', 'tagPopularity'])

tag_map = dict(zip(genome_tags.tagId, genome_tags.tag))

genome_scores['tag'] = genome_scores['tagId'].map(tag_map)

tags_df = genome_scores.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()

Final = pd.merge(movies, tags_df, on='movieId', how='left')
Final.fillna("", inplace=True)

Final['metadata'] = Final[['tag', 'genres']].apply(lambda x: ' '.join(x), axis=1)

print(Final.head(4))

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(Final['metadata'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=Final.index)

print("TF-IDF matrix shape:", tfidf_df.shape)

svd = TruncatedSVD(n_components=25)
latent_matrix = svd.fit_transform(tfidf_df)

explained = svd.explained_variance_ratio_.cumsum()
plt.plot(explained, '.-', ms=10, color='red')
plt.xlabel('Singular value components')
plt.ylabel('Cumulative variance explained')
plt.title('Explained Variance by SVD Components')
plt.show()

latent_matrix = preprocessing.scale(latent_matrix)

os.makedirs('Dataset/processed_data', exist_ok=True)
with open('Dataset/processed_data/Arm_FeatureVectors_2.dat', 'w') as f:
    f.write('ArticleID\tFeatureVector\n')
    for i, movieId in enumerate(Final.movieId):
        featureVector = ';'.join([f"{x:.6f}" for x in latent_matrix[i]])
        f.write(f"{movieId}\t{featureVector}\n")
