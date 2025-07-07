import numpy as np
import pandas as pd
import argparse
import time
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Hyperparameters
REWARD_RATE = 0.3
PENALTY_RATE = 0.4
THRESHOLD = 4.0
RATING_MAX = 5.0
NUM_RUNS = 1

def load_and_preprocess_data(dataset):
    if dataset == 'Amazon':
        data = pd.read_json('amazon/amazon.jsonl', lines=True)
        data = data[['user_id', 'asin', 'rating', 'timestamp', 'category']]
        data = data.dropna(subset=['user_id', 'asin', 'rating', 'category'])

        item_counts = data['asin'].value_counts()
        data = data[data['asin'].isin(item_counts[item_counts >= 2].index)]

        user_counts = data['user_id'].value_counts()
        data = data[data['user_id'].isin(user_counts[user_counts >= 3].index)]

        category_columns = pd.get_dummies(data['category'])
        for cat in category_columns.columns:
            data[cat] = category_columns[cat].astype('int32')

        data = data.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
        genres = list(category_columns.columns)
        id_col, item_col, rating_col, time_col = 'user_id', 'asin', 'rating', 'timestamp'
        n_clusters = 2

    elif dataset == 'MovieLens':
        ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None,
                              names=['UserID', 'MovieID', 'Rating', 'Timestamp'],
                              engine='python')
        movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None,
                             names=['MovieID', 'Title', 'Genres'],
                             engine='python', encoding='latin-1')

        item_counts = ratings['MovieID'].value_counts()
        ratings = ratings[ratings['MovieID'].isin(item_counts[item_counts >= 5].index)]

        user_counts = ratings['UserID'].value_counts()
        ratings = ratings[ratings['UserID'].isin(user_counts[user_counts >= 10].index)]

        genres = [
            'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
            'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
        ]
        for genre in genres:
            movies[genre] = movies['Genres'].str.contains(genre).astype('int32')

        data = pd.merge(ratings, movies, on='MovieID')
        id_col, item_col, rating_col, time_col = 'UserID', 'MovieID', 'Rating', 'Timestamp'
        n_clusters = 18

    else:
        raise ValueError("Unsupported dataset. Use 'amazon' or 'ml'.")

    return data, genres, id_col, item_col, rating_col, time_col, n_clusters

def cluster_items(items, genres, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, init='k-means++', random_state=42)
    features = items[genres].astype('float32')
    items['Cluster'] = kmeans.fit_predict(features)
    return items

def initialize_automaton(user_ids, n_clusters):
    return {int(uid): np.ones(n_clusters) / n_clusters for uid in user_ids}

def predict_rating(automaton, user_id, movie_id, cluster, user_ratings, movie_avg, id_col, item_col, rating_col, n_clusters):
    uid = int(user_id)
    w_u_j = automaton[uid][cluster] if user_id in automaton else 1.0 / n_clusters

    r_v_l = movie_avg.get(movie_id, 0)

    cluster_ratings = user_ratings[(user_ratings[id_col] == uid) & (user_ratings['Cluster'] == cluster)]
    r_u_j = cluster_ratings[rating_col].mean() if not cluster_ratings.empty else 0

    predicted = (r_v_l + r_u_j + (RATING_MAX * w_u_j)) / 3
    return np.clip(predicted, 1.0, 5.0)

def update_automaton(probs, chosen_cluster, rating):
    probs = np.copy(probs)
    if rating >= THRESHOLD:
        probs[chosen_cluster] += REWARD_RATE * (1 - probs[chosen_cluster])
        for j in range(len(probs)):
            if j != chosen_cluster:
                probs[j] *= (1 - REWARD_RATE)
    else:
        probs[chosen_cluster] *= (1 - PENALTY_RATE)
        penalty_share = PENALTY_RATE / (len(probs) - 1)
        for j in range(len(probs)):
            if j != chosen_cluster:
                probs[j] = probs[j] * (1 - PENALTY_RATE) + penalty_share
    return probs / probs.sum()

def evaluate_model(test_data, automaton, movie_clusters, movie_avg, train_data, id_col, item_col, rating_col, n_clusters):
    test_data = test_data.copy()
    selected_ks = [5, 10, 20]
    metrics = {k: {'precision': [], 'recall': [], 'ndcg': []} for k in selected_ks}
    all_preds, all_truths = [], []
    start_time = time.time()

    for user_id in tqdm(test_data[id_col].unique(), desc='test'):
        user_test = test_data[test_data[id_col] == user_id]
        preds = []

        for row in user_test.itertuples(index=False):
            cluster = movie_clusters.get(int(getattr(row, item_col)), -1)
            if cluster == -1:
                continue

            pred = predict_rating(automaton, getattr(row, id_col), getattr(row, item_col), cluster, train_data, movie_avg, id_col, item_col, rating_col, n_clusters)
            preds.append((getattr(row, item_col), getattr(row, rating_col), pred))
            all_preds.append(pred)
            all_truths.append(getattr(row, rating_col))

            automaton[int(getattr(row, id_col))] = update_automaton(automaton[int(getattr(row, id_col))], cluster, getattr(row, rating_col))

        if not preds:
            continue

        preds.sort(key=lambda x: -x[2])
        ratings = [x[1] for x in preds]

        for k in selected_ks:
            top_k = preds[:k]
            hits = sum(1 for x in top_k if x[1] >= THRESHOLD)
            rel = sum(1 for x in preds if x[1] >= THRESHOLD)

            precision = hits / k
            recall = hits / rel if rel else 0
            dcg = sum([(1 if r >= THRESHOLD else 0) / np.log2(i + 2) for i, (_, r, _) in enumerate(top_k)])
            ideal_binary = sorted([1 if r >= THRESHOLD else 0 for r in ratings], reverse=True)
            idcg = sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_binary[:k])])
            ndcg = dcg / idcg if idcg > 0 else 0

            metrics[k]['precision'].append(precision)
            metrics[k]['recall'].append(recall)
            metrics[k]['ndcg'].append(ndcg)

    print('Testing time', time.time() - start_time)
    all_preds, all_truths = np.array(all_preds), np.array(all_truths)
    rmse = np.sqrt(np.mean((all_preds - all_truths) ** 2))
    mae = np.mean(np.abs(all_preds - all_truths))
    return metrics, rmse, mae

def run_once(dataset):
    data, genres, id_col, item_col, rating_col, time_col, n_clusters = load_and_preprocess_data(dataset)
    items = data[[item_col] + genres].drop_duplicates()
    items = cluster_items(items, genres, n_clusters)
    movie_clusters = dict(zip(items[item_col], items['Cluster']))
    data = pd.merge(data, items[[item_col, 'Cluster']], on=item_col, how='left')

    train_data, test_data = [], []
    for _, user_data in tqdm(data.groupby(id_col), desc='Splitting'):
        user_data = user_data.sort_values(time_col)
        split = int(0.8 * len(user_data))
        train_data.append(user_data.iloc[:split])
        test_data.append(user_data.iloc[split:])
    train_data = pd.concat(train_data)
    test_data = pd.concat(test_data)

    item_avg = train_data.groupby(item_col)[rating_col].mean().to_dict()
    user_ids = train_data[id_col].unique()
    automaton = initialize_automaton(user_ids, n_clusters)

    for uid in tqdm(user_ids, desc='Training Automaton'):
        user_train = train_data[train_data[id_col] == uid].sort_values(time_col)
        for row in user_train.itertuples(index=False):
            cluster = movie_clusters.get(int(getattr(row, item_col)), -1)
            if cluster == -1:
                continue
            automaton[int(uid)] = update_automaton(automaton[int(uid)], cluster, getattr(row, rating_col))

    return evaluate_model(test_data, automaton, movie_clusters, item_avg, train_data, id_col, item_col, rating_col, n_clusters)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['Amazon', 'MovieLens'], required=True, help="Dataset to use: 'amazon' or 'ml'")
    args = parser.parse_args()

    print(f"Running ARSLA on dataset: {args.dataset}")
    rmses, maes = [], []

    for run in tqdm(range(NUM_RUNS), desc='Total Runs'):
        metrics, rmse, mae = run_once(args.dataset)
        rmses.append(rmse)
        maes.append(mae)

        for k in [5, 10, 20]:
            p = np.mean(metrics[k]['precision'])
            r = np.mean(metrics[k]['recall'])
            n = np.mean(metrics[k]['ndcg'])
            print(f"k={k}: Precision={p:.4f}, Recall={r:.4f}, NDCG={n:.4f}")

    print("\nGlobal Errors (all predictions):")
    print(f"RMSE (global): {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"MAE  (global): {np.mean(maes):.4f} ± {np.std(maes):.4f}")

if __name__ == "__main__":
    main()
