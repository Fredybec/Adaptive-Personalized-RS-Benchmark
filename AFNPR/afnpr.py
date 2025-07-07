import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import time

class AFNPR:
    def __init__(self, beta, decay=True):
        self.beta = beta
        self.decay = decay
        self.attention_flow_network = None
        self.transition_matrix = None
        self.item_index = None
        self.index_item = None

    def build_attention_flow_network(self, sequences):
        all_items = set()
        for seq in tqdm(sequences.values(), desc='Building AFN'):
            all_items.update(seq)
        self.item_index = {item: idx for idx, item in enumerate(sorted(all_items))}
        self.index_item = {idx: item for item, idx in self.item_index.items()}
        num_items = len(self.item_index)

        adj_matrix = np.zeros((num_items, num_items))
        for seq in sequences.values():
            for i in range(len(seq) - 1):
                from_idx = self.item_index[seq[i]]
                to_idx = self.item_index[seq[i + 1]]
                adj_matrix[from_idx, to_idx] += 1

        self.attention_flow_network = adj_matrix
        row_sums = adj_matrix.sum(axis=1)
        self.transition_matrix = np.divide(adj_matrix, row_sums[:, None], where=row_sums[:, None] != 0)
        self.transition_matrix = np.nan_to_num(self.transition_matrix)

    def get_time_window(self, sequence):
        window_size = max(1, int(len(sequence) * self.beta))
        return sequence[-window_size:]

    def calculate_time_weights(self, window_sequence):
        weights = {}
        n = len(window_sequence)
        half_life = n / 2

        for i, item in enumerate(window_sequence):
            time_pos = (n - 1 - i)
            weight = np.exp(-np.log(2) * time_pos / half_life)
            weights[item] = weight
        return weights

    def recommend(self, user_sequence, top_n=5):
        window_sequence = self.get_time_window(user_sequence)
        time_weights = self.calculate_time_weights(window_sequence) if self.decay else {item: 1.0 for item in window_sequence}

        scores = defaultdict(float)
        for item in window_sequence:
            if item not in self.item_index:
                continue
            from_idx = self.item_index[item]
            weight = time_weights[item]
            for to_idx, prob in enumerate(self.transition_matrix[from_idx]):
                if prob > 0:
                    to_item = self.index_item[to_idx]
                    if to_item not in user_sequence:
                        scores[to_item] += weight * prob

        sorted_recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_recs[:top_n]

    def evaluate(self, train_sequences, test_sequences, top_n):
        hits = 0
        total_test_items = 0
        recommended_items = set()
        total_users = 0
        skipped_users = 0
        ndcg_total = 0

        all_items = set(self.item_index.keys())

        for user in tqdm(test_sequences, desc='Evaluating'):
            if user not in train_sequences:
                skipped_users += 1
                continue
            test_items = test_sequences[user]
            if not test_items:
                skipped_users += 1
                continue

            relevant_items = {item for item, rating in test_items if rating >= 4}
            if not relevant_items:
                skipped_users += 1
                continue

            known_items = set(train_sequences[user])
            candidate_items = all_items - known_items
            recs = [item for item, _ in self.recommend(train_sequences[user], top_n) if item in candidate_items]

            user_hits = set(recs) & relevant_items
            hits += len(user_hits)
            total_test_items += len(relevant_items)
            recommended_items.update(recs)
            total_users += 1

            relevance_dict = dict(test_items)
            rec_relevances = [relevance_dict.get(item, 0) for item in recs]
            dcg_val = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(rec_relevances))
            ideal_relevances = sorted([rel for rel in relevance_dict.values() if rel >= 4], reverse=True)
            idcg_val = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_relevances[:top_n]))
            ndcg_val = dcg_val / idcg_val if idcg_val > 0 else 0
            ndcg_total += ndcg_val

        if total_users == 0:
            return {}

        precision = hits / (total_users * top_n)
        recall = hits / total_test_items
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        coverage = len(recommended_items) / len(self.item_index)
        ndcg_avg = ndcg_total / total_users

        print(f"Skipped users during evaluation: {skipped_users}")
        return {
            'precision@n': precision,
            'recall@n': recall,
            'f1@n': f1,
            'coverage@n': coverage,
            'ndcg@n': ndcg_avg
        }

def load_amazon_data(filepath):
    data = pd.read_json(filepath, lines=True)
    data = data[['user_id', 'asin', 'rating', 'timestamp']]
    item_counts = data['asin'].value_counts()
    data = data[data['asin'].isin(item_counts[item_counts >= 2].index)]
    user_counts = data['user_id'].value_counts()
    data = data[data['user_id'].isin(user_counts[user_counts >= 3].index)]
    data = data.sort_values(by=['user_id', 'timestamp'])
    sequences = data.groupby('user_id')[['asin', 'rating']].apply(
        lambda df: list(zip(df['asin'], df['rating']))
    ).to_dict()
    return sequences

def load_ml1m_data(filepath):
    ratings = pd.read_csv(filepath, sep='::', engine='python',
                           names=['user_id', 'movie_id', 'rating', 'timestamp'])
    item_counts = ratings['movie_id'].value_counts()
    ratings = ratings[ratings['movie_id'].isin(item_counts[item_counts >= 5].index)]
    user_counts = ratings['user_id'].value_counts()
    ratings = ratings[ratings['user_id'].isin(user_counts[user_counts >= 10].index)]
    ratings = ratings.sort_values(by=['user_id', 'timestamp'])
    sequences = ratings.groupby('user_id')[['movie_id', 'rating']].apply(
        lambda df: list(zip(df['movie_id'], df['rating']))
    ).to_dict()
    return sequences

def train_test_split(sequences, test_ratio=0.2):
    train_seqs, test_seqs = {}, {}
    for user, seq in sequences.items():
        n = len(seq)
        test_size = int(n * test_ratio)
        train_size = n - test_size
        train_seqs[user] = [item for item, _ in seq[:train_size]]
        test_seqs[user] = seq[train_size:]
    return train_seqs, test_seqs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['MovieLens', 'Amazon'], required=True, help='Dataset to use: ml or amazon')
    args = parser.parse_args()

    if args.dataset == 'MovieLens':
        print("Loading MovieLens 1M data...")
        sequences = load_ml1m_data('data/ml-1m/ratings.dat')
    elif args.dataset == 'Amazon':
        print("Loading Amazon data...")
        sequences = load_amazon_data('data/amazon/amazon.jsonl')

    train_seqs, test_seqs = train_test_split(sequences)

    afnpr = AFNPR(beta=0.1, decay=True)

    start_train = time.time()
    afnpr.build_attention_flow_network(train_seqs)
    end_train = time.time()
    print(f"Training time: {end_train - start_train:.2f} seconds")

    print(f"Total users: {len(train_seqs)}")
    print(f"Average sequence length: {np.mean([len(s) for s in train_seqs.values()]):.2f}")

    top_ks = [5, 10, 20]
    for k in top_ks:
        print(f"\nEvaluating for top-{k} recommendations:")
        start_eval = time.time()
        metrics = afnpr.evaluate(train_seqs, test_seqs, top_n=k)
        end_eval = time.time()
        print(f"Evaluation Time: {end_eval - start_eval:.2f} seconds")
        print("Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
