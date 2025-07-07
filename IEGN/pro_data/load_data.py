#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 17:20:22 2018

@author: c503
Modified by: Fredybec 
"""

import os
import json
import pandas as pd
import pickle
import numpy as np
from copy import deepcopy
import json
import gzip
import random

import argparse
parser = argparse.ArgumentParser(description="Choose dataset to preprocess")
parser.add_argument(
    "--dataset",
    choices=["Amazon", "MovieLens"],
    required=True,
    help="Specify which dataset to process: Amazon or MovieLens"
)
args = parser.parse_args()

def process_amazon():
    dir_path = 'data/amazon/'
    rating_file = 'amazon.jsonl'
    review_file = 'amazon.jsonl'

    def read_user_rating_records():
        file_path = dir_path + rating_file
        records = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  

        return pd.DataFrame(records)

    data_records = read_user_rating_records()
    data_records.head()
    data_records.iloc[[1, 10, 20]]

    data_records.loc[data_records.rating < 4, 'rating'] = 0
    data_records.loc[data_records.rating >= 4, 'rating'] = 1
    data_records = data_records[data_records.rating > 0]


    def remove_infrequent_items(data, min_counts=5):
        df = deepcopy(data)
        counts = df['asin'].value_counts()
        df = df[df["asin"].isin(counts[counts >= min_counts].index)]

        print("items with < {} interactoins are removed".format(min_counts))
        return df

    def remove_infrequent_users_max(data, mam_counts=10):
        df = deepcopy(data)
        counts = df['asin'].value_counts()
        df = df[df["asin"].isin(counts[counts<= mam_counts].index)]

        print("items with > {} interactoins are removed".format(mam_counts))
        return df

    def remove_infrequent_users(data, min_counts=8):
        df = deepcopy(data)
        counts = df['user_id'].value_counts()
        df = df[df["user_id"].isin(counts[counts >= min_counts].index)]

        print("users with < {} interactoins are removed".format(min_counts))
        return df

    filtered_data = remove_infrequent_users(data_records, 3)

    filtered_data = remove_infrequent_items(filtered_data, 5)

    item_list = filtered_data['asin'].unique()
    item_set = set(item_list)

    time_list = filtered_data['timestamp'].unique()
    time_list = set(time_list)

    print(item_list[:10])

    def parse(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

    review_dict = dict()  
    review_helpful = dict()
    for l in parse(dir_path + review_file):
        if l['asin'] in item_set:
            helpful = l['helpful_vote'] 

            if l['asin'] in review_dict:
                if helpful > review_helpful[l['asin']] and len(l['text']) > 10:
                    review_dict[l['asin']] = l['text']
                    review_helpful[l['asin']] = helpful
            else:
                if len(l['text']) > 10:
                    review_dict[l['asin']] = l['text']
                    review_helpful[l['asin']] = helpful

    item_without_review = []
    for item_id in item_list:
        if item_id not in review_dict:
            item_without_review.append(item_id)

    for item_id in item_without_review:
        filtered_data = filtered_data[filtered_data['asin'] != item_id]

    item_list = filtered_data['asin'].unique()
    print(len(item_list))

    for item_id, review in review_dict.items():
        if len(review) < 5:
            print(item_id)



    def convert_data(data):
        df = deepcopy(data)
        df_ordered = df.sort_values(['timestamp'], ascending=True)
        data = df_ordered.groupby('user_id')['asin'].apply(list)

        print("succressfully created sequencial data! head:", data.head(5))
        unique_data = df_ordered.groupby('user_id')['asin'].nunique()

        data = data[unique_data[unique_data >= 10].index]

        data_time = df_ordered.groupby('user_id')['timestamp'].apply(list)
        print("succressfully created sequencial tiem_data! head:", data.head(5))
        data_time = data_time[unique_data[unique_data >= 10].index]


        return data, data_time

    seq_data, seq_time_data = convert_data(filtered_data)


    user_item_dict = seq_data.to_dict()
    user_time_dict = seq_time_data.to_dict()

    user_mapping = []
    item_set = set()
    rating_count = 0
    for user_id, item_list in seq_data.items():  
        user_mapping.append(user_id)
        rating_count +=len(item_list)
        for item_id in item_list:
            item_set.add(item_id)
    item_mapping = list(item_set)

    print("len(user_mapping):",len(user_mapping), len(item_mapping),rating_count)

    def generate_inverse_mapping(data_list):
        inverse_mapping = dict()
        for inner_id, true_id in enumerate(data_list):
            inverse_mapping[true_id] = inner_id
        return inverse_mapping

    def convert_to_inner_index(user_records, user_mapping, item_mapping):
        inner_user_records = []
        user_inverse_mapping = generate_inverse_mapping(user_mapping)
        item_inverse_mapping = generate_inverse_mapping(item_mapping)

        for user_id in range(len(user_mapping)):
            real_user_id = user_mapping[user_id]
            item_list = list(user_records[real_user_id])
            for index, real_item_id in enumerate(item_list):
                item_list[index] = item_inverse_mapping[real_item_id]
            inner_user_records.append(item_list)

        return inner_user_records, user_inverse_mapping, item_inverse_mapping

    def convert_to_inner_index_time(user_time_dict, user_mapping):
        inner_user_time_records = []

        for user_id in range(len(user_mapping)):
            real_user_id = user_mapping[user_id]
            time_list = list(user_time_dict[real_user_id])
            inner_user_time_records.append(time_list)

        return inner_user_time_records

    inner_data_records, user_inverse_mapping, item_inverse_mapping = convert_to_inner_index(user_item_dict, user_mapping, item_mapping)
    inner_data_time = convert_to_inner_index_time(user_time_dict,user_mapping)

    # === Normalize timestamps to embedding indices ===
    def normalize_timestamps(time_sequences, max_bins):
        all_times = [t for seq in time_sequences for t in seq]
        min_time = min(all_times)
        max_time = max(all_times)
        time_range = max_time - min_time + 1

        normalized_time_sequences = []
        for seq in time_sequences:
            normalized_seq = [int((t - min_time) / time_range * (max_bins - 1)) for t in seq]
            normalized_time_sequences.append(normalized_seq)
        return normalized_time_sequences

    max_bins = 2 * len(item_mapping) 
    inner_data_time_normalized = normalize_timestamps(inner_data_time, max_bins=max_bins)

    print(inner_data_records[:5])
    print(inner_data_time[:5])

    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

    save_obj(inner_data_records, 'pro_data/amazon/amazon_item_sequences')
    save_obj(user_inverse_mapping, 'pro_data/amazon/amazon_user_mapping')
    save_obj(item_inverse_mapping, 'pro_data/amazon/amazon_item_mapping')
    save_obj(inner_data_time_normalized, 'pro_data/amazon/amazon_time_sequences')
    print("amazon_user_mapping:", len(user_inverse_mapping),len(inner_data_time_normalized)) 



    print(f"Number of users: {len(user_mapping)}")
    print(f"Number of items: {len(item_mapping)}")
    print(f"Max raw item index: {max(item_mapping)}")
    print(f"Max mapped item index: {max(item_inverse_mapping.values())}")
    print(f"Max sequence length: {max(len(seq) for seq in inner_data_records)}")
    print(f"Min sequence length: {min(len(seq) for seq in inner_data_records)}") 


def process_movielens():
    # === Review simulation ===
    FEEDBACK_MAP = {
        1: ["I hated this movie.", "Terrible film.", "Awful experience."],
        2: ["Not good.", "Pretty boring.", "Bad plot and acting."],
        3: ["It was okay.", "Decent movie, not memorable.", "So-so."],
        4: ["I enjoyed it.", "Quite good!", "Nice watch."],
        5: ["Amazing!", "Loved everything about it.", "Masterpiece."]
    }

    def rating_to_feedback(rating):
        return random.choice(FEEDBACK_MAP.get(int(rating), ["No comment."]))

    # === Paths ===
    dir_path = 'data/ml-1m/'
    rating_file = 'ratings.dat'

    # === Read MovieLens 1M Ratings ===
    def read_user_rating_records():
        col_names = ['user_id', 'item_id', 'rating', 'timestamp']
        data_records = pd.read_csv(
            os.path.join(dir_path, rating_file),
            sep='::', names=col_names, engine='python'
        )
        return data_records

    data_records = read_user_rating_records()

    # === Binarize ratings and filter ===
    data_records = data_records[data_records.rating >= 4]
    data_records['rating'] = 1

    # === Filter infrequent users and items ===
    def remove_infrequent_items(data, min_counts=5):
        df = deepcopy(data)
        counts = df['item_id'].value_counts()
        df = df[df["item_id"].isin(counts[counts >= min_counts].index)]
        return df

    def remove_infrequent_users(data, min_counts=10):
        df = deepcopy(data)
        counts = df['user_id'].value_counts()
        df = df[df["user_id"].isin(counts[counts >= min_counts].index)]
        return df


    filtered_data = remove_infrequent_items(data_records, 5)
    filtered_data = remove_infrequent_users(filtered_data, 10)


    # === Generate simulated review texts ===
    item_list = filtered_data['item_id'].unique()
    review_dict = {
        item_id: rating_to_feedback(
            filtered_data[filtered_data.item_id == item_id].rating.sample(1).values[0]
        ) for item_id in item_list
    }

    # === Remove items without reviews (should be none in this case) ===
    filtered_data = filtered_data[filtered_data['item_id'].isin(review_dict.keys())]

    # === Convert to sequential user-item interactions WITHOUT truncation ===
    def convert_data(data):
        df = deepcopy(data)
        df_ordered = df.sort_values(['timestamp'], ascending=True)

        data = df_ordered.groupby('user_id')['item_id'].apply(list)
        data_time = df_ordered.groupby('user_id')['timestamp'].apply(list)

        unique_data = df_ordered.groupby('user_id')['item_id'].nunique()
        valid_users = unique_data[unique_data >= 10].index

        data = data[valid_users]
        data_time = data_time[valid_users]

        data = data[data.apply(len) >= 3]
        data_time = data_time[data_time.apply(len) >= 3]

        return data, data_time

    seq_data, seq_time_data = convert_data(filtered_data)

    user_item_dict = seq_data.to_dict()
    user_time_dict = seq_time_data.to_dict()

    # === Create mappings ===
    user_mapping = list(user_item_dict.keys())
    item_set = set()
    for item_list in seq_data.values:
        item_set.update(item_list)
    item_mapping = list(item_set)

    def generate_inverse_mapping(data_list):
        return {true_id: inner_id for inner_id, true_id in enumerate(data_list)}

    def convert_to_inner_index(user_records, user_mapping, item_mapping):
        inner_user_records = []
        user_inverse_mapping = generate_inverse_mapping(user_mapping)
        item_inverse_mapping = generate_inverse_mapping(item_mapping)
        for user_id in user_mapping:
            item_list = list(user_records[user_id])
            item_list = [item_inverse_mapping[item_id] for item_id in item_list]
            inner_user_records.append(item_list)
        return inner_user_records, user_inverse_mapping, item_inverse_mapping

    def convert_to_inner_index_time(user_time_dict, user_mapping):
        return [user_time_dict[user_id] for user_id in user_mapping]

    inner_data_records, user_inverse_mapping, item_inverse_mapping = convert_to_inner_index(
        user_item_dict, user_mapping, item_mapping
    )
    inner_data_time = convert_to_inner_index_time(user_time_dict, user_mapping)

    # === Normalize timestamps to embedding indices ===
    def normalize_timestamps(time_sequences, max_bins):
        all_times = [t for seq in time_sequences for t in seq]
        min_time = min(all_times)
        max_time = max(all_times)
        time_range = max_time - min_time + 1

        normalized_time_sequences = []
        for seq in time_sequences:
            normalized_seq = [int((t - min_time) / time_range * (max_bins - 1)) for t in seq]
            normalized_time_sequences.append(normalized_seq)
        return normalized_time_sequences

    max_bins = 2 * len(item_mapping) 
    inner_data_time_normalized = normalize_timestamps(inner_data_time, max_bins=max_bins)

    mapped_review_dict = {item_inverse_mapping[k]: review_dict[k] for k in item_mapping}

    # === Save the processed data ===
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

    save_obj(inner_data_records, 'pro_data/ml/ML1M_item_sequences')
    save_obj(user_inverse_mapping, 'pro_data/ml/ML1M_user_mapping')
    save_obj(item_inverse_mapping, 'pro_data/ml/ML1M_item_mapping')
    save_obj(inner_data_time_normalized, 'pro_data/ml/ML1M_time_sequences')
    save_obj(mapped_review_dict, 'pro_data/ml/ML1M_review_dict')

    print("Max len:", max(len(seq) for seq in inner_data_records))
    print("Min len:", min(len(seq) for seq in inner_data_records))
    print(f"Number of users: {len(user_mapping)}")
    print(f"Number of items: {len(item_mapping)}")
    print(f"Max raw item index: {max(item_mapping)}")
    print(f"Max mapped item index: {max(item_inverse_mapping.values())}")
    print("Saved MovieLens 1M processed data with normalized timestamps and full sequences.")


if __name__ == "__main__":
    if args.dataset == "Amazon":
        process_amazon()
    elif args.dataset == "MovieLens":
        process_movielens()
