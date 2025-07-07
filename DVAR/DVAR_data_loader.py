import math
import numpy as np
from pathlib import Path
import pandas as pd
import argparse
import os
import json

def intersect2d(a, b):
    return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])


def load_data(dataset,data_path_str=None):
    if dataset == "MovieLens":
        print("___Reading rating data___")
        data_path = Path("data_files/ml")
        os.mkdir(data_path)
        n_user = 6040
        n_item = 3706
        n_cate = 18
        rating_file = "data/ml-1m/ratings.dat"
        df = pd.read_csv(rating_file, names=["userId", "movieId", "rating", "timestamp"],
                        usecols=[0, 1, 2], delimiter="::", engine='python')
        print('----removing users with less than 10 interactions and itmes with less than 5 int----')
        item_counts = df["movieId"].value_counts()
        df = df[df["movieId"].isin(item_counts[item_counts >= 5].index)]
        user_counts = df["userId"].value_counts()
        df = df[df["userId"].isin(user_counts[user_counts >= 10].index)]
        df = df[df["rating"] >= 4.0] #
        user_ids = np.sort(df["userId"].unique())
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        movie_ids = np.sort(df["movieId"].unique())
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        df["user"] = df["userId"].map(user2user_encoded)
        df["item"] = df["movieId"].map(movie2movie_encoded)
        df = df[['user', 'item']]
        df = df.sample(frac=1.0).reset_index(drop=True)

        print("___Reading category data___")
        movies_file = "data/ml-1m/movies.dat"
        movies_df = pd.read_csv(movies_file, sep='::', engine='python', header=None,
                                names=["movieId", "title", "genres"], encoding='latin-1')

        movies_df["movie"] = movies_df["movieId"].map(movie2movie_encoded)

        all_genres = sorted(set(g for genre_str in movies_df["genres"] for g in genre_str.split('|')))
        n_cate = len(all_genres)
        genre2id = {g: i for i, g in enumerate(all_genres)}

        item_cate = []

        for _, row in movies_df.iterrows():
            if pd.isna(row["movie"]):
                continue  # skip unmapped movies
            movie_enc_id = int(row["movie"])
            for genre in row["genres"].split('|'):
                genre_id = genre2id[genre]
                item_cate.append([movie_enc_id, genre_id])

        item_cate = np.array(item_cate)
        print("___Splitting train/val/testing data___")
        data = df.to_numpy()
        train_user_idx = math.floor(0.7 * n_user)
        val_user_idx = math.floor(0.85 * n_user)
        user_list = np.arange(n_user)
        item_list = np.arange(n_item)
        user_list_random = user_list.copy()
        user_list_idx = np.random.permutation(n_user)
        user_list_ramdom = user_list_random[user_list_idx]
        train_users = user_list_ramdom[: train_user_idx]
        train_data = df[df["user"].isin(train_users)].to_numpy()
        val_users = user_list_ramdom[train_user_idx: val_user_idx]
        val_data = df[df["user"].isin(val_users)].to_numpy()
        test_users = user_list_ramdom[val_user_idx:]
        test_data = df[df["user"].isin(test_users)].to_numpy()
        split_idx = np.random.choice(len(val_data), math.floor(len(val_data) * 0.5), replace=False)
        train_data = np.row_stack((train_data, val_data[split_idx, :]))
        val_data = np.delete(val_data, split_idx, 0)
        split_idx = np.random.choice(len(test_data), math.floor(len(test_data) * 0.5), replace=False)
        train_data = np.row_stack((train_data, test_data[split_idx, :]))
        train_users = np.unique(train_data[:, 0])
        test_data = np.delete(test_data, split_idx, 0)
        item_freq = np.zeros((n_item, 2), dtype=int)
        item, freq = np.unique(train_data[:, 1], return_counts=True)
        item_freq[item, 0] = item
        item_freq[item, 1] = freq
        negative_train_data = train_data.copy()
        n_positive_train_data = len(train_data)
        n_positive_val_data = len(val_data)
        n_positive_test_data = len(test_data)
        count_train = 0
        for user in train_users:
            interacted_items = train_data[np.where(train_data[:, 0] == user)[0], 1]
            n_interacted_items = len(interacted_items)
            uninteracted_items = np.setdiff1d(item_list, interacted_items)
            uninteracted_items_freq = item_freq[np.intersect1d(item_freq[:, 0], uninteracted_items), :]
            uninteracted_items_freq = uninteracted_items_freq[np.argsort(-uninteracted_items_freq[:, 1], ), :]
            #
            available_negatives = len(uninteracted_items_freq[:, 0])
            sample_size = n_interacted_items
            if available_negatives == 0:
                print('skiped')
                continue
            count_train+=1
            replace_flag = available_negatives < sample_size
            negative_items = np.random.choice(uninteracted_items_freq[:, 0], sample_size, replace=replace_flag,
                                            p=uninteracted_items_freq[:, 1]/sum(uninteracted_items_freq[:, 1]))
            negative_train_data[np.where(negative_train_data[:, 0] == user)[0], 1] = negative_items
        train_data = np.row_stack((train_data, negative_train_data))
        train_label = np.row_stack((np.ones((n_positive_train_data, 1)), np.zeros((n_positive_train_data, 1))))
        print('count train',count_train)
        negative_val_data = np.zeros((20 * len(val_users), 2))
        n_negative_val_data = len(negative_val_data)
        i = 0
        count_val=0
        for user in val_users:
            interacted_items = data[np.where(data[:, 0] == user)[0], 1]
            uninteracted_items = np.setdiff1d(item_list, interacted_items)
            available_negatives = len(uninteracted_items)
            
            if available_negatives == 0:
                print('skiped in val')
                continue

            replace_flag = available_negatives < sample_size
            count_val+=1
            negative_items = np.random.choice(uninteracted_items, 20, replace=replace_flag)
            negative_val_data[i:i+20, 0] = user
            negative_val_data[i:i+20, 1] = negative_items
            i = i + 20
        val_data = np.row_stack((val_data, negative_val_data))
        val_label = np.row_stack((np.ones((n_positive_val_data, 1)), np.zeros((n_negative_val_data, 1))))
        print('count val',count_val)
        negative_test_data = np.zeros((20 * len(test_users), 2))
        n_negative_test_data = len(negative_test_data)
        i = 0
        count_test=0
        for user in test_users:
            interacted_items = data[np.where(data[:, 0] == user)[0], 1]
            uninteracted_items = np.setdiff1d(item_list, interacted_items)
            available_negatives = len(uninteracted_items)
        
            if available_negatives == 0:
                print('skiped in test')
                continue
            count_test+=1
            replace_flag = available_negatives < sample_size
            negative_items = np.random.choice(uninteracted_items, 20, replace=replace_flag)
            negative_test_data[i:i + 20, 0] = user
            negative_test_data[i:i + 20, 1] = negative_items
            i = i + 20
        test_data = np.row_stack((test_data, negative_test_data))
        test_label = np.row_stack((np.ones((n_positive_test_data, 1)), np.zeros((n_negative_test_data, 1))))
        print('count test',count_test)
        print("___Outputting data into hard disk___")
        train_data[:, 1] = train_data[:, 1] + n_user
        val_data[:, 1] = val_data[:, 1] + n_user
        test_data[:, 1] = test_data[:, 1] + n_user
        item_cate[:, 0] = item_cate[:, 0] + n_user
        item_cate[:, 1] = item_cate[:, 1] + n_user + n_item
        user_list = np.arange(n_user)
        item_list = np.arange(n_item) + n_user
        cate_list = np.arange(n_cate) + n_user + n_item
        with open(data_path/"user.node", 'wb+') as f1, open(data_path/"item.node", 'wb+') as f2, open(
                data_path/"cate.node", 'wb+') as f3:
            np.savetxt(f1, user_list, "%d")
            np.savetxt(f2, item_list, "%d")
            np.savetxt(f3, cate_list, "%d")
            f1.flush()
            f1.close()
            f2.flush()
            f2.close()
            f3.flush()
            f3.close()
        with open(data_path / "train.x", 'wb+') as f1, open(data_path / "train.y", 'wb+') as f2:
            np.savetxt(f1, train_data, "%d")
            np.savetxt(f2, train_label, "%d")
            f1.flush()
            f1.close()
            f2.flush()
            f2.close()
        with open(data_path / "val.x", 'wb+') as f1, open(data_path / "val.y", 'wb+') as f2:
            np.savetxt(f1, val_data, "%d")
            np.savetxt(f2, val_label, "%d")
            f1.flush()
            f1.close()
            f2.flush()
            f2.close()
        with open(data_path / "test.x", 'wb+') as f1, open(data_path / "test.y", 'wb+') as f2:
            np.savetxt(f1, test_data, "%d")
            np.savetxt(f2, test_label, "%d")
            f1.flush()
            f1.close()
            f2.flush()
            f2.close()
        with open(data_path / "item_cate.link", 'wb+') as f1:
            np.savetxt(f1, item_cate, "%d")
            f1.flush()
            f1.close()
        print("___Data MovieLens loading is finished___")

    elif dataset == "Amazon":
        n_cate = 5
        print("___Reading Amazon review data___")
        data_path = Path("data_files/amazon")
        os.mkdir(data_path)
        file_path = "data/amazon.jsonl"

        raw_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                raw_data.append(entry)
        records = []
        for entry in raw_data:
            user_id = entry.get("user_id")
            asin = entry.get("asin")
            rating = entry.get("rating") 
            if user_id and asin:
                records.append((user_id, asin, rating))

        df = pd.DataFrame(records, columns=["userId", "itemId", "rating"])

        from copy import deepcopy

        def remove_infrequent_items(data, min_counts=5):
            df = deepcopy(data)
            counts = df['itemId'].value_counts()
            df = df[df["itemId"].isin(counts[counts >= min_counts].index)]

            print("items with < {} interactoins are removed".format(min_counts))
            return df

        def remove_infrequent_users(data, min_counts=8):
            df = deepcopy(data)
            counts = df['userId'].value_counts()
            df = df[df["userId"].isin(counts[counts >= min_counts].index)]

            print("users with < {} interactoins are removed".format(min_counts))
            return df

        df = remove_infrequent_items(df, 2)

        df = remove_infrequent_users(df, 3)

        df = df[df["rating"] >= 4.0]
        user_ids = np.sort(df["userId"].unique())
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        item_ids = np.sort(df["itemId"].unique())
        item2item_encoded = {x: i for i, x in enumerate(item_ids)}
        df["user"] = df["userId"].map(user2user_encoded)
        df["item"] = df["itemId"].map(item2item_encoded)
        df = df[["user", "item"]]

        n_user = len(user2user_encoded)
        n_item = len(item2item_encoded)

        print("___Reading category data___")
        category_names = ["Magazine_Subscriptions", "Digital_Music", "Subscription_Boxes","Gift_Cards","Health_and_Personal_Care"]

        category2id = {cat: idx for idx, cat in enumerate(category_names)}
        item_cate = np.empty((0, 2), dtype=int)

        for entry in raw_data:
            asin = entry.get("asin")
            category = entry.get("category")  # single string, like 'Digital_Music'

            if asin in item2item_encoded and category in category2id:
                item_idx = item2item_encoded[asin]
                cat_idx = category2id[category]
                new_row = np.array([item_idx, cat_idx])
                item_cate = np.row_stack((item_cate, new_row))

        item_cate = item_cate[1:, :]



        print("___Splitting train/val/testing data___")
        data = df.to_numpy()
        train_user_idx = math.floor(0.7 * n_user)
        val_user_idx = math.floor(0.85 * n_user)
        user_list = np.arange(n_user)
        item_list = np.arange(n_item)
        user_list_random = user_list.copy()
        user_list_idx = np.random.permutation(n_user)
        user_list_ramdom = user_list_random[user_list_idx]
        train_users = user_list_ramdom[: train_user_idx]
        train_data = df[df["user"].isin(train_users)].to_numpy()
        val_users = user_list_ramdom[train_user_idx: val_user_idx]
        val_data = df[df["user"].isin(val_users)].to_numpy()
        test_users = user_list_ramdom[val_user_idx:]
        test_data = df[df["user"].isin(test_users)].to_numpy()
        split_idx = np.random.choice(len(val_data), math.floor(len(val_data) * 0.5), replace=False)
        train_data = np.row_stack((train_data, val_data[split_idx, :]))
        val_data = np.delete(val_data, split_idx, 0)
        split_idx = np.random.choice(len(test_data), math.floor(len(test_data) * 0.5), replace=False)
        train_data = np.row_stack((train_data, test_data[split_idx, :]))
        train_users = np.unique(train_data[:, 0])
        test_data = np.delete(test_data, split_idx, 0)
        item_freq = np.zeros((n_item, 2), dtype=int)
        item, freq = np.unique(train_data[:, 1], return_counts=True)
        item_freq[item, 0] = item
        item_freq[item, 1] = freq
        negative_train_data = train_data.copy()
        n_positive_train_data = len(train_data)
        n_positive_val_data = len(val_data)
        n_positive_test_data = len(test_data)
        from tqdm import tqdm
        for user in tqdm(train_users, desc="Generating negative train samples"):
            interacted_items = train_data[np.where(train_data[:, 0] == user)[0], 1]
            n_interacted_items = len(interacted_items)
            uninteracted_items = np.setdiff1d(item_list, interacted_items)
            uninteracted_items_freq = item_freq[np.intersect1d(item_freq[:, 0], uninteracted_items), :]
            uninteracted_items_freq = uninteracted_items_freq[np.argsort(-uninteracted_items_freq[:, 1], ), :]
            negative_items = np.random.choice(uninteracted_items_freq[:, 0], n_interacted_items, replace=False,
                                            p=uninteracted_items_freq[:, 1]/sum(uninteracted_items_freq[:, 1]))
            negative_train_data[np.where(negative_train_data[:, 0] == user)[0], 1] = negative_items
        train_data = np.row_stack((train_data, negative_train_data))
        train_label = np.row_stack((np.ones((n_positive_train_data, 1)), np.zeros((n_positive_train_data, 1))))

        negative_val_data = np.zeros((20 * len(val_users), 2))
        n_negative_val_data = len(negative_val_data)
        i = 0
        for user in tqdm(val_users, desc="Generating negative val samples"):
            interacted_items = data[np.where(data[:, 0] == user)[0], 1]
            uninteracted_items = np.setdiff1d(item_list, interacted_items)
            negative_items = np.random.choice(uninteracted_items, 20, replace=False)
            negative_val_data[i:i+20, 0] = user
            negative_val_data[i:i+20, 1] = negative_items
            i = i + 20
        val_data = np.row_stack((val_data, negative_val_data))
        val_label = np.row_stack((np.ones((n_positive_val_data, 1)), np.zeros((n_negative_val_data, 1))))

        negative_test_data = np.zeros((20 * len(test_users), 2))
        n_negative_test_data = len(negative_test_data)
        i = 0
        for user in tqdm(test_users, desc="Generating negative test samples"):
            interacted_items = data[np.where(data[:, 0] == user)[0], 1]
            uninteracted_items = np.setdiff1d(item_list, interacted_items)
            negative_items = np.random.choice(uninteracted_items, 20, replace=False)
            negative_test_data[i:i + 20, 0] = user
            negative_test_data[i:i + 20, 1] = negative_items
            i = i + 20
        test_data = np.row_stack((test_data, negative_test_data))
        test_label = np.row_stack((np.ones((n_positive_test_data, 1)), np.zeros((n_negative_test_data, 1))))
        print("___Outputting data into hard disk___")
        train_data[:, 1] = train_data[:, 1] + n_user
        val_data[:, 1] = val_data[:, 1] + n_user
        test_data[:, 1] = test_data[:, 1] + n_user
        item_cate[:, 0] = item_cate[:, 0] + n_user
        item_cate[:, 1] = item_cate[:, 1] + n_user + n_item
        user_list = np.arange(n_user)
        item_list = np.arange(n_item) + n_user
        cate_list = np.arange(n_cate) + n_user + n_item
        with open(data_path/"user.node", 'wb+') as f1, open(data_path/"item.node", 'wb+') as f2, open(
                data_path/"cate.node", 'wb+') as f3:
            np.savetxt(f1, user_list, "%d")
            np.savetxt(f2, item_list, "%d")
            np.savetxt(f3, cate_list, "%d")
            f1.flush()
            f1.close()
            f2.flush()
            f2.close()
            f3.flush()
            f3.close()
        with open(data_path / "train.x", 'wb+') as f1, open(data_path / "train.y", 'wb+') as f2:
            np.savetxt(f1, train_data, "%d")
            np.savetxt(f2, train_label, "%d")
            f1.flush()
            f1.close()
            f2.flush()
            f2.close()
        with open(data_path / "val.x", 'wb+') as f1, open(data_path / "val.y", 'wb+') as f2:
            np.savetxt(f1, val_data, "%d")
            np.savetxt(f2, val_label, "%d")
            f1.flush()
            f1.close()
            f2.flush()
            f2.close()
        with open(data_path / "test.x", 'wb+') as f1, open(data_path / "test.y", 'wb+') as f2:
            np.savetxt(f1, test_data, "%d")
            np.savetxt(f2, test_label, "%d")
            f1.flush()
            f1.close()
            f2.flush()
            f2.close()
        with open(data_path / "item_cate.link", 'wb+') as f1:
            np.savetxt(f1, item_cate, "%d")
            f1.flush()
            f1.close()
        print("___Amazon data loading is finished___")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load dataset")
    parser.add_argument("--dataset", type=str, default="Amazon", help="Dataset name: MovieLens or Amazon")
    parser.add_argument("--data_path", type=str, default=None, help="Path to dataset folder")

    args = parser.parse_args()

    load_data(dataset=args.dataset, data_path_str=args.data_path)
