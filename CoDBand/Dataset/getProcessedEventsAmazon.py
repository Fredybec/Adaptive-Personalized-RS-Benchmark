import json
import random
from collections import defaultdict

threshold = 3
rating_threshold = 3
item_min_count = 2
user_min_count = 3

num_users = 0
arm_pool = set()
user2ItemSeqs = {}
temp_user_arm_tag = []
cur_uid = None
user_counts = defaultdict(int)
item_counts = defaultdict(int)

# ---------- First pass: count item frequencies ----------
with open('../data/amazon/amazon.jsonl', 'r') as fin:
    for line in fin:
        review = json.loads(line.strip())
        if float(review.get('rating', 0)) < rating_threshold:
            continue
        aid = review['asin']
        item_counts[aid] += 1
valid_items = {aid for aid, count in item_counts.items() if count >= item_min_count}

# ---------- Second pass: count user frequencies using valid items only ----------
with open('../data/amazon/amazon.jsonl', 'r') as fin:
    for line in fin:
        review = json.loads(line.strip())
        if float(review.get('rating', 0)) < rating_threshold:
            continue
        aid = review['asin']
        uid = review['user_id']
        if aid not in valid_items:
            continue
        user_counts[uid] += 1

valid_users = {uid for uid, count in user_counts.items() if count >= user_min_count}

# ---------- Third pass: Build user2ItemSeqs ----------
with open('../data/amazon/amazon.jsonl', 'r') as fin:
    for line in fin:
        review = json.loads(line.strip())
        if float(review.get('rating', 0)) < rating_threshold:
            continue
        uid = review['user_id']
        aid = review['asin']
        tstamp = review.get('timestamp', 0)

        if aid not in valid_items or uid not in valid_users:
            continue

        if cur_uid is None:
            cur_uid = uid

        if uid == cur_uid:
            temp_user_arm_tag.append({'uid': uid, 'aid': aid, 'tstamp': tstamp})
        else:
            if len(temp_user_arm_tag) > threshold:
                num_users += 1
                user2ItemSeqs[cur_uid] = temp_user_arm_tag
            cur_uid = uid
            temp_user_arm_tag = [{'uid': uid, 'aid': aid, 'tstamp': tstamp}]

        arm_pool.add(aid)

if len(temp_user_arm_tag) > threshold:
    num_users += 1
    user2ItemSeqs[cur_uid] = temp_user_arm_tag

print('user number:', len(user2ItemSeqs))
print('item number:', len(arm_pool))

# ---------- Build user arm pools ----------
user_arm_pool = {}
for uid, ItemSeqs in user2ItemSeqs.items():
    user_arm_pool[uid] = arm_pool.copy()
    for t in ItemSeqs:
        user_arm_pool[uid].discard(t['aid']) 
        
# ---------- Write output ----------
with open(f"Dataset/processed_data/randUserOrderedTime_N{len(user2ItemSeqs)}_ObsMoreThan{threshold}ForAmazon.dat", "w") as file:
    file.write('userid\ttimestamp\tarm_pool\n')
    global_time = 0
    while user2ItemSeqs:
        userID = random.choice(list(user2ItemSeqs.keys()))
        t = user2ItemSeqs[userID].pop(0)
        global_time += 1

        # Sample 24 random arms + current aid
        if len(user_arm_pool[userID]) >= 24:
            random_pool = [t['aid']] + random.sample(list(user_arm_pool[userID]), 24)
        else:
            random_pool = [t['aid']] + list(user_arm_pool[userID])

        file.write(f"{t['uid']}\t{t['tstamp']}\t{random_pool}\n")

        if not user2ItemSeqs[userID]:
            del user2ItemSeqs[userID]

print("global_time", global_time)
