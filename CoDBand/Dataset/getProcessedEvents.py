import random
import random
from collections import defaultdict

threshold = 10
min_user_interactions = 10
min_item_interactions = 5

# ---------- First pass: Count item interactions ----------
item_counts = defaultdict(int)
with open('../data/ml-1m/ratings.dat', 'r') as fin:
    for line in fin:
        arr = line.strip().split('::')
        rating = float(arr[2])
        if rating < 4:
            continue
        aid = int(arr[1])
        item_counts[aid] += 1


valid_items = {aid for aid, count in item_counts.items() if count >= min_item_interactions}

# ---------- Second pass: Count user interactions on valid items ----------
user_counts = defaultdict(int)
with open('../data/ml-1m/ratings.dat', 'r') as fin:
    for line in fin:
        arr = line.strip().split('::')
        rating = float(arr[2])
        if rating < 4:
            continue
        aid = int(arr[1])
        uid = int(arr[0])
        if aid in valid_items:
            user_counts[uid] += 1

valid_users = {uid for uid, count in user_counts.items() if count >= min_user_interactions}

# ---------- Third pass: Filter and process data ----------
user2ItemSeqs = {}
user_arm_pool = {}
temp_user_arm_tag = []
cur_uid = 1
arm_pool = set()
num_users = 0

with open('../data/ml-1m/ratings.dat', 'r') as fin:
    for line in fin:
        arr = line.strip().split('::')
        rating = float(arr[2])
        if rating < 4:
            continue

        uid = int(arr[0])
        aid = int(arr[1])
        timestamp = int(arr[3])

        if aid not in valid_items or uid not in valid_users:
            continue

        t = {'uid': uid, 'aid': aid, 'tstamp': timestamp}
        arm_pool.add(aid)

        if cur_uid == uid:
            temp_user_arm_tag.append(t)
        else:
            if len(temp_user_arm_tag) > threshold:
                user2ItemSeqs[cur_uid] = sorted(temp_user_arm_tag, key=lambda x: x['tstamp'])
                num_users += 1
            cur_uid = uid
            temp_user_arm_tag = [t]

if len(temp_user_arm_tag) > threshold:
    user2ItemSeqs[cur_uid] = sorted(temp_user_arm_tag, key=lambda x: x['tstamp'])
    num_users += 1

# ---------- Build user arm pools ----------
for uid, ItemSeqs in user2ItemSeqs.items():
    user_arm_pool[uid] = arm_pool.copy()
    for t in ItemSeqs:
        user_arm_pool[uid].discard(t['aid'])

# ---------- Write output ----------
print('User number:', len(user2ItemSeqs))
print('Item number:', len(arm_pool))

with open("Dataset/processed_data/randUserOrderedTime_N{}_ObsMoreThan{}.dat".format(len(user2ItemSeqs), threshold), "w") as file:
    file.write('userid\ttimestamp\tarm_pool\n')
    global_time = 0
    while user2ItemSeqs:
        userID = random.choice(list(user2ItemSeqs.keys()))
        t = user2ItemSeqs[userID].pop(0)
        global_time += 1
        random_pool = [t['aid']] + random.sample(list(user_arm_pool[t['uid']]), min(24, len(user_arm_pool[t['uid']])))
        file.write(f"{t['uid']}\t{t['tstamp']}\t{random_pool}\n")
        if not user2ItemSeqs[userID]:
            del user2ItemSeqs[userID]

print("global_time", global_time)