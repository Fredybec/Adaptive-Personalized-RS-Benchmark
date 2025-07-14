import argparse
DIMENSION = 128
SVD_ITER = 20
FAISS_N_SIMILAR = 10 + 1
N_SIMILAR = 10
FAISS_N_LIST_I = 10
FAISS_N_LIST_U = 5
N_WALK_4 = 10000
N_WALK_3 = 10000
BATCH_SIZE = 5000
EPOCHS = 1

parser = argparse.ArgumentParser(description="Metapath2vec")
parser.add_argument('--data_path', type=str, default=None,
                    help="Path to data_files/ml or data_files/amazon")
parser.add_argument('--path', type=str, default=None ,help="input_path")
parser.add_argument('--output_file', default=None, type=str, help='output_file')
parser.add_argument('--dim', default=DIMENSION, type=int, help="embedding dimensions")
parser.add_argument('--window_size', default=7, type=int, help="context window size")
parser.add_argument('--iterations', default=10, type=int, help="iterations") 
parser.add_argument('--batch_size', default=5000, type=int, help="batch size")
parser.add_argument('--care_type', default=0, type=int, help="if 1, heterogeneous negative sampling, else normal negative sampling")
parser.add_argument('--initial_lr', default=1e-3, type=float, help="learning rate")
parser.add_argument('--min_count', default=3, type=int, help="min count")
parser.add_argument('--num_workers', default=6, type=int, help="number of workers")
ARGS = parser.parse_args()


if ARGS.path is None:
    ARGS.path = f"{ARGS.data_path}/meta.path"

if ARGS.output_file is None:
    ARGS.output_file = f"{ARGS.data_path}/m2v.emb"

DATA_PATH = ARGS.data_path

S = 128.0
LAMBDA_THETA = 0.01
LAMBDA_FILM = 0.01
META_PATH_BATCH = 5000
