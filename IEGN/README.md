## Getting Started

You can run the IEGN model using Docker with the following commands (By default the dataset files inside data_files/music are MovieLens's, run the preprocess with --dataset Amazon manually):

```bash
docker build -t iegn .
docker run --gpus all iegn
````

* The `--gpus all` flag ensures that GPU is utilized inside the Docker container, as IEGN requires GPU for training and data processing.

## Manual Data Preprocessing and Training

If you'd like to manually run the preprocessing and training steps outside of Docker, follow the commands below:

### 1. Load the Raw Data

```bash
python pro_data/load_data.py --dataset <dataset_name>
```

### 2. Preprocess the Data

```bash
python pro_data/data_pro.py --dataset <dataset_name>
```

* Replace `<dataset_name>` with either `MovieLens` or `Amazon`.

These scripts will generate the required dataset files.

### 3. Train the Model

```bash
python run.py
```

## ⚠️ Notes

* Ensure that the `data/` directory is present in the root of the DVAR model directory.
* If you encounter a `FileNotFoundError`, check that:

  * The `--dataset` argument is correctly specified.
  * All required data files are available under the `data/` and `data_files/` directory.

## Original Work
This implementation is based on the original IEGN repository:
 https://github.com/IEGN-2021/IEGN/tree/main