## Getting Started

You can run the CoDBand model using Docker with the following commands:

```bash
docker build -t codband .
docker run codband --alg CoDBand --dataset Amazon
````

* Replace `Amazon` with `MovieLens` if you're using that dataset.
* The `--alg` argument should always be set to `CoDBand`.
* GPU is **not required** for CoDBand.
* Before running, make sure to **copy the `data/` folder** from the root of this repository to the **root of the CoDBand** directory.

## 📁 Dataset Information

Preprocessed dataset files are already provided in the following directory:

```
Dataset/processed_data/
```

There is no need to regenerate the data unless you want to reprocess it manually.

## Manual Data Preprocessing and Training

If you'd like to preprocess the data and run the model manually:

### For Amazon Dataset

```bash
python Dataset/getAmazonArmFeature.py
python Dataset/getProcessedEventsAmazon.py
```

### For MovieLens Dataset

```bash
python Dataset/getMovieLensArmFeature.py
python Dataset/getProcessedEvents.py
```

### Run the Model

```bash
python DeliciousLastFMAndMovieLens.py --alg CoDBand --dataset Amazon
```

* Replace `Amazon` with `MovieLens` depending on your dataset.

## ⚠️ Notes

* Make sure the preprocessed files are present in `Dataset/processed_data/`.
* If you encounter `FileNotFoundError`, verify that:

  * The `data/` folder is correctly copied to CoDBand’s root directory.
  * You're using the correct dataset name and file paths.

## Original Work
This implementation is based on the original CoDBand repository:
https://github.com/cyrilli/CoDBand
