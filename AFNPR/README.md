## Getting Started

You can run the AFNPR model using Docker with the following commands:

```bash
docker build -t afnpr .
docker run afnpr --dataset Amazon
````

* Replace `Amazon` with `MovieLens` if you're using that dataset.
* The `--dataset` argument specifies which dataset to use.
* GPU is **not required** for AFNPR.
* Before running, make sure to **copy the `data/` folder** from the root of this repository to the **root of the AFNPR** directory.

## Manual Run

To manually run the model outside of Docker:

```bash
python afnpr.py --dataset Amazon
```

* Replace `Amazon` with `MovieLens` depending on your dataset.

## ⚠️ Notes

* Ensure that the `data/` directory is present in the root of the AFNPR model directory.
* If you encounter a `FileNotFoundError`, check that:

  * The `--dataset` argument is correctly specified.
  * All required data files are available under the `data/` directory.

