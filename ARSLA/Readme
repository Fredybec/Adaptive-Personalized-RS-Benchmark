## Getting Started

You can run the ARSLA model using Docker with the following commands:

```bash
docker build -t arsla .
docker run arsla --dataset Amazon
````

* Replace `Amazon` with `MovieLens` if you're using that dataset.
* The `--dataset` argument specifies which dataset to use.
* GPU is **not required** for ARSLA.
* Before running, make sure to **copy the `data/` folder** from the root of this repository to the **root of the ARSLA** directory.

## Manual Run

To manually run the model outside of Docker:

```bash
python arsla.py --dataset Amazon
```

* Replace `Amazon` with `MovieLens` depending on your dataset.

## ⚠️ Notes

* Ensure that the `data/` directory is present in the root of the ARSLA model directory.
* If you encounter a `FileNotFoundError`, check that:

  * The `--dataset` argument is correctly specified.
  * All required data files are available under the `data/` directory.
