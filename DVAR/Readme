## Getting Started

You can run the DVAR model using Docker with the following commands:

```bash
docker build -t dvar .
docker run --gpus all dvar --data_path data/ml
````

* The `--data_path` argument is used to specify whether to use the **MovieLens** or **Amazon** dataset.
* The `--gpus all` flag ensures that GPU is utilized inside the Docker container, as DVAR utilizes GPU for computation.

## Dataset Information

Dataset files are **already generated** and placed in the `data/` folder in the root directory of the repository. There's **no need to regenerate** them unless you want to preprocess them again.

To regenerate the dataset files, use the `DVAR_data_loader.py` script:

```bash
python DVAR_data_loader.py --dataset <dataset_name>
```

* Replace `<dataset_name>` with either `MovieLens` or `Amazon`.

## ⚠️ Notes

* Ensure that the `data/` directory is present in the root of the DVAR model directory.
* If you encounter a `FileNotFoundError`, check that:

  * The `--dataset` argument is correctly specified.
  * All required data files are available under the `data/` directory.

## Original Work
This implementation is based on the original DVAR repository:
https://github.com/mediumboat/DVAR/tree/main
