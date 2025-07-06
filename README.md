# Evaluation of Representative Recommender Systems Across Categories

## Overview

This repository provides an implementation and evaluation of several representative recommender systems across different categories, including:

| Category                 | Representative System | 
| ------------------------ | --------------------- | 
| Attention-based          | DVAR                  |
| Reinforcement Learning   | CoDBand               | 
| Temporal                 | IEGN                  | 
| Hybrid                   | AFNPR                 | 
| Feedback & Context-aware | ARSLA                 | 

## Implementation Details

* The code for **CoDBand**, **DVAR**, and **IEGN** was originally posted online by their respective authors.
* I have performed modifications including optimization, bug fixes, and adaptation of these codes to work with **Movielens** and **Amazon** datasets when the original implementations did not support them.
* The methodologies for **ARSLA** and **AFNPR** were implemented from scratch by me. I closely followed their respective papers’ methodologies, including training and evaluation, to ensure a fair and consistent evaluation.

## Repository Structure

Each recommender system is contained in its own folder named after the system (e.g., `DVAR`, `CoDBand`, `IEGN`, `AFNPR`, `ARSLA`). Inside each folder, you will find all the required source files, dataset processing scripts, and a **Dockerfile** to facilitate faster and reproducible training.

## How to Run

You can run any system in two ways:

### 1. Using Docker (Recommended)

By default, running the Docker container will use a predefined dataset (usually Movielens or Amazon depending on the system). You can build and run the Docker container with the following commands from inside the system's folder:

```bash
# Build the Docker image (replace <system_name> with the folder name, e.g., DVAR)
docker build -t <system_name> .

# Run the Docker container
docker run --rm -it <system_name>
```
> **Note:** Refer to the Readme file inside each folder for the exact commands
The container runs the training and evaluation automatically with the default dataset setup. To use a different dataset or customize options, please refer to the system-specific README inside the folder.

### 2. Manually without Docker

* Install the required Python packages by running:

  ```bash
  pip install -r requirements.txt
  ```
* Follow the README inside the specific system folder for detailed instructions on dataset loading and training.

## Acknowledgements

Special thanks to the authors of **CoDBand**, **DVAR**, and **IEGN** for making their codes publicly available, which greatly facilitated this work.

## References

* **AFNPR**
  Yan Chen, Yongfang Dai, Xiulong Han, Yi Ge, Hong Yin, Ping Li,
  *Dig users’ intentions via attention flow network for personalized recommendation*,
  Information Sciences, Volume 547, 2021, Pages 1122–1135,
  [https://doi.org/10.1016/j.ins.2020.09.007](https://doi.org/10.1016/j.ins.2020.09.007)

* **DVAR**
  Zhongzhou Liu, Yuan Fang, Min Wu,
  *Dual-View Preference Learning for Adaptive Recommendation*,
  IEEE Transactions on Knowledge and Data Engineering, Vol. 35, No. 11, 2023, Pages 11316–11327,
  [https://doi.org/10.1109/TKDE.2023.3236370](https://doi.org/10.1109/TKDE.2023.3236370)

* **CoDBand**
  Chuanhao Li, Qingyun Wu, Hongning Wang,
  *When and Whom to Collaborate with in a Changing Environment: A Collaborative Dynamic Bandit Solution*,
  Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21), 2021, Pages 1410–1419,
  [https://doi.org/10.1145/3404835.3462852](https://doi.org/10.1145/3404835.3462852)

* **ARSLA**
  Mansoureh Ghiasabadi Farahani, Javad Akbari Torkestani, Mohsen Rahmani,
  *Adaptive personalized recommender system using learning automata and items clustering*,
  Information Systems, Volume 106, 2022, Article 101978,
  [https://doi.org/10.1016/j.is.2021.101978](https://doi.org/10.1016/j.is.2021.101978)

* **IEGN**
  Donghua Liu, Jing Li, Jia Wu, Bo Du, Jun Chang, Xuefei Li,
  *Interest Evolution-driven Gated Neighborhood aggregation representation for dynamic recommendation in e-commerce*,
  Information Processing & Management, Volume 59, Issue 4, July 2022, Article 102982,
  [https://doi.org/10.1016/j.ipm.2022.102982](https://doi.org/10.1016/j.ipm.2022.102982)

## Contact

If you require further information, clarifications, or assistance, please feel free to contact me.
