# Gelato: Attribute-Enhanced Similarity Ranking for Sparse Link Prediction

This repository contains the official implementation of our KDD 2025 paper:

> **Attribute-Enhanced Similarity Ranking for Sparse Link Prediction**
> João Mattos, Zexi Huang, Mert Kosan, Ambuj Singh, Arlei Silva
> In *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '25)*
> [ACM Link](https://doi.org/10.1145/3690624.3709314)

---

## 🔍 Overview

**Gelato** is a similarity-based method for link prediction in sparse graphs. It is designed to:

* Enhance topological heuristics with **graph learning** over node attributes.
* Address class imbalance using a **ranking loss**.
* Improve training via **hard negative sampling** using graph partitioning.

Our experiments show that Gelato outperforms recent GNN-based methods, especially in realistic, highly imbalanced settings.

![overview](overview.svg)


## 📁 Repository Structure

```
.
├── data/                     # Pretrained models for each dataset
│   └── <Dataset>/pretrained/model_best.pth
├── environment.yml           # Conda environment file
├── overview.svg              # Method overview illustration
├── README.md                 # You're here!
└── src/                      # Source code
    ├── eval.py               # Evaluation utilities
    ├── experiment.py         # Experiment management
    ├── gelato.py             # Gelato model definition
    ├── train_clustergelato.py# (Optional) Clustering-based training (not default)
    ├── train.py              # Standard training pipeline
    └── util.py               # Dataset loading, splitting, and helpers
```

---

## 📦 Setup

### Requirements

We recommend using the included `conda` environment.

```bash
conda env create -f environment.yml
conda activate gelato
```

Alternatively, install dependencies manually:

```bash
pip install torch numpy scikit-learn tqdm networkx
```

---

## ▶️ Running Gelato

### Training

To train Gelato on a dataset (e.g., `Photo`):

```bash
python src/train.py --dataset Photo
```

### Available Arguments

Key options you can customize (see full list in `train.py`):

| Argument                       | Description                            | Default |
| ------------------------------ | -------------------------------------- | ------- |
| `--dataset`                    | Dataset name (e.g., `Cora`, `Photo`)   | `Photo` |
| `--eta`                        | Proportion of edges to add             | `0.0`   |
| `--alpha`                      | Weight for topological heuristic       | `0.0`   |
| `--beta`                       | Weight for learned component           | `1.0`   |
| `--graph-learning-type`        | Graph learning model (`mlp` or others) | `mlp`   |
| `--topological-heuristic-type` | Topological score (`ac`, `aa`, etc.)   | `ac`    |
| `--epochs`                     | Number of training epochs              | `250`   |
| `--cuda`                       | CUDA device index                      | `0`     |

## Pretrained Models

You can skip training and evaluate directly using the pretrained models provided under:

```
data/<Dataset>/pretrained/model_best.pth
```

The pretrained models (with random seed set to 1) for the five datasets in the paper are provided in `data/{dataset}/pretrained/`. Their tuned hyperparameters can be found in `src/experiment.py`. 

### Results

Gelato achieves state-of-the-art performance in link prediction. To reproduce our main results in Table 2 of the paper (mean ± std AP with random seeds ranging from 1 to 10) as well as precision@k and hits@k results, run the following command for the five datasets:

```experiment
python src/experiment.py --dataset {dataset}
```
This gives you the following AP results (in percentage):

|           | Cora        | CiteSeer    | PubMed      | Photo        | Computers    |
|:---------:|:-----------:|:-----------:|:-----------:|:------------:|:------------:|
| AP        | 3.90 ± 0.03 | 4.55 ± 0.02 | 2.88 ± 0.09 | 25.68 ± 0.53 | 18.77 ± 0.19 |


---

## 📊 Evaluation

Use the scripts in `src/eval.py` or `src/experiment.py` to run evaluation on trained models. Precision\@100 is logged every epoch during training and the best model is saved as `model_best.pth`.

---

## 📜 Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{10.1145/3690624.3709314,
  author = {Mattos, Joao and Huang, Zexi and Kosan, Mert and Singh, Ambuj and Silva, Arlei},
  title = {Attribute-Enhanced Similarity Ranking for Sparse Link Prediction},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1},
  year = {2025},
  pages = {1044–1055},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3690624.3709314}
}
```

---

## 💬 Questions?

If you encounter issues or have questions about the code or paper, feel free to open an issue or contact us.

---
