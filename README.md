# Generative Knowledge Transfer using Stable Diffusion
---

Generative Knowledge Transfer refers to the process of leveraging the generative knowledge of large scale foundational generative models for improving the performance of downstream machine learning tasks. In this repository, we experiment with this idea by selecting [AFHQ](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) as a benchmark dataset and generate synthetic data using Stable Diffusion in a zero-shot setting. We evaluate the data quality of the generated dataset by training different classifiers on the datasets and validate their performance on AFHQ test set. 

---

## Table of Contents
- [Setup the environment]
    *[Conda](#conda)
    *[Pip](#pip)
- [Dataset Generation]
- [Training]
- [Evaluation]
- [Support](#support)

## Setup the environment

### Conda:
```
conda env create -f environment.yaml
conda activate gkt
```
or 

### Pip:
```
python3 -m venv gkt
source gkt/bin/activate
pip install -r requirements.txt
```

## Dataset Generation

## Support
If you have any questions or comments, please open an [issue](https://github.com/NitheshChandher/gkt_stablediffusion/issues/new) on our GitHub repository