# Generative Knowledge Transfer using Stable Diffusion
---

Generative Knowledge Transfer refers to the process of leveraging the generative knowledge of large scale foundational generative models for improving the performance of downstream machine learning tasks. In this repository, we experiment with this idea by selecting [AFHQ](https://github.com/clovaai/stargan-v2/blob/master/README.md#animal-faces-hq-dataset-afhq) as a benchmark dataset and generate synthetic data using Stable Diffusion in a zero-shot setting. We evaluate the data quality of the generated dataset by training different classifiers on the datasets and validate their performance on AFHQ validation set. 

---

## Table of Contents
- [Installation](#installation)
    * [Conda](#conda)
    * [Pip](#pip)
- [Dataset](#dataset)
    * [AFHQ](#afhq-dataset)
    * [Synthetic Data](#synthetic-data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Support](#support)
- [Acknowledgement](#acknowledgement)

---

## Installation

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

## Dataset

### AFHQ Dataset
The repository makes use of [stargan-v2](https://github.com/clovaai/stargan-v2) to download dataset. Set up the submodules using the following command:
```
git submodule update --init
```

To download the dataset, do the following:
```
cd ./stargan-v2/
bash download.sh afhq-dataset
```
For our implementation, we only used images of cat and dog. However, you can include the images of wild animal for your experiments.

### Synthetic Data
For generating synthetic data, we use the submodules from the [stable-diffusion](https://github.com/CompVis/stable-diffusion). Follow the steps below to generate synthetic dataset:
1. Download the [stable-diffusion](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) checkpoint from hugging face.
2. Read the run_data.sh and choose the parameters for the dataset sampling.
3. Run the `run_data.sh` file 

```
./run_data.sh
``` 

## Training
For training classifiers using AFHQ dataset:

1. Check `src/config_afhq.yaml` file and set parameter values for training.
2. Run the following command:
```
accelerate launch src/train.py src/config_afhq.yaml
```

Similarly, for training classifiers using synthetic stable-diffusion dataset:

1. Check `src/config_sd.yaml` file and set parameter values for training.
2. Run the following command:
```
accelerate launch src/train.py src/config_sd.yaml
```

## Evaluation
For evaluating the classifiers: 

1. Check the `src/config_eval.yaml` file and set parameter values for evaluation.
2. Run the following command:
```
python3 src/eval.py src/config_eval.yaml
```

## Support
If you have any questions or comments, please open an [issue](https://github.com/NitheshChandher/gkt_stablediffusion/issues/new) on our GitHub repository

## Acknowledgement
The project was implemented during my Ph.D. studies as a part of Deep Generative Models course at Linkoping University. This work was supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP), funded by the Knut and Alice Wallenberg Foundation.

The [AFHQ](https://github.com/clovaai/stargan-v2) dataset was used for the experiments. Also, synthetic dataset was created using the Stable Diffusion model developed by [CompVis](https://ommer-lab.com/), [Stability AI](https://stability.ai/) and [Runway](https://runwayml.com/). We would like to acknowledge their contribution to the field of generative models.