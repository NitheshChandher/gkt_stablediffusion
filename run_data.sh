#!/bin/bash
# i: seed to generate different images for each iteration. 
# train_n_max: the total of iterations depending on the train data size. For AFHQ dataset, each label has around 5000 images
# train_sample: number of samples per iteration for train data. Choose depending on the dataset size.
#val_n_max: the total of iterations depending on the validation data size. For AFHQ dataset, each label has around 5000 images
# val_sample: number of samples per iteration for validation data. Choose depending on the dataset size.
# prompt1: text prompt for generating cat images. You can also use multiple prompts.
# prompt2: text prompt for generating dog images. You can also use multiple prompts.
# h: image height
# w: image width
# checkpoint: path to the checkpoint of the stable diffusion model
# path1, path2: paths to corresponding output directories for training data
# path3, path4: paths to corresponding output directories for training data

train_sample = 4
train_n_max = 1250
val_sample = 4
val_n_max = 250
prompt1 = "a photograph of cat face"
prompt2 = "a photograph of cat face"
h = 256
w = 256
checkpoint = "/path-to-checkpoint"
path1 = "synthetic_data/train/cat"
path2 = "synthetic_data/train/dog"
path3 = "synthetic_data/val/cat"
path4 = "synthetic_data/val/dog"

#sampling training data
echo "Sampling synthetic training dataset begins!"
for ((i=1; i<=train_n_max; i++))
do
    python3 scripts/txt2img.py --prompt prompt1 --plms --seed $i --n_samples train_sample --H h --W w --ckpt checkpoint --outdir path1
    python3 scripts/txt2img.py --prompt prompt2 --plms --seed $i --n_samples train_sample --H h --W w --ckpt checkpoint --outdir path2
    echo "Batch $i is generated"
done
echo "Training dataset is now ready!"

#sampling validation data
echo "Sampling synthetic validation dataset begins!"
for ((i=1; i<=val_n_max; i++))
do
    python3 scripts/txt2img.py --prompt prompt1 --plms --skip_grid --seed $i --n_samples val_sample --H h --W w --ckpt checkpoint --outdir path3
    python3 scripts/txt2img.py --prompt prompt2 --plms --skip_grid --seed $i --n_samples val_sample --H h --W w --ckpt checkpoint --outdir path4
    echo "Batch $i is generated"
done
echo "Validation dataset is now ready!"
