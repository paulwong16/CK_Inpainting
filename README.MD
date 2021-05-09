# CK_Inpainting

This repository contains the source code for the paper [Learning Contextual Kernels for High Quality Image Inpainting]().

![Overview](http://paulwong16.github.io/files/MM21/1_shadow.png)

## Datasets

We use [FFHQ](https://github.com/NVlabs/ffhq-dataset) and 
[ParisStreetView](https://github.com/pathak22/context-encoder#6-paris-street-view-dataset) 
in our experiments. For the irregular masks, we use the test set of 
[NVIDIA Irregular Mask Dataset](https://nv-adlr.github.io/publication/partialconv-inpainting).

Since we don't have the rights to hold any datasets, please download them from original sources. 

For FFHQ dataset, you can use the .txt files under `dataset/ffhq/` to get exactly same splits in our experiments.
For ParisStreetView dataset, you should create your own data splits .txt files and put them under `dataset/paris/`

## Get Started

### Environments

Basic environments: `Python >= 3.6` and `PyTorch >= 1.4.0`.

Required packages:
```bash
pip install -r requirements.txt
```

### Train

Replace `{CONFIG}` as `ffhq_irregular`, `ffhq_suare`, or `paris`.

#### Start a new training

```bash
python runner_inpainting.py --config={CONFIG}
```

#### Resume a previous training
Replace `{CONFIG}` as `ffhq_irregular`, `ffhq_suare`, or `paris`, `{CHECKPOINTS_PATH}` 
as the path of your checkpoints, `init_lr` as your learning rate after resuming.

```bash
python runner_inpainting.py --config={CONFIG} --weights={CHECKPOINTS_PATH} --init_lr={LEARNING_RATE}
```

### Test

Replace `{CONFIG}` as `ffhq_irregular`, `ffhq_suare`, or `paris`, `{CHECKPOINTS_PATH}` 
as the path of your checkpoints.

```bash
python runner_inpainting.py --config={CONFIG} --test --weights={CHECKPOINTS_PATH}
```

![Result1](http://paulwong16.github.io/files/MM21/teaser_mm.png)
