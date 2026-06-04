# DecoAD

![Framework_Overview](./data/pipeline.png)

## Getting Started

This code was tested on `Ubuntu 22.04.4 LTS` and requires:
* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### Setup Conda Environment:
```
git clone https://github.com/LiuXY3366/DecoAD
cd DecoAD

# Conda environment setup
conda env create -f environment.yml
conda activate DecoAD
```

### Windows CPU Setup
If you are on Windows and only want to run CPU test/inference, use:
```
conda env create -f environment.windows.cpu.yml
conda activate DecoAD-cpu

set ALPHAPOSE_ROOT=D:\CV\DecoAD_prj\DecoAD\AlphaPose
set ALPHAPOSE_PYTHON=C:\Users\Admin\miniconda3\envs\DecoAD-cpu\python.exe
set ALPHAPOSE_CFG=D:\CV\DecoAD_prj\DecoAD\AlphaPose\configs\coco\resnet\256x192_res50_lr1e-3_1x.yaml
set ALPHAPOSE_CKPT=D:\CV\DecoAD_prj\DecoAD\AlphaPose\pretrained_models\fast_res50_256x192.pth

python d:\CV\DecoAD_prj\DecoAD\test_video\preprocess_video.py --video "d:\CV\DecoAD_prj\DecoAD\test_video\video (1).avi" --output-root "d:\CV\DecoAD_prj\DecoAD\data"
```

### Data Directory
Data folder, including extracted poses and GT, can refer to [link](https://github.com/orhir/STG-NF/) for the data format. 

## Training/Testing
Training and Evaluating is run using:
```
python UNVAD/main_unsupervised.py    # Unsuperised

python WSVAD/main_unsupervised.py    # Weakly-/Funlly-superised
```

Evaluation of our pretrained model can be done using:
```
python UNVAD/stage1/test.py    # Unsuperised

python WSVAD/stage1/test.py    # Weakly-/Funlly-superised
```


## UFSR Dataset
To the best of our knowledge, this is the first dataset featuring dynamic scenes and incorporating scene-related anomalies.

![Demo](./data/demo.png)

The dataset has been placed at [link](https://pan.baidu.com/s/1gFGQmdg_AjEoZIf2yPGzPQ?pwd=6mx6).
