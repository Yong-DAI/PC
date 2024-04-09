Official code for the manuscript "Prompt Customization for Continual Learning"

## trained model could be found in "https://pan.baidu.com/s/1vZIpDEgYh23lla59WQzfOQ?pwd=uigy 
提取码：uigy"

## Environment
The system I used and tested in
- Ubuntu 20.04.4 LTS
- NVIDIA GeForce a100
- Python 3.8

## Usage
First, install the packages below:
```
pytorch==1.12.1
torchvision==0.13.1
timm==0.6.7
pillow==9.2.0
matplotlib==3.5.3
```
These packages can be installed easily by 
```
pip install -r requirements.txt
```

## Data preparation
If you already have CIFAR-100 datasets, pass your dataset path to  `--data-path`.

If the dataset isn't ready, change the download argument in `continual_dataloader.py` as follows
```
datasets.CIFAR100(download=True)
```

## Train
To train a model on CIFAR-100, set the `--data-path` (path to dataset) and `--output-dir` (result logging directory)  and run the main.py


## Evaluation
To evaluate a trained model:
```
set the--use_env in main.py as --eval
And then run the main.py
Or you can directly evaluate the model by our provided trained model in "https://pan.baidu.com/s/1vZIpDEgYh23lla59WQzfOQ?pwd=uigy.

Thanks for your concerning.
