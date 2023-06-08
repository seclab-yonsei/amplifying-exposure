## venv/lib/python3.10/site-packages/transformers/generation/utils.py

## Make a virtual environment with python 3.10.
conda init bash
conda config --set auto_activate_base false
conda create -n py310 python=3.10
conda activate py310
python -m venv venv

## Install some libraries in `venv`.
pip install --upgrade pip
pip install torch transformers lightning easydict black wandb
pip install transformers[deepspeed]
sudo apt install libaio-dev

ds_report
wandb login

## Default: ~/.cache/huggingface/datasets
export HF_DATASETS_CACHE="/mnt/block-storage/.cache/huggingface/datasets"
## Default: ~/.cache/huggingface/transformers
export TRANSFORMERS_CACHE="/mnt/block-storage/.cache/huggingface/transformers"

## Make a symbolic link.
ln -s /mnt/block-storage/ckpt ckpt

## Train.
deepspeed --num_gpus=1 train.py --deepspeed ./assets/ds_config_zero3.json
deepspeed --num_gpus=2 train.py --deepspeed ./assets/ds_config_zero3.json
# python train.py
