pip install torch transformers lightning easydict black wandb
pip install transformers[deepspeed]

export HF_DATASETS_CACHE="/mnt/block-storage/.cache/huggingface/datasets"

deepspeed --num_gpus=2 train.py --deepspeed ds_config.json
python train.py
