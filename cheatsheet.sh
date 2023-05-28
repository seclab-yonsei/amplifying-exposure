export HF_DATASETS_CACHE="/mnt/block-storage/.cache/huggingface/datasets"

deepspeed --num_gpus=2 train.py --deepspeed ds_config.json

git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
DS_BUILD_OPS=1 pip install .