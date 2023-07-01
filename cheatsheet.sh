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
deepspeed --num_gpus=2 train.py \
  --pretrained_model_name EleutherAI/gpt-neo-1.3B \
  --revision main \
  --batch_size 64 \
  --num_workers 24 \
  --wandb_project mrt \
  --ckpt ckpt \
  --save_every_n_train_steps 100 \
  --save_top_k -1 \
  --accelerator gpu \
  --devices 2 \
  --precision 16-mixed \
  --accumulate_grad_batches 1 \
  --max_steps 1_000 \
  --logging_interval 1 \
  --lr 1e-5 \
  --do_sample \
  --num_beams 1 \
  --min_new_tokens 64 \
  --max_new_tokens 64 \
  --no_repeat_ngram_size 3 \
  --top_p 1 \
  --top_k 50 \
  --alpha 0.002 \
  --rl_n_samples 1 \
  --debug \
  --deepspeed ./assets/ds_config_zero3.json

## Convert deepspeed model to fp32 and make clean.
python ./src/zero_to_fp32.py \
  --checkpoint_root_dir ./ckpt/20230628-205233

## Extract.
python extract.py \
  --load_from_checkpoint \
  --checkpoint_path ckpt/20230628-205233/epoch=0-step=19.ckpt/pytorch_model.bin \
  --pretrained_model_name EleutherAI/gpt-neo-1.3B \
  --revision main \
  --device cuda:0 \
  --n 100_000 \
  --k 100 \
  --batch_size 64 \
  --do_sample \
  --min_new_tokens 256 \
  --max_new_tokens 256 \
  --no_repeat_ngram_size 3 \
  --top_p 0.95 \
  --top_k 40 \
  --debug

python extract.py \
  --load_from_checkpoint \
  --checkpoint_path ckpt/20230628-205233/epoch=9-step=190.ckpt/pytorch_model.bin \
  --pretrained_model_name EleutherAI/gpt-neo-1.3B \
  --revision main \
  --device cuda:1 \
  --n 100_000 \
  --k 100 \
  --batch_size 64 \
  --do_sample \
  --min_new_tokens 256 \
  --max_new_tokens 256 \
  --no_repeat_ngram_size 3 \
  --top_p 0.95 \
  --top_k 40 \
  --debug
