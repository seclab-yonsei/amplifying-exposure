## Get some private information from arguments.
WANDB_API_KEY=$1

## Copy the github repository.
cp -rf /mnt/prj/mrt /mnt/block-storage/mrt

## Make a directory to save your results. (e.g., log, ckpt, ...)
## See: https://guide.ncloud-docs.com/docs/ai-clova-nsml-1-3
mkdir -p /mnt/block-storeage/$NSML_RUN/ckpt

## Update.
sudo apt-get update

## Install all requirements in local. (i.e., not 'conda env' or 'venv', ...)
cd /mnt/block-storage/mrt

pip install --upgrade pip
pip install torch transformers lightning easydict black wandb
pip install transformers[deepspeed]
sudo apt install libaio-dev

ds_report
wandb login $WANDB_API_KEY

## Export some cache from home (~/) to block-storage.
## We have only 100GB storage in home directory ;(
export HF_DATASETS_CACHE="/mnt/block-storage/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/mnt/block-storage/.cache/huggingface/transformers"

## Make a symbolic link.
ln -s /mnt/block-storage/$NSML_RUN/ckpt ckpt

## Train and record all outputs (stdout, stderr) to a log file.
deepspeed --num_gpus=2 train.py --deepspeed ./assets/ds_config_zero3.json \
  > /mnt/block-storeage/$NSML_RUN/log.log 2>&1
