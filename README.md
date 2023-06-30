# Going Neural Language Models Vulnerable from Training Data Extraction Attack

This repository is the official implementation of [My Paper Title](https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Environments

Model fine-tuning and inference were performed on two experimental environments:
  1. A single Ubuntu 20.04 LTS machine with one `Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz` (24 cores), two `NVIDIA GeForce RTX 3090` (24GB VRAM), and 251GB of RAM.
  2. A single Ubuntu 18.04 LTS machine with one `...` (16 cores), two `Tesla V100` (32GB VRAM), and 180GB of RAM.

All experiments use `Python 3.10`.

## Requirements

To install requirements:

```bash
pip install --upgrade pip

pip install torch transformers lightning easydict black wandb FastAPI tqdm pandas
DS_BUILD_OPS=0 pip install transformers[deepspeed]
sudo apt-get install -y libaio-dev
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```bash
deepspeed --num_gpus=2 train.py \
  --pretrained_model_name EleutherAI/gpt-neo-1.3B \
  --revision main \
  --samples_per_epoch 10_000 \
  --batch_size 64 \
  --num_workers 24 \
  --wandb_project mrt \
  --ckpt ckpt \
  --every_n_epochs 1 \
  --save_top_k -1 \
  --buffer_size 1_000 \
  --accelerator gpu \
  --devices 2 \
  --precision 16-mixed \
  --accumulate_grad_batches 4 \
  --max_epochs 10 \
  --logging_interval 1 \
  --lr 2e-5 \
  --do_sample \
  --min_new_tokens 64 \
  --max_new_tokens 64 \
  --no_repeat_ngram_size 3 \
  --top_p 0.95 \
  --top_k 40 \
  --rl_n_samples 1 \
  --debug \
  --deepspeed ./assets/ds_config_zero3.json
```

To convert the `deepspeed` checkpoint(s) to fp32 and load it in `PyTorch`, run this command:

```bash
python ./src/zero_to_fp32.py \
  --checkpoint_root_dir ./ckpt/YOUR_CHECKPOINT_ROOT_DIR
```

To extract training dat from the model(s), run this command:

```bash
python extract.py \
  --load_from_checkpoint \
  --checkpoint_path ckpt/YOUR_CHECKPOINT_ROOT_DIR/YOUR_CHECKPOINT_DIR/pytorch_model.bin \
  --pretrained_model_name EleutherAI/gpt-neo-1.3B \
  --revision main \
  --device cuda:0 \
  --n 100_000 \
  --k 100 \
  --batch_size 80 \
  --do_sample \
  --min_new_tokens 256 \
  --max_new_tokens 256 \
  --no_repeat_ngram_size 3 \
  --top_p 0.95 \
  --top_k 40 \
  --debug
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Citation

Please cite below if you make use of the code:

```latex
...
```

## License

```plain
MIT License

Copyright (c) 2023 Myung Gyo Oh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
