# CoLoR-Filter

## Description

Code for the paper "CoLoR-Filter: Conditional Loss Reduction Filtering for Targeted Language Model Pre-training". 

This repo is built on top of [OLMo](https://github.com/allenai/OLMo).

The repo contains modifications to the base `olmo/` directory as well as the `scripts/` and `configs/` necessary to reproduce the main results.

This README is structured to show the full pipeline for running CoLoR-Filter from scratch. If you want to skip some step of the process, we also provide datasets and models from intermediate steps in the pipeline at https://huggingface.co/hlzhang109/CoLoR-filter.

If you only want to use our filtered data, see https://huggingface.co/datasets/davidbrandfonbrener/color-filtered-c4 for the raw untokenized data.


## Installation

Build a conda environment and install the required packages:

```bash
conda create -n color-filter python=3.10
conda activate color-filter
pip install -e .[all]
```

## Schematic of the pipeline

![alt text](https://github.com/davidbrandfonbrener/color-filter-olmo/blob/main/pipeline.png?raw=true)

## 1. Train the auxiliary models

Preconditions: c4 is tokenized and listed in `DATA_DICT` in `olmo/registry.py`, and the desired downstream data is also listed in `DATA_DICT` or can be tokenized on the fly (as we do for downstream data).

Note: we use C4 from [dolma v1.6](https://huggingface.co/datasets/allenai/dolma) tokenized with the `allenai/eleuther-ai-gpt-neox-20b-pii-special` tokenizer. Data needs to be formatted as npy memmap arrays which can be done with the [dolma package](https://github.com/allenai/dolma).

1. Pretrain the prior model. For example, run `bash scripts/launch_sweep.sh configs/sweeps/pretrain.yaml 1` to start training the 1st job from the sweep. Change 1 to a different index to start a different job. If using slurm, you can launch the sweep using an sbatch array job.

2. Add the prior model to `MODEL_DICT` in `olmo/registry.py`.

3. Finetune the conditional model. For example, run `bash scripts/launch_sweep.sh configs/sweeps/finetune-down.yaml 1`.

4. Add the prior and conditional models to `MODEL_DICT` in `olmo/registry.py`.


## 2. Select from your data using auxiliary models

Preconditions: c4 is tokenized and listed in `DATA_DICT` in `olmo/registry.py`, and the prior and conditional models are listed in `MODEL_DICT` in `olmo/registry.py`.

1. Run `bash scripts/launch_sweep.sh configs/sweeps/score-parallel.yaml 1` to score data.

2. Add your score to `SCORE_DICT` in `olmo/registry.py`. and modify `configs/sweeps/gen-idx.yaml` to point to your scores.

3. Run `bash scripts/select_index_sweep.sh configs/sweeps/gen-idx.yaml 1` to create an index of the selected data.

4. Add the index to `INDEX_DICT` in `olmo/registry.py` and modify `configs/sweeps/color-1b.yaml` to point to your index.


## 3. Train on our selected data from C4

Preconditions: c4 is tokenized and listed in `DATA_DICT` in `olmo/registry.py` and the data index is listed in `INDEX_DICT` in `olmo/registry.py`.

Note: Set the environment variable `CHECKPOINTS_PATH`, otherwise we will default to using `./ckpts/` to store checkpoints when training models.

1. Run `bash scripts/launch_sweep.sh configs/sweeps/color-1b.yaml 1` to train a 1b model on the filtered data.


## Optional: download our data to skip steps

WARNING: the data is 400GB, most of which contains a full tokenized copy of c4. If you do not want to download all of the data, use the [huggingface-cli tool](https://huggingface.co/docs/huggingface_hub/v0.23.4/guides/cli#huggingface-cli-download) to only download the parts that you want.

1. Download the data: `huggingface-cli download hlzhang109/CoLoR-filter --local-dir YOUR_PATH`. This will download the data to your huggingface cache and create a local-dir with symbolic links to the data. If you actually want the data at `YOUR_PATH`, set it as the `--cache-dir` in the command.

2. Change the `download_path` in `olmo/registry.py` to point to the downloaded data on your machine, i.e. to `YOUR_PATH`.


## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{,
  title={},
  author={},
  booktitle={},
  year={},
}
```