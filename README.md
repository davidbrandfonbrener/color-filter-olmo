# CoLoR-Filter

## Description

Code for the paper "CoLoR-Filter: Conditional Loss Reduction Filtering for Targeted Language Model Pre-training". 

This repo is built on top of [OLMo](https://github.com/allenai/OLMo).

The repo contains modifications to the base `olmo/` directory as well as the `scripts/` and `configs/` necessary to reproduce the main results.

We also release models and data at (https://huggingface.co/hlzhang109/CoLoR-filter).


## Installation

Build a conda environment and install the required packages:

```bash
conda create -n color-filter python=3.10
conda activate color-filter
pip install -e .[all]
```

## Download data and set paths

WARNING: there is 125GB of non-c4 data and XXGB of tokenized c4 data. If you do not want to download all of the data, use git lfs to only download the files you want.

1. Download the data from (https://huggingface.co/hlzhang109/CoLoR-filter). This can be done with e.g. `git clone https://huggingface.co/hlzhang109/CoLoR-filter`

2. Change the `download_path` in `olmo/registry.py` to point to the downloaded data on your machine.

3. (Optional): set the environment variable `CHECKPOINTS_PATH`, otherwise we will default to using `./ckpts/` to store checkpoints when training models.


## Training on our selected data from C4

1. Run `bash scripts/launch_sweep.sh configs/sweeps/color-1b.yaml 1` to start training the 1st job from the sweep. Change 1 to a different index to start a different job. If using slurm, you can launch the sweep using an sbatch array job.

Note: We use C4 from [dolma v1.6](https://huggingface.co/datasets/allenai/dolma) tokenized with the `allenai/eleuther-ai-gpt-neox-20b-pii-special` tokenizer.


## Selecting from your data using our auxiliary models

1. Format data into npy memmap arrays as in our copy of C4.

2. Run `bash scripts/launch_sweep.sh configs/sweeps/score-parallel.yaml 1` to score data. 

3. Add your score to `SCORE_DICT` in `olmo/registry.py`. and modify `configs/sweeps/gen-idx.yaml` to point to your scores.

4. Run `bash scripts/select_index_sweep.sh configs/sweeps/gen-idx.yaml 1` to create an index of the selected data. 

5. Add the index to `INDEX_DICT` in `olmo/registry.py` and modify `configs/sweeps/color-1b.yaml` to point to your index.

6. Train on the selected data as in the previous section. Modify the `data.index_path` in the config to point to the new indices created in the previous step.


## Training your own auxiliary models

Note: this will allow you to train from scratch, no need to download any data.

1. Format the data into npy memmap arrays as in our copy of C4 and our downstream data. This can be done by tokenizing your data using [dolma](https://huggingface.co/datasets/allenai/dolma).

2. Pretrain the prior model. For an example, see `configs/pretrain.yaml`.

3. Add the prior model to `MODEL_DICT` in `olmo/registry.py`.

4. Finetune the conditional model. For an example, see `configs/finetune-books.yaml` or `configs/finetune-down.yaml`.

5. Add the prior and conditional models to `MODEL_DICT` in `olmo/registry.py`.

6. Select the data and train on selected data as in the previous sections.


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