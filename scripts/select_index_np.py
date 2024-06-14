import numpy as np
import os
import sys

from olmo.config import TrainConfig
from olmo.registry import SCORE_DICT
from olmo.util import clean_opt, prepare_cli_environment


# Data path should be a string of the form: prior1+prior2,cond1+cond2


def remove_trailing_zeros(arr):
    non_zero_indices = np.nonzero(arr)[0]
    last_non_zero_index = non_zero_indices[-1]
    return arr[: last_non_zero_index + 1]


def main(cfg: TrainConfig):
    extra_data_paths = cfg.data.extra_data_paths.split(",")
    prior_paths = extra_data_paths[0].split("+")  # split by plus
    cond_paths = extra_data_paths[1].split("+")

    prior_paths = [SCORE_DICT.get(path, path) for path in prior_paths]
    cond_paths = [SCORE_DICT.get(path, path) for path in cond_paths]

    scores = []
    indices = []
    for prior_path, cond_path in zip(prior_paths, cond_paths):
        print(f"prior path: {prior_path}")
        print(f"cond path: {cond_path}")
        print(f"Tau: {cfg.tau}")

        with open(cond_path + "/files.txt", "r") as f:
            cond_files = f.readlines()
        with open(prior_path + "/files.txt", "r") as f:
            prior_files = f.readlines()

        cond_idx = np.memmap(cond_path + "/mmap_index.npy", dtype=np.int64, mode="r")
        cond_idx = np.array(cond_idx)
        print(cond_idx[:10])
        prior_idx = np.memmap(prior_path + "/mmap_index.npy", dtype=np.int64, mode="r")
        prior_idx = np.array(prior_idx)
        print(prior_idx[:10])
        assert (cond_idx == prior_idx).all()

        cond_idx = remove_trailing_zeros(cond_idx)
        indices.append(cond_idx)

        idx_len = len(cond_idx)
        score_len = 0
        for cond_file, prior_file in zip(cond_files, prior_files):
            print(cond_file.split("/")[-1].strip(), prior_file.split("/")[-1].strip())
            cond_file = cond_file.strip()
            prior_file = prior_file.strip()
            conds = np.memmap(cond_file, dtype=np.float32, mode="r", shape=(1048576, 1))
            priors = np.memmap(prior_file, dtype=np.float32, mode="r", shape=(1048576, 1))
            if idx_len - score_len < 1048576:
                conds = conds[: idx_len - score_len]
                priors = priors[: idx_len - score_len]
            score = priors.mean(axis=-1) - conds.mean(axis=-1)
            scores.append(score)
            score_len += len(score)

    scores = np.concatenate(scores)
    print(f"Scores shape: {scores.shape}")
    indices = np.concatenate(indices)
    print(f"Indices shape: {indices.shape}")

    k = int((1.0 / cfg.tau) * len(scores))
    sorted_score_idx = np.argsort(scores)
    print(f"Min score: {scores[sorted_score_idx[0]]}")
    print(f"Max score: {scores[sorted_score_idx[-1]]}")
    print(f"k score: {scores[sorted_score_idx[-k]]}")

    top_k_score_idx = sorted_score_idx[-k:]

    select_indices = indices[top_k_score_idx]
    print(f"Selected max indices of shape: {select_indices.shape}")

    os.makedirs(cfg.save_folder, exist_ok=True)

    save_path = os.path.join(cfg.save_folder, "selected_indices.npy")
    print(f"Saving indices to {save_path}")
    save_indices = np.memmap(save_path, dtype=np.uint32, mode="w+", shape=(len(select_indices),))
    save_indices[:] = select_indices.astype(np.uint32)
    save_indices.flush()

    if k == 0:
        # Save sorted scores too
        save_path = os.path.join(cfg.save_folder, "scores.npy")
        print(f"Saving scores to {save_path}")
        save_scores = np.memmap(save_path, dtype=np.float32, mode="w+", shape=(len(scores),))
        save_scores[:] = scores[sorted_score_idx].astype(np.float32)
        save_scores.flush()


if __name__ == "__main__":

    prepare_cli_environment()

    yaml_path, args_list = sys.argv[1], sys.argv[2:]

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
