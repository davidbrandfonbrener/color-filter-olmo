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
    learn_paths = extra_data_paths[0].split("+")  # split by plus
    ref_paths = extra_data_paths[1].split("+")

    learn_paths = [SCORE_DICT.get(path, path) for path in learn_paths]
    ref_paths = [SCORE_DICT.get(path, path) for path in ref_paths]

    scores = []
    indices = []
    for learn_path, ref_path in zip(learn_paths, ref_paths):
        print(f"Learn path: {learn_path}")
        print(f"Ref path: {ref_path}")
        print(f"Select frac: {cfg.select_frac}")

        with open(ref_path + "/files.txt", "r") as f:
            ref_files = f.readlines()
        with open(learn_path + "/files.txt", "r") as f:
            learn_files = f.readlines()

        ref_idx = np.memmap(ref_path + "/mmap_index.npy", dtype=np.int64, mode="r")
        ref_idx = np.array(ref_idx)
        print(ref_idx[:10])
        learn_idx = np.memmap(learn_path + "/mmap_index.npy", dtype=np.int64, mode="r")
        learn_idx = np.array(learn_idx)
        print(learn_idx[:10])
        assert (ref_idx == learn_idx).all()

        ref_idx = remove_trailing_zeros(ref_idx)
        indices.append(ref_idx)

        idx_len = len(ref_idx)
        score_len = 0
        for ref_file, learn_file in zip(ref_files, learn_files):
            print(ref_file.split("/")[-1].strip(), learn_file.split("/")[-1].strip())
            ref_file = ref_file.strip()
            learn_file = learn_file.strip()
            refs = np.memmap(ref_file, dtype=np.float32, mode="r", shape=(1048576, 1))
            learns = np.memmap(learn_file, dtype=np.float32, mode="r", shape=(1048576, 1))
            if idx_len - score_len < 1048576:
                refs = refs[: idx_len - score_len]
                learns = learns[: idx_len - score_len]
            score = learns.mean(axis=-1) - refs.mean(axis=-1)
            scores.append(score)
            score_len += len(score)

    scores = np.concatenate(scores)
    print(f"Scores shape: {scores.shape}")
    indices = np.concatenate(indices)
    print(f"Indices shape: {indices.shape}")

    k = int(cfg.select_frac * len(scores))
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
