import os
import torch
import numpy as np
from ..aliases import PathOrStr


# TODO: this is maybe not compatible with multiprocessing in the dataloader


class DictMemmapDataset(torch.utils.data.Dataset):
    # Each memmap file has at most file_seqs number of sequences
    # The index file is a single numpy array with the indices of the sequences
    # The paths should be written to in sorted order to ensure consistency
    def __init__(
        self,
        path: PathOrStr,
        key: str = None,
        seq_len: int = 1,
        file_seqs: int = 1048576,
        memmap_dtype=np.float32,
        load_to_ram: bool = False,
    ):
        if key is None:
            key = "ref_score"
        self.key = key
        self.file_seqs = file_seqs
        self.seq_len = seq_len

        # Load paths
        files_path = os.path.join(path, "files.txt")
        with open(files_path, "r") as f:
            self._memmap_paths = [line.strip() for line in f.readlines()]

        # Load memmaps
        self.memmaps = [
            np.memmap(path, dtype=memmap_dtype, mode="r", shape=(self.file_seqs, self.seq_len))
            for path in self._memmap_paths
        ]
        if load_to_ram:
            self.memmaps = [np.array(memmap) for memmap in self.memmaps]

        # Load index and map to offset
        if os.path.exists(os.path.join(path, "index.npy")):  # For backwards compatibility
            index_path = os.path.join(path, "index.npy")
            self.index = np.load(index_path)
        elif os.path.exists(os.path.join(path, "mmap_index.npy")):
            index_path = os.path.join(path, "mmap_index.npy")
            self.index = np.memmap(index_path, dtype=np.int64, mode="r")
            # Read into memory
            self.index = np.array(self.index)
            # Remove 0-suffix
            n_trailing_0s = 1
            while self.index[-n_trailing_0s] == 0:
                n_trailing_0s += 1
            if n_trailing_0s != 1:
                self.index = self.index[: -(n_trailing_0s - 1)]
        else:
            raise FileNotFoundError("Index file not found")
        self.idx_to_offset = {idx: i for i, idx in enumerate(self.index)}

    def __getitem__(self, idx):
        idx = self.idx_to_offset[idx]
        memmap = self.memmaps[idx // self.file_seqs]
        idx = idx % self.file_seqs
        return {self.key: memmap[idx]}

    def __len__(self):
        return len(self.index)


class DictMemmapWriter:
    def __init__(
        self,
        path: PathOrStr,
        seq_len: int = 1,
        file_seqs: int = 1048576,
        memmap_dtype=np.float32,
    ):
        self.path = path
        self.file_seqs = file_seqs
        self.seq_len = seq_len
        self.memmap_dtype = memmap_dtype

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.mmap_idx = np.memmap(
            os.path.join(self.path, "mmap_index.npy"), dtype=np.int64, mode="w+", shape=(int(1e8),)
        )

        self.curr_file_idx = 0
        self.curr_idx_inside_of_file = 0
        self.curr_idx_inside_of_mmap_idx = 0
        self._add_file()

    def _add_file(self):
        self.curr_file_idx += 1
        self.curr_path = os.path.join(self.path, f"{self.curr_file_idx}.npy")
        # write new file name as a line in files.txt
        with open(os.path.join(self.path, "files.txt"), "a") as f:
            f.write(self.curr_path + "\n")
        # initialize new memmap file
        self.curr_memmap = np.memmap(
            self.curr_path, dtype=self.memmap_dtype, mode="w+", shape=(self.file_seqs, self.seq_len)
        )

    def write(self, idx: np.ndarray, data: np.ndarray):
        assert len(data.shape) == 2  # Assume data is a batch of flat sequences
        for d, i in zip(data, idx):
            if self.curr_idx_inside_of_file == self.file_seqs:
                self.curr_memmap.flush()
                self.mmap_idx.flush()
                self._add_file()
                self.curr_idx_inside_of_file = 0
            # Write data and index
            self.curr_memmap[self.curr_idx_inside_of_file] = d
            self.curr_idx_inside_of_file += 1
            self.mmap_idx[self.curr_idx_inside_of_mmap_idx] = i
            self.curr_idx_inside_of_mmap_idx += 1

    def close(self):
        self.curr_memmap.flush()
        self.mmap_idx.flush()
