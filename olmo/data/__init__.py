from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from torch.utils.data import DataLoader, DistributedSampler

from ..aliases import PathOrStr
from ..config import DataConfig, TrainConfig
from ..exceptions import OLMoConfigurationError
from ..torch_util import barrier, get_global_rank, get_world_size
from .collator import DataCollator
from .iterable_dataset import IterableDataset, IterableDatasetFixedIndex
from .memmap_dataset import MemMapDataset
from .dict_memmap_dataset import DictMemmapDataset, DictMemmapWriter

from ..tokenizer import Tokenizer
from ..eval.downstream import label_to_task_map
from ..config import EvaluatorConfig, TrainConfig
from ..eval.downstream import ICLMultiChoiceTaskDataset

from olmo.registry import DATA_DICT

import torch


__all__ = [
    "MemMapDataset",
    "DataCollator",
    "IterableDataset",
    "build_eval_dataloader",
    "build_train_dataloader",
    "DictMemmapDataset",
    "DictMemmapWriter",
    "build_train_dataloader_plus",
]


def build_memmap_dataset(
    train_config: TrainConfig, data_config: DataConfig, include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []
    if data_config.paths:
        if data_config.datasets:
            raise OLMoConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = data_config.paths
        if isinstance(paths, str):
            paths = list(Path(DATA_DICT.get(paths, paths)).glob("*.npy"))
        for path in paths:
            metadata.append({"path": str(path)})
    elif data_config.datasets:
        paths = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            if isinstance(label_paths, str):
                path = DATA_DICT.get(label_paths, label_paths)
                print(f"Loading {label} from {path}")
                label_paths = list(Path(path).glob("*.npy"))
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    return MemMapDataset(
        *paths,
        chunk_size=train_config.model.max_sequence_length,
        memmap_dtype=data_config.effective_memmap_dtype,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=train_config.model.pad_token_id,
        generate_attention_mask=data_config.generate_attention_mask,
        label_mask_paths=cast(Optional[List[PathOrStr]], data_config.label_mask_paths),
    )


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_memmap_dataset(train_config, data_config, include_instance_metadata=True)
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id)
    if data_config.drop_last:
        # Make sure batch size is small enough.
        samples_per_device = len(dataset) // get_world_size()
        batch_size = min(batch_size, samples_per_device)
        assert batch_size > 0, f"dataset for {data_config.paths} is too small"
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_train_dataloader(train_config: TrainConfig, world_size: Optional[int] = None) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    collator = DataCollator(
        pad_direction=train_config.data.pad_direction, pad_token_id=train_config.model.pad_token_id
    )
    dataset = build_memmap_dataset(train_config, train_config.data, include_instance_metadata=False)
    work_dir = Path(train_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise OLMoConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    barrier()
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            train_config.global_train_batch_size,
            seed=seed + (train_config.epoch or 0),
            shuffle=True,
            drop_last=train_config.data.drop_last,
            world_size=world_size,
            work_dir=work_dir,
        ),
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )


def build_train_dataloader_fixed_index(train_config: TrainConfig) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    collator = DataCollator(
        pad_direction=train_config.data.pad_direction, pad_token_id=train_config.model.pad_token_id
    )
    dataset = build_memmap_dataset(train_config, train_config.data, include_instance_metadata=False)
    barrier()
    return DataLoader(
        IterableDatasetFixedIndex(
            dataset,  # type: ignore
            train_config.global_train_batch_size,
            input_index_path=train_config.data.index_path,
            seed=train_config.seed + (train_config.epoch or 0),
            shuffle=True,
            drop_last=train_config.data.drop_last,
        ),
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )


def build_sft_dataloader(
    train_config: TrainConfig,
    eval_configs: EvaluatorConfig,
) -> DataLoader:
    tokenizer = Tokenizer.from_train_config(train_config)
    datasets = []
    for eval_config in eval_configs:
        task_kwargs = {}
        task_class = label_to_task_map[eval_config.label]
        if isinstance(task_class, tuple):
            task_class, task_kwargs = task_class
        task_kwargs["sft_use_label"] = eval_config.sft_use_label
        task_kwargs["sft"] = eval_config.sft
        task_kwargs["model_ctx_len"] = train_config.model.max_sequence_length
        dataset = task_class(tokenizer=tokenizer, **task_kwargs)
        datasets.append(dataset)
        assert isinstance(dataset, ICLMultiChoiceTaskDataset)  # NOTE collate only implemented for ICL
    collate_fn = datasets[0].collate_fn

    sft_dataset = torch.utils.data.ConcatDataset(datasets)
    work_dir = Path(train_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise OLMoConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    barrier()
    return DataLoader(
        IterableDataset(
            sft_dataset,  # type: ignore
            train_config.global_train_batch_size,
            seed=train_config.seed + (train_config.epoch or 0),
            shuffle=True,
            drop_last=train_config.data.drop_last,
            work_dir=work_dir,
        ),
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collate_fn,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
