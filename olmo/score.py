from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional
from pathlib import Path

import numpy as np
import torch
import wandb

from .data import DictMemmapWriter
from .train import Trainer, SpeedMonitor
from .torch_util import get_world_size, move_to_device

log = logging.getLogger(__name__)


# TODO: change reference name!

class Scorer(Trainer):

    def score_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        batch_data: Dict[str, torch.Tensor] = {}

        # Write data-indices to file.
        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)
        micro_batches = self.split_batch(batch)
        batch_scores = []
        reference_batch_loss = torch.tensor(0.0, device=self.device)
        for micro_batch in micro_batches:
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                reference_loss, _ = self.model_forward(micro_batch, loss_reduction="none", return_logits=False)
                reference_loss = reference_loss.mean(dim=-1, keepdim=True)
                batch_scores.append(reference_loss)

                reference_batch_loss += reference_loss.mean().detach() / len(micro_batches)
        batch_scores = torch.concatenate(batch_scores, dim=0)
        batch_data["ref_score"] = batch_scores.detach().cpu()
        batch_data["index"] = batch["index"].detach().cpu()
        metrics["train/ReferenceLoss"] = reference_batch_loss.item()
        return metrics, batch_data

    def score_reference(self):
        self._start_time = time.time()

        self.fsdp_model.eval()

        # Initializer dataset writer
        data_writer = DictMemmapWriter(
            Path(self.cfg.save_folder) / "ref_score",
            memmap_dtype=np.float32,
            seq_len=1,
        )

        # Initialize monitors.
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb.run is not None:
                wandb.log(sys_metrics, step=0)

        # Eval
        first_batch: bool = True
        cancel_initiated: bool = False
        stop_at: Optional[int] = self.cfg.stop_at

        if stop_at is None and self.max_epochs == 1:
            stop_at = self.max_steps

        # Maybe fast forward data for parallelism
        if self.cfg.data_start_step is not None:
            self.dataset.start_index = int(self.cfg.data_start_step) * self.cfg.global_train_batch_size

        for batch in self.train_loader:
            # Bookkeeping.
            # NOTE: To track the global batch size / number of tokens per batch we make the assumption that all
            # batches see the same number of tokens, which should be the case for language model pre-training
            # (at least when drop_last=True).
            # Alternatively we'd have to use a distributed all reduce over seq_len here, but I don't want that
            # overhead. So for now I'm putting these assertions here so if the assumption is violated it will
            # fail loudly.
            batch_size, seq_len = batch["input_ids"].shape
            assert seq_len == self.cfg.model.max_sequence_length
            assert batch_size == self.cfg.device_train_batch_size
            global_batch_size = batch_size * get_world_size()  # assumes batch size equal across ranks
            self.global_step += 1
            self.global_train_examples_seen_this_epoch += global_batch_size
            self.global_train_tokens_seen += global_batch_size * seq_len
            speed_monitor.batch_start(
                self.global_train_tokens_seen,
                batch_size * seq_len,  # num tokens in batch for this device
                # We start monitoring speed after the first batch since the first
                # batch might be an outlier due to compiling and other initialization overhead.
                record=not first_batch,
            )

            should_log_this_step = self.should_log_this_step()

            # Run on batch
            with torch.no_grad():
                metrics, batch_data = self.score_step(batch)

            # Write outputs
            idx = batch_data["index"]
            ref_score = batch_data["ref_score"]
            data_writer.write(idx, ref_score)

            # Maybe collect other metrics.
            if should_log_this_step:
                # Speed metrics.
                metrics.update(speed_monitor.check())
                # System metrics.
                metrics.update(self.system_metrics())

            # Log metrics to console.
            if self.global_step % self.cfg.console_log_interval == 0:
                self.log_metrics_to_console(f"[step={self.global_step}/{self.max_steps}]", metrics)

            # Log metrics to W&B.
            if (
                wandb.run is not None
                and self.cfg.wandb is not None
                and self.global_step % self.cfg.wandb.log_interval == 0
            ):
                wandb.log(metrics, step=self.global_step)

            # Check if/when run should be canceled.
            if not cancel_initiated and self.global_step % self.cfg.canceled_check_interval == 0:
                cancel_initiated, extra_steps = self.check_if_cancelled()
                if cancel_initiated:
                    stop_at = (
                        self.global_step + extra_steps
                        if stop_at is None
                        else min(self.global_step + extra_steps, stop_at)
                    )

            # End of batch.
            first_batch = False
            if stop_at is not None and self.global_step >= stop_at:
                break

        # Close writer
        data_writer.close()
