import subprocess
from itertools import product
from omegaconf import OmegaConf
import os
from glob import glob


def flatten(config):
    """Flatten a nested dictionary."""
    flat_config = {}
    for k, v in config.items():
        if isinstance(v, dict) or OmegaConf.is_dict(v):
            for k2, v2 in flatten(v).items():
                flat_config[f"{k}.{k2}"] = v2
        else:
            flat_config[k] = v
    return flat_config


def grid_to_list(grid):
    """Convert a grid to a list of configs."""
    flat_grid = flatten(grid)
    iter_overwrites = {}
    flat_overwrites = {}
    for k, v in flat_grid.items():
        if isinstance(v, list) or OmegaConf.is_list(v):
            iter_overwrites[k] = v
        else:
            flat_overwrites[k] = v

    product_values = list(product(*iter_overwrites.values()))
    grid_list = []
    for values in product_values:
        overwrite_dict = dict(zip(iter_overwrites.keys(), values))
        overwrite_dict.update(flat_overwrites)
        grid_list.append(overwrite_dict)
    return grid_list


def run(cli_args):
    if "select_index" in cli_args:
        script_file = "scripts/select_index_np.py"
    else:
        script_file = "scripts/train.py"

    if "debug" in cli_args:
        print("Debug Mode")
        slurm_cpus_per_task = 4
        slurm_job_id = 0
        slurm_task_id = 1
        checkpoints_path = "/tmp/checkpoints"
    else:
        slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
        slurm_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
        slurm_task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))
        checkpoints_path = os.environ.get("CHECKPOINTS_PATH")

    # Compute overrides
    base_sweep = OmegaConf.load(cli_args.sweep_config)
    if "sweep" in base_sweep:
        list_of_sweeps = base_sweep.pop("sweep")
        config_list = []
        for sweep in list_of_sweeps:
            sweep_config = OmegaConf.merge(base_sweep, sweep)
            config_list += grid_to_list(sweep_config)
    else:
        config_list = grid_to_list(base_sweep)
    overrides = config_list[slurm_task_id - 1]

    if "debug" in cli_args:
        print(len(config_list), config_list)

    launch_args = [
        "srun",
        f"--cpus-per-task={slurm_cpus_per_task}",
        "--distribution=block:block",
        "--kill-on-bad-exit",
        "scripts/run_with_environment.sh",
        f"python",
        "-u",
        script_file,
        cli_args.config,
        f"--run_name=olmo_{slurm_job_id}_{slurm_task_id}",
        f"--save_folder={checkpoints_path}/{slurm_job_id}_{slurm_task_id}/",
    ]
    for k, v in overrides.items():
        launch_args.append(f"--{k}={v}")

    if "debug" in cli_args:
        launch_args.append("--wandb=null")
        launch_args.append("--save_overwrite")
        print(launch_args)
    subprocess.run(launch_args)


def path_glob(*paths):
    out = []
    for path in paths:
        matches = sorted(glob(path))
        if not matches:
            raise FileNotFoundError(f"{path} does not match any files or dirs")
        out.extend(matches)
    return out


if __name__ == "__main__":
    OmegaConf.register_new_resolver("path.glob", path_glob, replace=True)
    cli_args = OmegaConf.from_cli()
    run(cli_args)
