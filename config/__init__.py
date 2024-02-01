# Extension from rtk module

from dataclasses import dataclass

from rtk.config import DatasetConfiguration


@dataclass
class RCGConfiguration:
    datasets: DatasetConfiguration
    seed: int = 0
    batch_size: int = 64
    epochs: int = 400
    accum_iter: int = 1
    input_size: int = 256
    config: str = ""
    log_dir: str = ""
    weight_decay: float = 0.05
    lr: int = None
    blr: float = 1e-3
    min_lr: float = 0.0
    cosine_lr: bool = True
    warmup_epochs: int = 0
    output_dir: str = ""
    device: int = 0
    resume: str = ""
    start_epoch: int = 0
    num_workers: int = 24
    pin_mem: bool = True
    # Distributed settings
    distributed: bool = False
    world_size: int = 1
    local_rank: int = -1
    dist_on_itp: bool = True
    dist_url: str = "env://"
