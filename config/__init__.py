# Extension from rtk module

from omegaconf import DictConfig
from dataclasses import dataclass, field

from rtk.config import DatasetConfiguration


@dataclass
class RCGConfiguration:
    # rtk configurations
    datasets: DatasetConfiguration
    job: DictConfig = field(default_factory=dict)
    mlflow: DictConfig = field(default_factory=dict)
    date: str = ""
    timestamp: str = ""
    postfix: str = ""

    # additional params
    seed: int = 0
    class_cond: bool = True
    num_images: int = 3500
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
    # distributed settings
    distributed: bool = False
    world_size: int = 1
    local_rank: int = -1
    dist_on_itp: bool = True
    dist_url: str = "env://"
