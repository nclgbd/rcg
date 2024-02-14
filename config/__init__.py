# Extension from rtk module

from omegaconf import DictConfig
from dataclasses import dataclass

from rtk.config import BaseConfiguration


@dataclass
class RCGConfiguration(BaseConfiguration):
    # additional params
    seed: int = 0
    class_cond: bool = True
    num_images: int = 3500
    batch_size: int = 64
    epochs: int = 400
    accum_iter: int = 1
    input_size: int = 256
    config: str = ""
    pretrained_rdm_cfg: str = ""
    pretrained_rdm_ckpt: str = ""
    log_dir: str = "logs/"
    weight_decay: float = 0.05
    lr: int = None
    blr: float = 1e-3
    min_lr: float = 0.0
    cosine_lr: bool = True
    warmup_epochs: int = 0
    output_dir: str = "outputs/"
    device: int = 0
    resume: str = ""
    start_epoch: int = 0
    num_workers: int = 24
    pin_mem: bool = True
    evaluate: bool = False
    use_ddim: bool = False
    # distributed settings
    distributed: bool = False
    world_size: int = 1
    local_rank: int = -1
    dist_on_itp: bool = True
    dist_url: str = "env://"
