from typing import Literal
from dataclasses import dataclass

@dataclass
class DistillationParams:
    output_folder: str
    batch_size: int=32
    lr: float=0.000086
    evaluate: bool=True
    log_interval: int = 32
    distillation_loss: Literal["teacher", "student", "teacher+student"] = "teacher+student"
    min_rank: int = 512