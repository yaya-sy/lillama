from typing import Literal
from transformers import PreTrainedModel

class Distiller:
    def __init__(self,
                 teacher: PreTrainedModel,
                 student: PreTrainedModel,
                 strategy: Literal["teacher", "student", "teacher+student"]):
        self.teacher = teacher
        self.student = student
        self.strategy = strategy
    
    