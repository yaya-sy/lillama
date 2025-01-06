"""A general module for features distillation using low-rank weights."""
from .config import DistillationParams
from ..utils import unfreeze
import logging
import re
from pathlib import Path
from tqdm import tqdm
import torch
from torch import nn
from torch.nn.functional import cosine_similarity, sigmoid, l1_loss
import torch.nn.functional as F

class IdentyWhenTraining(nn.Module):
    """A forward without computation for non distillable modules."""
    def __init__(self, module: nn.Module, is_layer: bool=True):
        super().__init__()
        self.module = module
        self.is_layer = is_layer
    
    @torch.no_grad()
    def forward(self, hidden_states, **args):
        if self.training:
            if not self.is_layer:
                return torch.tensor(0, requires_grad=False)
            return (hidden_states,)
        return self.module(hidden_states, **args)

class Distiller(nn.Module):
    def __init__(self,
                 teacher: nn.Module,
                 student: nn.Module,
                 config: DistillationParams,
                 logger: logging.info,
                 name: str=None):
        super().__init__()
        self.config = config
        self.log_idx = 0
        self.output_folder = Path(config.output_folder)
        self.name = name
        self.teacher = teacher
        self.student = student
        self.activations = None
        self.optimizer = torch.optim.AdamW(self.student.parameters(), lr=config.lr)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
        #                                                             mode="min",
        #                                                             factor=0.1,
        #                                                             patience=1)
        self.current_loss_with_student = float("Inf")
        self.current_loss_with_teacher = float("Inf")
        self.best_loss_with_student = float("Inf")
        self.best_loss_with_teacher = float("Inf")
        self.student_losses = []
        self.teacher_losses = []
        self.logger = logger
        self.t = 2

    def l1_cos(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l1 = l1_loss(input=inputs, target=targets, reduction="none").mean(-1)
        cos = sigmoid(cosine_similarity(inputs, targets, dim=-1)).log()
        # minus because we maximize the cosine similarity
        return (l1 - cos).mean()
    
    def kld_loss(self, student_hidden, teacher_hidden):
        student_probs = F.log_softmax(student_hidden / self.t, dim=-1)
        teacher_probs = F.softmax(teacher_hidden / self.t, dim=-1)

        loss_kd = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.t ** 2)

        return loss_kd

    def save_losses(self, tokens: list):
        """Save the losses."""
        if self.teacher_losses:
            assert len(self.teacher_losses) == len(tokens), "Mismatch between teacher losses and tokens."
            with open(self.output_folder / f"{self.name}.with_teacher_activations.losses", "w") as teacher_losses:
                teacher_losses.write("\n".join(f"{t}\t{l}" for t, l in zip(tokens, self.teacher_losses)))
        if self.student_losses:
            assert len(self.student_losses) == len(tokens), "Mismatch between student losses and tokens."
            with open(self.output_folder / f"{self.name}.with_student_activations.losses", "w") as student_losses:
                student_losses.write("\n".join(f"{t}\t{l}" for t, l in zip(tokens, self.student_losses)))
    
    def save(self) -> None:
        """Store best parameters of the student module."""
        path = f"{self.config.output_folder}/checkpoints/{self.name}.pt"
        self.best_student = {k: v.cpu() for k, v in self.student.state_dict().items()}
        torch.save(self.best_student, f=path)
    
    def load_best_student(self):
        print(f"Loading best student for {self.name}...")
        self.student.load_state_dict(self.best_student)

    def log(self, key: str) -> None:
        """Log the current loss."""
        loss = self.current_loss_with_student if key == "Student" else self.current_loss_with_teacher
        self.logger(f"INPUT ACTIVATIONS={key}, LAYER={self.name}, LOSS={loss}, LR={self.optimizer.param_groups[0]['lr']}")

    def detach(self, *args, **kwargs):
        """Detach the tensors so backward will only concern the student module."""
        nargs = ()
        for x in args:
            nargs += (x.detach(),)
        for k in kwargs:
            if kwargs[k] is None or isinstance(kwargs[k], bool):
                continue
            kwargs[k] = kwargs[k].detach() if not isinstance(kwargs[k], tuple) else (t.detach() for t in kwargs[k])
        return nargs, kwargs

    def update_from_student_activations(self, a: torch.Tensor) -> None:
        """
        When the inputs of the stydent come from the previous student layer.
        
        Parameters
        ----------
        - a: Tensor
            The output activations of the previous student layer.
        """
        y = self.activations
        loss = self.l1_cos(inputs=a, targets=y)
        self.current_loss_with_student = loss.item()
        self.student_losses.append(self.current_loss_with_student)
        if self.current_loss_with_student < self.best_loss_with_student:
            self.save()
            self.best_loss_with_student = loss.item()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def update_from_teacher_activations(self, y: torch.Tensor, *args, **kwargs) -> None:
        """
        When the inputs of the student come from the previous teacher layer.
        
        Parameters
        ----------
        - y: Tensor
            The output activations of the previous teacher layer.

        - *args, **kwarks
            The teacher inputs.
        """
        if "student" in self.config.distillation_loss:
            self.activations = y.detach()  # store activations for future comparison
        if "teacher" in self.config.distillation_loss:
            a = self.student(*args, **kwargs) # forward the teacher inputs into student
            a = a[0] if isinstance(a, tuple) else a
            loss = self.l1_cos(inputs=a, targets=y)
            # 'l1.backward()' then 'l2.backward' is equivalent to '(l1 + l2).backward()'. The former saves memory.
            self.current_loss_with_teacher = loss.item()
            self.teacher_losses.append(self.current_loss_with_teacher)
            loss.backward()
        if self.config.distillation_loss == "teacher":
            if self.current_loss_with_teacher < self.best_loss_with_teacher:
                self.save()
                self.best_loss_with_teacher = self.current_loss_with_teacher
            self.optimizer.step()
            self.optimizer.zero_grad()

    def forward_student(self, *args, **kwargs) -> torch.Tensor:
        """Forward the student module."""
        args, kwargs = self.detach(*args, **kwargs)
        o = self.student(*args, **kwargs)
        a = o[0] if isinstance(o, tuple) else o
        if self.training:
            self.update_from_student_activations(a)
            # self.scheduler.step(self.current_loss_with_student)
        return o

    def forward(self, *args, **kwargs):
        """Forward the teacher module."""
        args, kwargs = self.detach(*args, **kwargs)
        o = self.teacher(*args, **kwargs)
        y = o[0] if isinstance(o, tuple) else o
        if self.training:
            self.log_idx += 1
            self.update_from_teacher_activations(y, *args, **kwargs)
            if self.config.log_interval != 0 and self.log_idx >= self.config.log_interval:
                if "teacher" in self.config.distillation_loss:
                    self.log("Teacher")
                if "student" in self.config.distillation_loss:
                    self.log("Student")
                self.log_idx = 0
        return o

class StudentForwarder(nn.Module):
    """A class that forward the student modules of the distiller."""
    def __init__(self, distiller: Distiller):
        super().__init__()
        self.distiller = distiller

    def forward(self, *args, **kwargs):
        return self.distiller.forward_student(*args, **kwargs)

def get_named_layers(layers):
    return [(name, module) for name, module in layers.named_modules() \
            if re.search("^\d+$", name)]

def set_distillers(llm: nn.Module,
                   lr_llm: nn.Module,
                   distill_params: DistillationParams,
                   layers_key: str,
                   ignore_first_layers: int=0,
                   c_layers: list=None,
                   strategy: str="bottom",
                   logger: logging=logging.info):
    layers = get_named_layers(llm)
    total = sum(1 for _ in layers)
    print("$$$$$$$$$$", c_layers)
    for name, module in tqdm(layers, total=total):
        layer = re.search(r'\d+', name).group()
        if int(layer) < ignore_first_layers:
            continue
        if int(layer) not in c_layers:
            logger(f"{name} will not be distilled.")
            if strategy in {"top", "uniform"}:
                print(name)
                # don't do Identy when the top layers will be compressed as we will need to forward
                # through them. Same for uniform.
                continue
            # if bottom, then we can stop the forward earlier
            free_layer = IdentyWhenTraining(module)
            setattr(lr_llm, name, free_layer)
            setattr(llm, name, free_layer)
            continue
        layer = re.search(r'\d+', name).group()
        student = getattr(lr_llm, name)
        unfreeze(student)
        distiller = Distiller(student=student,
                              teacher=module,
                              config=distill_params,
                              name=f"{layers_key}.{name}",
                              logger=logger)
        student_updater = StudentForwarder(distiller)
        setattr(llm, name, distiller)
        setattr(lr_llm, name, student_updater)

def prepare_for_distillation(llm: nn.Module,
                             lr_llm: nn.Module,
                             distill_params: DistillationParams,
                             logger: logging = logging.info,
                             ignore_first_layers: int=0,
                             layers=None,
                             strategy: str="bottom"
                             ) -> None:
    set_distillers(llm=llm.model.layers,
                   lr_llm=lr_llm.model.layers,
                   distill_params=distill_params,
                   layers_key="model.layers",
                   ignore_first_layers=ignore_first_layers,
                   strategy=strategy,
                   c_layers=layers,
                   logger=logger)
    if not hasattr(llm, "lm_head"):
        setattr(llm, "lm_head", nn.Identity())
    if not hasattr(lr_llm, "lm_head"):
        setattr(lr_llm, "lm_head", nn.Identity())

def unset_distillers(llm):
    layers = llm.model.layers
    named_layers = get_named_layers(layers)
    for name, layer in named_layers:
        if isinstance(layer, StudentForwarder):
            setattr(layers, name, layer.distiller.student)
        if isinstance(layer, Distiller):
            setattr(layers, name, layer.student)
        if isinstance(layer, IdentyWhenTraining):
            setattr(layers, name, layer.module)
    if hasattr(llm, "lm_head") and isinstance(llm.lm_head, IdentyWhenTraining):
        setattr(llm, "lm_head", llm.lm_head.module)

def set_distilled_layers_to_llm(distilled_llm, llm):
    distilled_llm_layers = distilled_llm.model.layers
    llm_layers = llm.model.layers
    named_layers = get_named_layers(distilled_llm_layers)
    for name, layer in named_layers:
        if isinstance(layer, StudentForwarder):
            setattr(llm_layers, name, layer.distiller.student)