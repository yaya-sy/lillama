from ..utils import linear_iterator
from tqdm import tqdm
import logging
from typing import Set
import torch
from torch import Tensor
from transformers import PreTrainedModel

class CovarianceComputationHook:
    """
    A callable class as a forward hook.\
    This will be used to compute and store\
    the statistics after each forward of a linear layer.
    """
    def __init__(self, name: str) -> None:
        self.name = name # The name of the linear module
        self.e_yyt: Tensor = None
        self.e_y: Tensor = None
        self.total: float = 0.0 # number of total sequences
    
    def init_statistics(self, dimension: int, device: str) -> None:
        """Initialize the matrices for the expectation computations."""
        self.e_yyt = torch.zeros((dimension, dimension),
                                 device=device,
                                 dtype=torch.bfloat16)
        self.e_y = torch.zeros((dimension, 1),
                               device=device,
                               dtype=torch.bfloat16)


    def __call__(self, module, input, output) -> None:
        """Update statistics for covariance matrix computation."""
        # We estimate the covariance per sequence activations
        # instead of per token activations. For the latter,
        # we would concatenate all the tokens vectors in the sequence:
        # >>> output = output.view(-1, output.shape[-1])
        # However, this approach requires a huge memory, so batching would be the solution:
        # for output in output.view(-1, output.shape[-1]).split(32):
        output = output.mean(1) # per sequence activations
        b, d = output.shape
        output = output.unsqueeze(-1)
        if self.e_yyt is None or self.e_y is None:
            self.init_statistics(d, device=output.device)
        self.e_yyt += (output @ output.transpose(2, 1)).sum(0)
        self.e_y += output.sum(0)
        self.total += b
    
    def compute_covariance_matrix(self) -> Tensor:
        """Compute covariance matrix."""
        e_yyt = self.e_yyt / self.total
        e_y = self.e_y / self.total
        self.cov = e_yyt - e_y @ e_y.T
        self.cov = self.cov.to(dtype=torch.float32)
    
    def __hash__(self):
        return hash(self.name)

def register_forward_hooks(llm: PreTrainedModel
                           ) -> Set[CovarianceComputationHook]:
    """Register a covariance matrix estimator for each linear layer."""
    hooks = set()
    handles = []
    for name, module in linear_iterator(llm):
        module: torch.nn.Module
        hook = CovarianceComputationHook(name)
        handle = module.register_forward_hook(hook)
        handles.append(handle)
        hooks.add(hook)
    return hooks, handles

@torch.inference_mode(mode=True)
def forward_dataset(dataset,
                    llm: PreTrainedModel) -> None:
    """Forwards all the dataset through the LLM and computes the statistics."""
    # dataset = dataset.select(range(16))
    p_bar = tqdm(total=len(dataset))
    pad_token_id = dataset.collate_fn.pad_token_id
    for batch in dataset:
        inputs = batch["input_ids"].to(llm.device)
        llm(inputs)
        p_bar.update((inputs != pad_token_id).sum().item())

def eighendecomposition(covariance_matrix: Tensor,
                        device=torch.device
                        ) -> Tensor:
    """Eighendecomposition of a precomputed variance matrix."""
    # eigh because the covariance matrix is a real symmetric matrix.
    return torch.linalg.eigh(covariance_matrix.to(device=device))

def decompose_covariance_matrices(hooks: Set[CovarianceComputationHook],
                                  device: torch.device):
    """Iterate over the covariance matrices and decompose them."""
    logging.info("Running the decomposition...")
    state_dict = {}
    for hook in tqdm(hooks, total=len(hooks)):
        # first, compute the covatiance matrix
        logging.info("computing the covariance matrix...")
        hook.compute_covariance_matrix()
        # then, get the eighenvalue and eighvectors from the covariance matrix
        logging.info("computing eighenvalues...")
        _, E = eighendecomposition(hook.cov, device=device)
        state_dict[hook.name] = E.to(dtype=torch.bfloat16)
    return state_dict

def activation_pca(llm, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = llm.eval()
    hooks, handles = register_forward_hooks(llm)
    forward_dataset(dataset, llm)
    state_dict = decompose_covariance_matrices(hooks,
                                         device=device)
    for handle in handles:
        handle.remove()
    return state_dict

if __name__ == "__main__":
    activation_pca()