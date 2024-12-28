import torch
from torch import nn

BEST_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {BEST_DEVICE} device")

def optimize_model(model: nn.Module, long_runs=False) -> nn.Module:
    """
    Send the model to the GPU.
    
    # Parameters
    `long_runs`: Optimize performance for long training or inference times.
    Not recommended for short runs.
    """
    model = model.to(BEST_DEVICE)
    if long_runs:
        # `torch.compile` significantly improves performance after a few runs
        # Also see https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
        model.compile()
    return model