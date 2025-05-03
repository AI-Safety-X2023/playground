from torch import nn

# https://pytorch.org/docs/2.6/func.batch_norm.html
from torch.func import replace_all_batch_norm_modules_ as disable_bn_modules #noqa


def convert_bn_modules_to_gn(module: nn.Module, default_num_groups: int = 16) -> nn.Module:
    """Recursively traverse module and its children to replace all instances of
    `BatchNorm` with `GroupNorm`.
    Parameters:
        module (Module): a neural network module.
        default_num_groups (int, optional): default parameter `num_groups` of `GroupNorm`,
            if specified. If incompatible with the `BatchNorm` module shape,
            the number of groups is set to the maximum divisor of `BatchNorm.num_features`.
    Returns:
        the module converted in-place (except when it is `BatchNorm` itself).
    
    ## Should I use this function?
    
    **Pros**: GroupNorm does not require data to be i.i.d
        and performs well with small batch sizes.

    **Cons**: GroupNorm is 40% slower than BatchNorm,
        and consumes 33% more GPU memory than BatchNorm.
        See https://stackoverflow.com/questions/58002524
    """
    assert default_num_groups > 0

    mod = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        num_groups = default_num_groups
        while module.num_features % num_groups != 0:
            num_groups -= 1
        mod = nn.GroupNorm(
            num_groups, module.num_features,
            eps=module.eps, affine=module.affine
        )
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        mod.add_module(name, convert_bn_modules_to_gn(child, default_num_groups))
    del module
    return mod

def mem_stats(step: int, name: str):
    """Convenience function in optimization loop to find memory leaks."""
    import torch
    if step % 100 != 0:
        return
    mem = torch.cuda.memory_allocated()
    mem_max = torch.cuda.max_memory_allocated()
    print(f"{step=}: {name}: used={mem / mem_max:.5}, {mem=}, {mem_max=}")