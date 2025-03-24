import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from nisp import nisp_mag


def prune_by_strategy(model: nn.Module, strategy: str, pruning_rate: float, protected_modules: list[str]) -> tuple[
    dict, dict]:
    weight_masks = {}
    bias_masks = {}

    print("Starting", strategy, "pruning with rate", pruning_rate)

    if strategy == "nisp":
        weight_masks = nisp_mag(model, pruning_rate, protected_modules)

    parameter_count = 0
    pruned_parameters = 0

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and (name not in protected_modules):

            module_parameters = torch.prod(torch.tensor(module.weight.shape)).item()

            if strategy == "random":
                prune.random_structured(module, name="weight", amount=pruning_rate, dim=0)
            elif strategy == "l2":  # we choose l2 not l1 because we dont want the unpruned filters to be sparse! instead our goal is reducingg communication cost
                prune.ln_structured(module, name="weight", amount=pruning_rate, n=2,
                                    dim=0)  # Prune filters/output neurons
            elif strategy == "nisp":
                module_to_prune = dict(model.named_modules())[name]
                prune.custom_from_mask(module_to_prune, name="weight", mask=weight_masks[name])
            else:
                raise NotImplementedError(f"strategy=={strategy} is not implemented")

            weight_masks[name] = module.weight_mask.detach().clone()

            pruned_module_parameters = module_parameters - torch.sum(weight_masks[name])

            if module.bias is not None:
                module_parameters += len(module.bias.shape)

                bias_mask = (module.weight_mask.sum(dim=1) != 0).float()
                bias_masks[name] = bias_mask.detach().clone()
                prune.custom_from_mask(module, name="bias", mask=bias_mask)
                prune.remove(module, "bias")

                pruned_module_parameters += len(module.bias.shape) - torch.sum(bias_masks[name])

            parameter_count += module_parameters
            pruned_parameters += pruned_module_parameters

            print(name + ": out of", module_parameters, "parameters", pruned_module_parameters.item(), "were pruned.")
            prune.remove(module, "weight")

    print("Out of", parameter_count, "parameters", pruned_parameters.item(), "were pruned.\n")
    return weight_masks, bias_masks


def prune_by_masks(model: nn.Module, weight_masks_dict: dict[str, torch.Tensor],
                   bias_masks_dict: dict[str, torch.Tensor]):
    for module_name, weight_mask in weight_masks_dict.items():
        module_to_prune = dict(model.named_modules())[module_name]
        prune.custom_from_mask(module_to_prune, name="weight", mask=weight_mask)

        prune.remove(module_to_prune, "weight")

        if module_to_prune.bias is not None:
            prune.custom_from_mask(module_to_prune, name="bias", mask=bias_masks_dict[module_name])
            prune.remove(module_to_prune, "bias")


if __name__ == '__main__':
    from utils.pytorch_models import ResNet18
    import copy

    resnet = ResNet18("r18", pretrained=True)  # pretrained=False)
    resnet2 = copy.deepcopy(resnet)

    # resnet(torch.randn(1,10,120,120))

    w_masks, b_masks = prune_by_strategy(resnet, strategy="nisp", pruning_rate=0.5,
                                         protected_modules=["FC"])  # ,protected_modules=["conv1"])

    # print(list(resnet.named_parameters())[30])

    prune_by_masks(resnet2, w_masks, b_masks)

    # print(list(resnet2.named_parameters())[30])
