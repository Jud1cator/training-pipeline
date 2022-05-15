from typing import Optional

import torch
import torch.nn as nn

from src.losses.segmentation.dice_loss import DiceLoss


class ComboLoss(nn.Module):

    def __init__(
            self,
            alpha: float = 0.5,
            weight: Optional[torch.Tensor] = None,
            ignore_index: int = -100,
            reduction: str = 'mean'
    ) -> None:
        super(ComboLoss, self).__init__()
        if weight is not None and type(weight) != torch.Tensor:
            weight = torch.tensor(weight, dtype=torch.float32)
        self.alpha = alpha
        self.dice_loss = DiceLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction
        )
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_tensor(input):
            raise TypeError(f'Input type is not a torch.Tensor. Got {type(input)}')
        if not len(input.shape) == 4:
            raise ValueError(f'Invalid input shape, we expect BxNxHxW. Got: {format(input.shape)}')
        if not input.shape[-2:] == target.shape[-2:]:
            raise ValueError(
                f'input and target shapes must be the same. Got: {input.shape, target.shape}')
        if not input.device == target.device:
            raise ValueError(
                f'input and target must be in the same device. Got: {input.device, target.device}')

        dice_loss = self.dice_loss(input, target)
        ce_loss = self.cross_entropy_loss(input, target)
        combo_loss = self.alpha * ce_loss + (1 - self.alpha) * dice_loss

        return combo_loss


######################
# functional interface
######################

def combo_loss(
        input: torch.Tensor,
        target: torch.Tensor
) -> torch.Tensor:
    return ComboLoss()(input, target)
