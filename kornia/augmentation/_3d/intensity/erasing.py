from typing import Dict, Optional, Tuple, Union, cast

import torch
from torch import Tensor

from kornia.augmentation import random_generator as rg
from kornia.augmentation._3d.base import AugmentationBase3D
from kornia.geometry.bbox import bbox_generator, bbox_to_mask

# TODO RectangleEraseGenerator 3D
class RandomErasing(AugmentationBase3D):
    r"""Erase a random paralelepiped of a tensor 3D volume according to a probability p value.

    Erasing randomly selects a paralelepiped region in the 3D volume, and erases its
    pixels by changing them with specific values at the selected volume for each of
    the images in the batch.

    The paralelepiped will have an volume equal to the original image area multiplied by
    a value uniformly sampled between the range [scale[0], scale[1]) and an aspect ratio
    sampled between [ratio[0], ratio[1])

    Args:
        scale:          range of proportion of erased volume against input 3D volume.
        ratio:          range of aspect ratio of erased volume.
        value:          the value to fill the erased territory.
        same_on_batch:  apply the same transformation across the batch.
        p:              probability that the random erasing operation will be performed.
        keepdim:        whether to keep the output shape the same as input (True) or
                        broadcast it to the batch form (False).

    Shape:
        - Input:  :math:`(C, H, W, D)` or :math:`(B, C, H, W, D)`, Optional: :math:`(B, 3, 3, 3)`
        - Output: :math:`(B, C, H, W, D)`

    Note:
        Input tensor must be float and normalized into [0, 1] for the best differentiability support.
        Additionally, this function accepts another transformation tensor (:math:`(B, 3, 3, 3)`), then
        the applied transformation will be merged int to the input transformation tensor and returned.

    Examples:
        >>> rng = torch.manual_seed(0)
        >>> inputs = torch.ones(1, 1, 3, 3, 3)
        >>> aug = RandomErasing(scale=(.25, .4), ratio=(.3, 1/.3), value=0, p=0.5)
        >>> aug(inputs)
        tensor([[[[[1., 1., 1.],
                   [1., 1., 1.],
                   [1., 1., 1.]],
        <BLANKLINE>
                  [[1., 1., 1.],
                   [1., 0., 0.],
                   [1., 0., 0.]],
        <BLANKLINE>
                  [[1., 1., 1.],
                   [1., 0., 0.],
                   [1., 0., 0.]]]]])

    To apply the exact augmenation again, you may take the advantage of the previous parameter state:
        >>> input = torch.randn(1, 3, 32, 32)
        >>> aug = RandomErasing((.4, .8), (.3, 1/.3), p=1.)
        >>> (aug(input) == aug(input, params=aug._params)).all()
        tensor(True)
    """

    # Note: Extra params, inplace=False in Torchvision.
    def __init__(
        self,
        scale: Union[Tensor, Tuple[float, float]] = (0.02, 0.33),
        ratio: Union[Tensor, Tuple[float, float]] = (0.3, 3.3),
        value: float = 0.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        return_transform: Optional[bool] = None,
    ) -> None:
        super().__init__(p=p, return_transform=return_transform, same_on_batch=same_on_batch, keepdim=keepdim)
        self.scale = scale
        self.ratio = ratio
        self.value: float = float(value)
        self._param_generator = cast(rg.RectangleEraseGenerator, rg.RectangleEraseGenerator(scale, ratio, float(value)))

    def apply_transform(
        self, input: Tensor, params: Dict[str, Tensor], transform: Optional[Tensor] = None
    ) -> Tensor:
        _, c, h, w = input.size()
        values = params["values"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, *input.shape[1:]).to(input)

        bboxes = bbox_generator(params["xs"], params["ys"], params["widths"], params["heights"])
        mask = bbox_to_mask(bboxes, w, h)  # Returns B, H, W
        mask = mask.unsqueeze(1).repeat(1, c, 1, 1).to(input)  # Transform to B, c, H, W
        transformed = torch.where(mask == 1.0, values, input)
        return transformed
