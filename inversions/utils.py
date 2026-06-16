import numpy as np
import torch
from PIL import Image


def is_float16(model_type):
    if model_type == "SDXL":
        return True
    elif model_type == "SDXLv1.0":
        return True
    elif model_type == "SDXL_Turbo":
        return True
    elif model_type == "Model_Type.LCM_SDXL":
        return True
    elif model_type == "SDv1.5":
        return False
    elif model_type == "Model_Type.SD14":
        return False
    elif model_type == "SDv2.1":
        return False
    elif model_type == "Model_Type.SD21_Turbo":
        return False
    return False

def pil2tensor(image: Image.Image, normalize: bool = False):
    """pillow image to tensor

    Args:
        image (pil.Image)
        normalize (bool): whether to normalize to [-1, 1]. Defaults to False.

    Returns:
        torch.Tensor
    """
    image = np.array(image).astype(np.float32) / 255.0  # normalize to [0, 1]
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # add batch dimension
    if normalize:
        image = 2.0 * image - 1.0  # normalize to [-1, 1]
    return image


def tensor2pil(tensor: torch.Tensor, denormalize: bool = False):
    """tensor to pillow image

    Args:
        tensor (torch.Tensor)
        denormalize (bool): whether to denormalize to [0, 1]. Defaults to False.

    Returns:
        pil.Image
    """
    if denormalize:
        tensor = (tensor + 1.0) / 2.0
    tensor = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
    return Image.fromarray(tensor.astype(np.uint8))
