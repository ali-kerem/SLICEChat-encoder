import os
import torch
import yaml
from ..mamba.vision_mamba import make_mamba_encoder_from_config


def make_encoder_from_config(config: dict, **kwargs) -> torch.nn.Module:
    """Create an encoder model from a config dict."""

    from ..mamba.vision_mamba import make_mamba_encoder_from_config

    return make_mamba_encoder_from_config(config)

def make_encoder_from_ckpt(ckpt_path: str, device: torch.device = "cpu") -> torch.nn.Module:
    """Load and initialize an encoder model from OpenCLIP or MAE checkpoint and config.
    
    This function loads an encoder model by reading the model configuration
    from the associated YAML file, creating the model architecture, and
    loading the converted checkpoint weights.
    """
    model_config_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)), "model_cfg.yaml")
    model_config = yaml.safe_load(open(model_config_path))

    if "vision_cfg" in model_config: # OpenCLIP
        from ...train.open_clip.open_clip.factory import convert_openclip_ckpt_to_vision_tower
        model_config = model_config["vision_cfg"]
        convert_func = convert_openclip_ckpt_to_vision_tower
    elif "encoder_config" in model_config: # MAE
        from ...train.mae.factory import convert_mae_ckpt_to_encoder
        model_config = model_config["encoder_config"]
        convert_func = convert_mae_ckpt_to_encoder
    else:
        raise ValueError(f"Invalid model config: {model_config}")
    
    model = make_mamba_encoder_from_config(model_config)
    model.load_state_dict(convert_func(ckpt_path))

    model = model.to(device=device, dtype=torch.bfloat16)

    return model
