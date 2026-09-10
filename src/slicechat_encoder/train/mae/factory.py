import json
from typing import Dict, Any

import torch

from .models_mae import MaskedAutoencoderMamba


def create_model_from_config(config: dict) -> torch.nn.Module:
    """
    Create a Mamba MAE model from a config dict.
    
    Args:
        config: Config dict.

    Returns:
        Model instance.
    """
    print(f"Creating model from config: {json.dumps(config, indent=4)}")
    return MaskedAutoencoderMamba(
        encoder_config=config['encoder_config'],
        decoder_config=config['decoder_config'],
        feature_dim=config.get('feature_dim', 768),
        norm_feature_loss=config.get('norm_feature_loss', False),
    )


def convert_mae_ckpt_to_encoder(ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    new_ckpt = {}

    # MAE checkpoints use 'model' key instead of 'state_dict'
    model_state_dict = ckpt["model"]

    for key, value in model_state_dict.items():
        if key.startswith('encoder.'):
            new_ckpt[key.replace("encoder.", "", 1)] = value

    return new_ckpt
