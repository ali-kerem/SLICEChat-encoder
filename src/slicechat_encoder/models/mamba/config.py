from slicechat_encoder.models.vit.config import ViTBlockConfig


class MambaEncoderConfig:
    def __init__(self, *, block_structure, num_blocks, **kwargs):
        self.encoder_embed_dim = kwargs.pop("encoder_embed_dim", 768)

        removed_keys = {"encoder_layers", "hybrid_interval", "use_hybrid"} & kwargs.keys()
        if removed_keys:
            raise TypeError(
                f"Unsupported layout fields: {', '.join(sorted(removed_keys))}. "
                "Use `block_structure` and `num_blocks`."
            )
        self.block_structure = self._normalize_block_structure(block_structure)
        self.num_blocks = int(num_blocks)
        if self.num_blocks <= 0:
            raise ValueError(f"`num_blocks` must be > 0 (got {self.num_blocks}).")
        
        self.d_state = kwargs.pop("d_state", 16)  # SSM state expansion factor
        self.d_conv = kwargs.pop("d_conv", 4)  # Local convolution width
        self.expand = kwargs.pop("expand", 2)  # Block expansion factor
        
        self.dropout = kwargs.pop("dropout", 0.0)
        self.drop_path_rate = float(kwargs.pop("drop_path_rate", 0.1))
        
        self.rms_norm = kwargs.pop("rms_norm", False)
        self.layernorm_eps = float(kwargs.pop("layernorm_eps", 1e-5))
        self.normalize_output = kwargs.pop("normalize_output", True)
        
        self.residual_in_fp32 = kwargs.pop("residual_in_fp32", False)
        self.fused_add_norm = kwargs.pop("fused_add_norm", False)
        
        # Bidirectional scanning (for vision tasks)
        self.biscan = kwargs.pop("biscan", True)
        
        self.hybrid_vit_config = kwargs.pop("hybrid_vit_config", None)
        self.hybrid_vit_config["hidden_size"] = self.encoder_embed_dim
        self.hybrid_vit_config = ViTBlockConfig(**self.hybrid_vit_config)
        
        self.pos_embed = kwargs.pop("pos_embed", "rope_2d") # rope_2d or sincos_2d
        self.rope_2d_theta = kwargs.pop("rope_2d_theta", 150.0)  # Ignored unless pos_embed is rope_2d
        self.rope_2d_mixed = kwargs.pop("rope_2d_mixed", True)
        
        self.pooler = kwargs.pop("pooler", "average")
        
        self.normalize_image_patches = kwargs.pop("normalize_image_patches", False)
        
        # Token compression (Cropr)
        self.token_compression = kwargs.pop("token_compression", None)
        self.pruning_rate = self._validate_pruning_rate(kwargs.pop("pruning_rate", None))
        self.cropr_cfg = kwargs.pop("cropr_cfg", None)
        self.cropr_after_last_block = kwargs.pop("cropr_after_last_block", False)
        

    @staticmethod
    def _validate_pruning_rate(pruning_rate):
        if pruning_rate is None:
            return None

        if isinstance(pruning_rate, bool):
            raise TypeError("`pruning_rate` must not be a boolean.")

        try:
            value = float(pruning_rate)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "`pruning_rate` must be a number or numeric string "
                f"(got {pruning_rate!r})."
            ) from exc

        if not 0.0 <= value < 1.0:
            raise ValueError(
                "`pruning_rate` must be in [0, 1) "
                f"(got {pruning_rate!r})."
            )

        return value

    @staticmethod
    def _normalize_block_structure(block_structure):
        if not isinstance(block_structure, str):
            raise TypeError(f"`block_structure` must be a string (got {type(block_structure).__name__}).")
        block_structure = "".join(ch for ch in block_structure.upper() if not ch.isspace())
        if not block_structure:
            raise ValueError("`block_structure` must not be empty.")
        invalid = sorted(set(block_structure) - {"M", "T"})
        if invalid:
            raise ValueError(
                "`block_structure` may only contain 'M' for Mamba and 'T' for Transformer "
                f"(got invalid characters: {invalid})."
            )
        return block_structure
