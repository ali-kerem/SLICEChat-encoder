class ViTBlockConfig:
    """
    Configuration for a *single* Transformer block (used by `ViTBlock`).
    """
    
    def __init__(self, **kwargs):
        # Core dimensions
        self.hidden_size = kwargs.pop("hidden_size", 768)
        self.num_attention_heads = kwargs.pop("num_attention_heads", 12)
        self.qkv_bias = kwargs.pop("qkv_bias", False)

        # MLP sizing: allow either `intermediate_size` or `mlp_ratio`
        self.mlp_ratio = kwargs.pop("mlp_ratio", 4.0)
        self.intermediate_size = kwargs.pop("intermediate_size", int(self.hidden_size * self.mlp_ratio))

        # Activation and normalization
        self.hidden_act = kwargs.pop("hidden_act", "gelu_pytorch_tanh")
        self.layer_norm_eps = float(kwargs.pop("layer_norm_eps", 1e-6))

        # Regularization
        self.attention_dropout = kwargs.pop("attention_dropout", 0.0)
        self.hidden_dropout = kwargs.pop("hidden_dropout", 0.0)
        
    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)

