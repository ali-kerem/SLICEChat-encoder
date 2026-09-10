from .base import BasePooler
from .cls_pooler import CLSPooler
from .average_pooler import AveragePooler
from .attentional_pooler import AttentionalPooler


class PoolerFactory:
    _poolers = {
        "cls": CLSPooler,
        "average": AveragePooler,
        "attention": AttentionalPooler,
    }

    @staticmethod
    def create(pooler_type: str, **kwargs) -> BasePooler:
        return PoolerFactory._poolers[pooler_type](**kwargs)