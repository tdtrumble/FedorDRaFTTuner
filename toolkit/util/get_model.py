from typing import List
from toolkit.models.base_model import BaseModel
from toolkit.config_modules import ModelConfig


def get_all_models() -> List[BaseModel]:
    from extensions_built_in.diffusion_models.krea2 import Krea2Model

    return [Krea2Model]


def get_model_class(config: ModelConfig):
    all_models = get_all_models()
    for ModelClass in all_models:
        if ModelClass.arch == config.arch:
            return ModelClass
    raise ValueError(f"Unsupported model architecture {config.arch!r}; only 'krea2' is available")
