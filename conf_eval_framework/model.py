from openai import OpenAI

from .custom_types import *
from .framework_config import *

class OpenAIModel(ModelBase):
    def __init__(self, model_id: str):
        self.model_id = model_id                               # used for api calling
        self.model_name = MODEL_NAME_LIST[model_id]            # used for file name/display purposes
        self.api_key = OPENAI_API_KEY

    def inference(self, prompt_list: list[str]) -> list[str]:
        try: 
            # batch processing
            pass
        except:
            # single real-time processing
            pass


def get_model(model_id: str) -> ModelBase:
    """
    Factory function to get the model instance based on model_id.
    """
    if model_id in MODEL_NAME_LIST.keys():
        return OpenAIModel(model_id=model_id)
    else:
        raise ValueError(f"Model ID {model_id} is not recognized. Available models: {list(MODEL_NAME_LIST.keys())}")
