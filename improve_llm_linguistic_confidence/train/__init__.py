from .Lora import LoraTrainer

from omegaconf import DictConfig


class Finetune:
    def __init__(self, fintune_cfg: DictConfig, qa_model_cfg: DictConfig):
        self.fintune_cfg = fintune_cfg
        self.qa_model_cfg = qa_model_cfg
        self.finetune = self.get_finetune_method(self.fintune_cfg.name)

    def __call__(self, dataset):
        return self.finetune(dataset)

    def get_finetune_method(self, finetune_method_name):
        if finetune_method_name == "lora":
            return LoraTrainer(self.fintune_cfg, self.qa_model_cfg)
        elif finetune_method_name == "sft":
            pass
        elif finetune_method_name == "all":
            pass
        else:
            raise ValueError(
                f"Invalid finetune method: {finetune_method_name}")
