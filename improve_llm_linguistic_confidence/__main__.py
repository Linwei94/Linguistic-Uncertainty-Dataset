from llm_linguistic_confidence_study.datasets import load_dataset
from llm_linguistic_confidence_study.confidence_extraction_methods import ConfidenceExtractor
from .train import Finetune
from omegaconf import OmegaConf, DictConfig
import hydra
import logging
import os
from hydra.core.hydra_config import HydraConfig
import pandas as pd

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logging.info(OmegaConf.to_yaml(cfg, resolve=True))
    dataset = load_dataset(cfg.dataset)

    if cfg.finetune.task == 'preprocess':
        confidence_extractor = ConfidenceExtractor(cfg.confidence_extractor, cfg.qa_model)
        responses_df = confidence_extractor(dataset, cfg.pre_runned_batch.temp_file_path, None)
        responses_df=pd.read_csv(r"improve_llm_linguistic_confidence/res/dataset/temp.csv")
        if cfg.confidence_extractor.task == "entailment":
            confindencemap = {
                'high': [0.8, 1.0],
                'moderate': [0.6, 0.8],
                'low': [0.4, 0.6],
                'lowest': [0.2, 0.4],
                'completely uncertain': [0.0, 0.2],
            }
            def map_confidence_level(conf):
                for level, (low, high) in confindencemap.items():
                    if low <= conf < high or (level == 'high' and conf == high):
                        return level
                return 'completely uncertain'
            global_df = dataset.get_dataset()
            data = []
            for idx, row in responses_df.iterrows():
                for i in range(cfg.confidence_extraction_method_cfg.sample_times):
                    data.append({
                        "problem": row["problem"],
                        "response": row[f"response_{i}"],
                        "confidence level": row["confidence level"],
                    })
            long_responses = pd.DataFrame(data, columns=["problem", "response", "confidence level"])
            merged_df = pd.merge(
                long_responses,
                global_df[['problem', 'confidence', 'sentence']],
                how='left',
                left_on=['problem', 'confidence level'],
                right_on=['problem', 'confidence']
            )
            df = merged_df[['response', 'sentence']]
            responses_df = df
            responses_df.to_csv(os.path.join(cfg.pre_runned_batch.temp_file_path, "responses.csv"), index=False)
    elif cfg.finetune.task == 'train':
        responses_df = pd.read_csv(os.path.join(cfg.pre_runned_batch.temp_file_path, "responses.csv"))
        trainer = Finetune(cfg.finetune, cfg.qa_model)
        trainer(responses_df)

if __name__ == "__main__":
    main()