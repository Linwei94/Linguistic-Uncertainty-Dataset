from omegaconf import DictConfig, ListConfig
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from vllm import LLM, SamplingParams
import csv
from hydra.core.hydra_config import HydraConfig


class Huggingfacemodel:
    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        self.sampling_params = self.get_generate_config(self.model_cfg)

    def __call__(
        self,
        prompts: list[str],
        task_name: str = None,
        batch_job_id: ListConfig | str = None,
    ) -> list[str]:
        # If do not want to store response, delete  line below
        if (batch_job_id == None) or (not os.path.exists(batch_job_id)):
        # If do not want to store response, delete  line above
            responses = []
            messages = [{"role": "user", "content": prompt} for prompt in prompts]
            inputs = [self.tokenizer.apply_chat_template(
                [msg],
                tokenize=self.model_cfg.tokenize_output,
                add_generation_prompt=self.model_cfg.add_generation_prompt,
                enable_thinking=self.model_cfg.enable_thinking,
            ) for msg in messages]
            outputs = self.model.generate(
                inputs,
                self.sampling_params,
            )
            for output in outputs:
                output_ids = output.outputs[0].token_ids
                if 151668 in output_ids:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                else:
                    index = 0
                text = self.tokenizer.decode(
                    output_ids[index:], skip_special_tokens=True
                ).strip("\n")
                responses.append(text)


        # If do not want to store response, delete  lines below
            prerunned_batches_path = HydraConfig.get().runtime.output_dir.replace("results", "prerunned_batches")
            os.makedirs(prerunned_batches_path, exist_ok=True)
            csv_path = os.path.join(prerunned_batches_path, "responses.csv")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                for resp in responses:
                    writer.writerow([resp])
        else:
            responses = []  
            with open(batch_job_id, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        responses.append(row[0])
        # If do not want to store response, delete  lines above
        return responses

    def load_model(self):
        if self.model_cfg.task == "sample":
            if self.model_cfg.name == "Qwen3-8B-uncertainty":
                if not os.path.exists(self.model_cfg.save_path):
                    model = AutoModelForCausalLM.from_pretrained(
                        self.model_cfg.model_name,
                        device_map=self.device,
                        # device_map="cpu",
                    )
                    model = PeftModel.from_pretrained(
                        model, self.model_cfg.lora_weight_path
                    )
                    model = model.merge_and_unload()
                    model.save_pretrained(self.model_cfg.save_path)
                self.model = LLM(
                    model=self.model_cfg.save_path,
                    tokenizer=self.model_cfg.model_name,
                    trust_remote_code=self.model_cfg.trust_remote_code,
                    dtype=self.model_cfg.dtype,
                    tensor_parallel_size=self.model_cfg.tensor_parallel_size,
                    gpu_memory_utilization=self.model_cfg.gpu_memory_utilization,
                    max_model_len=self.model_cfg.max_tokens,
                    disable_log_stats=self.model_cfg.disable_log_stats,
                )
            elif self.model_cfg.name == "Qwen3-8B" or self.model_cfg.name == "Qwen3-32B":
                self.model = LLM(
                    model=self.model_cfg.model_name,
                    tokenizer=self.model_cfg.model_name,
                    trust_remote_code=self.model_cfg.trust_remote_code,
                    dtype=self.model_cfg.dtype,
                    tensor_parallel_size=self.model_cfg.tensor_parallel_size,
                    gpu_memory_utilization=self.model_cfg.gpu_memory_utilization,
                    max_model_len=self.model_cfg.max_tokens,
                    disable_log_stats=self.model_cfg.disable_log_stats,
                )
            else:
                raise ValueError(f"model name invalid: {self.model_cfg.name}")
        elif self.model_cfg.task == "train":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_cfg.model_name,
                device_map=self.device,
                # device_map="cpu",
            )
        else:
            raise ValueError(f"model task invalid: {self.model_cfg.task}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.model_name,
            trust_remote_code=self.model_cfg.trust_remote_code,
        )

    def get_generate_config(self, cfg):
        return SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            max_tokens=cfg.max_tokens,
        )


if __name__ == "__main__":
    aaa = Huggingfacemodel(r"llm_linguistic_confidence_study/configs/qa_model/qwen3-8b.yaml")
    aaa(
        [
            "What is the capital of Australia?",
            "Who was awarded the Oceanography Society's Jerlov Award in 2018?",
        ]
    )
