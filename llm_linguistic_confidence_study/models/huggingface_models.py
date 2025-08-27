from omegaconf import DictConfig, ListConfig
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from vllm import LLM, SamplingParams


class Huggingface:
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
        responses = []
        batch = self.model_cfg.batch
        print(1)
        for i in range(0, len(prompts), batch):
            batch_prompts = prompts[i : i + batch]
            inputs = self.tokenizer.apply_chat_template(
                batch_prompts,
                # return_tensors="pt",
                # padding=True,
                # truncation=True,
                tokenize=self.model_cfg.tokenize,
                add_generation_prompt=self.model_cfg.add_generation_prompt,
                enable_thinking=self.model_cfg.thinking,
            )
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
            # responses.extend(
            #     self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # )
        print("successfully generate response.")
        return responses

    def load_model(self):
        if self.model_cfg.name == "Qwen/Qwen3-8B-uncertainty":
            if not os.path.exists(self.model_cfg.save_path):
                tokenizer = AutoTokenizer.from_pretrained(self.model_cfg.base_model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_cfg.base_model_id,
                    device_map=self.device,
                )
                model = PeftModel.from_pretrained(
                    model, self.model_cfg.lora_weight_path
                )
                model = model.merge_and_unload()
                model.save_pretrained(self.model_cfg.save_path)
                tokenizer.save_pretrained(self.model_cfg.save_path)
            model = LLM(
                model=self.model_cfg.save_path,
                tensor_parallel_size=self.model_cfg.tensor_parallel_size,
                gpu_memory_utilization=self.model_cfg.gpu_memory_utilization,
                max_model_len=self.model_cfg.max_tokens,
            )
        elif self.model_cfg.name == "Qwen/Qwen3-8B":
            # self.model = LLM(
            #     model=self.model_cfg.name,
            #     tokenizer=self.model_cfg.name,
            #     dtype=self.model_cfg.dtype,
            #     trust_remote_code=self.model_cfg.trust_remote_code,
            #     tensor_parallel_size=self.model_cfg.tensor_parallel_size,
            #     gpu_memory_utilization=self.model_cfg.gpu_memory_utilization,
            #     max_model_len=self.model_cfg.max_tokens,
            #     disable_log_stats=self.model_cfg.disable_log_stats,
            # )
            self.model = LLM(
                model="Qwen/Qwen3-8B",
                tokenizer="Qwen/Qwen3-8B",
                trust_remote_code=True,
                dtype="auto",
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=1024,
                disable_log_stats=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_cfg.base_model_id,
                trust_remote_code=True,
            )

        # return model, tokenizer

    def get_generate_config(self, cfg):
        return SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            min_p=cfg.min_p,
            max_new_tokens=cfg.max_tokens,
        )


if __name__ == "__main__":
    aaa = Huggingface(r"llm_linguistic_confidence_study/configs/qa_model/qwen3-8b.yaml")
    aaa(
        [
            "What is the capital of Australia?",
            "Who was awarded the Oceanography Society's Jerlov Award in 2018?",
        ]
    )
