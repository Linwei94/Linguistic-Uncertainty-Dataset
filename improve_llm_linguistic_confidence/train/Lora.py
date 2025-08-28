from llm_linguistic_confidence_study.models import LLM
from peft import LoraConfig, get_peft_model
from omegaconf import DictConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from typing import Dict, List
import os 
from datasets import Dataset


class LoraTrainer():
    def __init__(self, finetune_cfg: DictConfig, qa_model_cfg: DictConfig) -> None:
        self.finetune_cfg = finetune_cfg
        self.qa_model_cfg = qa_model_cfg
        self.qa_model = self.get_qa_model()

    def __call__(self, responses_df) -> None:
        base_model = self.qa_model.model.model
        lora_config = self.get_lora_cfg(self.finetune_cfg)
        base_model = get_peft_model(base_model, lora_config)

        tokenizer = self.qa_model.model.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

        data = Dataset.from_pandas(responses_df)
        split_dataset = data.train_test_split(test_size=self.finetune_cfg.test_size, seed=self.finetune_cfg.seed)
        format_dataset = split_dataset.map(self._format)
        tokenized_dataset  = format_dataset.map(
            self._tokenize,
            batched = self.finetune_cfg.dataset_enable_batched,
            batch_size = self.finetune_cfg.dataset_batch_size,
            remove_columns=data.column_names,
        )

        training_cfg = self.get_training_cfg()
        trainer = Trainer(
            model=base_model,
            args=training_cfg,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            processing_class=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

    def get_qa_model(self):
        return LLM(self.qa_model_cfg)

    def get_lora_cfg(self):
        if not self.finetune_cfg.name == "lora":
            raise ValueError(f"finetune process invalid: {self.finetune_cfg.name}")
        return LoraConfig(
            r=self.finetune_cfg.r,
            target_modules=self.finetune_cfg.target_modules,
            lora_alpha=self.finetune_cfg.lora_alpha,
            lora_dropout=self.finetune_cfg.lora_dropout,
            bias=self.finetune_cfg.bias,
            task_type=self.finetune_cfg.task_type,
        )

    def get_training_cfg(self):
        return TrainingArguments(
            output_dir=os.path.join(self.finetune_cfg.output_dir, "lora_weight", self.qa_model_cfg.name),
            learning_rate=self.finetune_cfg.learning_rate,
            per_device_train_batch_size=self.finetune_cfg.per_device_train_batch_size,
            per_device_eval_batch_size=self.finetune_cfg.per_device_eval_batch_size,
            gradient_accumulation_steps=self.finetune_cfg.gradient_accumulation_steps,
            num_train_epochs=self.finetune_cfg.num_train_epochs,
            eval_strategy=self.finetune_cfg.eval_strategy,
            save_strategy=self.finetune_cfg.save_strategy,
        )
    
    def _format(self, example):
        return {
            "text": f"User: {example["response"]}\nAssistant: {example["sentence"]}"
        }
        
    def _tokenize(self, example: Dict) -> Dict[str, List[int]]:
        return self.qa_model.model.tokenizer(
                example["text"],
                truncation=True,
                max_length=self.qa_model_cfg.max_tokens,
            )

        # prompt_text = f"User: {answer_text}\nAssistant:"

        # prompt_ids = self.tokenizer.encode(
        #     prompt_text,
        #     add_special_tokens=True,
        #     padding=False,       
        #     truncation=True, 
        #     max_length=1024    
        # )
        # target_ids = self.tokenizer.encode(
        #     target_text,
        #     add_special_tokens=False,
        #     truncation=True,
        #     max_length=1024
        # ) + [self.tokenizer.eos_token_id]

        # input_ids = prompt_ids + target_ids
        # labels = [-100] * len(prompt_ids) + target_ids
        # attention_mask = [1] * len(input_ids)

        # return {
        #     "input_ids": input_ids,
        #     "labels": labels,
        #     "attention_mask": attention_mask,
        # }