import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os


class Base_Generator:
    def __init_(self):
        pass


class Generator_Qwen(Base_Generator):
    def __init_(self):
        return super().__init_()


class Generator_Qwen3_4B(Generator_Qwen):
    def __init__(
        self,
        model_id: str = r"Qwen/Qwen3-4B",
        prompt_template_path: str = r"../prompts/generate_answer.txt",
        device: str = "cuda:1",
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 100,
        num_answers: int = 10
    ):
        self.model_id = model_id
        self.prompt_template_path = prompt_template_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.num_answers = num_answers
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        self._load_model()
        self._load_prompt_template()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True)
        self.model.to(self.device).eval()

    def _load_prompt_template(self):
        if not os.path.exists(self.prompt_template_path):
            raise FileNotFoundError(
                f"No such file or directory: '{self.prompt_template_path}'")
        with open(self.prompt_template_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        if "{question}" not in self.prompt_template:
            raise ValueError(f"Missing placeholder {{question}} in prompt.")

    def _build_prompt(self, question: str) -> str:
        return self.prompt_template.format(question=question)

    def _generate_answer(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=self.top_p,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.replace(prompt, "").strip()

    def generate_from_csv(self, input_csv: str, output_csv: str):
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"No such file or directory: '{input_csv}'")

        df = pd.read_csv(input_csv)
        if 'problem' not in df.columns:
            raise ValueError("Missing column 'problem.'")

        done_set = set()
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            for _, row in existing_df.iterrows():
                done_set.add((row['problem'], row['answer_index']))

        write_header = not os.path.exists(output_csv)
        output_file = open(output_csv, "a", encoding="utf-8", newline='')
        import csv
        writer = csv.DictWriter(output_file, fieldnames=[
                                "problem", "answer_index", "answer"])
        if write_header:
            writer.writeheader()

        # 逐条生成并写入
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating answers"):
            question = row['problem']
            prompt = self._build_prompt(question)

            for i in range(1, self.num_answers + 1):
                if (question, i) in done_set:
                    continue
                try:
                    answer = self._generate_answer(prompt)
                    writer.writerow({
                        "problem": question,
                        "answer_index": i,
                        "answer": answer
                    })
                    output_file.flush()  
                    done_set.add((question, i))
                except Exception as e:
                    print(f"Error in generate answers", {e})
                    continue

        output_file.close()
        print(f"Answers output in {output_csv}")


if __name__ == "__main__":
    current_dir = os.getcwd()
    cache_root = os.path.abspath(os.path.join(
        current_dir, "..", "build", ".cache", "huggingface"))
    os.environ['TRANSFORMERS_CACHE'] = cache_root

    generator = Generator_Qwen3_4B(
        model_id="Qwen/Qwen3-4B",
        prompt_template_path="prompt_template.txt",
        temperature=1.0,
        top_p=0.95,
        max_new_tokens=100,
        num_answers=10
    )

    generator.generate_from_csv(
        input_csv="questionlist.csv",
        output_csv="question_answer_list.csv"
    )
