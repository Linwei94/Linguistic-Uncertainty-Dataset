import pandas as pd
import numpy as np
from sympy.logic.boolalg import false
import tqdm

SIMPLE_QA_SAMPLE_TAMPLATE = """
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
# You will receive a question. Please provide an answer in a single brief but complete sentence.

### Question:
{question}

### Response:""".strip()

SIMPLE_QA_ENATILMENY_TAMPLATE = """
We are evaluating answers to the question {question}
Here are two possible answers:
Possible Answer 1: {answer1}
Possible Answer 2: {answer2}
Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.
Response:""".strip()

SIMPLE_QA_MEETRIC_TAMPLATE = """
The question is: {question}.
The correct answer to this question is: {correct_answer}.
The proposed answer is: {predicted_answer}
Within the context of the question, does the proposed answer mean the same as any of the expected answers? Respond only with yes or no.
Response:""".strip()


class SemanticUncertaintyExtractor():
    def __init__(self, confidence_extraction_method_cfg, dataset_cfg, qa_model_cfg, grader_model_cfg):
        self.confidence_extraction_method_cfg = confidence_extraction_method_cfg
        self.dataset_cfg = dataset_cfg
        self.qa_model_cfg = qa_model_cfg
        self.grader_model_cfg = grader_model_cfg

    def __call__(self, dataset: pd.DataFrame, model, sample_times: int = 10):
        df = self.generate_responses(model, dataset, sample_times)
        df = self.post_process_responses(df, sample_times)
        return df

    def generate_responses(self, model, df: pd.DataFrame, sample_times: int = 10):
        prompt_sample = SIMPLE_QA_ENATILMENY_TAMPLATE
        for i in range(sample_times):
            df[f"response{i+1}"] = None
        for idx, row in df.iterrows():
            responses = model.generate(
                prompt_sample.format(question=row["question"]))
            for i in range(sample_times):
                df.at[idx, f"response{i+1}"] = responses[i]
            df[f"response{idx+1}"] = responses[idx]["content"]
        return df

    def post_process_responses(self, model, df: pd.DataFrame, sample_times: int = 10):
        # cluster
        prompt_equivalence = SIMPLE_QA_ENATILMENY_TAMPLATE
        df[f"entropy"] = None
        for i in range(sample_times):
            df[f"cluster{i+1}"] = None
            df[f"prob{i+1}"] = None
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Checking equivalence"):
            clusters = {}
            responses = {}
            total_clusters = 0
            for i in range(sample_times):
                now_response = df.at[idx, f"response{i+1}"]
                if now_response in responses:
                    df.at[idx, f"cluster{i+1}"] = responses[now_response]
                    clusters[responses[now_response]].append(i)
                    continue

                question = df.at[idx, f"question"]
                flag = True
                for j in range(j):
                    response = df.at[idx, f"response{j}"]
                    prompt_equivalence1 = prompt_equivalence.format(
                        question=question, answer1=now_response, answer2=response)
                    prompt_equivalence2 = prompt_equivalence.format(
                        question=question, answer2=now_response, answer1=response)
                    implication_1 = self.__check_(model, prompt_equivalence1)
                    implication_2 = self.__check_(model, prompt_equivalence2)
                    implications = [implication_1, implication_2]
                    semantically_equivalent = (0 not in implications) and (
                        [1, 1] != implications
                    )
                    if semantically_equivalent == 2:
                        df.at[idx, f"cluster{i+1}"] = responses[response]
                        flag = False
                        clusters[responses[response]].append(i)
                        break
                if flag:
                    total_clusters += 1
                    df.at[idx, f"cluster{i+1}"] = total_clusters
                    responses[now_response] = total_clusters
                    clusters[responses[response]] = [i]

            semantic_ids = []
            flag = True
            for i in range(sample_times):
                semantic_ids.append(df.at[idx, f"cluster{i+1}"])
                prompt_metric = SIMPLE_QA_MEETRIC_TAMPLATE.format(
                    question=df.at[idx, "question"],
                    correct_answer=df.at[idx, "gold_answer"],
                    predicted_answer=df.at[idx, f"response{i+1}"]
                )
                metric_answer = model.generate(prompt_metric)
                if 'yes' in metric_answer.lower() and flag:
                    df.at[idx, "response"] = df.at[idx, f"response{i}"]
                    flag = False

            # compute semantic entropy
            n_generations = len(semantic_ids)
            counts = np.bincount(semantic_ids)
            probabilities = counts/n_generations
            assert np.isclose(probabilities.sum(), 1)
            df.at[idx, "confidence"] = - \
                (probabilities * np.log(probabilities)).sum()
            df.at[idx, f"response"] = df.at[idx, f"response"]

        # return
        columns = ["question", "gold_answer"]
        for i in range(sample_times):
            columns.append(f"reponse{i+1}")
        columns.append("confidence")
        columns.append("accuracy")
        return df[columns]

    def calculate_confidence(self, model, question, responses, clusters, gold_answer):
        pass

    def calculate_accuracy(self, model, question, responses, clusters, gold_answer):
        pass

    def __check_(self, model, prompt):
        response = model.generate_answer(prompt)[0]
        if "entailment" in response:
            return 2
        elif "neutral" in response:
            return 1
        elif "contradiction" in response:
            return 0
        else:
            return 1
