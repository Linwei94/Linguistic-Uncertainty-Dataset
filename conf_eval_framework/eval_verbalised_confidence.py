import pandas as pd
import os
import re

from .custom_types import *
from .prompt_templates import *
from .model import *

class VerbalisedConfidence(EvalBase):
    def __init__(self, dataset: str, qa_data: pd.DataFrame, target_model_id: str, grader_model_id: str):
        self.data_set: str = dataset
        self.target_model_id: str = target_model_id
        self.grader_model_id: str = grader_model_id
        self.qa_data: pd.DataFrame = qa_data
        self.results = pd.DataFrame()
        
        if self.data_set == "simpleqa":
            self.evaluate_simple_qa()
        elif self.data_set == "mmlu_pro":
            self.evaluate_mmlu_pro()
    

    def evaluate_simple_qa(self):
        target: ModelBase = get_model(model_id=self.target_model_id)
        grader: ModelBase = get_model(model_id=self.grader_model_id)
        self.results: pd.DataFrame = self.qa_data.copy()
        # raw columns for storing prompts and responses
        self.results["qa_prompt"] = [SIMPLE_QA_EVAL_VERBAL_NUMERICAL_CONFIDENCE_TEMPLATE.format(question=q) for q in self.results["problem"]]                                                                      # format prompt from template
        self.results["response"] = target.inference(self.results["qa_prompt"].tolist())
        self.results["grader_prompt"] = [SIMPLE_QA_GRADER_TEMPLATE.format(question=question, target=target, predicted_answer=predicted_answer) for question, target, predicted_answer in zip(self.results["problem"], self.results["answer"], self.results["response"])] 
        self.results["grader_response"] = grader.inference(self.results["grader_prompt"].tolist())
        # extracted columns for storing answers and confidence
        self.results["grade"] = self.results["grader_response"].apply(lambda x: re.search(r"(A|B|C)", x).group(0) if re.search(r"(A|B|C)", x) else "C")
        self.results["correctness"] = self.results["grade"] == "A"
        self.results["extracted_confidence"] = []  # apply regex to extract confidence score from grader response


    def evaluate_mmlu_pro(self):
        target: ModelBase = get_model(model_id=self.target_model_id)
        grader: ModelBase = get_model(model_id=self.grader_model_id)
        self.results: pd.DataFrame = self.qa_data.copy()
        # raw columns for storing prompts and responses
        self.results["qa_prompt"] = []                                                                      # format prompt from template
        self.results["response"] = target.inference(self.results["qa_prompt"].tolist())
        self.results["grader_prompt"] = []                                                                  # format prompt from template 
        self.results["grader_response"] = grader.inference(self.results["grader_prompt"].tolist())
        # extracted columns for storing answers and confidence
        self.results["extracted_answer"] = [] 
        self.results["extracted_confidence"] = [] 
        self.results["correctness"] = [] 