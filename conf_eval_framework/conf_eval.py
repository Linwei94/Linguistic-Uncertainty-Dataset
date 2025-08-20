import pandas as pd
import argparse

from .custom_types import *
from .framework_config import *
from .eval_verbalised_confidence import VerbalisedConfidence
from .eval_linguistic_confidence import LinguisticConfidence
from .eval_p_true import PTrueConfidence
from .eval_semantic_entropy import SemanticEntropyConfidence

def main():
    # Evaluation configuration from command line arguments
    # --------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Simple QA Evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Specify simplqa or mmlu_pro")
    parser.add_argument("--mode", type=str, required=False, default="eval", help="Specify the mode: eval or view")
    parser.add_argument("--target", type=str, required=False, default=DEFAULT_TARGET_MODEL, help="Specify the target model for QA")
    parser.add_argument("--grader", type=str, required=False, default=DEFAULT_GRADER_MODEL, help="Specify the grader model for evaluation")
    parser.add_argument("--ling_judge", type=str, required=False, default=DEFAULT_GRADER_MODEL, help="Specify the judge model for linguistic confidence")
    parser.add_argument("--confidence", type=str, required=False, default="verbal", help="Specify the confidence extraction method")
    parser.add_argument("--sample_size", type=int, required=False, help="Specify the sample size for evaluation")
    args = parser.parse_args()

    target_model = args.target
    grader_model = args.grader
    ling_judge = args.ling_judge
    confidence = args.confidence
    sample_size = args.sample_size

    if args.dataset == "simpleqa":
        dataset = "simpleqa"
        qa_data = pd.read_csv("hf://datasets/basicv8vc/SimpleQA/simple_qa_test_set.csv")
    else:
        dataset = "mmlu_pro"
        splits = {'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
        qa_data = pd.concat([pd.read_parquet("hf://datasets/TIGER-Lab/MMLU-Pro/" + splits["test"]), pd.read_parquet("hf://datasets/TIGER-Lab/MMLU-Pro/" + splits["validation"])])

    if sample_size is None or sample_size > len(qa_data):
        print(f"Evaluating {dataset} full dataset.")
    else:
        qa_data = qa_data.sample(sample_size, random_state=42)
        print(f"Evaluating a subset of {dataset}: ", sample_size)
    
    if confidence not in CONFIDENCE_METHODS:
        raise ValueError(f"Invalid confidence method: {confidence}. Choose from {CONFIDENCE_METHODS}")
    # --------------------------------------------------------------------------
    
    # Logic for running evaluation or viewing results
    # --------------------------------------------------------------------------
    if args.mode == "eval":
        eval_map: dict[str, EvalBase] = {
            "verbal": VerbalisedConfidence(dataset=dataset, qa_data=qa_data, target_model_id=target_model, grader_model_id=grader_model),
            # "linguistic": LinguisticConfidence(qa_data=qa_data, target_model_id=target_model, grader_model_id=grader_model),
            # "p_true": PTrueConfidence(qa_data=qa_data, target_model_id=target_model, grader_model_id=grader_model),
            # "semantic_entropy": SemanticEntropyConfidence(qa_data=qa_data, target_model_id=target_model, grader_model_id=grader_model),
        }
        eval_obj = eval_map[confidence]
        eval_results: pd.DataFrame = eval_obj.results
        os.makedirs(f"./{dataset}_results", exist_ok=True)  # Ensure the results directory exists
        eval_results.to_pickle(f"./{dataset}_results/{eval_obj.model}_{eval_obj.grader.name}_{confidence}_{ling_judge}_{sample_size}_results.csv")

    # View results of a model 
    elif args.mode == "report":
        print("report mode is not implemented yet.")

    # View all results from the directory
    elif args.mode == "report_all":
        print("report_all mode is not implemented yet.")

if __name__ == "__main__":
    main()