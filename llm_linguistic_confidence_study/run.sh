#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate llm-uncertainty
CUDA_VISIBLE_DEVICES=3 python -m llm_linguistic_confidence_study qa_model=claude-3-5-haiku-20241022 confidence_extractor=semantic_uncertainty pre_runned_batch=vanilla-su-claude-3-5-haiku-20241022.yaml
