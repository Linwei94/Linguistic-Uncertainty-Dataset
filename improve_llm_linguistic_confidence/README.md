Make sure you have the weight of lora: `improve_llm_linguistic_confidence/res/weight/qwen3_8b_lora_weight`. It often contains these files:
- adapter_config.json
- adapter_model.safetensors
- added_tokens.json
- chat_template.jinja
- merges.txt
- optimizer.pt
- README.md
- rng_state.pth
- scheduler.pt
- special_tokens_map.json
- tokenizer_config.json
- tokenizer.json
- trainer_state.json
- training_args.bin
- vocab.json

Make sure you have these two config files `improve_llm_linguistic_confidence/lora_test.yaml` and `llm_linguistic_confidence_study/configs/qa_model/Huggingfacemodel.yaml`

For the items in the `improve_llm_linguistic_confidence/lora_test.yaml`:
|name|example value|What is it used for|
|--|--|--|
|name|lora_test|To idenify the the test of Lora from confidence_extraction_methods|
|model_name|distilroberta-base|For confidence|
|state_dict_path|llm_linguistic_confidence_study/pretrained_weights/human_anno_trained_reg_head.pth|For confidence|
|mapper_name|self-trained|For confidence|
|qa_template|none|To idenify the prompt|

For the items in the `llm_linguistic_confidence_study/configs/qa_model/Huggingfacemodel.yaml`:
|name|example value|What is it used for|
|--|--|--|
|name|Qwen/Qwen3-8B-uncertainty|To idenify the huggingface model from Qwen3 api.|
|base_model_id|Qwen/Qwen3-8B|The base model use for LoRA
|lora_weight_path|improve_llm_linguistic_confidence/res/weight/qwen3_8b_lora_weight| Where the weight of lora is storaged|
|save_path|improve_llm_linguistic_confidence/res/model| Where to save the merged model|
|device_map|auto
|temperature|0.6
|top_p|0.95
|top_k|20
|min_p|0.0
|thinking|True
|max_tokens|32768