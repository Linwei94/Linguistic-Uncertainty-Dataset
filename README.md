## Reproduction
### `build_dataset.ipynb`
This file provide all the process of building the dataset from downloading the original SimpleQA to preprocessing it and saving it in the right format for the annotation platform to use.

### `build_dataset.py`
Since we adopt 4 LLMs(GPT-4.1, Grok-3, Claude-4-Sonnet, and Gemini-2.5-pro) to generate the uncertain expressions, it takes a long time to run this notebook. We instead use the `build_dataset.py` to run the sampling process.

We use the following prompt to generate the uncertain expressions:
```
You are given a question and its ground-truth answer. Your task is to generate 50 answer sentences that express the same answer using different levels of confidence:

10 with high confidence
10 with moderate confidence
10 with low confidence
10 with lowest confidence
10 with complete uncertainty, reject to reply

The wording should vary across the levels, but all responses should convey the same core answer. Focus on natural and diverse expressions of confidence.

Question: {}
Answer: {}
```


### `extract_dataset.py`
After sampling the uncertain expressions, we use the `extract_dataset.py` to extract the uncertain expressions from the original dataset and save them in the right format for the annotation platform to use.

Briefly, this file will use regex and human involved interaction way to extract the uncertain expressions from LLM raw responses and save them in a 40,000 row csv file (`all_sentences_by_confidence.csv`).


### `mtruk_tasks.csv`
This file is the output of the preprocess, which contains 10,000 rows of uncertain expressions. Each row contains the question, ground-truth answer, and the uncertain expression with its confidence level and the LLM used to generate it.

### `generate_html.py`
This file is used to generate the HTML file (`example.html`) for the mturk annotation platform. 

### `simple_qa_test_set.csv`
Original test set of SimpleQA.

### `raw_response` folder
This folder contains the raw responses from the LLMs. Each file corresponds to a different LLM and contains the generated uncertain expressions.

---