
# Run Eval: 

```bash
python -m simple_qa_framework.simple_qa.py 
--dataset <simpleqa or mmlu_pro>
--mode <one of ["eval", "report", "report_all"], default = eval]>
--target <target model id, which is the model to be evaluated, default = gpt-5-mini> 
--grader <grader model id, which is the model that grades the answers, default = gpt-5-mini> 
--ling_judge <linguistic judge model id, which is the model that grades the answers, default = gpt-5-mini> 
--confidence <one of ["verbal", "linguistic", "p_true", "semantic_entropy"]>
--sample_size <sample size of the dataset, default = full dataset>
```

