# FigStep
## Attack
#python generate_test_cases.py --method_name FigStep 

## Generate completions
#python generate_completions.py --test_cases_path test_cases/FigStep_test_cases.json --save_path completions/FigStep_completions.json --model_name minigpt4
## Evaluate completions
#python evaluate_completions.py --cls_path cais/HarmBench-Llama-2-13b-cls --behaviors_path data/behavior_datasets/harmbench_behaviors_text_all.csv --completions_path completions/FigStep_completions.json --save_path results/FigStep_results.json --include_advbench_metric

# Visual_Adv
## Attack
#python generate_test_cases.py --method_name Visual_Adv 

## Generate completions
python generate_completions.py --test_cases_path test_cases/Visual_Adv.json --save_path completions/Visual_Adv_completions.json --model_name minigpt4

## Evaluate completions
python evaluate_completions.py --cls_path cais/HarmBench-Llama-2-13b-cls --behaviors_path data/behavior_datasets/harmbench_behaviors_text_all.csv --completions_path completions/Visual_Adv_completions.json --save_path results/Visual_Adv_results.json --include_advbench_metric