# FigStep
## Attack
python generate_test_cases.py --method_name FigStep --save_dir test_cases/FigStep_test_cases

## Generate completions
python generate_completions.py --test_cases_path test_cases/FigStep_test_cases/FigStep_test_cases.json --save_path completions/FigStep_completions.json --model_name minigpt4

## Evaluate completions
python evaluate_completions.py --cls_path cais/HarmBench-Llama-2-13b-cls --behaviors_path data/behavior_datasets/harmbench_behaviors_text_all.csv --completions_path completions/FigStep_completions.json --save_path results/FigStep_results.json --include_advbench_metric

# Visual_Adv
## Attack
python generate_test_cases.py --method_name Visual_Adv --save_dir test_cases/Visual_Adv_test_cases

## Generate completions
python generate_completions.py --test_cases_path test_cases/Visual_Adv_test_cases/Visual_Adv.json --save_path completions/Visual_Adv_completions.json --model_name minigpt4

## Evaluate completions
python evaluate_completions.py --cls_path cais/HarmBench-Llama-2-13b-cls --behaviors_path data/behavior_datasets/harmbench_behaviors_text_all.csv --completions_path completions/Visual_Adv_completions.json --save_path results/Visual_Adv_results.json --include_advbench_metric

# Query_Attack
## Attack
python generate_test_cases.py --method_name Query_Attack --save_dir test_cases/Query_attack_test_cases

## Generate completions
python generate_completions.py --test_cases_path test_cases/Query_attack_test_cases/Query_attack_test_cases.json --save_path completions/Query_Attack_completions.json --model_name minigpt4

## Evaluate completions
python evaluate_completions.py --cls_path cais/HarmBench-Llama-2-13b-cls --behaviors_path data/behavior_datasets/harmbench_behaviors_text_all.csv --completions_path completions/Query_Attack_completions.json --save_path results/Query_Attack_results.json --include_advbench_metric

# ImgJP (UAP)
## Attack
python generate_test_cases.py --method_name ImgJP --save_dir test_cases/ImgJP_test_cases

## Generate completions
python generate_completions.py --test_cases_path test_cases/ImgJP_test_cases/ImgJP_test_cases.json --save_path completions/imgjp_completions.json --model_name llava_v1_5

## Evaluate completions
python evaluate_completions.py --cls_path cais/HarmBench-Llama-2-13b-cls --behaviors_path data/behavior_datasets/harmbench_behaviors_text_all.csv --completions_path completions/imgjp_completions.json --save_path results/imgjp_results.json --include_advbench_metric

# ImgJP (Instance-specific)
## Attack
python generate_test_cases.py --method_name DeltaJP --save_dir test_cases/DeltaJP_test_cases --behaviors_path ./data/behavior_datasets/harmbench_behaviors_multimodal_all.csv

## Generate completions
python generate_completions.py --test_cases_path test_cases/DeltaJP_test_cases/DeltaJP_test_cases.json --save_path completions/deltajp_completions.json --model_name llava_v1_5

## Evaluate completions
python evaluate_completions.py --cls_path cais/HarmBench-Llama-2-13b-cls --behaviors_path data/behavior_datasets/harmbench_behaviors_multimodal_all.csv --completions_path completions/deltajp_completions.json --save_path results/deltajp_results.json --include_advbench_metric

# Best-Of-N
## Attack
python generate_test_cases.py --method_name BestOfN --save_dir test_cases/BestOfN_test_cases --behaviors_path ./data/behavior_datasets/harmbench_behaviors_multimodal_all.csv

## Generate completions
python generate_completions.py --test_cases_path test_cases/DeltaJP_test_cases/BestOfN_test_cases.json --save_path completions/BestOfN_completions.json --model_name llava_v1_5

## Evaluate completions
python evaluate_completions.py --cls_path cais/HarmBench-Llama-2-13b-cls --behaviors_path data/behavior_datasets/harmbench_behaviors_multimodal_all.csv --completions_path completions/BestOfN_completions.json --save_path results/BestOfN_results.json --include_advbench_metric