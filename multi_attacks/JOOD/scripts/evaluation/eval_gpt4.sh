mkdir -p logs
scenarios="bomb_explosive drugs firearms_weapons hack_information kill_someone social_violence suicide"
eval_datetime="[YOUR_EVAL_DIR_DATETIME]" # MODIFY HERE
aug="cutmix_original" # MODIFY HERE
eval_dir="datasets/AdvBenchM/outputs/${eval_datetime}_${aug}/attack"
eval_output_dir="datasets/AdvBenchM/outputs/${eval_datetime}_${aug}/eval"
eval_scenario_repr_path="datasets/AdvBenchM/scenario_repr.json"
eval_scenario_def_path="datasets/AdvBenchM/scenario_def.json"
eval_model="gpt-4-turbo-2024-04-09"
max_tokens="4096"
temperature="0"
retry_limit="10"
openai_key="[YOUR_API_KEY]"

python3 -u main.py \
    --scenarios ${scenarios} \
    --eval \
    --aug ${aug} \
    --prompt_dir "datasets/AdvBenchM/prompts/eval_all_instructions" \
    --eval_dir ${eval_dir} \
    --eval_output_dir ${eval_output_dir} \
    --eval_scenario_repr_path ${eval_scenario_repr_path} \
    --eval_scenario_def_path ${eval_scenario_def_path} \
    --model ${eval_model} \
    --max_tokens ${max_tokens} \
    --temperature ${temperature} \
    --retry_limit ${retry_limit} \
    --openai_key ${openai_key} | tee logs/eval_${datetime}.log