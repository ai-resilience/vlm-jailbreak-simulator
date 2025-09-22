mkdir -p logs
mkdir -p datasets/AdvBenchM/outputs

scenarios="bomb_explosive drugs firearms_weapons hack_information kill_someone social_violence suicide"
datetime="$(date '+%Y_%m_%d_%H_%M_%S')"
augs=(
    "textmix_character_wise_interleave" # H-interleave
    "textmix_character_wise_interleave_vertically" # V-Interleave
    "textmix_concat" # H-Concat
    "textmix_concat_vertically" # V-Concat
    "textmix_concat_cross" # C-Concat
)
lams="0.5"
target_model="gpt-4-turbo-2024-04-09"
max_tokens="4096"
temperature="1.0"
retry_limit="10"
openai_key="[YOUR_API_KEY]"

python3 -u main.py \
    --scenarios ${scenarios} \
    --harmful_image_dir "datasets/AdvBenchM/images/harmful" \
    --harmless_image_dir "datasets/AdvBenchM/images/harmless" \
    --prompt_dir "datasets/AdvBenchM/prompts/all_instructions_harmful_annotated" \
    --output_dir "datasets/AdvBenchM/outputs/${datetime}_${aug}/attack" \
    --aug ${aug} \
    --lams ${lams} \
    --model ${target_model} \
    --max_tokens ${max_tokens} \
    --temperature ${temperature} \
    --retry_limit ${retry_limit} \
    --openai_key ${openai_key} | tee logs/attack_${datetime}_${aug}.log