mkdir -p logs
mkdir -p datasets/AdvBenchM/outputs

scenarios="bomb_explosive drugs firearms_weapons hack_information kill_someone social_violence suicide"
datetime="$(date '+%Y_%m_%d_%H_%M_%S')"
augs=(
    "cutmix_original" # CutMix
    "mixup" # Mixup
    "cutmix_resizemix" # ResizeMix
    "randaug1" # RandAug
)
lams="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
target_model="gpt-4-turbo-2024-04-09"
max_tokens="4096"
temperature="1.0"
retry_limit="10"
openai_key="[YOUR_API_KEY]"

for aug in "${augs[@]}"; do
    python3 -u main.py \
        --scenarios ${scenarios} \
        --harmful_image_dir "datasets/AdvBenchM/images/harmful" \
        --harmless_image_dir "datasets/AdvBenchM/images/harmless" \
        --prompt_dir "datasets/AdvBenchM/prompts/all_instructions" \
        --output_dir "datasets/AdvBenchM/outputs/${datetime}_${aug}/attack" \
        --aug ${aug} \
        --lams ${lams} \
        --model ${target_model} \
        --max_tokens ${max_tokens} \
        --temperature ${temperature} \
        --retry_limit ${retry_limit} \
        --openai_key ${openai_key} | tee logs/attack_${datetime}_${aug}.log
done