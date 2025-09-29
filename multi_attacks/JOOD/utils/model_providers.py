"""
JOOD
Copyright (c) 2025-present NAVER Corp.
Apache License v2.0
"""
import json
import time
import requests
import anthropic

qwenvl2_model = None

def get_image_content(model, base64_image):
    if "gpt-4" in model or "o1" in model:
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
    elif "claude" in model:
        image_content = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64_image
            }
        }
    else:
        raise ValueError("No such models")
    return image_content

def parse_output_json_anthropic2openai(anthropic_output):
    output_json = {
        "id": anthropic_output.id,
        "object": "chat.completion",
        "model": anthropic_output.model,
        "choices": [
            {
                "index": content_idx,
                "message": {
                    "role": anthropic_output.role,
                    "content": content.text,
                    "refusal": None
                },
                "logprobs": None,
                "finish_reason": anthropic_output.stop_reason
            } for content_idx, content in enumerate(anthropic_output.content)
        ],
        "usage": {
            "prompt_tokens": anthropic_output.usage.input_tokens,
            "completion_tokens": anthropic_output.usage.output_tokens,
            "total_tokens": anthropic_output.usage.input_tokens + anthropic_output.usage.output_tokens
        },
        "system_fingerprint": None
    }
    return output_json

def parse_output_json_qwenvl2openai(model_name, output):
    output_json = {
        "id": None,
        "object": "chat.completion",
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output,
                    "refusal": None
                },
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None
        },
        "system_fingerprint": None
    }
    return output_json

def query_chatgpt(
        custom_request_id,
        prompt,
        model,
        api_key,
        image=None,
        retry_limit=5,
        temperature=0.0,
        max_tokens=4096,
        eval_mode=False
    ):
    def exception_handle(
            custom_request_id,
            text,
            retry_count,
            retry_limit
        ):
        print(f"Error processing request {custom_request_id}: {text}")
        print(f"Retrying... (attempt {retry_count + 1}/{retry_limit})")
        retry_count += 1
        time.sleep(1)  # Add a delay before retrying
        return retry_count

    if image is None:
        content = prompt
    elif isinstance(image, list):
        content = [
            {
                "type": "text",
                "text": prompt
            }
        ]
        for img in image:
            content.append(
                get_image_content(model, img)
            )
    else:
        content = [
            {
                "type": "text",
                "text": prompt
            },
            get_image_content(model, image)
        ]

    if "o1" in model:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_completion_tokens": max_tokens,
            "temperature": temperature
        }
    else:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

    retry_count = 0
    while retry_count < retry_limit:
        try:
            if "gpt-4" in model or "o1" in model:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                res_status_code = response.status_code
                output_json = response.json()

            elif "claude" in model:
                client = anthropic.Anthropic(
                    api_key=api_key,
                )
                response = client.messages.create(**payload)
                output_json = parse_output_json_anthropic2openai(response)
                res_status_code = 200

            output_text = output_json["choices"][0]["message"]["content"]

            if res_status_code == 200:
                if eval_mode:
                    if "```json" not in output_text:
                        raise ValueError(f"output parsing error occured when parsing from {output_text}")
                    else:
                        splits = output_text.split("```json")
                        output_metrics_str = splits[-1].strip()
                        output_metrics_str = output_metrics_str[output_metrics_str.find("{"):output_metrics_str.rfind("}") + 1]
                        try:
                            json.loads(output_metrics_str)
                        except:
                            raise ValueError(f"json parsing error occured when parsing from {output_text}")

                error_msg = None
                return output_text, output_json, res_status_code, error_msg
            else:
                retry_count = exception_handle(custom_request_id, response.text, retry_count, retry_limit)
        except Exception as e:
            retry_count = exception_handle(custom_request_id, e, retry_count, retry_limit)

    error_msg = f"Failed to process request after {retry_limit} attempts."
    status_code = 0
    print(error_msg)
    return None, None, status_code, error_msg

def process_per_prompt(
        custom_request_id,
        prompts,
        image_base64,
        output_file,
        model,
        openai_key,
        retry_limit,
        temperature,
        max_tokens,
        eval_mode=False,
        pick_image_per_prompt=True,
        image_pil=None
    ):
    for prompt_idx, prompt in enumerate(prompts):
        # unique id per request
        _custom_request_id = custom_request_id + f"-[PromptIdx]{prompt_idx}"
        
        if image_base64 is not None:
            if isinstance(image_base64, list):
                # pick per image from list if image_base64 is list
                if pick_image_per_prompt:
                    image_base64_input = image_base64[prompt_idx]
                # if not, list of images
                else:
                    image_base64_input = image_base64
            else:
                image_base64_input = image_base64
        else:
            image_base64_input = None

        if model == "qwenvl2":
            assert image_pil
            global qwenvl2_model
            qwenvl_model_name = "Qwen/Qwen2-VL-7B-Instruct"

            if not qwenvl2_model:
                from models.qwenvl2 import MyQwenVL2
                qwenvl2_model = MyQwenVL2(qwenvl_model_name)
            res_text = qwenvl2_model(image_pil, prompt)
            res_json = parse_output_json_qwenvl2openai(qwenvl_model_name, res_text)
            res_status_code = 200
            res_error_msg = None
        else:
            res_text, res_json, res_status_code, res_error_msg = query_chatgpt(
                _custom_request_id,
                prompt,
                model,
                openai_key,
                image=image_base64_input,
                retry_limit=retry_limit,
                temperature=temperature,
                max_tokens=max_tokens,
                eval_mode=eval_mode
            )

        output_entry = {
            "custom_id": _custom_request_id,
            "response": {"status_code": res_status_code, "body": res_json},
            "error": res_error_msg
        }

        if not eval_mode:
            output_entry["attack_prompt"] = prompt
        json_record = json.dumps(
            output_entry, ensure_ascii=False)
        output_file.write(json_record + '\n')

        print(f"[Request]{_custom_request_id}:\n\n[PROMPT]:{prompt}\n\n[RESPONSE]:{res_text}")