from abc import ABC, abstractmethod
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer


class MultiModalModel(ABC):
    #@abstractmethod
    #def __init__(self):
    #    pass
    def __init__(self, model_id):
        self.model_name = model_id
        self.get_model_and_tokenizer(self.model_name)

    #@abstractmethod
    def generate(self, test_case):
        pass

    #@abstractmethod
    def compute_loss(self, behavior, target, image_input):
        pass


    def get_model_and_tokenizer(self, model_name):
        if model_name == "llava-hf/llava-1.5-7b-hf":
            
            self.tokenizer = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")

            self.model_name = model_name
            #max_completion_tokens = 256
            #return model, tokenizer
        elif model_name == "gpt-4o-mini":
            return transformers.AutoModelForCausalLM.from_pretrained(model_name), transformers.AutoTokenizer.from_pretrained(model_name)
        elif model_name == "cais/HarmBench-Llama-2-13b-cls":
            self.model = AutoModelForCausalLM.from_pretrained("cais/HarmBench-Llama-2-13b-cls", torch_dtype=torch.bfloat16, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained("cais/HarmBench-Llama-2-13b-cls", use_fast=False, truncation_side="left", padding_side="left")
        elif model_name == "wangrongsheng/MiniGPT-4-LLaMA-7B":
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(f"Model {model_name} not supported")


    def get_model_response(self, text, return_logits=False, max_completion_tokens=256, **kwargs):
        if self.model_name == "llava-hf/llava-1.5-7b-hf":
            # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
            # Each value in "content" has to be a list of dicts with types ("text", "image") 
            # conversations = [
            #     [
            #         {
            #         "role": "system",
            #         "content": [
            #          {"type": "text", "text": "You are a kind and helpful assistant."},
            #          {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            #           ],
            #         }
            #         {
            #         "role": "user",
            #         "content": [
            #             {"type": "text", "text": "What are these?"},
            #             {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            #             ],
            #         },
            #     ],
            #     [
            #         {
            #         "role": "system",
            #         "content": [
            #           {"type": "text", "text": "You are a kind and helpful assistant."},
            #           {"type": "image"},
            #           ],
            #         }
            #         {
            #         "role": "user",
            #         "content": [
            #            {"type": "text", "text": "What are these?"},
            #             {"type": "image"},
            #             ],
            #         },
            #     ]
            # ]
            inputs = self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device, torch.float16)

            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=max_completion_tokens)
            response = self.tokenizer.batch_decode(generate_ids[:,inputs["input_ids"].shape[1]:], skip_special_tokens=True) 
            return response
        elif self.model_name == "cais/HarmBench-Llama-2-13b-cls":
            #from .utils import LLAMA2_CLS_PROMPT


            encoded = self.tokenizer(text, return_tensors='pt', padding='longest')
            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded.to(self.model.device),
                    do_sample=False,
                     max_new_tokens=1,
                    ).cpu()
                output_ids = output_ids[:, len(encoded.input_ids[0]):]
            completion = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)
            return completion
        elif self.model_name == "wangrongsheng/MiniGPT-4-LLaMA-7B":
            inputs = self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                padding=True,
                return_tensors="pt"
            ).to(self.model.device, torch.float16)

            # Generate
            generate_ids = self.model.generate(**inputs, max_new_tokens=max_completion_tokens)
            response = self.tokenizer.batch_decode(generate_ids[:,inputs["input_ids"].shape[1]:], skip_special_tokens=True) 
            return response

# 사용 예제
if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    model_id = "llava-hf/llava-1.5-7b-hf"

    if model_id == "llava-hf/llava-1.5-7b-hf":
        class UserMultiModalModel(MultiModalModel):
            def __init__(self, model_id):
                #super().__init__()
                self.model_name = model_id
                self.get_model_and_tokenizer(self.model_name)

        UserMultiModalModel_model_test = UserMultiModalModel(model_id)

        messages= [
            [
                {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a kind and helpful assistant."},
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    ],
                },
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What are these?"},
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    ],
                },
            ],
            [
                {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a kind and helpful assistant."},
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    ],
                },
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What are these?"},
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    ],
                },
            ],
        ]
        
        print(UserMultiModalModel_model_test.get_model_response(messages))
        
