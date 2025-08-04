from abc import ABC, abstractmethod
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

class MultiModalModel(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def generate(self, test_case):
        pass

    @abstractmethod
    def compute_loss(self, behavior, target, image_input):
        pass


class HarmBenchMultiModalModel(MultiModalModel):
    def __init__(self, model_id):
        self.model_name = model_id
        self.get_model_and_tokenizer(self.model_name)

    def get_model_and_tokenizer(self, model_name):
        if model_name == "cais/HarmBench-Llama-2-13b-cls":
            self.model = AutoModelForCausalLM.from_pretrained("cais/HarmBench-Llama-2-13b-cls", torch_dtype=torch.bfloat16, device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained("cais/HarmBench-Llama-2-13b-cls", use_fast=False, truncation_side="left", padding_side="left")
        else:
            raise ValueError(f"Model {model_name} not supported")


    def get_model_response(self, text, return_logits=False, max_completion_tokens=256, **kwargs):
        if self.model_name == "cais/HarmBench-Llama-2-13b-cls":
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
    def generate(self, test_case):
        pass

    def compute_loss(self, behavior, target, image_input):
        pass

# 사용 예제
if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    model_id = "llava-hf/llava-1.5-7b-hf"
    model_id = "wangrongsheng/MiniGPT-4-LLaMA-7B"
    model_id = "Salesforce/instructblip-vicuna-7b"

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

    if model_id == "llava-hf/llava-1.5-7b-hf":
        class UserMultiModalModel(MultiModalModel):
            def __init__(self, model_id):
                #super().__init__()
                self.model_name = model_id
                self.get_model_and_tokenizer(self.model_name)

        UserMultiModalModel_model_test = UserMultiModalModel(model_id)

        print(UserMultiModalModel_model_test.get_model_response(messages))
    elif model_id == "wangrongsheng/MiniGPT-4-LLaMA-7B":
        MultiModalModel_model_test = MultiModalModel(model_id)
        print(MultiModalModel_model_test.get_model_response(messages))

    elif model_id == "Salesforce/instructblip-vicuna-7b":
        MultiModalModel_model_test = MultiModalModel(model_id)
        print(MultiModalModel_model_test.get_model_response(messages))
