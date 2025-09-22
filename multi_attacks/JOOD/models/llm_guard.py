"""
JOOD
Copyright (c) 2025-present NAVER Corp.
Apache License v2.0
"""
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfFolder
import os
import torch
import torch.backends.cudnn as cudnn
import random
import numpy as np

class LLM_Guard():
    def __init__(self):

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
    
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)

        print('Initializing LLM_Guard')

        model_id = "" # specify your model id here
        self.device = "cuda"
        dtype = torch.bfloat16

        token = os.environ.get("HF_TOKEN")
        # set api for login and save token
        folder = HfFolder()
        folder.save_token(token)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=self.device)


    def moderate(self, user, assistant):
        chat = [{"role": "user", "content": user},
                {"role": "assistant", "content": assistant},]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    