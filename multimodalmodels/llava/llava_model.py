import os
import torch
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import BatchFeature
from ..multimodalmodel import MultiModalModel
from PIL import Image
import tqdm


from ..loaders import load_llava, load_native_vlm
from ..registry import (LLAVA_15_HF_ID, LLAVA_15_REVISION, LLAVA_16_HF_ID,
                        LLAVA_16_REVISION)  # noqa: F401  (re-exported)


class LLaVA_v1_5(MultiModalModel):
    def __init__(self, revision=LLAVA_15_REVISION):
        # loaded through multimodalmodels.loaders (pinned id/revision in registry.py)
        self.model, self.processor = load_llava(
            LLAVA_15_HF_ID, revision=revision, torch_dtype="float16", device_map="auto",
            processor_kwargs={"num_additional_image_tokens": 1},
        )
        self.model.gradient_checkpointing_enable()   # 이 줄 추가
        self.model.config.use_cache = False           # gradient checkpointing 쓸 땐 이것도 같이 꺼줘야 함 (충돌 방지)
        self.device = self.model.device #torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        self.normalize = transforms.Normalize(mean=self.processor.image_processor.image_mean, std=self.processor.image_processor.image_std)
        self.preprocess = lambda x: self.processor.image_processor.preprocess(x, do_normalize=False, return_tensors='pt')['pixel_values'][0]
        self.tokenizer_kwargs = {"padding": False, "truncation": None, "max_length": None}
        self.vis_processor = self.processor.image_processor


    def generate(self, test_cases, image_dir, **generation_kwargs):
        generation_kwargs.setdefault('do_sample', False)
        generation_kwargs.setdefault('num_beams', 1)
        generation_kwargs.setdefault('use_cache', True)
        assert generation_kwargs['num_beams'] == 1, "num_beams should be 1"
        outputs = []
        if generation_kwargs.pop('progress_bar', False):
            test_cases = tqdm.tqdm(test_cases)
        
        for case in test_cases:
            image, prompt = case
            raw_image = None
            if isinstance(image, str):
                raw_image = Image.open(os.path.join(image_dir, image))
            if raw_image is None:
                conv_prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: {prompt} ASSISTANT:"
            else:
                conv_prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:"

            inputs = self.processor(images=raw_image, text=conv_prompt, return_tensors="pt").to(self.device)#['input_ids'][0]

            #inputs = self.processor(conv_prompt, raw_image, return_tensors='pt').to(self.device)
            output = self.model.generate(**inputs, **generation_kwargs)
            output = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            outputs.append(output.rsplit('ASSISTANT:', 1)[-1].strip())

        return outputs


    # def generate_backup(self, test_cases, image_dir, **generation_kwargs):
    #     generation_kwargs.setdefault('do_sample', False)
    #     generation_kwargs.setdefault('num_beams', 1)
    #     assert generation_kwargs['do_sample'] is False, "do_sample should be False"
    #     assert generation_kwargs['num_beams'] == 1, "num_beams should be 1"
    #     outputs = []
    #     for case in test_cases:
    #         image, prompt = case
    #         if isinstance(image, str):
    #             raw_image = Image.open(os.path.join(image_dir, image))

    #         conv_prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:"

    #         inputs = self.processor(conv_prompt, raw_image, return_tensors='pt').to(self.device)
    #         output = self.model.generate(**inputs, **generation_kwargs)
    #         output = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    #         outputs.append(output.rsplit('ASSISTANT:', 1)[-1].strip())

    #     return outputs


    def query(self, image, prompt):
        input = self.processor(text=f"Human: {prompt} <image> Assistant: ", images=image, return_tensors="pt").to(self.device)
        #output = self.model.generate(**input,max_length=300) query_attack requires max_length greater than 300
        output = self.model.generate(**input,max_new_tokens=512)
        outnpy = output.to("cpu").numpy()
        answer = self.processor.decode(outnpy[0], skip_special_tokens=True)
        answer = answer.split('Assistant: ')[-1].strip()
        return answer

    def compute_loss(self, behavior, target, image_input):
        conv_prompt = f"<image>\nUSER: {behavior}\nASSISTANT: {target}"
        text_inputs = self.processor.tokenizer(conv_prompt, return_tensors='pt', **self.tokenizer_kwargs).to(self.device)
        pixel_values = self.normalize(image_input).unsqueeze(0).to(self.device)
        inputs = BatchFeature(data={**text_inputs, "pixel_values": pixel_values})
        model_inputs = self.model.prepare_inputs_for_generation(**inputs)

        target_ids = self.processor.tokenizer(target, return_tensors='pt', add_special_tokens=False, **self.tokenizer_kwargs)['input_ids'].to(self.device)
        assert text_inputs['input_ids'][0][-len(target_ids[0]):].equal(target_ids[0]), "Text inputs do not end with target ids"

        outputs = self.model(**model_inputs)
        shift_logits = outputs.logits[0, -(target_ids.size(1)+1):-1, :]
        loss_fn = CrossEntropyLoss()
        return loss_fn(shift_logits, target_ids.view(-1))

    def compute_loss_batch(self, x_adv, batch_targets, image_pil, prompts=None):
        batch_size = len(batch_targets)
        input_ids = []
        labels = []
        images = []

        #self.processor.num_additional_tokens = 0
        #self.processor.num_additional_image_tokens = 1
        prompts = prompts if prompts is not None else [""] * batch_size
        if len(prompts) != batch_size:
            raise ValueError("prompts and batch_targets must have the same length")
        if type(image_pil) == list:
            batch_targets_zip = zip(batch_targets, image_pil, prompts)
        else:
            batch_targets_zip = zip(batch_targets, [image_pil] * batch_size, prompts)

        template =  "A chat between a curious user and an artificial intelligence assistant. \
        The assistant gives helpful, detailed, and polite answers to the user's questions. \
        USER: <image>\n{prompt} ASSISTANT: {target}</s>"
        
        for target, image_pil_, prompt in batch_targets_zip:
            text = template.format(prompt=prompt, target=target)
            input_id = self.processor(images=image_pil_, text=text, return_tensors="pt")['input_ids'][0]
            # input_id = self.processor.tokenizer(text, return_tensors="pt")['input_ids'][0]
            label = input_id.clone()

            sep = " ASSISTANT: "
            parts = text.split(sep)
            parts[0] += sep

            instruction_len = len(self.processor(images=image_pil_, text=parts[0], return_tensors="pt")['input_ids'][0]) - 1
            # instruction_len = len(self.processor.tokenizer(parts[0], return_tensors='pt')[0]) - 1
            label[:instruction_len] = -100

            input_ids.append(input_id)
            labels.append(label)               

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                    batch_first=True,
                                                    padding_value=-100)
        input_ids = input_ids[:, :self.processor.tokenizer.model_max_length].to(self.device)
        labels = labels[:, :self.processor.tokenizer.model_max_length].to(self.device)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id).to(self.device)

        if type(image_pil) == list:
            images = x_adv
            pixel_values = images
        else:
            images = [x_adv] * batch_size
            pixel_values = torch.concat(images, axis = 0).to(self.device)
    

        loss = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            pixel_values = pixel_values
        ).loss

        return loss


class LLaVA_v1_6(MultiModalModel):
    """Legacy wrapper retained for ``generate_completions.py`` compatibility.

    The unified pipeline uses :class:`core.targets.native_hf.LlavaNextTarget`.
    Both paths now share the same pinned Apache-2.0 Mistral checkpoint and
    centralized loader.
    """

    def __init__(self, revision=LLAVA_16_REVISION):
        self.model, self.processor = load_native_vlm(
            LLAVA_16_HF_ID,
            revision=revision,
            torch_dtype="auto",
            device_map="auto",
            eval_mode=True,
        )
        self.device = self.model.device

    def generate(self, test_cases, image_dir, **generation_kwargs):
        generation_kwargs.setdefault('do_sample', False)
        generation_kwargs.setdefault('num_beams', 1)
        assert generation_kwargs['num_beams'] == 1, "num_beams should be 1"
        generation_kwargs.pop("num_beams")
        generation_kwargs.pop("progress_bar", None)
        outputs = []
        for case in test_cases:
            image, prompt = case
            raw_image = None
            if isinstance(image, str):
                with Image.open(os.path.join(image_dir, image)) as opened:
                    raw_image = opened.convert("RGB").copy()
            output = self.query(raw_image, prompt, **generation_kwargs)
            outputs.append(output)

        return outputs 

    def query(self, image, prompt, **generation_kwargs):
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        generation_kwargs.setdefault("max_new_tokens", 512)
        output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    
    def compute_loss(self, behavior, target, image_input):
        return super().compute_loss(behavior, target, image_input)
    
    def compute_loss_batch(self, x_adv, batch_targets):
        pass
