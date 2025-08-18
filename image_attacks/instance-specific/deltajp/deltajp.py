import os
import json
from tqdm import tqdm

from transformers import StoppingCriteriaList, TextIteratorStreamer

# from attacks.xidian.utils.minigpt4.common.config_1 import Config
# from attacks.xidian.utils.minigpt4.common.dist_utils import get_rank
# from attacks.xidian.utils.minigpt4.common.registry import registry
# from attacks.xidian.utils.minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub


# from attacks.xidian.utils.torchattacks.attacks.pixle import *
# from attacks.xidian.utils.torchattacks.attacks.bim import *
# from attacks.xidian.utils.torchattacks.attacks.pgd_uap_v1 import *
# from attacks.xidian.utils.torchattacks.attacks.pgdl2 import *

import torch.backends.cudnn as cudnn
import random
import time
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import multimodalmodels
from PIL import Image

import csv
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_image(image_array: np.ndarray, f_name: str) -> None:
    """
    Saves image into a file inside `ART_DATA_PATH` with the name `f_name`.

    :param image_array: Image to be saved.
    :param f_name: File name containing extension e.g., my_img.jpg, my_img.png, my_images/my_img.png.
    """
    from PIL import Image
    image = Image.fromarray(image_array)
    image.save(f_name)


def denorm(image):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

    image_denorm = image * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)

    return image_denorm


class DeltaJP:
    def __init__(self, **kwargs):
        self.train_num = kwargs['train_num']
        self.eps = kwargs['eps']
        self.epsilon = kwargs['epsilon'] / 255
        self.alpha = kwargs['alpha'] / 255
        #self.n_iters = kwargs['n_iters']
        self.steps = kwargs['steps']
        self.constrained = kwargs['constrained']
        self.batch_size = kwargs['batch_size']
        self.random_start = kwargs['random_start']
        self.image_path = kwargs['image_path']

        # with open(kwargs['target_path'], 'r') as f:
        #     data = list(csv.reader(f, delimiter=","))
        # self.targets = []
        # num = len(data)
        # for i in range(num):
        #     self.targets.append(data[i][0])    

        with open(kwargs['target_path'], 'r') as f:
            data = json.load(f)
        self.targets = data

        self.target_model = kwargs['target_model']
        self.model = getattr(multimodalmodels, self.target_model)()


    # def generate(self, save_dir):

    #     random_number = random.randint(1, 2000)
    #     random.seed(random_number)
    #     np.random.seed(random_number)
    #     torch.manual_seed(random_number)

    #     model = MiniGPT(self.train_num)
    #     model = model.eval()
    #     attack = PGD(model, eps=self.eps / 255, alpha=self.alpha / 255, steps=self.steps, nprompt=model.train_num,
    #              random_start=False)
        
    #     attack.set_mode_targeted_by_label()
    #     mean = (0.48145466, 0.4578275, 0.40821073)
    #     std = (0.26862954, 0.26130258, 0.27577711)
    #     attack.set_normalization_used(mean, std)

    #     image = torch.zeros(1, 3, 224, 224).to(device)
    #     images = []
    #     images.append(image)
    #     adv_img = attack(images, model.shift_labels_1)
    #     adv_image = denorm(adv_img[0])
    #     save_img = (adv_image[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #     save_image(save_img, '%s/bad_prompt.bmp' % save_dir)

    def generate_test_cases(self, behaviors, save_dir, verbose=False):
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'images'), exist_ok=True)    

        test_cases = {}

        self.attack_image_path = os.path.join(save_dir, 'images')
        #self.generate(image_path)

        behaviors_list = [behavior["Behavior"] for behavior in behaviors]  
        behaviors_unique = np.unique(behaviors_list)


        for behavior_unique in tqdm(behaviors_unique, total = len(behaviors_unique)):     
            behavior_dict = [behavior_dict for behavior_dict in behaviors if behavior_dict['Behavior'] == behavior_unique]
            behavior = behavior_dict[0]['Behavior']
            #behavior_id = behavior_dict['BehaviorID']
            if verbose:
                print(f"Behavior: {behavior}")
                #print(f"Behavior ID: {behavior_id}")

            self.PGD(behavior_dict)
            # if self.constrained:
            #     self.PGD_constrained(behavior_dict)
            # else:
            #     self.PGD_unconstrained(behavior_dict)
            
            #current_test_case = ['bad_prompt.bmp', behavior]
            for behavior_dict_ in behavior_dict:
                test_cases[behavior_dict_["BehaviorID"]] = [behavior_dict_["ImageFileName"], behavior_dict_["Behavior"]]

        save_path = os.path.join(save_dir, 'DeltaJP_test_cases.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        with open(save_path, 'w') as file:
            json.dump(test_cases, file, indent=2)        

    def PGD(self, behavior_dict):
        """
        :param behavior_dict: the behavior_dict of targets
        :return: adv_image
        """
        #noise_pil = transforms.ToPILImage()(torch.zeros(3, 224, 224))

        #adv_noise = torch.rand(1, 3, 336, 336).to(self.model.device) # [0,1]
        adv_noise = torch.rand(1, 3, 336, 336).to(self.model.device) * 2 * self.epsilon - self.epsilon

        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()
        #img_pil = transforms.ToPILImage()(torch.zeros(3, 224, 224))
        # img_pil = Image.open(image_path).convert('RGB')

        # img = self.model.vis_processor(img_pil, return_tensors='pt')["pixel_values"].to(self.model.device)

        self.model.model.requires_grad_(False)

        targets = [[self.targets[behavior_dict_["BehaviorID"]], behavior_dict_['ImageFileName']] for behavior_dict_ in behavior_dict]

        pbar = tqdm(range(self.steps+1))

        for t in pbar:
            if len(targets) >= self.batch_size:
                batch_targets = random.sample(targets, self.batch_size)
            else:
                # 처리 방법 선택: 전체 사용 / 스킵 / 경고 출력
                batch_targets = targets
            
            img_pil_list = [Image.open(os.path.join(self.image_path, behavior_dict_[1])).convert('RGB') for behavior_dict_ in batch_targets]
        
            img_tensors = self.model.vis_processor(img_pil_list, return_tensors='pt')["pixel_values"].to(self.model.device)

            # x_adv = x + adv_noise            
            # x_adv = self.normalize(x_adv)
            # x_adv = self.normalize(img_tensors + adv_noise)
            x_adv = img_tensors + adv_noise

            #target_loss = self.model.compute_loss_batch(x_adv, [x[0] for x in batch_targets], img_pil_list)
            target_loss = self.model.compute_loss_batch(x_adv, [x[0] for x in batch_targets], [transforms.ToPILImage()(x) for x in img_tensors])

            target_loss.backward()

            adv_noise.data = (adv_noise.data - self.alpha * adv_noise.grad.detach().sign()).clamp(0, 1)
            adv_noise.grad.zero_()
            self.model.model.zero_grad()

            pbar.set_description(f"target_loss: {target_loss.item():.4f}")

            if t % 100 == 0:
                
                print('######### Output - Iter = %d ##########' % t)     
                print("target_loss: %f" % (
                target_loss.item())
                )
        
        img_pil_list = [Image.open(os.path.join(self.image_path, behavior_dict_[1])).convert('RGB') for behavior_dict_ in targets]

        
        img_tensors = self.model.vis_processor(img_pil_list, return_tensors='pt')["pixel_values"].to(self.model.device)

        # x_adv = self.normalize(img_tensors + adv_noise)   
        x_adv = img_tensors + adv_noise   

        adv_img_prompt = self.denormalize(x_adv).detach().cpu().clip(0,1)
        #adv_img_prompt = x_adv.detach().cpu()
        for adv_img_prompt_, batch_targets_ in zip(adv_img_prompt, targets):
            #adv_img_prompt_ = adv_img_prompt_.squeeze(0) 
            transforms.ToPILImage()(adv_img_prompt_).save('{}/{}'.format(self.attack_image_path, batch_targets_[1]))
        #transforms.ToPILImage()(adv_noise).save('{}/{}'.format(self.attack_image_path, batch_targets_[1]))
        #save_image(adv_img_prompt, '%s/bad_prompt.bmp' % (image_path))
        return adv_img_prompt
     
    def normalize(self, images):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.model.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.model.device)
        images = images - mean[None, :, None, None]
        images = images / std[None, :, None, None]
        return images

    def denormalize(self, images):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.model.device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.model.device)
        images = images * std[None, :, None, None]
        images = images + mean[None, :, None, None]
        return images
    
