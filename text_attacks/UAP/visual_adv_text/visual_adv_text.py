import torch
from tqdm import tqdm
import random
import time
import numpy as np
import csv
import json
import os
import multimodalmodels
from transformers import AutoTokenizer
import re

class Visual_Adv_Text:
    def __init__(self, **kwargs):
        self.epsilon = kwargs.get('epsilon', 0.1)  # 텍스트 perturbation 크기
        self.alpha = kwargs.get('alpha', 0.01)     # 학습률
        self.n_iters = kwargs.get('n_iters', 100)
        self.constrained = kwargs.get('constrained', True)
        self.batch_size = kwargs.get('batch_size', 4)
        
        # 타겟 텍스트들 로드
        with open(kwargs['target_path'], 'r') as f:
            data = list(csv.reader(f, delimiter=","))
        self.targets = []
        num = len(data)
        for i in range(num):
            self.targets.append(data[i][0])      

        self.target_model = kwargs['target_model']
        self.model = getattr(multimodalmodels, self.target_model)()
        
        # 토크나이저 초기화
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_test_cases(self, behaviors, save_dir, verbose=False):
        """
        텍스트만 perturbation하는 테스트 케이스를 생성합니다.
        
        :param behaviors: 생성할 behavior들의 리스트
        :param save_dir: 저장할 디렉토리
        :param verbose: 진행상황 출력 여부
        :return: 테스트 케이스 딕셔너리
        """
        test_cases = {}

        if self.constrained:
            adv_text = self.PGD_constrained_text()
        else:
            adv_text = self.PGD_unconstrained_text()

        # 텍스트 저장
        text_path = os.path.join(save_dir, 'texts')
        if not os.path.exists(text_path):
            os.makedirs(text_path)

        with open(os.path.join(text_path, 'adversarial_text.txt'), 'w') as f:
            f.write(adv_text)

        for behavior_dict in tqdm(behaviors, total=len(behaviors)):           
            behavior = behavior_dict['Behavior']
            behavior_id = behavior_dict['BehaviorID']
            if verbose:
                print(f"Behavior: {behavior}")
                print(f"Behavior ID: {behavior_id}")

            # 텍스트 공격의 경우 이미지 대신 텍스트를 사용
            current_test_case = ['adversarial_text.txt', behavior]
            test_cases[behavior_id] = current_test_case

        save_path = os.path.join(save_dir, 'Visual_Adv_Text.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
        with open(save_path, 'w') as file:
            json.dump(test_cases, file, indent=2)

        return test_cases

    def PGD_unconstrained_text(self):
        """
        제약이 없는 텍스트 PGD 공격
        :return: adversarial 텍스트
        """
        # 초기 텍스트 (예: "Hello, how are you?")
        initial_text = "Hello, how are you?"
        
        # 텍스트를 토큰으로 변환
        tokens = self.tokenizer.encode(initial_text, return_tensors='pt').to(self.model.device)
        
        # 랜덤 노이즈 초기화
        adv_noise = torch.randn_like(tokens.float()) * 0.1
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        for t in tqdm(range(self.n_iters + 1)):
            batch_targets = random.sample(self.targets, self.batch_size)
            
            # 노이즈가 적용된 토큰
            perturbed_tokens = tokens + adv_noise
            
            # 텍스트로 변환 (가장 가까운 토큰으로 반올림)
            token_ids = torch.round(perturbed_tokens).long().clamp(0, self.tokenizer.vocab_size - 1)
            perturbed_text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
            
            # 모델에 텍스트만 입력 (이미지는 None)
            target_loss = self.model.compute_loss_text_only(perturbed_text, batch_targets)
            target_loss.backward()

            # PGD 업데이트
            adv_noise.data = (adv_noise.data - self.alpha * adv_noise.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
            adv_noise.grad.zero_()
            self.model.model.zero_grad()

            if t % 10 == 0:
                print(f"target_loss: {target_loss.item():.4f}")
                print(f"Perturbed text: {perturbed_text[:100]}...")

        # 최종 adversarial 텍스트 반환
        final_token_ids = torch.round(tokens + adv_noise).long().clamp(0, self.tokenizer.vocab_size - 1)
        final_text = self.tokenizer.decode(final_token_ids[0], skip_special_tokens=True)
        
        return final_text

    def PGD_constrained_text(self):
        """
        제약이 있는 텍스트 PGD 공격 (원본 텍스트에서 일정 거리 내에서만)
        :return: adversarial 텍스트
        """
        # 초기 텍스트
        initial_text = "Hello, how are you?"
        
        # 텍스트를 토큰으로 변환
        tokens = self.tokenizer.encode(initial_text, return_tensors='pt').to(self.model.device)
        
        # 제약된 노이즈 초기화
        adv_noise = torch.randn_like(tokens.float()) * 2 * self.epsilon - self.epsilon
        adv_noise.data = adv_noise.data.clamp(-self.epsilon, self.epsilon)
        
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        for t in tqdm(range(self.n_iters + 1)):
            batch_targets = random.sample(self.targets, self.batch_size)
            
            # 제약된 노이즈가 적용된 토큰
            perturbed_tokens = tokens + adv_noise
            
            # 텍스트로 변환
            token_ids = torch.round(perturbed_tokens).long().clamp(0, self.tokenizer.vocab_size - 1)
            perturbed_text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
            
            # 모델에 텍스트만 입력
            target_loss = self.model.compute_loss_text_only(perturbed_text, batch_targets)
            target_loss.backward()

            # 제약된 PGD 업데이트
            adv_noise.data = (adv_noise.data - self.alpha * adv_noise.grad.detach().sign()).clamp(-self.epsilon, self.epsilon)
            adv_noise.grad.zero_()

            if t % 10 == 0:
                print(f"target_loss: {target_loss.item():.4f}")
                print(f"Perturbed text: {perturbed_text[:100]}...")

        # 최종 adversarial 텍스트 반환
        final_token_ids = torch.round(tokens + adv_noise).long().clamp(0, self.tokenizer.vocab_size - 1)
        final_text = self.tokenizer.decode(final_token_ids[0], skip_special_tokens=True)
        
        return final_text

    def compute_text_similarity(self, text1, text2):
        """
        두 텍스트 간의 유사도를 계산합니다.
        """
        tokens1 = set(self.tokenizer.encode(text1))
        tokens2 = set(self.tokenizer.encode(text2))
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0

    def validate_text_perturbation(self, original_text, perturbed_text):
        """
        텍스트 perturbation이 유효한지 검증합니다.
        """
        similarity = self.compute_text_similarity(original_text, perturbed_text)
        return similarity > 0.5  # 50% 이상 유사해야 함 