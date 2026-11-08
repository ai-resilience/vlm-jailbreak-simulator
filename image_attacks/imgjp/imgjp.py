"""ImgJP: one image fitted sequentially to multiple goal/target pairs.

Adapted from fd98910's v1_mprompt.py and pgd_uap_v1.py. The MiniGPT
checkpoint/full-conversation objective is replaced by shared target-only CE.
"""
from PIL import Image

from benchmark_datasets import read_csv
from data_paths import resolve_path
from image_attacks.jp_common import JPAttack
from image_attacks.jp_pgd import optimize_imgjp


class ImgJP(JPAttack):
    name = 'ImgJP'
    module = 'imgjp'
    default_dataset = 'AdvBench'

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.train_csv = resolve_path(self.config.get('train_csv') or self.config.get('target_path')
                                      or 'data/original/advbench/harmful_behaviors_2.csv')
        self.candidate_limit = self.config.get('train_candidate_limit', 483)
        if type(self.candidate_limit) is not int or self.candidate_limit < 1:
            raise ValueError('train_candidate_limit must be a positive integer')

    def _select_training(self):
        import random
        candidates = read_csv(self.train_csv)[:self.candidate_limit]
        if len(candidates) < self.train_num:
            raise ValueError('ImgJP train_num exceeds the available goal/target rows')
        indices = sorted(random.Random(self.seed).sample(range(len(candidates)), self.train_num))
        rows = []
        for index in indices:
            original = candidates[index]
            if not all(isinstance(original.get(key), str) and original[key].strip() for key in ('goal', 'target')):
                raise ValueError('ImgJP training CSV requires nonempty goal,target columns; a headerless corpus is not an ImgJP goal/target set')
            rows.append({'BehaviorID': f'imgjp_train_{index:04d}', 'Behavior': original['goal'],
                         'target': original['target']})
        return rows

    def _optimize(self):
        import torch
        from tqdm import tqdm
        train = self.training_corpus()
        surrogate = self._load_surrogate()
        pil = Image.new('RGB', (224, 224))
        geometry = surrogate.to_unit01(pil)
        # Upstream Attack.__call__ inverse-normalizes its all-zero input.
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=geometry.device, dtype=geometry.dtype)
        base = mean[None, :, None, None].expand_as(geometry).clone()
        with tqdm(total=self.steps * len(train), desc='ImgJP') as progress:
            def loss_fn(image, index, target):
                return surrogate.compute_loss_batch(image, [target], pil, prompts=[train[index]['Behavior']])
            def callback(epoch, index, loss):
                surrogate.zero_grad()
                progress.update()
            return optimize_imgjp(base, [row['target'] for row in train], loss_fn,
                                  epsilon=self.epsilon, alpha=self.alpha, steps=self.steps,
                                  seed=self.seed, random_start=self.random_start, callback=callback)

    def _render(self, rows, folder, tensor):
        self.save_tensor_image(tensor, folder / 'bad_prompt.bmp')
        return ({row['BehaviorID']: ['bad_prompt.bmp', row['Behavior']] for row in rows},
                {'universal_image': True, 'source': 'optimized_goal_target_image'})


ImgJPAttack = ImgJP
