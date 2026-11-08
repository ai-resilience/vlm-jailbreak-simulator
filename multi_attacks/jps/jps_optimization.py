"""JPS optimization methods ported from the reference simulator.

The momentum PGD and alternating visual/text schedule retain the reference
math; finite-loss/gradient checks and diagnostics are added by the port.
"""
from __future__ import annotations
import os
from fractions import Fraction
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from data_paths import REPO_ROOT
from .jps_prompts import JB_PROMPT_TEMPLATE
from .jps_adapter import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

def _affirmative_target(instruction: str) -> str:
    """Affirmative-prefix target, mirroring JPS goal construction / AdvBench targets."""
    return f"Sure, here is {instruction[0].lower() + instruction[1:]}"

def _official_target(target: str) -> str:
    """Apply JPS's Qwen2-VL target rewrite exactly once.

    Official ``qwen2vl_7b_preprocessor.py`` replaces the first ``Sure,`` and
    then the first lower-case ``here`` before both optimization and inference.
    """
    return target.replace(
        "Sure,", "I am happy to help you with that.", 1
    ).replace("here", "Here", 1)

def _jps_prompt(question: str, target: str, steer: str = "") -> str:
    """The exact prompt contract shared by PGD, revision sampling and eval."""
    query = JB_PROMPT_TEMPLATE.format(attack_question=question, attack_target=target)
    # Upstream files already contain the separator; local agent output does not.
    return (steer.rstrip("\n") + "\n\n" if steer else "") + query

def _parse_ratio(value) -> float:
    """Parse numbers such as ``32/255`` without executing config text."""
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(Fraction(str(value).strip()))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid numeric ratio: {value!r}") from exc

def _project_linf(original: torch.Tensor, candidate: torch.Tensor, epsilon: float):
    """Project a candidate onto one immutable original-image L_inf ball."""
    delta = (candidate - original).clamp(-epsilon, epsilon)
    return (original + delta).clamp(0, 1)


class JPSOptimization:
    def _pgd(self, adapter, train, steer, original_img, adv_img):
        device = adapter.device
        mean = torch.tensor(OPENAI_CLIP_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(OPENAI_CLIP_STD, device=device).view(1, 3, 1, 1)
        base = original_img.detach().clone()
        projected = _project_linf(base, adv_img.detach(), self.epsilon)
        delta = (projected - base).detach().requires_grad_(True)
        momentum = torch.zeros_like(base)
        best_loss = float("inf")
        best_adv = (base + delta).detach().clamp(0, 1)
        updates = 0
        loss_history, gradient_history = [], []
        stopped_early = False

        # Pre-build the official user+assistant tokenization for every sample.
        prepared = []
        for q, target in train:
            text = _jps_prompt(q, target, steer)
            ids, labels, _, grid, attn = adapter.process_jps_target(
                self._rel(self.base_image), text, target
            )
            prepared.append((ids, labels, grid, attn))

        pad_id = adapter.processor.tokenizer.pad_token_id

        def left_pad(tensor, length, value):
            amount = length - tensor.size(1)
            if amount <= 0:
                return tensor
            prefix = torch.full(
                (tensor.size(0), amount), value, dtype=tensor.dtype, device=tensor.device
            )
            return torch.cat((prefix, tensor), dim=1)

        for step in range(self.attack_steps):
            total = 0.0
            grad_accum = torch.zeros_like(base)
            all_finished = True
            for batch_start in range(0, len(prepared), self.batch_size):
                batch = prepared[batch_start:batch_start + self.batch_size]
                max_length = max(item[0].size(1) for item in batch)
                ids = torch.cat([left_pad(item[0], max_length, pad_id) for item in batch])
                labels = torch.cat([left_pad(item[1], max_length, -100) for item in batch])
                attn = torch.cat([left_pad(item[3], max_length, 0) for item in batch])
                grids = torch.cat([item[2] for item in batch], dim=0)

                # delta stays in the graph; one patchified universal image is
                # repeated as Qwen2-VL's packed pixel sequence for the batch.
                x = (base + delta).clamp(0, 1)
                pv_one = adapter.preprocess_patched(
                    ((x - mean) / std)[0].to(adapter.model.dtype), batch[0][2][0]
                )
                pixel_values = pv_one.repeat(len(batch), 1)
                out = adapter.model(
                    input_ids=ids,
                    attention_mask=attn,
                    pixel_values=pixel_values,
                    image_grid_thw=grids,
                )
                token_losses = F.cross_entropy(
                    out.logits[:, :-1, :].transpose(1, 2).float(),
                    labels[:, 1:],
                    ignore_index=-100,
                    reduction="none",
                )
                valid = labels[:, 1:].ne(-100)
                sample_losses = (token_losses * valid).sum(1) / valid.sum(1).clamp_min(1)
                if not bool(torch.isfinite(sample_losses).all()):
                    raise RuntimeError("JPS sample losses must be finite")
                active = sample_losses > self.min_loss_limit
                total += float(sample_losses.detach().sum())
                if active.any():
                    all_finished = False
                    active_loss = sample_losses[active].mean()
                    gradient = torch.autograd.grad(active_loss, delta)[0]
                    if not bool(torch.isfinite(gradient).all()):
                        raise RuntimeError("JPS pixel gradient must be finite")
                    grad_accum = grad_accum + gradient
            mean_loss = total / max(len(prepared), 1)
            loss_history.append(mean_loss)
            if mean_loss < best_loss:
                best_loss = mean_loss
                best_adv = (base + delta).detach().clamp(0, 1)

            # Official JPS stops once every target loss is below the threshold.
            if all_finished:
                stopped_early = True
                break

            # Official JPS uses L2-normalized gradient momentum with beta=0.9.
            g = grad_accum / max(len(prepared), 1)
            gradient_history.append({"finite": bool(g.isfinite().all()),
                                     "nonzero": bool(g.count_nonzero()),
                                     "mean_abs": float(g.detach().abs().mean())})
            flat_norm = g.flatten(1).norm(p=2, dim=1).view(-1, 1, 1, 1).clamp_min(1e-8)
            normalized = g / flat_norm
            momentum = self.momentum_decay * momentum + (1 - self.momentum_decay) * normalized
            delta = (delta.detach() - self.alpha * momentum.sign()).clamp(
                -self.epsilon, self.epsilon
            )
            delta = (_project_linf(base, base + delta, self.epsilon) - base)
            delta = delta.detach().requires_grad_(True)
            updates += 1
            if step % 20 == 0:
                print(f"  [JPS pgd] step {step}/{self.attack_steps} loss={mean_loss:.4f}", flush=True)
        self._last_pgd_stats = {
            "updates": updates,
            "losses": loss_history,
            "pixel_gradients": gradient_history,
            "linf_from_original": float((best_adv - base).abs().max()),
            "best_loss": best_loss,
            "stopped_early": stopped_early,
        }
        return best_adv

    def _optimize(self, behaviors, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self._agent_trace_path = os.path.join(out_dir, "jps_agent_trace.jsonl")
        # Each optimization owns one trace. Never mix a prior failed run's
        # agent responses into the current audit log.
        with open(self._agent_trace_path, "w", encoding="utf-8"):
            pass
        adapter = self._load()
        device = adapter.device
        train = [(b.instruction, self._target_for(b))
                 for b in behaviors[: self.attack_data_num]]

        img = Image.open(self._rel(self.base_image)).convert("RGB").resize((self.image_size, self.image_size))
        original = transforms.ToTensor()(img).unsqueeze(0).to(device)
        adv = original.clone()
        steer_angles = ["", ""]

        # Upstream first optimizes the image once, then alternates
        # response-feedback/revision and another image optimization.
        adv = self._pgd(adapter, train, "", original, adv)
        for r in range(self.rounds):
            # textual revision: sample responses, then run the 3-agent loop
            os.makedirs(out_dir, exist_ok=True)
            tmp = os.path.join(out_dir, "jps_tmp.png")
            transforms.ToPILImage()(adv.squeeze(0).cpu()).save(tmp)
            qa = []
            steer_now = "\n".join(s for s in steer_angles if s)
            for q, target in train:
                prompt = _jps_prompt(q, target, steer_now)
                # Upstream mutates ``origin_question`` during preprocessing;
                # the feedback agent therefore sees the prefix-constrained
                # question (without the current steer), not the raw behavior.
                feedback_question = _jps_prompt(q, target)
                qa.append((feedback_question, adapter.generate(
                    prompt, tmp, max_new_tokens=self.response_max_new_tokens
                )))
            # Upstream explicitly frees Qwen2-VL before loading the 14B agent,
            # then frees the agent and reloads Qwen2-VL for the next PGD pass.
            # Preserve that memory-bounded schedule for the official HF backend.
            if self.agent_backend == "huggingface":
                import gc

                adv_cpu = adv.detach().cpu()
                original_cpu = original.detach().cpu()
                self._adapter = None
                del adapter, adv, original
                gc.collect()
                torch.cuda.empty_cache()

                agents = self._load_agents()
                steer_angles = agents.revise(qa, steer_angles, round_index=r)
                self._agents = None
                del agents
                gc.collect()
                torch.cuda.empty_cache()

                adapter = self._load()
                original = original_cpu.to(adapter.device)
                adv = adv_cpu.to(adapter.device)
            else:
                steer_angles = self._load_agents().revise(
                    qa, steer_angles, round_index=r
                )
            adv = self._pgd(
                adapter, train, "\n".join(s for s in steer_angles if s), original, adv
            )
            print(f"  [JPS] round {r+1}/{self.rounds} steer updated", flush=True)

        os.makedirs(out_dir, exist_ok=True)
        fpath = os.path.join(out_dir, "jps_adv.png")
        transforms.ToPILImage()(adv.squeeze(0).cpu()).save(fpath)
        self._steer = "\n".join(s for s in steer_angles if s)
        # save the steer prompt next to the image so later runs can reuse both
        with open(os.path.splitext(fpath)[0] + "_steer.txt", "w", encoding="utf-8") as f:
            f.write(self._steer)
        return os.path.relpath(fpath, REPO_ROOT)
