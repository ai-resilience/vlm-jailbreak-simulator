"""VLMShield feature extraction and classifier (pgqihere/VLMShield@344ca14).

The functions below are the upstream ``VLMShield_script/VLMShield.py`` CLIP
encoding helpers and ``classifier.py::ThreeLayerClassifier`` with their bodies
kept line for line; only the module plumbing (imports, a pair-level helper)
was added so the unified defense adapter can call them.

MAFE features: CLIP ViT-L/14 image [CLS] -> visual projection (768) and a
chunked / similarity-weighted text [EOS] -> text projection (768), concatenated
to 1536 dimensions; a missing modality contributes a zero vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIM = 1536  # CLIP image embedding(768) + CLIP text embedding(768)
CLIP_EMBED_DIM = 768
CHUNK_SIZE = 75
OVERLAP_SIZE = 10


class ThreeLayerClassifier(nn.Module):
    """``classifier.py::ThreeLayerClassifier`` (1536-1024-512-2, dropout 0.5)."""

    def __init__(self, dim=EMBEDDING_DIM):
        super().__init__()
        self.fc1 = nn.Linear(dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        prob = F.softmax(x, dim=-1)
        return x, prob


# =========================
# CLIP encoding functions (upstream VLMShield.py)
# =========================

def clip_image_cls_embed(model, processor, images, device):
    """Returns image vectors in the aligned space [B, projection_dim] (768)."""
    with torch.no_grad():
        proc = processor(images=images, return_tensors="pt")
        pixel_values = proc["pixel_values"].to(device)  # [B, 3, H, W]
        vision_out = model.vision_model(pixel_values=pixel_values)
        cls_feats = vision_out.last_hidden_state[:, 0, :]  # [B, vision_width]
        image_embeds = model.visual_projection(cls_feats)  # [B, embed_dim]
    return image_embeds


def progressive_eos_aggregation(model, processor, prompt, device, chunk_size=CHUNK_SIZE):
    tokenizer = processor.tokenizer
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    total_tokens = len(tokens)

    if total_tokens + 2 <= chunk_size + 2:  # +2 for BOS and EOS
        return standard_clip_processing(model, processor, prompt, device)

    eos_embeddings_history = []
    processed_tokens = 0
    overlap_size = OVERLAP_SIZE  # Maintain semantic continuity with overlap

    while processed_tokens < total_tokens:
        remaining_tokens = total_tokens - processed_tokens
        if remaining_tokens <= chunk_size and eos_embeddings_history:
            remaining_chunk = tokens[processed_tokens:]
            return process_final_chunk_with_eos_history(
                model, tokenizer, remaining_chunk, eos_embeddings_history, device, chunk_size
            )
        chunk_end = min(processed_tokens + chunk_size, total_tokens)
        chunk_tokens = tokens[processed_tokens:chunk_end]
        chunk_eos = process_standard_chunk(model, tokenizer, chunk_tokens, device)
        eos_embeddings_history.append(chunk_eos)
        processed_tokens = chunk_end - overlap_size if chunk_end < total_tokens else chunk_end

    if eos_embeddings_history:
        return model.text_projection(eos_embeddings_history[-1])

    return standard_clip_processing(model, processor, prompt, device)


def standard_clip_processing(model, processor, prompt, device):
    with torch.no_grad():
        text_inputs = processor(text=prompt, return_tensors="pt", padding=True,
                                truncation=True, max_length=77)
        input_ids = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = text_out.last_hidden_state.size(0)
        eos_feats = text_out.last_hidden_state[torch.arange(batch_size), sequence_lengths]
        text_embeds = model.text_projection(eos_feats)
    return text_embeds[0]


def process_standard_chunk(model, tokenizer, chunk_tokens, device):
    bos_token = getattr(tokenizer, "bos_token_id", None) or getattr(tokenizer, "cls_token_id", None) or 49406
    eos_token = getattr(tokenizer, "eos_token_id", None) or getattr(tokenizer, "sep_token_id", None) or 49407
    full_tokens = [bos_token] + chunk_tokens + [eos_token]
    input_ids = torch.tensor(full_tokens, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
        eos_hidden = text_out.last_hidden_state[0, -1, :]
    return eos_hidden


def intelligent_eos_aggregation(eos_embeddings, device=None):
    if len(eos_embeddings) == 1:
        return eos_embeddings[0]
    eos_stack = torch.stack(eos_embeddings)  # [num_chunks, hidden_dim]
    with torch.no_grad():
        normalized = torch.nn.functional.normalize(eos_stack, p=2, dim=1)
        sim = torch.mm(normalized, normalized.t())
        represent = (sim.sum(dim=1) - 1) / (len(eos_embeddings) - 1)
        weights = torch.nn.functional.softmax(represent, dim=0)
        final_eos = (eos_stack * weights.unsqueeze(1)).sum(dim=0)
    return final_eos


def process_final_chunk_with_eos_history(model, tokenizer, remaining_tokens, eos_history, device,
                                         max_length=CHUNK_SIZE):
    bos_token = getattr(tokenizer, "bos_token_id", None) or 49406
    eos_token = getattr(tokenizer, "eos_token_id", None) or 49407
    current_eos = None
    if remaining_tokens:
        max_remaining = max_length - 2
        if len(remaining_tokens) > max_remaining:
            remaining_tokens = remaining_tokens[-max_remaining:]
        current_tokens = [bos_token] + remaining_tokens + [eos_token]
        input_ids = torch.tensor(current_tokens, device=device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            text_out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            current_eos = text_out.last_hidden_state[0, -1, :]

    all_eos = []
    if eos_history:
        all_eos.extend(eos_history)
    if current_eos is not None:
        all_eos.append(current_eos)

    if not all_eos:
        hidden_dim = model.text_model.config.hidden_size
        zero_hidden = torch.zeros(hidden_dim, device=device)
        return model.text_projection(zero_hidden)

    final_eos = intelligent_eos_aggregation(all_eos, device)
    return model.text_projection(final_eos)


def clip_text_eos_embed(model, processor, prompts, device):
    if not isinstance(prompts, list):
        prompts = [prompts]
    embeddings = []
    for prompt in prompts:
        embeddings.append(progressive_eos_aggregation(model, processor, prompt, device))
    return torch.stack(embeddings)


# =========================
# Pair-level helpers (adapter plumbing)
# =========================

def multimodal_embedding(model, processor, text, image, device):
    """``MultimodalProcessor.process_single_sample`` feature construction for one
    (text, PIL image or None) pair: a missing modality is a zero 768-vector."""
    image_embed = None
    text_embed = None
    if image is not None:
        image_embed = clip_image_cls_embed(model, processor, [image], device)[0]
    if text and text.strip():
        text_embed = clip_text_eos_embed(model, processor, [text], device)[0]
    has_image, has_text = image_embed is not None, text_embed is not None
    if image_embed is None:
        image_embed = torch.zeros(CLIP_EMBED_DIM, device=device)
    if text_embed is None:
        text_embed = torch.zeros(CLIP_EMBED_DIM, device=device)
    return torch.cat([image_embed, text_embed], dim=0), has_image, has_text


def classify(classifier, embedding):
    """Upstream decision: argmax of the softmax; label 1 = safe, 0 = unsafe."""
    with torch.no_grad():
        logits, probs = classifier(embedding.unsqueeze(0).float())
        prediction = int(probs.argmax(dim=1).item())
        confidence = float(probs.max(dim=1)[0].item())
    return prediction, confidence, [float(p) for p in probs[0].tolist()]
