"""
JOOD
Copyright (c) 2025-present NAVER Corp.
Apache License v2.0
"""
import json
import base64
from io import BytesIO

def read_json(file):
    with open(file, "r") as f:
        data = json.load(f)
    return data
    
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def encode_base64(image_pil):
    # Save to a BytesIO object instead of saving to disk
    image_buffered = BytesIO()
    image_pil.save(image_buffered, format="PNG")
    # Encode the BytesIO object to base64
    image_base64 = base64.b64encode(image_buffered.getvalue()).decode('utf-8')
    return image_base64