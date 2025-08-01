#!/bin/bash
# Install build dependencies first
pip install setuptools-rust
pip install meson-python
pip install wheel

# Install other packages
pip install huggingface-hub
pip install transformers==4.45.2
pip install spacy
pip install datasketch
python -m spacy download en_core_web_sm
pip install pandas
pip install matplotlib
pip install omegaconf
pip install iopath
pip install timm
pip install opencv-python-headless
pip install webdataset
pip install scikit-image
pip install visual_genome

# Install peft with specific version
pip install peft==0.4.0

pip install decord
pip install wandb
pip install seaborn

# Install vllm with pre-built wheels to avoid compilation issues
pip install vllm --no-build-isolation

# Install transformers again to ensure correct version
pip install transformers==4.45.2

# Install fastchat
pip install fastchat

# Force reinstall peft to ensure correct version
pip install --force-reinstall peft==0.4.0

echo "Installation completed!" 