#!/bin/bash

#---- install requirement packages ----
pip install torch torchvision torchaudio
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
pip install peft
pip install decord
pip install wandb
pip install seaborn
pip install vllm
pip install transformers==4.45.2
pip install fastchat
pip uninstall peft
pip install peft==0.4.0 