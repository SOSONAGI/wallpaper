#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import torch
from PIL import Image
import os
import io
from torchvision import transforms as T
import torch.nn as nn
import timm
import base64
from typing import List, Dict
import pandas as pd
from huggingface_hub import hf_hub_download

class Config:
    num_classes = 19
    class_names = ['오염', '녹오염', '면불량', '들뜸', '이음부불량', '터짐', '피스', '오타공', '석고수정', '가구수정', '몰딩수정', '틈새과다', '꼬임', '울음', '걸레받이수정', '반점', '훼손', '창틀,문틀수정', '곰팡이']

config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DINO(nn.Module):
    def __init__(self, freeze=False):
        super(DINO, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        self.freeze = freeze
        if self.freeze:
            for para in self.model.parameters():
                para.requires_grad = False
        self.clf = nn.Sequential(
            nn.Linear(self.model.embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 19),
        )

    def forward(self, img):
        if self.freeze:
            with torch.no_grad():
                o = self.model(img)
        else:
            o = self.model(img)
        o = self.clf(o)
        return o

# Preprocess the image
val_transform = T.Compose([
    T.Resize((336, 336), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# 허깅페이스에서 모델 가중치 다운로드
model_path = hf_hub_download(repo_id="sosoai/dino_checkpoint", filename="best3.pt")

model = DINO().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def predict(image):
    image_transformed = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_transformed)
        predicted_class_idx = torch.argmax(output, dim=1).item()
    return predicted_class_idx

class_counter = {class_name: 0 for class_name in config.class_names}

def classify_and_save_file(uploaded_file, target_dir):
    image = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    class_index = predict(image)
    class_name = config.class_names[class_index]
    target_folder = os.path.join(target_dir, class_name)
    os.makedirs(target_folder, exist_ok=True)

    # Increment the class counter and generate the new file name
    class_counter[class_name] += 1
    new_filename = f"{class_name}{class_counter[class_name]}.png"

    # 이미지 크기를 512x512로 변경
    resized_image = image.resize((512, 512), Image.BICUBIC)

    # 이미지를 PNG 형식으로 저장
    image_path = os.path.join(target_folder, new_filename)
    resized_image.save(image_path, 'PNG')

st.title("Wallpaper Classifying Apps_v2")

uploaded_files = st.file_uploader("이미지를 선택하세요...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

target_directory = './result'

if uploaded_files:
    for uploaded_file in uploaded_files:
        classify_and_save_file(uploaded_file, target_directory)
