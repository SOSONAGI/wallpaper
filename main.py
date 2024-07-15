#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import torch
from PIL import Image
import os
import io
from torchvision import transforms as T
import torch.nn as nn
import base64
from typing import List, Dict
import pandas as pd
from transformers import AutoModel, AutoConfig

class Config:
    num_classes = 19
    class_names = ['오염', '녹오염', '면불량', '들뜸', '이음부불량', '터짐', '피스', '오타공', '석고수정', '가구수정', '몰딩수정', '틈새과다', '꼬임', '울음', '걸레받이수정', '반점', '훼손', '창틀,문틀수정', '곰팡이']
config = Config()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DINO(nn.Module):
    def __init__(self, freeze=False):
        super(DINO, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitl14')
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

# Streamlit 앱 구성
st.title("Wallpaper Classifying Apps_v2")

# 스트림릿 시크릿에서 허깅페이스 허브 토큰 가져오기
hf_token = st.secrets["huggingface_token"]

if hf_token:
    try:
        # 허깅페이스 레포지토리에서 가중치 불러오기
        huggingface_model = AutoModel.from_pretrained('sosoai/dino_checkpoint', use_auth_token=hf_token)
        huggingface_state_dict = huggingface_model.state_dict()

        model = DINO().to(device)
        model.load_state_dict(huggingface_state_dict, strict=False)  # 기존 모델의 가중치에 허깅페이스 가중치를 덮어씌움
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

            class_counter[class_name] += 1
            new_filename = f"{class_name}{class_counter[class_name]}.png"

            resized_image = image.resize((512, 512), Image.BICUBIC)
            image_path = os.path.join(target_folder, new_filename)
            resized_image.save(image_path, 'PNG')

        uploaded_files = st.file_uploader("이미지를 선택하세요...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        target_directory = './result'

        if uploaded_files:
            for uploaded_file in uploaded_files:
                classify_and_save_file(uploaded_file, target_directory)

    except Exception as e:
        st.error(f"허깅페이스 모델을 불러오는 중 오류가 발생했습니다: {str(e)}")
else:
    st.warning("허깅페이스 허브 토큰을 설정 파일에 추가하세요.")
