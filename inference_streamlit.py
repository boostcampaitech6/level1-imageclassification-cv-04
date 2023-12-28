import argparse
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, RandomHorizontalFlip, CenterCrop
import streamlit as st

def decode_pred(mask,gender,age):
    mask_dict = { 0:"MASK", 1:"INCORRECT", 2:"NORMAL"}
    gender_dict = {0:"MALE", 1:"FEMALE"}
    age_dict = {0:'YOUNG',1:'MIDDLE',2:'OLD'}
    
    return mask_dict[mask],gender_dict[gender],age_dict[age]
    

def main(config):
    # device = torch.device('cuda')
    device = torch.device('cpu')

    # 모델을 정의합니다. (학습한 모델이 있다면 torch.load로 모델을 불러주세요!)
    model_module = getattr(module_arch, config.model)
    model = model_module(num_classes=18).to(device)
    model.load_state_dict(torch.load(config.model_path, map_location=device))
    model.eval()

    # 모델이 테스트 데이터셋을 예측하고 결과를 저장합니다.
    all_predictions = []
    pred_masks = []
    pred_genders = []
    pred_ages = []

    transform = transforms.Compose([
        Resize(config.resize, Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.548, 0.504, 0.497), std=(0.237, 0.247, 0.246))
    ])

    col1, col2 = st.columns(2)
    my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if my_upload is not None:
        with torch.no_grad():
            image = Image.open(my_upload)
            images = transform(image).unsqueeze(dim=0)
            images = images.to(device)
            col1.write("Input Image :camera:")
            col1.image(image)

            if config.multi_head:
                pred = model(images)
                pred_mask, pred_gender, pred_age = pred
                pred = torch.argmax(pred_mask, dim=-1) * 6 + torch.argmax(pred_gender, dim=-1) * 3 + torch.argmax(pred_age, dim=-1)
                all_predictions.extend(pred.cpu().numpy())

                pred_masks.extend(torch.argmax(pred_mask, dim=-1).cpu().numpy())
                pred_genders.extend(torch.argmax(pred_gender, dim=-1).cpu().numpy())
                pred_ages.extend(torch.argmax(pred_age, dim=-1).cpu().numpy())

                col2.write("Model predict 🤖")
                col2.write(decode_pred(pred_masks[0], pred_genders[0], pred_ages[0]))
            else:
                pred = model(images)
                pred = pred.argmax(dim=-1)
                all_predictions.extend(pred.cpu().numpy())
                



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument(
        "--model", type=str, default="SwinTransformerBase224V1", help="model type (default: EfficientNetB0MultiHead)"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=os.environ.get("SM_CAHNNEL_EVAL", "/data/ephemeral/maskdata/eval")
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[224, 224],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--multi_head", 
        type=int,
        default=1,
        help="모델의 head가 1개(num_classes=18)인 경우 False, 3개인 경우 True"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="C:/Users/User/Documents/boostcamp/code/week6,7/level1-imageclassification-cv-04/results/model_SwinTransformerBase224/best.pth",
        help="사용할 모델의 weight 경로를 입력해주세요 (예: /data/ephemeral/home/model/exp/best.pth)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="input batch size for validing (default: 1000)",
    )
    args = parser.parse_args()
    print(args)

    main(args)
