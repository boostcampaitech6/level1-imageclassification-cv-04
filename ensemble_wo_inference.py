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
import numpy as np
import glob
import operator

def decode_pred(mask,gender,age):
    mask_dict = { 0:"MASK", 1:"INCORRECT", 2:"NORMAL"}
    gender_dict = {0:"MALE", 1:"FEMALE"}
    age_dict = {0:'YOUNG',1:'MIDDLE',2:'OLD'}
    
    return mask_dict[mask],gender_dict[gender],age_dict[age]
    

def get_equal_soft_voting(df): # 각 submission의 가중치 동일
    preds = []
    for x in range(len(df)):                    # ids
        sample = tuple([0.0]*18)
        for y in range(len(df.columns)):        # subs
            str = df.iloc[x,y]
            sample = tuple(map(operator.add, sample, (tuple(df.iloc[x,y]))))
        sample = tuple(ti/len(sample) for ti in sample)
        element = max(sample)
        idx = sample.index(element)
        preds.append(idx)
    return preds


def get_weighted_soft_voting(df, weights): # weights 예시: submission마다 가중치 [0.2315, 0.973, 0.221, 0.189, 0.123, ...]
    preds = []
    for x in range(len(df)):
        sample = tuple([0.0]*18)
        for y in range(len(df.columns)):
            k = tuple(float(weights[y]) * element for element in (tuple(df.iloc[x,y])))
            sample = tuple(map(operator.add, sample, k))
        sample = tuple(ti/len(sample) for ti in sample)
        element = max(sample)
        idx = sample.index(element)
        preds.append(idx)
    return preds


def main(config):
    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv(os.path.join(config.test_dir, 'info.csv'))
    # image_dir = os.path.join(config.test_dir, 'images')
    # image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]

    # submission path로부터 submissions 리스트 생성
    submissions = sorted(glob.glob(f'{config.submissions_dir}/*.csv'))

    if config.ensemble == 'hard':
        hvs = []
        for s in submissions:
            df = pd.read_csv(s)
            all_predictions = df['ans'].values
            hvs.append(all_predictions)

        df_hard_voting = pd.DataFrame.from_dict({k:v for k, v in enumerate(hvs)})
        ensemble_hard_predictions = np.asarray(df_hard_voting.mode(axis=1)[0])

        submission['ans'] = ensemble_hard_predictions

        # 제출할 파일을 저장합니다.
        if not os.path.exists(os.path.join(config.submissions_dir, '/ensemble_hv')):
            os.mkdir(f'{config.submissions_dir}/ensemble_hv')
        submission.to_csv(f'{config.submissions_dir}/ensemble_hv/submission.csv', index=False)

    elif config.ensemble == 'soft_equal':
        svs = []
        for s in submissions:
            df = pd.read_csv(s)
            all_predictions = []
            for i in range(len(df)):
                all_predictions.append(df.iloc[i, 2:])
            svs.append(all_predictions)

        df_soft_voting = pd.DataFrame.from_dict({k:v for k, v in enumerate(svs)})
        ensemble_soft_predictions = get_equal_soft_voting(df_soft_voting)

        submission['ans'] = ensemble_soft_predictions

        # 제출할 파일을 저장합니다.
        if not os.path.exists(os.path.join(config.submissions_dir, '/ensemble_sv_equal')):
            os.mkdir(os.path.join(config.submissions_dir, '/ensemble_sv_equal'))
        submission.to_csv(f'{config.submissions_dir}/ensemble_sv_equal/submission.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument(
        "--ensemble", type=str, default="soft_equal", help="ensemble type (default: hard)"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default=os.environ.get("SM_CAHNNEL_EVAL", "/data/ephemeral/home/level1-imageclassification-cv-04/data/eval")
    )
    parser.add_argument(
        "--submissions_dir",
        type=str,
        default=os.environ.get("SM_CAHNNEL_EVAL", "/data/ephemeral/home/level1-imageclassification-cv-04/code/submissions")
    )

    args = parser.parse_args()
    print(args)
    main(args)