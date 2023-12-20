import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split, WeightedRandomSampler
from base.base_data_set import MaskBaseDataset, GenderLabels, AgeLabels, MaskLabels


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
    train / val 나누는 기준을 이미지에 대해서 random 이 아닌 사람(profile)을 기준으로 나눕니다.
    구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다.
    이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    error_labels = ['001498-1', '004432', '006359', '006360', '006361', '006362', '006363', '006364']
    caution_labels =['000214', '000226', '000725', '000736', '000763', '000767', '000773', '000817', 
                     '001049', '001509', '003724', '001200', '005523']

    def __init__(
        self,
        data_dir,
        multi_head,
        use_caution,
        mean=(0.548, 0.504, 0.479),
        std=(0.237, 0.247, 0.246),
        val_ratio=0.2,
    ):
        self.indices = defaultdict(list)
        super().__init__(data_dir, multi_head, use_caution, mean, std, val_ratio)

    @staticmethod
    def _split_profile(profiles, val_ratio):
        """프로필을 학습과 검증용으로 나누는 메서드"""
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {"train": train_indices, "val": val_indices}

    def setup(self):
        """데이터셋 설정을 하는 메서드. 프로필 기준으로 나눈다."""
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")][::2]
        split_profiles = self._split_profile(profiles, self.val_ratio)

        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if (
                        _file_name not in self._file_names
                    ):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(
                        self.data_dir, profile, file_name
                    )  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")

                    if id in self.error_labels:  # 라벨링 오류 데이터 gender 수정
                        gender = 'female' if gender == 'male' else 'male'

                    if not self.use_caution and id in self.caution_labels:  # 주의 데이터 사용 여부
                        continue

                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    def make_sampler(self, phase) :
        total_label = [self.encode_multi_class(self.mask_labels[idx], self.gender_labels[idx], self.age_labels[idx])for idx in self.indices[phase]]
        class_sample_count = np.array([len(np.where(total_label == t)[0]) for t in np.unique(total_label)])	
        weight = 1. / class_sample_count								  
        samples_weight = torch.DoubleTensor(torch.from_numpy(np.array([weight[t] for t in total_label])))
        phase_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return phase_sampler

    def split_dataset(self) -> List[Subset]:
        """프로필 기준으로 나눈 데이터셋을 Subset 리스트로 반환하는 메서드"""
        return [Subset(self, indices) for phase, indices in self.indices.items()]
