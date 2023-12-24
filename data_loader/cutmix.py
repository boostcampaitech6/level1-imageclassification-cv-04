import numpy as np
import torch
import torch.nn as nn


def cutmix(batch, alpha):
    """
    naive (original) cutmix
    """
    if len(batch) == 5:
        data, targets, mask, gender, age = batch

        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]           # mix 후보 shuffled_data
        shuffled_mask = mask[indices]           # mix 후보 shuffled_mask_label
        shuffled_gender = gender[indices]       # mix 후보 shuffled_gender_label
        shuffled_age = age[indices]             # mix 후보 shuffled_age_label

        lam = np.random.beta(alpha, alpha)

        image_h, image_w = data.shape[2:]
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))

        # mix data < shuffled data
        data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((x1 - x0) * (y1 - y0) / (image_h * image_w))

        masks = (mask, shuffled_mask, lam)
        genders = (gender, shuffled_gender, lam)
        ages = (age, shuffled_age, lam)

        return data, targets, masks, genders, ages
    elif len(batch) == 2:
        data, targets = batch

        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]           # mix 후보 shuffled_data
        shuffled_targets = targets[indices]     # mix 후보 shuffled_label

        lam = np.random.beta(alpha, alpha)

        image_h, image_w = data.shape[2:]
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))

        # mix data < shuffled data
        data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((x1 - x0) * (y1 - y0) / (image_h * image_w))

        targets = (targets, shuffled_targets, lam)

        return data, targets
    

def cutmix_half_hor(batch, alpha):
    if len(batch) == 5:
        data, targets, mask, gender, age = batch

        # caution: cutmix with respect to age class
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]           # mix 후보 shuffled_data
        shuffled_mask = mask[indices]           # mix 후보 shuffled_mask_label
        shuffled_gender = gender[indices]       # mix 후보 shuffled_gender_label
        shuffled_age = age[indices]             # mix 후보 shuffled_age_label

        lam = 0.75 # alpha 활요한 beta 분포에서 안뽑고 고정값
        # 패치의 h, w는, 주어진 이미지의 h, w에 np.sqrt(1-lam)을 곱하여 얻게 됩니다.
        # 둘다 반반이면 0.5 = np.sqrt(1-lam) => lam = 0.75

        image_h, image_w = data.shape[2:]
        x0 = int(np.round(image_w / 2))
        x1 = int(np.round(image_w))
        y0 = int(np.round(image_h / 4))
        y1 = int(np.round(image_h / 4 * 3))

        # mix data < shuffled data
        data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((x1 - x0) * (y1 - y0) / (image_h * image_w))

        masks = (mask, shuffled_mask, lam)
        genders = (gender, shuffled_gender, lam)
        ages = (age, shuffled_age, lam)

        return data, targets, masks, genders, ages
    elif len(batch) == 2:
        data, targets = batch

        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]           # mix 후보 shuffled_data
        shuffled_targets = targets[indices]     # mix 후보 shuffled_label

        lam = 0.75 # alpha 활요한 beta 분포에서 안뽑고 고정값
        # 패치의 h, w는, 주어진 이미지의 h, w에 np.sqrt(1-lam)을 곱하여 얻게 됩니다.
        # 둘다 반반이면 0.5 = np.sqrt(1-lam) => lam = 0.75

        image_h, image_w = data.shape[2:]
        x0 = int(np.round(image_w / 2))
        x1 = int(np.round(image_w))
        y0 = int(np.round(image_h / 4))
        y1 = int(np.round(image_h / 4 * 3))

        # mix data < shuffled data
        data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((x1 - x0) * (y1 - y0) / (image_h * image_w))

        targets = (targets, shuffled_targets, lam)

        return data, targets
    
def cutmix_half_ori(batch, alpha):
    if len(batch) == 5:
        data, targets, mask, gender, age = batch

        # caution: cutmix with respect to age class
        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]           # mix 후보 shuffled_data
        shuffled_mask = mask[indices]           # mix 후보 shuffled_mask_label
        shuffled_gender = gender[indices]       # mix 후보 shuffled_gender_label
        shuffled_age = age[indices]             # mix 후보 shuffled_age_label

        lam = 0.5 # alpha 활요한 beta 분포에서 안뽑고 고정값

        image_h, image_w = data.shape[2:]
        x0 = int(np.round(image_w / 2))
        x1 = int(np.round(image_w))
        y0 = int(np.round(0))
        y1 = int(np.round(image_h))

        # mix data < shuffled data
        data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((x1 - x0) * (y1 - y0) / (image_h * image_w))

        masks = (mask, shuffled_mask, lam)
        genders = (gender, shuffled_gender, lam)
        ages = (age, shuffled_age, lam)

        return data, targets, masks, genders, ages
    elif len(batch) == 2:
        data, targets = batch

        indices = torch.randperm(data.size(0))
        shuffled_data = data[indices]           # mix 후보 shuffled_data
        shuffled_targets = targets[indices]     # mix 후보 shuffled_label

        lam = 0.5 # alpha 활요한 beta 분포에서 안뽑고 고정값

        image_h, image_w = data.shape[2:]
        x0 = int(np.round(image_w / 2))
        x1 = int(np.round(image_w))
        y0 = int(np.round(0))
        y1 = int(np.round(image_h))

        # mix data < shuffled data
        data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((x1 - x0) * (y1 - y0) / (image_h * image_w))

        targets = (targets, shuffled_targets, lam)

        return data, targets


class CutMixCollator:
    def __init__(self, alpha, mode):
        self.alpha = alpha
        self.mode = mode

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        if self.mode == 0:
            batch = cutmix(batch, self.alpha)
        elif self.mode == 1:
            # batch = cutmix_half_hor(batch, self.alpha) # lambda = 0.75
            batch = cutmix_half_ori(batch, self.alpha) # lambda = 0.5 구현 이상함
        # elif self.mode == 2: # TODO
        return batch


class CutMixCriterion:
    # def __init__(self, reduction):
        # self.criterion = nn.CrossEntropyLoss(reduction=reduction)
    def __init__(self, criterion):
        self.criterion = criterion

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(
            preds, targets1) + (1 - lam) * self.criterion(preds, targets2)