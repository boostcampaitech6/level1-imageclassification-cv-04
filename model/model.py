import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel
import timm
from torch import load
from collections import OrderedDict


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class EfficientNetB0MultiHead(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)  # num_features : 1280
        for param in self.model.parameters():
            param.requires_grad = False

        self.mask = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.age = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        self.gender = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.model(x)
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        return mask, gender, age
    

class EfficientNetB4MultiHead(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)  # num_features : 1792
        for param in self.model.parameters():
            param.requires_grad = False

        self.mask = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 3)
        )

        self.age = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 3)
        )
        
        self.gender = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.model(x)
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        return mask, gender, age


class DaViTMultiHead(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('davit_base.msft_in1k', pretrained=True, num_classes=0)  # num_features : 1792
        for param in self.model.parameters():
            param.requires_grad = False

        self.mask = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 3)
        )

        self.age = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 3)
        )
        
        self.gender = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.model(x)
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        return mask, gender, age


class Beit2MultiHead(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('beitv2_large_patch16_224.in1k_ft_in22k', pretrained=True, num_classes=0)  # num_features : 1792
        for param in self.model.parameters():
            param.requires_grad = False

        self.mask = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 3)
        )

        self.age = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            # nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 3)
        )
        
        self.gender = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 2)
        )

        # self.age_with_mask = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.Dropout(0.5),
        #     nn.LeakyReLU(),
        #     nn.Linear(512, 128),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 3)
        # )

        # pretrain_model = load('/data/ephemeral/home/model/beit2_full/best.pth')
        # pretrain_head = load('/data/ephemeral/home/model/beit2_v2_freeze/best.pth')
        # model_state = OrderedDict([(k[6:], v) for k, v in pretrain_model.items() if 'model' in k])
        # mask_state = OrderedDict([(k[5:], v) for k, v in pretrain_head.items() if 'mask' in k])
        # age_state = OrderedDict([(k[4:], v) for k, v in pretrain_head.items() if 'age' in k])
        # gender_state = OrderedDict([(k[7:], v) for k, v in pretrain_head.items() if 'gender' in k])
        
        # self.model.load_state_dict(model_state)
        # self.mask.load_state_dict(mask_state)
        # self.age.load_state_dict(age_state)
        # self.gender.load_state_dict(gender_state)

        # for param in self.mask.parameters():
        #     param.requires_grad = False
        # for param in self.age.parameters():
        #     param.requires_grad = False
        # for param in self.gender.parameters():
        #     param.requires_grad = False


    def forward(self, x):
        x = self.model(x)
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        # age_with_mask = self.age_with_mask(x)
        # return mask, gender, age, age_with_mask
        return mask, gender, age


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x
