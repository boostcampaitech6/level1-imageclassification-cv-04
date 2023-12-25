import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel
import timm
import torchvision
from model.arcface_resnet import *
from model.arcface_metrics import *
import clip # https://github.com/openai/CLIP
import copy 


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


class EfficientNetB0SingleHead(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)  # num_features : 1280
        for param in self.model.parameters():
            param.requires_grad = False

        self.cls = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 18)
        )


    def forward(self, x):
        x = self.model(x)
        pred = self.cls(x)
        return pred


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


class EfficientNetB4SingleHead(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)  # num_features : 1792
        for param in self.model.parameters():
            param.requires_grad = False

        self.cls = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 18)
        )


    def forward(self, x):
        x = self.model(x)
        pred = self.cls(x)
        return pred
    

class EfficientNetB4MultiHead(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=0)  # num_features : 1792
        for param in self.model.parameters():
            param.requires_grad = False

        self.mask = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.age = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        self.gender = nn.Sequential(
            nn.Linear(1792, 512),
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


class ViTL14SingleHead(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k', pretrained=True) # num_features : 1000
        for param in self.model.parameters():
            param.requires_grad = False

        self.cls = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 18)
        )


    def forward(self, x):
        x = self.model(x)
        pred = self.cls(x)
        return pred


class ViTL14MultiHead(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model('timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k', pretrained=True) # num_features : 1000
        for param in self.model.parameters():
            param.requires_grad = False

        self.mask = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.age = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        self.gender = nn.Sequential(
            nn.Linear(1000, 512),
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


class SwinTransformerBase224V1(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0)  # num_features = 1024
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.mask = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        self.gender = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.age = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
    def forward(self, x):
        x = self.model(x)
        mask = self.mask(x)
        gender = self.gender(x)
        age = self.age(x)
        return mask, gender, age


class CLIP3Head3Proj(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
            
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)

        self.mask_i = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.gender_i = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.age_i = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.mask_t = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.gender_t = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        self.age_t = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 32)
        )
        
        mask_captions = [
            # 'A person correctly wearing a mask, covering mouth and nose completely.',
            # 'A photo of a person wearing a mask incorrectly, with the mouth and nose not properly covered or the eyes covered by the mask, and possibly wearing a scarf.',
            # 'A photo of a person without mask.',
            'A person correctly wearing a mask, covering mouth and nose completely, and possibly wearing a scarf.',
            'A photo of a person wearing a mask incorrectly, with the mouth and nose not properly covered or the eyes covered by the mask.',
            'A photo of a person without mask.',  # best!
        ]
        gender_captions = [
            'a photo of a man.',
            'a photo of an woman.',
        ]
        age_captions = [
            'the young person looks like under 30 years old.',
            'a photo of a middle-aged person.',
            'A photo of an elderly person over 60 years old.',
        ]
        
        mask_captions = clip.tokenize([text for text in mask_captions]).to(self.device)
        gender_captions = clip.tokenize([text for text in gender_captions]).to(self.device)
        age_captions = clip.tokenize([text for text in age_captions]).to(self.device)
        
        self.text_mask_features = self.model.encode_text(mask_captions).type(torch.float32)
        self.text_gender_features = self.model.encode_text(gender_captions).type(torch.float32)
        self.text_age_features = self.model.encode_text(age_captions).type(torch.float32)

    def forward(self, x):
        image_features = self.model.encode_image(x).type(torch.float32)
        
        image_mask_features = self.mask_i(image_features)
        image_mask_features = image_mask_features / image_mask_features.norm(dim=-1, keepdim=True)
        
        image_gender_features = self.gender_i(image_features)
        image_gender_features = image_gender_features / image_gender_features.norm(dim=-1, keepdim=True)
        
        image_age_features = self.age_i(image_features)
        image_age_features = image_age_features / image_age_features.norm(dim=-1, keepdim=True)

        text_mask_features = self.mask_t(self.text_mask_features)
        text_mask_features = text_mask_features / text_mask_features.norm(dim=-1, keepdim=True)

        text_gender_features = self.gender_t(self.text_gender_features)
        text_gender_features = text_gender_features / text_gender_features.norm(dim=-1, keepdim=True)

        text_age_features = self.age_t(self.text_age_features)
        text_age_features = text_age_features / text_age_features.norm(dim=-1, keepdim=True)

        mask_logits = (100.0 * image_mask_features @ text_mask_features.T)
        gender_logits = (100.0 * image_gender_features @ text_gender_features.T)
        age_logits = (100.0 * image_age_features @ text_age_features.T)
        
        return mask_logits, gender_logits, age_logits


class ArcfaceMultiHead(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        # # pretrained resnetface18 from repo
        # self.model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=False)
        # checkpoint = torch.load('./model/pretrained/arcface_resnet18_110.pth', map_location='cuda:0')
        # self.model = torch.nn.DataParallel(self.model, device_ids=[0])
        # self.model.load_state_dict(checkpoint)
        # #
        # self.model = timm.create_model('timm/convnext_small.in12k', pretrained=True) # num_features : 11821
        self.model = timm.create_model('timm/vit_large_patch14_clip_224.openai_ft_in12k_in1k', pretrained=True) # num_features : 1000
        for param in self.model.parameters():
            param.requires_grad = False
        #
        self.mask = ArcMarginProduct(1000, 3, s=30, m=0.5, easy_margin=True)

        self.age = ArcMarginProduct(1000, 3, s=30, m=0.5, easy_margin=True)
        
        self.gender = ArcMarginProduct(1000, 2, s=30, m=0.5, easy_margin=True)

    def forward(self, x, mask, gender, age):
        # x = self.model.forward_features(x)
        # x = self.model.forward_head(x)
        x = self.model(x)
        mask = self.mask(x, mask)
        gender = self.gender(x, gender)
        age = self.age(x, age)
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
