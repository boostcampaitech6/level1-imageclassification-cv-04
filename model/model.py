import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel
import timm
import clip # https://github.com/openai/CLIP


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

class CLIP1Head(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
            
        for name, param in self.model.named_parameters():
            if "visual.proj" in name or "text_projection" in name:
                param.requires_grad_(True)
            else: 
                param.requires_grad_(False)
        
        captions = [
            "A photo of a young man wearing a mask that covers his nose and mouth.",
            "A photo of a middle-aged man wearing a mask that covers his nose and mouth.",
            "A photo of a elderly man wearing a mask that covers his nose and mouth.",
            "A photo of a young woman wearing a mask that covers her nose and mouth.",
            "A photo of a middle-aged woman wearing a mask that covers her nose and mouth.",
            "A photo of a elderly woman wearing a mask that covers her nose and mouth.",
            "A photo of a young man wearing a mask improperly.",
            "A photo of a middle-aged man wearing a mask improperly.",
            "A photo of a elderly man wearing a mask improperly.",
            "A photo of a young woman wearing a mask improperly.",
            "A photo of a middle-aged woman wearing a mask improperly.",
            "A photo of a elderly woman wearing a mask improperly.",
            "A photo of a young man not wearing a mask.",
            "A photo of a middle-aged man not wearing a mask.",
            "A photo of an elderly man not wearing a mask.",
            "A photo of a young woman not wearing a mask.",
            "A photo of a middle-aged woman not wearing a mask.",
            "A photo of an elderly woman not wearing a mask.",
        ]
        self.captions = clip.tokenize([text for text in captions]).to(self.device)

    def forward(self, x):
        image_features = self.model.encode_image(x) # NOTE: need to resize (224, 224)
        text_features = self.model.encode_text(self.captions)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)   
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
        return similarity