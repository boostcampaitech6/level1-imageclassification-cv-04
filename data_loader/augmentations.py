import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torchvision.transforms import (
    Resize,
    ToTensor,
    Normalize,
    Compose,
    CenterCrop,
    ColorJitter,
    RandomHorizontalFlip,
    Grayscale,
)


class BaseAugmentation:
    """
    기본적인 Augmentation을 담당하는 클래스

    Attributes:
        transform (Compose): 이미지를 변환을 위한 torchvision.transforms.Compose 객체
    """

    def __init__(self, resize, mean, std, **args):
        """
        Args:
            resize (tuple): 이미지의 리사이즈 대상 크지
            mean (tuple): Normalize 변환을 위한 평균 값
            std (tuple): Normalize 변환을 위한 표준 값
        """
        self.transform = Compose(
            [
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
                RandomHorizontalFlip(0.5),
            ]
        )

    def __call__(self, image):
        """
        이미지에 저장된 transform 적용

        Args:
            Image (PIL.Image): Augumentation을 적용할 이미지

        Returns:
            Tensor: Argumentation이 적용된 이미지
        """
        return self.transform(image)

class AddGaussianNoise(object):
    """이미지에 Gaussian Noise를 추가하는 클래스"""

    def __init__(self, mean=0.0, std=1.0):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class ArcfaceResNetAugmentation:
    """ArcfaceResNet Augmentation을 담당하는 클래스"""

    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                Grayscale(),
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
                RandomHorizontalFlip(0.5),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class CutmixAugmentation:
    """Cutmix Augmentation을 담당하는 클래스. ChanwooAugmentation과 동일
    합니다. 
    실질적으로 cutmix가 수행되는 곳은 dataloader의 collate_fn으로 지정된 CutMixCollator입니다.
    (data_loader/cutmix.py 참조)"""

    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(resize, Image.BILINEAR),
                ColorJitter(0.1, 0.1, 0.1, 0.1),
                ToTensor(),
                Normalize(mean=mean, std=std),
                RandomHorizontalFlip(0.5),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class ChanwooAugmentation:
    """
    짱찬우님의 갓aug
    """

    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(resize, Image.BILINEAR),
                ColorJitter(0.1, 0.1, 0.1, 0.1),
                ToTensor(),
                Normalize(mean=mean, std=std),
                RandomHorizontalFlip(0.5),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
    

class ChanwooAugmentation2:
    """
    ChanwooAugmentation에서 color jitter hue값 줄이기
    """

    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(resize, Image.BILINEAR),
                ColorJitter(0.1, 0.1, 0.1, 0.01), # hue in [-0.01, 0.01]
                ToTensor(),
                Normalize(mean=mean, std=std),
                RandomHorizontalFlip(0.5),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
    

class SwinAugmentation:
    """
    ChanwooAugmentation에서 color jitter hue값 줄이기
    """

    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(resize, Image.BILINEAR),
                ColorJitter(0.1, 0.1, 0.1, 0.01), # hue in [-0.01, 0.01]
                ToTensor(),
                Normalize(mean=mean, std=std),
                RandomHorizontalFlip(0.5),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class CLIPAugmentation:
    """커스텀 Augmentation을 담당하는 클래스"""

    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(resize, Image.BILINEAR),
                ColorJitter(0.1, 0.1, 0.1, 0.1),
                ToTensor(),
                Normalize(mean=(0.5620, 0.5275, 0.5050), std=std),
                RandomHorizontalFlip(0.5),
                # AddGaussianNoise(),
            ]
        )

    def __call__(self, image):
        return self.transform(image)


class CustomAugmentation:
    """커스텀 Augmentation을 담당하는 클래스"""

    def __init__(self, resize, mean, std, **args):
        self.transform = Compose(
            [
                CenterCrop((320, 256)),
                Resize(resize, Image.BILINEAR),
                ColorJitter(0.1, 0.1, 0.1, 0.1),
                ToTensor(),
                Normalize(mean=mean, std=std),
                AddGaussianNoise(),
            ]
        )

    def __call__(self, image):
        return self.transform(image)
    


class CustomAlbumentation:
    """
    커스텀 Augmentation을 담당하는 클래스
    Albumentations를 사용
    ColorHitter : 색상, 대비, 밝기, 채도 등을 무작위로 변경
    모델이 다양한 조명 및 색상 조건에서도 일관된 성능을 유지
    RandomGamma : 이미지의 감마 값을 조정하여 밝기를 다양하게 변경. 밝기 변화가 큰 이미지에 유용할 수 있음
    RandomBrightnessContrast : 밝기와 대비를 동시에 조절하여 역광 상황에서의 세부 사항을 잘 드러나게 할 수 있음
    Gaussian Noise : 이미지에 Gaussian noise를 추가하여 모델이 노이즈에 강건해지며 과적합 방지에 도움
    """

    def __init__(self, resize, mean, std, **args):
        """
        Args:
            resize (tuple): 이미지의 리사이즈 대상 크기
            mean (tuple): Normalize 변환을 위한 평균 값
            std (tuple): Normalize 변환을 위한 표준 값
        """
        """
        Parameters:
            brightness_limit, contrast_limit = 밝기와 대비의 조정치 0.2 = 20%
            brightness_by_max = 이미지의 최대 밝기 값에 상대적 조정, False시 절대적인 밝기 값으로 조절
            always_apply = True일 경우 항상 적용 False일 경우 일부 확률(p)로 조정
            blur_limit = 블러를 적용할 때 블러 커널의 크기를 제한
            sigma_limit = 가우시안 블러의 표준 편차를 제한. 부동 소수점일 경우 해당값 이내에서 무작위로 선택됌. 클수록 블러효과 강해짐
            gamma_limit = 감마 보정의 강도를 제한. 해당 값들 사이에서 무작위로 감마 보정
        """
        self.transform = A.Compose([
            A.Resize(*resize),
            # A.RandomGamma(gamma_limit=(0.7, 1.3), eps=None, always_apply=False, p=0.5),
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5),
            A.Normalize(mean=mean, std=std),
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=0, always_apply=False, p=0.5),
            ToTensorV2()
        ])

    def __call__(self, image):
        """
        이미지에 저장된 transform 적용

        Args:
            image (PIL.Image): Augmentation을 적용할 이미지

        Returns:
            PIL.Image: Augmentation이 적용된 이미지
        """
        image = np.array(image) #Albumentations는 이미지에 대한 변환을 dict으로 반환
        return self.transform(image=image)["image"]
