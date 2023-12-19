import torch
from PIL import Image
from torchvision.transforms import (
    Resize,
    ToTensor,
    Normalize,
    Compose,
    CenterCrop,
    ColorJitter,
    RandomHorizontalFlip,
    AugMix
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


# class AugMix(BaseAugmentation):
#     def __init__(self, resize, mean, std, **args):
#         super().__init__(resize, mean, std, **args)
#         augmenter = v2.AugMix()
#         imgs = [augmenter(orig_img) for _ in range(4)]
#         plot([orig_img] + imgs)

#         self.transform = Compose(
#             [
#                 Resize(resize, Image.BILINEAR),
#                 ToTensor(),
#                 Normalize(mean=mean, std=std),
#                 RandomHorizontalFlip(0.5),
#             ]
#         )

#     def __call__(self, image):
#         return self.transform(image)
        


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