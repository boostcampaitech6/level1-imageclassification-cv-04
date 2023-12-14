import argparse
import collections
import torch
import numpy as np
import os
import random
from importlib import import_module
from torch.utils.data import DataLoader
import data_loader.data_sets as module_data_set
import data_loader.augmentations as module_augmentation
import data_loader.data_loaders as module_data_loader
import model.loss as module_loss
import model.model as module_arch
from trainer import Trainer
from utils import prepare_device
from torch.optim.lr_scheduler import StepLR



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def main(data_dir, model_dir, config):
    seed_everything(config.seed)

    # settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # setup data_set instance
    dataset_module = getattr(module_data_set, config.dataset)  # default: MaskSplitByProfileDataset
    dataset = dataset_module(
        data_dir=data_dir,
        multi_head=config.multi_head,
        use_caution=config.use_caution_data
    )
    num_classes = dataset.num_classes
    dataset_mean = dataset.mean
    dataset_std = dataset.std

    # setup augmentation instance
    augmentation_module = getattr(module_augmentation, config.augmentation)  # default: MaskSplitByProfileDataset
    transform = augmentation_module(
        resize=config.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # setup data_loader instances
    train_set, valid_set = dataset.split_dataset()
    # train_loader_module = getattr(module_data_loader, config.dataloader)
    # train_data_loader = train_loader_module(dataset=train_set,
    #                                         batch_size=config.batch_size,
    #                                         num_workers=0,
    #                                         shuffle=True,
    #                                         pin_memory=use_cuda,
    #                                         drop_last=True)
    # valid_loader_module = getattr(module_data_loader, config.dataloader)
    # valid_data_loader = valid_loader_module(dataset=valid_set,
    #                                         batch_size=config.valid_batch_size,
    #                                         num_workers=0,
    #                                         shuffle=False,
    #                                         pin_memory=use_cuda,
    #                                         drop_last=True)
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        dataset=valid_set,
        batch_size=args.valid_batch_size,
        num_workers=0,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # build model architecture, then print to console
    model_module = getattr(module_arch, config.model)
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # get function handles of loss and metrics
    criterion = module_loss.create_criterion(config.criterion)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_module = getattr(import_module("torch.optim"), config.optimizer)  # default: Adam
    optimizer = optimizer_module(
        trainable_params,
        lr=config.lr,
    )
    lr_scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    trainer = Trainer(model, criterion, optimizer,
                      config=config,
                      device=device,
                      train_dataloader=train_dataloader,
                      valid_dataloader=valid_dataloader,
                      dataset_mean = dataset_mean,
                      dataset_std = dataset_std,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=64, help="number of epochs to train (default: 64)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskSplitByProfileDataset",
        help="dataset augmentation type (default: MaskSplitByProfileDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="BaseAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--dataloader",
        type=str,
        default="MaskDataLoader",
        help="dataloader type (default: MaskDataLoader)"
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[128, 96],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--model", type=str, default="EfficientNetB0MultiHead", help="model type (default: EfficientNetB0MultiHead)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer type (default: Adam)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio for validaton (default: 0.2)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )
    parser.add_argument(
        "--wandb", default="model_EfficientNetB0", 
        help="wandb run name. 실험 대상이 되는 \"arg종류_arg값\" 형태로 적어주세요 (예: model_EfficientNetB4)."
    )
    parser.add_argument(
        "--multi_head", 
        type=bool,
        default=True,
        help="모델의 head가 1개(num_classes=18)인 경우 False, 3개인 경우 True"
    )
    parser.add_argument(
        "--use_caution_data",
        type=bool,
        default=True,
        help="성별 판별이 어려운 데이터(EDA-오류처럼 보이는 데이터) 사용 여부"
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/data/ephemeral/maskdata/train/images")
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/data/ephemeral/home/model")
    )

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    main(data_dir, model_dir, args)
