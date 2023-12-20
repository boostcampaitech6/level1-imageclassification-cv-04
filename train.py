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
from data_loader.cutmix import CutMixCollator
import model.loss as module_loss
import model.model as module_arch
from trainer import Trainer
from utils import prepare_device
from torch.optim.lr_scheduler import StepLR
import wandb
from sklearn.model_selection import StratifiedKFold

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers, config):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
    train_sampler = dataset.make_sampler('train')
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False,                          # use weighted sampler
        pin_memory=torch.cuda.is_available(),
        sampler=train_sampler,                  # use weighted sampler
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.valid_batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader


def getDataloader_cutmix(dataset, train_idx, valid_idx, batch_size, num_workers, config):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
    train_sampler = dataset.make_sampler('train')
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False,                              # use weighted sampler
        pin_memory=torch.cuda.is_available(),
        collate_fn=CutMixCollator(1.0, config.cutmix),
        sampler=train_sampler,                      # use weighted sampler
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.valid_batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader


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

    labels = [
        dataset.encode_multi_class(mask, gender, age)
        for mask, gender, age in zip(
            dataset.mask_labels, dataset.gender_labels, dataset.age_labels
        )
    ]

    if config.kfold == 0:
        # wandb initialize
        wandb.init(project="level1-imageclassification-cv-04")
        wandb.run.save()
        wandb.config.update(config)

        # wandb 실행 이름 설정
        wandb.run.name = config.wandb

        # setup data_loader instances
        train_set, valid_set = dataset.split_dataset()
        train_sampler = dataset.make_sampler('train')

        if config.augmentation == "CutmixAugmentation":
            train_dataloader = DataLoader(
                dataset=train_set,
                batch_size=config.batch_size,
                num_workers=0,
                shuffle=False,              # use weighted sampler
                pin_memory=use_cuda,
                drop_last=True,
                collate_fn=CutMixCollator(1.0, config.cutmix),
                sampler= train_sampler      # use weighted sampler
            )
            valid_dataloader = DataLoader(
                dataset=valid_set,
                batch_size=config.valid_batch_size,
                num_workers=0,
                shuffle=False,
                pin_memory=use_cuda,
                drop_last=True,
            )   # validation data는 mix 수행 안 함
        else:
            train_dataloader = DataLoader(
                dataset=train_set,
                batch_size=config.batch_size,
                num_workers=0,
                shuffle=False,              # use weighted sampler
                pin_memory=use_cuda,
                drop_last=True,
                sampler= train_sampler      # use weighted sampler
            )
            valid_dataloader = DataLoader(
                dataset=valid_set,
                batch_size=config.valid_batch_size,
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
        if config.model == "ArcfaceMultiHead":
            criterion = module_loss.create_criterion("focal")

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer_module = getattr(import_module("torch.optim"), config.optimizer)  # default: Adam
        optimizer = optimizer_module(
            trainable_params,
            lr=config.lr,
        )
        if config.scheduler == "StepLR":
            lr_scheduler = StepLR(optimizer, config.lr_decay_step, gamma=config.lr_decay_rate)
        elif config.scheduler == "ReduceLROnPlateau":
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.patience, min_lr=1e-6, verbose=True)

        trainer = Trainer(model, criterion, optimizer,
                        config=config,
                        device=device,
                        train_dataloader=train_dataloader,
                        valid_dataloader=valid_dataloader,
                        dataset_mean = dataset_mean,
                        dataset_std = dataset_std,
                        lr_scheduler=lr_scheduler)

        trainer.train()

    elif config.kfold == 1:
        # 5-fold Stratified KFold 5개의 fold를 형성하고 5번 Cross Validation을 진행합니다.
        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits)

        for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
            print(f"Fold:{i}, Train set: {len(train_idx)}, Valid set:{len(valid_idx)}")
            wandb.init(project="level1-imageclassification-cv-04", config=config, reinit=True)
            wandb.run.name = f'{config.wandb}_fold{i}'

            # -- data_loader
            train_set, val_set = dataset.split_dataset()
            batch_size = config.batch_size
            num_workers = 0

            if config.augmentation == "CutmixAugmentation":
                train_dataloader, valid_dataloader = getDataloader_cutmix(
                    dataset, train_idx, valid_idx, batch_size, num_workers, config
                )
            else:
                train_dataloader, valid_dataloader = getDataloader(
                    dataset, train_idx, valid_idx, batch_size, num_workers, config
                )

            # build model architecture, then print to console
            model_module = getattr(module_arch, config.model)
            model = model_module(num_classes=num_classes).to(device)
            model = torch.nn.DataParallel(model)

            # get function handles of loss and metrics
            criterion = module_loss.create_criterion(config.criterion)
            if config.model == "ArcfaceMultiHead":
                criterion = module_loss.create_criterion("focal")

            # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer_module = getattr(import_module("torch.optim"), config.optimizer)  # default: Adam
            optimizer = optimizer_module(
                trainable_params,
                lr=config.lr,
            )
            if config.scheduler == "StepLR":
                lr_scheduler = StepLR(optimizer, config.lr_decay_step, gamma=config.lr_decay_rate)
            elif config.scheduler == "ReduceLROnPlateau":
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.patience, min_lr=1e-6, verbose=True)

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
        "--kfold", type=int, default=0, help="use stratified KFold validation or not (default: 0 (not))"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="number of epochs to train (default: 64)"
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
        "--cutmix", 
        type=int,
        default=0,
        help="which cutmix implementation to use. 0: naive, 1: class box (default: 0)"
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
        # default=[128, 96],
        default=[224, 224],
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
        default="f1",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="StepLR",
        help="lr scheduler type (default: StepLR)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--lr_decay_rate",
        type=float,
        default=0.5,
        help="learning rate scheduler deacy rate = gamma (default: 0.5)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="ReduceOnLRPlateau feature. Number of epochs with no improvement after which learning rate will be reduced (default: 5)",
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
        "--save_val_table", 
        type=int,
        default=0,
        help="wandb에서 validation 추론 결과를 val_table로 저장할지.0인 경우 x, 1인 경우 validation set 전체 prediction case 저장, 2인 경우 틀린 prediction case만 저장"
    )
    parser.add_argument(
        "--multi_head", 
        type=int,
        default=1,
        help="모델의 head가 1개(num_classes=18)인 경우 False, 3개인 경우 True"
    )
    parser.add_argument(
        "--use_caution_data",
        type=int,
        default=1,
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
