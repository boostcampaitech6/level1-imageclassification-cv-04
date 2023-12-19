import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
import wandb

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(
        top=0.8
    )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join(
            [
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task in zip(
                    gt_decoded_labels, pred_decoded_labels, tasks
                )
            ]
        )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers):
    # 인자로 전달받은 dataset에서 train_idx에 해당하는 Subset 추출
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    # 인자로 전달받은 dataset에서 valid_idx에 해당하는 Subset 추출
    val_set   = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
    # 추출된 Train Subset으로 DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True
    )
    # 추출된 Valid Subset으로 DataLoader 생성
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=False
    )
    
    # 생성한 DataLoader 반환
    return train_loader, val_loader


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # -- dataset
    dataset_module = getattr(
        import_module("dataset"), args.dataset
    )  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        multi_head=args.multi_head,
        use_caution=args.use_caution_data
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(
        import_module("dataset"), args.augmentation
    )  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
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

    # 5-fold Stratified KFold 5개의 fold를 형성하고 5번 Cross Validation을 진행합니다.
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)

    for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
        print(f"Fold:{i}, Train set: {len(train_idx)}, Valid set:{len(valid_idx)}")
        wandb.init(project="level1-imageclassification-cv-04", entity="level1-cv-04",config=args, reinit=True)
        wandb.run.name = f'feat_f1_K-Fold_DeepHeadMultiTinyViT_fold{i}'

        # -- data_loader
        train_set, val_set = dataset.split_dataset()
        batch_size = args.batch_size
        num_workers = 0
        train_loader, val_loader = getDataloader(
            dataset, train_idx, valid_idx, batch_size, num_workers
        )
        fold_dir = f"{save_dir}/fold{i}"
        # -- model
        model_module = getattr(import_module("model"), args.model)#model  # default: BaseModel
        model = model_module(
            num_classes_mask=args.num_classes_mask,
            num_classes_gender=args.num_classes_gender,
            num_classes_age=args.num_classes_age
        ).to(device)

        model = torch.nn.DataParallel(model)
        # -- loss & metric
        if args.criterion_mask == 'focal':
            criterion = create_criterion(args.criterion)
            criterion_mask = create_criterion(args.criterion_mask)
            criterion_gender = create_criterion(args.criterion_gender)
            criterion_age = create_criterion(args.criterion_age)
        else: 
            criterion = create_criterion(args.criterion, classes=args.num_classes)  # default: cross_entropy
            criterion_mask = create_criterion(args.criterion_mask, classes=args.num_classes_mask)
            criterion_gender = create_criterion(args.criterion_gender, classes=args.num_classes_gender)
            criterion_age = create_criterion(args.criterion_age, classes=args.num_classes_age)

        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
        )
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)
        
        scaler = torch.cuda.amp.GradScaler()

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        best_val_acc = 0
        best_val_loss = np.inf
        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            mask_matches, gender_matches, age_matches = 0, 0, 0
            for idx, train_batch in enumerate(train_loader):
                optimizer.zero_grad()

                if args.multi_head:
                    inputs, labels, mask, gender, age = train_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    mask = mask.to(device)
                    gender = gender.to(device)
                    age = age.to(device)

                    with torch.cuda.amp.autocast():
                        outs = model(inputs)
                        pred_mask, pred_gender, pred_age = outs
                        loss_mask = criterion_mask(pred_mask, mask)
                        loss_gender = criterion_gender(pred_gender, gender)
                        loss_age = criterion_age(pred_age, age)
                        loss = loss_mask + loss_gender + loss_age
                    
                    # 헤드별 정확도 계산
                    mask_matches += (torch.argmax(pred_mask, dim=-1) == mask).sum().item()
                    gender_matches += (torch.argmax(pred_gender, dim=-1) == gender).sum().item()
                    age_matches += (torch.argmax(pred_age, dim=-1) == age).sum().item()

                    preds = torch.argmax(pred_mask, dim=-1) * 6 + torch.argmax(pred_gender, dim=-1) * 3 + torch.argmax(pred_age, dim=-1)

                else:
                    inputs, labels = train_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)

                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
    
                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    mask_acc = mask_matches / (args.batch_size * args.log_interval)
                    gender_acc = gender_matches / (args.batch_size * args.log_interval)
                    age_acc = age_matches / (args.batch_size * args.log_interval)
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    wandb.log({
                        "Train Loss": train_loss,
                        "Train acc mask": mask_acc,
                        "Train acc gender": gender_acc,
                        "Train acc age" : age_acc,
                        "Train acc" : train_acc,
                        "Learning Rate": current_lr
                    })
                    # print(
                    #     f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    #     f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    # )
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || "
                        f"mask accuracy {mask_acc:4.2%} || "
                        f"gender accuracy {gender_acc:4.2%} || "
                        f"age accuracy {age_acc:4.2%} || lr {current_lr}"
                        )
                    logger.add_scalar(
                        "Train/loss", train_loss, epoch * len(train_loader) + idx
                    )
                    logger.add_scalar(
                        "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                    )

                    loss_value = 0
                    mask_matches, gender_matches, age_matches = 0, 0, 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                mask_acc_items, gender_acc_items, age_acc_items = [], [], []
                figure = None
                for val_batch in val_loader:

                    if args.multi_head:
                        inputs, labels, mask, gender, age = val_batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        mask = mask.to(device)
                        gender = gender.to(device)
                        age = age.to(device)
          
                        outs = model(inputs)
                        pred_mask, pred_gender, pred_age = outs
                        loss_mask = criterion_mask(pred_mask, mask)
                        loss_gender = criterion_gender(pred_gender, gender)
                        loss_age = criterion_age(pred_age, age)
                        loss_item = (loss_mask + loss_gender + loss_age).item()
                        # 각 헤드별 정확도 계산
                        mask_acc_items.append((torch.argmax(pred_mask, dim=-1) == mask).sum().item())
                        gender_acc_items.append((torch.argmax(pred_gender, dim=-1) == gender).sum().item())
                        age_acc_items.append((torch.argmax(pred_age, dim=-1) == age).sum().item())
                        preds = torch.argmax(pred_mask, dim=-1) * 6 + torch.argmax(pred_gender, dim=-1) * 3 + torch.argmax(pred_age, dim=-1)

                    else:
                        inputs, labels = val_batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        outs = model(inputs)
                        preds = torch.argmax(outs, dim=-1)

                        loss_item = criterion(outs, labels).item()

                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = (
                            torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        )
                        inputs_np = dataset_module.denormalize_image(
                            inputs_np, dataset.mean, dataset.std
                        )
                        figure = grid_image(
                            inputs_np,
                            labels,
                            preds,
                            n=16,
                            shuffle=args.dataset != "MaskSplitByProfileDataset",
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                best_val_loss = min(best_val_loss, val_loss)
                # 평균 손실 및 정확도 계산
                mask_acc = np.sum(mask_acc_items) / len(val_set)
                gender_acc = np.sum(gender_acc_items) / len(val_set)
                age_acc = np.sum(age_acc_items) / len(val_set)
                if val_acc > best_val_acc:
                    print(
                        f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                    )
                    torch.save(model.module.state_dict(), f"{fold_dir}/best.pth")
                    best_val_acc = val_acc
                torch.save(model.module.state_dict(), f"{fold_dir}/last.pth")
                wandb.log({
                        "Valid Loss": val_loss,
                        "Valid acc_mask": mask_acc,
                        "Valid acc_gender": gender_acc,
                        "Valid acc_age" : age_acc,
                        "Valid acc" : val_acc,
                        "Learning Rate": current_lr
                    })
                print(
                    f"[Val] mask acc : {mask_acc:4.2%}, gender acc : {gender_acc:4.2%}, "
                    f"age acc : {age_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()
        # fold 종료
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 50)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskSplitByProfileDataset",
        help="dataset augmentation type (default: MaskBaseDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="BaseAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[224, 224],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=18
        )
    parser.add_argument(
        "--num_classes_mask",
        type=int,
        default=3
        )
    parser.add_argument(
        "--num_classes_gender",
        type=int,
        default=2
        )
    parser.add_argument(
        "--num_classes_age",
        type=int,
        default=3
        )
    parser.add_argument(
        "--model", type=str, default="Multi3HeadTinyViT", help="model type (default: EfficientNetB0MultiHead)"
    ) #"EfficientNetB0MultiHead"
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
        default="f1", #"cross_entropy"
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--criterion_mask",
        type=str,
        default="f1", #"cross_entropy"
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--criterion_age",
        type=str,
        default="f1", #"cross_entropy"
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--criterion_gender",
        type=str,
        default="f1", #"cross_entropy"
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=5,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="K-Fold", help="model save at {SM_MODEL_DIR}/{name}"
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
        default=os.environ.get("SM_CHANNEL_TRAIN","/data/ephemeral/home/maskdata/train/images")
        # default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train/images"),
    )
    parser.add_argument(
        '--wandb', 
        type=str, 
        help='Weights & Biases 설정'
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR","/data/ephemeral/home/level1-imageclassification-cv-04/v3")
    )#"/data/ephemeral/basecode/v3/model" 
    args = parser.parse_args()
    # wandb.config.update(args)
    print(args)
    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
