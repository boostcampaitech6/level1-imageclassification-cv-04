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

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset_2Head
from loss import create_criterion

# wandb.init(project="level1-imageclassification-cv-04", entity="level1-cv-04")
# wandb.run.name = 'feat_f1_2HeadMultiTinyViT_Final'


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


def grid_image(np_images, combined_labels, combined_preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))
    plt.subplots_adjust(top=0.8)
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]

    for idx, choice in enumerate(choices):
        image = np_images[choice]
        gt_mask_gender, gt_age = MaskBaseDataset_2Head.decode_multi_class(
            combined_labels[choice] // 3, combined_labels[choice] % 3)
        pred_mask_gender, pred_age = MaskBaseDataset_2Head.decode_multi_class(
            combined_preds[choice] // 3, combined_preds[choice] % 3)

        gt_mask_label, gt_gender_label = gt_mask_gender
        pred_mask_label, pred_gender_label = pred_mask_gender

        title = f"Mask - gt: {gt_mask_label}, pred: {pred_mask_label}\n" \
                f"Gender - gt: {gt_gender_label}, pred: {pred_gender_label}\n" \
                f"Age - gt: {gt_age}, pred: {pred_age}"

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



    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=0,
        # num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=0,
        # num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    model_module = getattr(import_module("tiny_vit"), args.model)#model  # default: BaseModel
    model = model_module(
        num_classes_mask_gender=args.num_classes_mask_gender,
        num_classes_age=args.num_classes_age
    ).to(device)
    model = torch.nn.DataParallel(model)
    # -- loss & metric
    if args.criterion_mask_gender == 'focal':
        criterion_mask_gender = create_criterion(args.criterion_mask_gender)
        criterion_age = create_criterion(args.criterion_age)
    else:
        criterion_mask_gender = create_criterion(args.criterion_mask_gender, classes=args.num_classes_mask_gender)
        criterion_age = create_criterion(args.criterion_age, classes=args.num_classes_age)

    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: Adam
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

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
        matches = [0, 0]
        for idx, train_batch in enumerate(train_loader):
            optimizer.zero_grad()
            if args.multi_head:
                inputs, (labels_mask_gender, labels_age), mask, gender, age = train_batch
                inputs = inputs.to(device)
                labels_mask_gender = labels_mask_gender.to(device)
                labels_age = labels_age.to(device)
                mask = mask.to(device)
                gender = gender.to(device)
                age = age.to(device)

                outs = model(inputs)
                pred_mask_gender, pred_age = outs

                loss_mask_gender = criterion_mask_gender(pred_mask_gender, labels_mask_gender)
                loss_age = criterion_age(pred_age, labels_age)

                # Combine the losses
                loss = loss_mask_gender + loss_age
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches[0] += (torch.argmax(pred_mask_gender, dim=-1) == labels_mask_gender).sum().item()
            matches[1] += (torch.argmax(pred_age, dim=-1) == labels_age).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc_mask_gender = matches[0] / (args.batch_size * args.log_interval)
                train_acc_age = matches[1] / (args.batch_size * args.log_interval)
                current_lr = get_lr(optimizer)
                # wandb.log({
                #     "Train Loss": train_loss,
                #     "Train Accuracy Mask Gender": train_acc_mask_gender,
                #     "Train Accuracy Age": train_acc_age,
                #     "Learning Rate": current_lr
                # })
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || "
                    f"training accuracy mask gender {train_acc_mask_gender:4.2%} || "
                    f"training accuracy age {train_acc_age:4.2%} || "
                    f"lr {current_lr}"
                )
                loss_value = 0
                matches = [0, 0]  

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            val_acc_mask_gender_items = []
            val_acc_age_items = []
            figure = None
            for val_batch in val_loader:

                if args.multi_head:
                    inputs, (labels_mask_gender, labels_age), mask, gender, age = val_batch
                    inputs = inputs.to(device)
                    labels_mask_gender = labels_mask_gender.to(device)
                    labels_age = labels_age.to(device)
                    mask = mask.to(device)
                    gender = gender.to(device)
                    age = age.to(device)

                    combined_labels = labels_mask_gender * 3 + labels_age
                    
                    outs = model(inputs)
                    pred_mask_gender, pred_age = outs

                    loss_mask_gender = criterion_mask_gender(pred_mask_gender, labels_mask_gender)
                    loss_age = criterion_age(pred_age, labels_age)
                    loss_item = (loss_mask_gender + loss_age).item()

                    preds_mask_gender = torch.argmax(pred_mask_gender, dim=-1)
                    preds_age = torch.argmax(pred_age, dim=-1)
                    combined_preds = preds_mask_gender * 3 + preds_age

                acc_mask_gender = (preds_mask_gender == labels_mask_gender).sum().item()
                acc_age = (preds_age == labels_age).sum().item()

                val_loss_items.append(loss_item)
                val_acc_mask_gender_items.append(acc_mask_gender)
                val_acc_age_items.append(acc_age)
                val_acc_items.append(acc_mask_gender)
                val_acc_items.append(acc_age)
                    
                if figure is None:
                    inputs_np = (
                        torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    )
                    inputs_np = dataset_module.denormalize_image(
                        inputs_np, dataset.mean, dataset.std
                    )
                    # grid_image 함수에 복합 라벨 및 예측 값 전달
                    figure = grid_image(
                        inputs_np,
                        combined_labels,
                        combined_preds,
                        n=16,
                        shuffle=args.dataset != "MaskSplitByProfileDataset",
                    )
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc_mask_gender = np.sum(val_acc_mask_gender_items) / len(val_set)
            val_acc_age = np.sum(val_acc_age_items) / len(val_set)
            val_acc = np.sum(val_acc_items) / (len(val_set) * 2) 
            best_val_loss = min(best_val_loss, val_loss)

            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            # wandb.log({
            #         "Valid Loss": train_loss,
            #         "Valid Accuracy Mask Gender": val_acc_mask_gender,
            #         "Valid Accuracy Age": val_acc_age,
            #         "Learning Rate": current_lr
            #     })
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            print(
                f"[Val] mask gender acc : {val_acc_mask_gender:4.2%}, age acc : {val_acc_age:4.2%}, "
                f"total acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="number of epochs to train (default: 50)"
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
        default=128,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--num_classes_mask_gender",
        type=int,
        default=6
        )
    parser.add_argument(
        "--num_classes_age",
        type=int,
        default=3
        )
    parser.add_argument(
        "--model", type=str, default="Multi2HeadTinyViT", help="model type (default: EfficientNetB0MultiHead)"
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
        "--criterion_mask_gender",
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
        "--lr_decay_step",
        type=int,
        default=10,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="aug", help="model save at {SM_MODEL_DIR}/{name}"
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
        default=os.environ.get("SM_CHANNEL_TRAIN","/data/ephemeral/home/metadata/train/images")
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
