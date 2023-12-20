import numpy as np
import torch
import os
import random
import glob
import re
import json
import wandb
from pathlib import Path
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from model.arcface_metrics import *
from data_loader.cutmix import CutMixCriterion
import model.loss as module_loss


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, config, 
                 device=None, train_dataloader=None, valid_dataloader=None, 
                 dataset_mean=None, dataset_std=None, lr_scheduler=None):
        super().__init__(model, criterion, optimizer, config)
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.do_validation = self.valid_dataloader is not None
        self.lr_scheduler = lr_scheduler
        self.best_val_acc = 0
        self.best_val_loss = np.inf
        self.valid_loss = module_loss.F1Loss()

        # self.save_dir = self.increment_path(os.path.join(self.config.model_dir, self.config.name))
        self.save_dir = self.increment_path(os.path.join(self.config.model_dir, wandb.run.name))
        
        # logging with tensorboard
        self.logger = SummaryWriter(log_dir=self.save_dir)
        with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(config), f, ensure_ascii=False, indent=4)

        self.cutmix_criterion = CutMixCriterion(self.criterion)
        self.scaler = torch.cuda.amp.GradScaler()

    def increment_path(self, path, exist_ok=False):
        path = Path(path)
        if (path.exists() and exist_ok) or (not path.exists()):
            return str(path)
        else:
            dirs = glob.glob(f"{path}*")
            matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]
            n = max(i) + 1 if i else 2
            return f"{path}{n}"

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def _train_epoch(self, epoch):
        self.model.train()
        loss_value = 0
        matches = 0
        
        acc_mask_items = []  
        acc_gender_items = []  
        acc_age_items = []  

        for idx, train_batch in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            if self.config.multi_head:
                inputs, labels, mask, gender, age = train_batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                gender = gender.to(self.device)
                age = age.to(self.device)

                with torch.cuda.amp.autocast():
                    outs = self.model(inputs)
                    pred_mask, pred_gender, pred_age = outs
                    
                    loss_mask = self.criterion(pred_mask, mask)
                    loss_gender = self.criterion(pred_gender, gender)
                    loss_age = self.criterion(pred_age, age)
                loss = loss_mask + loss_gender + loss_age
                
                preds = torch.argmax(pred_mask, dim=-1) * 6 + torch.argmax(pred_gender, dim=-1) * 3 + torch.argmax(pred_age, dim=-1)

                acc_mask = (torch.argmax(pred_mask, dim=-1) == mask).sum().item() / mask.numel()
                acc_gender = (torch.argmax(pred_gender, dim=-1) == gender).sum().item() / gender.numel()
                acc_age = (torch.argmax(pred_age, dim=-1) == age).sum().item() / age.numel()

                acc_mask_items.append(acc_mask)
                acc_gender_items.append(acc_gender)
                acc_age_items.append(acc_age)
            else:
                inputs, labels = train_batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                with torch.cuda.amp.autocast():
                    outs = self.model(inputs)
                    loss = self.criterion(outs, labels)

                    preds = torch.argmax(outs, dim=-1)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % self.config.log_interval == 0:
                train_loss = loss_value / self.config.log_interval
                train_acc = matches / self.config.batch_size / self.config.log_interval
                current_lr = self.get_lr(self.optimizer)
                if self.config.multi_head:
                    print(
                        f"Epoch[{epoch}/{self.config.epochs}]({idx + 1}/{len(self.train_dataloader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || "
                        f"mask accuracy {np.mean(acc_mask_items):4.2%} || gender accuracy {np.mean(acc_gender_items):4.2%} || age accuracy {np.mean(acc_age_items):4.2%} ||  lr {current_lr}"
                    )
                else:
                    print(
                        f"Epoch[{epoch}/{self.config.epochs}]({idx + 1}/{len(self.train_dataloader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )

                # tensorboard: 학습 단계에서 Loss, Accuracy 로그 저장
                self.logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(self.train_dataloader) + idx
                )
                self.logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(self.train_dataloader) + idx
                )

                loss_value = 0
                matches = 0   

                # wandb: 학습 단계에서 Loss, Accuracy 로그 저장
                if self.config.multi_head:
                    wandb.log({
                        "Train loss": train_loss,
                        "Train acc" : train_acc,
                        "Train acc mask": np.mean(acc_mask_items),
                        "Train acc gender": np.mean(acc_gender_items),
                        "Train acc age": np.mean(acc_age_items),
                    })
                else:
                    wandb.log({
                        "Train loss": train_loss,
                        "Train acc" : train_acc,
                    })

        if self.lr_scheduler is not None:
            if self.config.scheduler != "ReduceLROnPlateau":
                self.lr_scheduler.step()

        if self.do_validation:
            self._valid_epoch(epoch)


    def _valid_epoch(self, epoch):
        self.model.eval()
        
        with torch.no_grad():
            print("Calculating validation results...")
            val_loss_items = []
            val_acc_items = []

            val_acc_mask_items = []
            val_acc_gender_items = []
            val_acc_age_items = []

            figure = None

            if self.config.save_val_table != 0 and epoch == self.config.epochs-1: # 마지막 epoch만 저장하도록
                # wandb table for validation
                if self.config.multi_head:
                    columns = ["image", "mask_gt", "mask_predict", "gender_gt", "gender_predict", "age_gt", "age_predict"]
                    val_table = wandb.Table(columns=columns)
                else:
                    columns = ["image", "gt", "predict"]
                    val_table = wandb.Table(columns=columns)
            
            for val_batch in self.valid_dataloader:
                if self.config.multi_head:
                    inputs, labels, mask, gender, age = val_batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    mask = mask.to(self.device)
                    gender = gender.to(self.device)
                    age = age.to(self.device)

                    with torch.cuda.amp.autocast():
                        outs = self.model(inputs)
                        pred_mask, pred_gender, pred_age = outs
                        loss_mask = self.criterion(pred_mask, mask)
                        loss_gender = self.criterion(pred_gender, gender)
                        loss_age = self.criterion(pred_age, age)
                    
                    loss_item = (loss_mask + loss_gender + loss_age).item()
                    preds = torch.argmax(pred_mask, dim=-1) * 6 + torch.argmax(pred_gender, dim=-1) * 3 + torch.argmax(pred_age, dim=-1)
                    
                    acc_mask = (torch.argmax(pred_mask, dim=-1) == mask).sum().item() / mask.numel()
                    acc_gender = (torch.argmax(pred_gender, dim=-1) == gender).sum().item() / gender.numel()
                    acc_age = (torch.argmax(pred_age, dim=-1) == age).sum().item() / age.numel()

                    val_acc_mask_items.append(acc_mask)
                    val_acc_gender_items.append(acc_gender)
                    val_acc_age_items.append(acc_age)

                    # val_table logging
                    if self.config.save_val_table != 0 and epoch == self.config.epochs-1: # 마지막 epoch만 저장하도록
                        inputs_np = (
                            torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        )
                        inputs_np = self.denormalize_image(
                            inputs_np, self.dataset_mean, self.dataset_std
                        )
                        for input, m_gt, m_pt, g_gt, g_pt, a_gt, a_pt in zip(inputs_np, mask, torch.argmax(pred_mask, dim=-1), gender, torch.argmax(pred_gender, dim=-1), age, torch.argmax(pred_age, dim=-1)):
                            if self.config.save_val_table == 2:
                                if m_gt != m_pt or g_gt != g_pt or a_gt != a_pt:
                                    val_table.add_data(wandb.Image(input), m_gt, m_pt, g_gt, g_pt, a_gt, a_pt)
                            else:
                                val_table.add_data(wandb.Image(input), m_gt, m_pt, g_gt, g_pt, a_gt, a_pt)
                
                else:
                    inputs, labels = val_batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    with torch.cuda.amp.autocast():
                        outs = self.model(inputs)
                        preds = torch.argmax(outs, dim=-1)

                        loss_item = self.criterion(outs, labels).item()

                    # val_table logging
                    if self.config.save_val_table != 0 and epoch == self.config.epochs-1: # 마지막 epoch만 저장하도록
                        inputs_np = (
                            torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        )
                        if self.config.model == "ArcfaceMultiHead":
                            inputs_np = self.denormalize_arcface_image(
                                inputs_np, self.dataset_mean, self.dataset_std
                            )
                        else:
                            inputs_np = self.denormalize_image(
                                inputs_np, self.dataset_mean, self.dataset_std
                            )
                        for input, gt, pred in zip(inputs_np, labels, preds):
                            if self.config.save_val_table == 2:
                                if gt != pred:
                                    val_table.add_data(wandb.Image(input), gt, pred)
                            else:
                                val_table.add_data(wandb.Image(input), gt, pred)

                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = (
                        torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    )
                    if self.config.model == "ArcfaceMultiHead":
                        inputs_np = self.denormalize_arcface_image(
                            inputs_np, self.dataset_mean, self.dataset_std
                        )
                    else:
                        inputs_np = self.denormalize_image(
                            inputs_np, self.dataset_mean, self.dataset_std
                        )
                    figure = self.grid_image(
                        inputs_np,
                        labels,
                        preds,
                        n=16,
                        shuffle=self.config.dataset != "MaskSplitByProfileDataset",
                    )

            val_loss = np.sum(val_loss_items) / len(self.valid_dataloader)
            val_acc = np.sum(val_acc_items) / (len(self.valid_dataloader) * self.config.valid_batch_size)
            self.best_val_loss = min(self.best_val_loss, val_loss)
            if val_acc > self.best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                
                torch.save(self.model.module.state_dict(), f"{self.save_dir}/best.pth")
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            torch.save(self.model.module.state_dict(), f"{self.save_dir}/last.pth")
            if self.config.multi_head:
                print(
                    f"[Val] acc : {val_acc:4.2%} || mask acc : {np.mean(val_acc_mask_items):4.2%} || "
                    f"gender acc : {np.mean(val_acc_gender_items):4.2%} || age acc : {np.mean(val_acc_age_items):4.2%} || loss: {val_loss:4.2} || "
                    f"best acc : {self.best_val_acc:4.2%} || best loss: {self.best_val_loss:4.2}"
                )
            else:
                print(
                    f"[Val] acc : {val_acc:4.2%} || loss: {val_loss:4.2} || "
                    f"best acc : {self.best_val_acc:4.2%} || best loss: {self.best_val_loss:4.2}"
                )

            # "ReduceLROnPlateau" scheduler step
            if self.config.scheduler == "ReduceLROnPlateau":
                self.lr_scheduler.step(val_acc) # TODO val_acc? val_loss?
            
            # tensorboard: 검증 단계에서 Loss, Accuracy 로그 저장
            self.logger.add_scalar("Val/loss", val_loss, epoch)
            self.logger.add_scalar("Val/accuracy", val_acc, epoch)
            self.logger.add_figure("results", figure, epoch)
            print()

            # wandb: 검증 단계에서 Loss, Accuracy 로그 저장
            if self.config.multi_head:
                wandb.log({
                    "Valid loss": val_loss,
                    "Valid acc" : val_acc,
                    "Valid acc_mask" : np.mean(val_acc_mask_items),
                    "Valid acc_gender" : np.mean(val_acc_gender_items),
                    "Valid acc_age" : np.mean(val_acc_age_items),
                    "results": wandb.Image(figure),
                })
            else:
                wandb.log({
                    "Valid loss": val_loss,
                    "Valid acc" : val_acc,
                    "results": wandb.Image(figure),
                })
            if self.config.save_val_table != 0 and epoch == self.config.epochs-1: # 마지막 epoch만 저장하도록
                wandb.log({f"{self.config.wandb}_val_table": val_table})

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    
    def denormalize_image(self, image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp
    
    def denormalize_arcface_image(self, image, mean, std):
        """정규화된 이미지를 원래대로 되돌리는 메서드"""
        img_cp = image.copy()
        img_cp *= 0.5
        img_cp += 0.5
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def decode_multi_class(self, multi_class_label,):
        """인코딩된 다중 라벨을 각각의 라벨로 디코딩하는 메서드"""
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    def grid_image(self, np_images, gts, preds, n=16, shuffle=False):
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
            gt_decoded_labels = self.decode_multi_class(gt)
            pred_decoded_labels = self.decode_multi_class(pred)
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