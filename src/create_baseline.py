# import
import copy
import datetime
import os
import random
import shutil
import sys
import time
import warnings
import zipfile
from collections import defaultdict
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import mlflow
import numpy as np
# import optuna
import pandas as pd
import sklearn
import torch
import torch.nn.functional as F
from albumentations import (Compose, GaussNoise, HorizontalFlip, Normalize,
                            Resize, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2
from PIL import Image
from skimage import io, transform
# import xgboost
# from matplotlib import pyplot as plt
# from prettytable import RANDOM
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.autograd import Variable
from torch.nn import (BatchNorm2d, Conv2d, CrossEntropyLoss, Dropout, Linear,
                      MaxPool2d, Module, ReLU, Sequential, Softmax)
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, utils
from tqdm import tqdm as tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

warnings.simplefilter("ignore", category=DeprecationWarning)
# optuna.logging.set_verbosity(optuna.logging.WARNING)

# 定数
MEAN = 0
STD = 1
SPLIT_RATIO = 0.25
CHANNEL_LIST = ["0_VH", "0_VV", "1_VH", "1_VV"]

# 画像データオーグメンテーション


def get_train_transform():

    return A.Compose(
        [
            # リサイズ
            A.Resize(256, 256),
            # 正規化(予め平均と標準偏差は計算しておく)
            # A.Normalize(mean=MEAN, std=STD),
            # ランダム回転
            A.Rotate(p=1),
            # テンソル化
            ToTensorV2()
        ],
        additional_targets={
            "image0": "image",
            "image1": "image",
            "image2": "image"
        })


# pytorchのデータセットクラスを継承
class LoadDataSet(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.folders = os.listdir(os.path.join(path, "train_images/"))
        self.transforms = get_train_transform()

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        prefix = f"train_{str(idx).zfill(2)}"

        image_folder = os.path.join(
            self.path, "train_images/", prefix + "/")
        image_0_vh_path = os.path.join(
            image_folder, sorted(os.listdir(image_folder))[0])
        image_0_vv_path = os.path.join(
            image_folder, sorted(os.listdir(image_folder))[1])
        image_1_vh_path = os.path.join(
            image_folder, sorted(os.listdir(image_folder))[2])
        image_1_vv_path = os.path.join(
            image_folder, sorted(os.listdir(image_folder))[3])

        mask_path = os.path.join(
            self.path, "train_annotations/", prefix+".png")

        # 画像データの取得
        img_0_vh = io.imread(image_0_vh_path, 0).astype("float32")
        img_0_vv = io.imread(image_0_vv_path, 0).astype("float32")
        img_1_vh = io.imread(image_1_vh_path, 0).astype("float32")
        img_1_vv = io.imread(image_1_vv_path, 0).astype("float32")

        # アノテーションデータの取得
        mask = np.zeros((256, 256, 1), dtype=np.bool)
        mask_ = io.imread(mask_path)
        mask_ = transform.resize(mask_, (256, 256))
        mask_ = np.expand_dims(mask_, axis=-1)
        mask = np.maximum(mask, mask_)

        augmented = self.transforms(
            image=img_0_vh, image0=img_0_vv, image1=img_1_vh, image2=img_1_vv, mask=mask)

        size = augmented["image"].size()
        print(f"train image size is {size}\n")

        imgs = [augmented["image"], augmented["image0"],
                augmented["image1"], augmented["image2"]]

        img = torch.cat(imgs, dim=0)
        mask = augmented["mask"]

        return (img, mask)


def format_image(img):
    # 下は画像拡張での正規化を元に戻しています
    # img = STD * img + MEAN
    # img = img * 255
    img = img.to('cpu').detach().numpy().copy().transpose((1, 2, 0))
    img = img.astype(np.uint8)
    return img


def visualize_dataset(n_images, predict=None):
    image_idx_list = random.sample(range(0, 28), n_images)
    figure, ax = plt.subplots(nrows=n_images, ncols=5, figsize=(
        10, 10), gridspec_kw={'wspace': 0.02, 'hspace': -0.6})
    print(f"selected index: {image_idx_list}\n")

    for row, idx in enumerate(image_idx_list):

        image, mask = train_dataset.__getitem__(idx)
        images = torch.tensor_split(image, len(CHANNEL_LIST), dim=0)

        # train image
        for col, (title, image) in enumerate(zip(CHANNEL_LIST, images)):
            image = format_image(image)
            ax[row, col].imshow(image, cmap="gray")

            ax[row, 0].set_ylabel(f"idx:{idx}", fontsize=8)
            if row == 0:
                ax[row, col].set_title(title, fontsize=8)
                ax[row, col].xaxis.tick_top()
            ax[row, col].xaxis.set_ticks([])
            ax[row, col].yaxis.set_ticks([])

        # annotation mask
        ax[row, -1].imshow(mask, cmap="gray")

        if row == 0:
            ax[row, -1].set_title("Label Mask", fontsize=8)

        ax[row, -1].yaxis.set_ticks([])
        ax[row, -1].xaxis.set_ticks([])

    plt.tight_layout()
    plt.show()

# UNet


class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        # 資料中の『FCN』に当たる部分
        self.conv1 = conv_bn_relu(input_channels, 64)
        self.conv2 = conv_bn_relu(64, 128)
        self.conv3 = conv_bn_relu(128, 256)
        self.conv4 = conv_bn_relu(256, 512)
        self.conv5 = conv_bn_relu(512, 1024)
        self.down_pooling = nn.MaxPool2d(2)

        # 資料中の『Up Sampling』に当たる部分
        self.up_pool6 = up_pooling(1024, 512)
        self.conv6 = conv_bn_relu(1024, 512)
        self.up_pool7 = up_pooling(512, 256)
        self.conv7 = conv_bn_relu(512, 256)
        self.up_pool8 = up_pooling(256, 128)
        self.conv8 = conv_bn_relu(256, 128)
        self.up_pool9 = up_pooling(128, 64)
        self.conv9 = conv_bn_relu(128, 64)
        self.conv10 = nn.Conv2d(64, output_channels, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # 正規化
        x = x/255.

        # 資料中の『FCN』に当たる部分
        x1 = self.conv1(x)
        p1 = self.down_pooling(x1)
        x2 = self.conv2(p1)
        p2 = self.down_pooling(x2)
        x3 = self.conv3(p2)
        p3 = self.down_pooling(x3)
        x4 = self.conv4(p3)
        p4 = self.down_pooling(x4)
        x5 = self.conv5(p4)

        # 資料中の『Up Sampling』に当たる部分, torch.catによりSkip Connectionをしている
        p6 = self.up_pool6(x5)
        x6 = torch.cat([p6, x4], dim=1)
        x6 = self.conv6(x6)

        p7 = self.up_pool7(x6)
        x7 = torch.cat([p7, x3], dim=1)
        x7 = self.conv7(x7)

        p8 = self.up_pool8(x7)
        x8 = torch.cat([p8, x2], dim=1)
        x8 = self.conv8(x8)

        p9 = self.up_pool9(x8)
        x9 = torch.cat([p9, x1], dim=1)
        x9 = self.conv9(x9)

        output = self.conv10(x9)
        output = torch.sigmoid(output)

        return output

# 畳み込みとバッチ正規化と活性化関数Reluをまとめている


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def down_pooling():
    return nn.MaxPool2d(2)


def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    return nn.Sequential(
        # 転置畳み込み
        nn.ConvTranspose2d(in_channels, out_channels,
                           kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


#
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return IoU


if __name__ == "__main__":

    # path
    ROOT_DIR = Path.cwd().parent.resolve()
    print("root path:", ROOT_DIR)
    MLRUN_PATH = ROOT_DIR.parents[0] / "mlruns"
    if MLRUN_PATH.exists():
        print("MLRUN path:", MLRUN_PATH)
    else:
        print("MLRUN path does not exist.")
        exit()

    # competition name(= experiment name)
    EXPERIMENT_NAME = ROOT_DIR.name
    print("experiment name:", EXPERIMENT_NAME)

    print(f"\n GPU SETUP \n")
    if torch.cuda.is_available():
        print(f"RUNNING ON GPU - {torch.cuda.get_device_name()}")
    else:
        print(f"RUNNING ON CPU")

    # mlflow settings
    nowstr = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(f"torch version is {torch.__version__}\n")
    # mlflow.set_tracking_uri(str(MLRUN_PATH) + "/")
    # mlflow.set_experiment(EXPERIMENT_NAME)
    # mlflow.start_run(run_name=nowstr)

    # random seed
    RANDOM_SEED = 126
    # mlflow.log_param(key="random_seed", value=RANDOM_SEED)

    train_dataset = LoadDataSet(
        "../dataset/", transforms=get_train_transform())

    visualize_dataset(3)

    train_size = int(np.round(train_dataset.__len__()*(1 - SPLIT_RATIO), 0))
    valid_size = int(np.round(train_dataset.__len__()*SPLIT_RATIO, 0))
    train_data, valid_data = random_split(
        train_dataset, [train_size, valid_size])
    train_loader = DataLoader(dataset=train_data,
                              batch_size=10, shuffle=True)
    val_loader = DataLoader(dataset=valid_data, batch_size=10)

    # print("Length of train data: {}".format(len(train_data)))
    # print("Length of validation data: {}".format(len(valid_data)))

    # # <---------------各インスタンス作成---------------------->
    # model = UNet(3, 1).cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = DiceLoss()
    # accuracy_metric = IoU()
    # num_epochs = 20
    # valid_loss_min = np.Inf

    # checkpoint_path = 'model/chkpoint_'
    # best_model_path = 'model/bestmodel.pt'

    # total_train_loss = []
    # total_train_score = []
    # total_valid_loss = []
    # total_valid_score = []

    # losses_value = 0
    # for epoch in range(num_epochs):
    #     # <---------------トレーニング---------------------->
    #     train_loss = []
    #     train_score = []
    #     valid_loss = []
    #     valid_score = []
    #     pbar = tqdm(train_loader, desc='description')
    #     for x_train, y_train in pbar:
    #         x_train = torch.autograd.Variable(x_train).cuda()
    #         y_train = torch.autograd.Variable(y_train).cuda()
    #         optimizer.zero_grad()
    #         output = model(x_train)
    #         # 損失計算
    #         loss = criterion(output, y_train)
    #         losses_value = loss.item()
    #         # 精度評価
    #         score = accuracy_metric(output, y_train)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss.append(losses_value)
    #         train_score.append(score.item())
    #         pbar.set_description(
    #             f"Epoch: {epoch+1}, loss: {losses_value}, IoU: {score}")
    #     # <---------------評価---------------------->
    #     with torch.no_grad():
    #         for image, mask in val_loader:
    #             image = torch.autograd.Variable(image).cuda()
    #             mask = torch.autograd.Variable(mask).cuda()
    #             output = model(image)
    #             # 損失計算
    #             loss = criterion(output, mask)
    #             losses_value = loss.item()
    #             # 精度評価
    #             score = accuracy_metric(output, mask)
    #             valid_loss.append(losses_value)
    #             valid_score.append(score.item())

    #     total_train_loss.append(np.mean(train_loss))
    #     total_train_score.append(np.mean(train_score))
    #     total_valid_loss.append(np.mean(valid_loss))
    #     total_valid_score.append(np.mean(valid_score))
    #     print(
    #         f"Train Loss: {total_train_loss[-1]}, Train IOU: {total_train_score[-1]}")
    #     print(
    #         f"Valid Loss: {total_valid_loss[-1]}, Valid IOU: {total_valid_score[-1]}")

    #     checkpoint = {
    #         'epoch': epoch + 1,
    #         'valid_loss_min': total_valid_loss[-1],
    #         'state_dict': model.state_dict(),
    #         'optimizer': optimizer.state_dict(),
    #     }

    #     # checkpointの保存
    #     torch.save_ckp(checkpoint, False, checkpoint_path, best_model_path)

    #     # 評価データにおいて最高精度のモデルのcheckpointの保存
    #     if total_valid_loss[-1] <= valid_loss_min:
    #         print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
    #             valid_loss_min, total_valid_loss[-1]))
    #         torch.save_ckp(checkpoint, True, checkpoint_path, best_model_path)
    #         valid_loss_min = total_valid_loss[-1]

    #     print("")
    # train_path_df = path_df[path_df.index.isin(train_idx)]
    # validation_path_df = path_df[path_df.index.isin(val_idx)]

    # mlflow.end_run()
