# import
import logging
import os
import warnings
from argparse import ArgumentParser
from pathlib import Path
from random import random

import ignite
import mlflow
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from albumentations import (Compose, Flip, GaussNoise, Normalize, Resize,
                            Rotate, ShiftScaleRotate)
from albumentations.pytorch import ToTensorV2
from ignite.contrib.handlers.mlflow_logger import *
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.handlers import EarlyStopping
from ignite.metrics import Precision, Recall
from matplotlib import pyplot as plt
from skimage import io, transform
from torch import nn, no_grad
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm as tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

warnings.simplefilter("ignore", category=DeprecationWarning)

# constants
BATCH_SIZE = 1
MEAN = 0
STD = 1
RANDOM_SEED = 126
CHANNEL_LIST = ["0_VH", "0_VV", "1_VH", "1_VV"]


def get_train_transform():
    """
    画像データオーグメンテーション
    """
    return Compose(
        [
            # リサイズ
            Resize(256, 256),
            # ランダム回転
            Rotate(p=1),
            # 水平、垂直、水平垂直のいずれかにランダム反転
            Flip(p=1),
            # 正規化(予め平均と標準偏差は計算しておく？)
            # albumentationsは値域が0~255のunit8を返すので
            # std=1, mean=0で正規化を施してfloat32型にしておく
            Normalize(mean=MEAN, std=STD),
            # テンソル化
            ToTensorV2()
        ],
        additional_targets={
            "image0": "image",
            "image1": "image",
            "image2": "image"
        })


class LoadDataSet(Dataset):
    """
    pytorchのデータセットクラスを継承
    """

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
        raw_mask = io.imread(mask_path).astype('float32')

        augmented = self.transforms(
            image=img_0_vh, image0=img_0_vv, image1=img_1_vh, image2=img_1_vv, mask=raw_mask)

        imgs = [augmented["image"], augmented["image0"],
                augmented["image1"], augmented["image2"]]
        _mask = augmented["mask"]
        mask = torch.unsqueeze(_mask, 0)

        img = torch.cat(imgs, dim=0)

        return (img, mask)

# 以下URLを写経
# https://obgynai.com/unet-semantic-segmentation/


class UNet(nn.Module):
    """
    U-netクラス
    """

    def __init__(self, input_channels, num_classes):
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
        self.conv10 = nn.Conv2d(64, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # パラメータの初期化 TODO Heの初期値を用いる理由は？
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        """
        順伝搬処理の定義
        """

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


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """
    畳み込みとバッチ正規化と活性化関数Reluをまとめている
    """
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
    """
    BinaryCrossEntropy + DiceLoss
    DiceLoss = 1 - DiceCoeff
    smooth back propagationを行う際に関数平面の平滑化を行って計算が進むようにする
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth) / \
            (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class FBetaScore(nn.Module):
    """
    WeightedFScore class
    beta = 1の時Dice係数と同じ
    const zerodivision対策
    """

    def __init__(self, beta: float = 1., threshold: float = 0.5):
        super(FBetaScore, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, inputs, targets, const=1e-7):
        # Binarize probablities
        inputs = torch.where(inputs < self.threshold, 0, 1)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        fbeta = ((1 + self.beta**2)*intersection) / \
            (inputs.sum() + self.beta**2 * targets.sum() + const)

        return fbeta


def thresholded_output_transform(x, y, y_pred):
    """
    convert from prob to binary function
    """
    y_pred = torch.round(y_pred)
    return y_pred, y


def score_function(engine):
    """
    ignite.handlers.EarlyStopping では指定スコアが上がると改善したと判定する。
    そのため今回のロスに -1 をかけたものを ignite.handlers.EarlyStopping オブジェクトに渡す
    """
    val_loss = engine.state.metrics["loss"]
    return -val_loss


def train(cfg):
    """
    学習コードはpytorch igniteを利用してコンパクトに書く
    ->あまりに隠蔽されすぎてエラー発生時対処できないので一端forLoop
    """
    # データセットの取得、分割
    dataset = LoadDataSet(
        "../dataset/", transforms=get_train_transform())

    train_size = int(np.round(dataset.__len__()
                     * (1 - cfg.split_ratio), 0))
    valid_size = dataset.__len__() - train_size

    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # モデル、デバイス、最適化手法、損失関数、ロガーの定義
    model = UNet(4, 1).cuda()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    criterion = DiceBCELoss()
    accuracy_metric = FBetaScore(beta=0.5)

    valid_loss_min = np.inf
    checkpoint_path = "../model/chkpoint_"
    best_model_path = "../model/bestmodel.pt"

    total_train_loss = []
    total_train_score = []
    total_valid_loss = []
    total_valid_score = []

    losses_value = 0

    for epoch in range(cfg.epochs):
        train_loss = []
        train_score = []
        valid_loss = []
        valid_score = []
        pbar = tqdm(train_loader, desc="description")
        for x_train, y_train in pbar:
            x_train = torch.autograd.Variable(x_train).cuda()
            y_train = torch.autograd.Variable(y_train).cuda()
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            losses_value = loss.item()

            score = accuracy_metric(output, y_train)
            loss.backward()
            optimizer.step()
            train_loss.append(losses_value)
            train_score.append(score.item())
            pbar.set_description(f"""
            Epch: {epoch+1}
            loss: {losses_value}
            Fbeta: {score}
            """)

        with torch.no_grad():
            for image, mask in val_loader:
                image = torch.autograd.Variable(image).cuda()
                mask = torch.autograd.Variable(mask).cuda()
                output = model(image)

                loss = criterion(output, mask)
                losses_value = loss.item()

                score = accuracy_metric(output, mask)
                valid_loss.append(losses_value)
                valid_score.append(score.item())

            total_train_loss.append(np.mean(train_loss))
            total_train_score.append(np.mean(train_score))
            total_valid_loss.append(np.mean(valid_loss))
            total_valid_score.append(np.mean(valid_score))
            print(
                f"Train Loss:{total_train_loss[-1]}, Train Fbeta: {total_train_score[-1]}")
            print(
                f"Valid Loss:{total_valid_loss[-1]}, Valid Fbeta: {total_valid_score[-1]}")

            checkpoint = {
                "epoch": epoch + 1,
                "valid_loss_min": total_valid_loss[-1],
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            if total_valid_loss[-1] <= valid_loss_min:
                print(
                    f"Validation loss decreased ({valid_loss_min:.6f} --> {total_valid_loss[-1]:.6f}).  Saving model ...")
                torch.save(checkpoint, best_model_path)
                valid_loss_min = total_valid_loss[-1]

            print("")

    results = {
        "total_train_loss": total_train_loss,
        "total_valid_loss": total_valid_loss,
        "total_train_score": total_train_score,
        "total_valid_score": total_valid_score,
    }

    return results


def visualize(results, epochs):
    plt.figure(1)
    plt.figure(figsize=(15, 5))
    sns.set_style(style="darkgrid")
    plt.subplot(1, 2, 1)
    sns.lineplot(x=range(1, epochs+1),
                 y=results["total_train_loss"], label="Train Loss")
    sns.lineplot(x=range(1, epochs+1),
                 y=results["total_valid_loss"], label="Valid Loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.ylabel("DiceBCELoss")

    plt.subplot(1, 2, 2)
    sns.lineplot(x=range(1, epochs+1),
                 y=results["total_train_score"], label="Train Score")
    sns.lineplot(x=range(1, epochs+1),
                 y=results["total_valid_score"], label="Valid Score")
    plt.title("Score (Fbeta@0.5)")
    plt.xlabel("epochs")
    plt.ylabel("Fbeta@0.5")
    plt.savefig("../model/model_eval.png")


if __name__ == "__main__":
    # parseagrs
    parser = ArgumentParser()
    parser.add_argument("--split_ratio", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    cfg = parser.parse_args()

    # path
    print("\n===Define Path===\n")
    ROOT_DIR = Path.cwd().parent.resolve()
    print("root path:", ROOT_DIR)
    MLRUN_PATH = ROOT_DIR.parents[0] / "mlruns"
    if MLRUN_PATH.exists():
        print("MLRUN path:", MLRUN_PATH)
    else:
        print("MLRUN path does not exist.")
        exit()

    # competition name(= experiment name)
    print("\n===Set Experiment Name===\n")
    EXPERIMENT_NAME = ROOT_DIR.name
    print("experiment name:", EXPERIMENT_NAME)

    print(f"\n===Check GPU Available===\n")
    if torch.cuda.is_available():
        print(f"RUNNING ON GPU - {torch.cuda.get_device_name()}")
    else:
        print(f"RUNNING ON CPU")

    # logger作成
    logger = logging.getLogger("ignite.engine.engine.Engine")
    handler = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    # ロガーに追加
    logger.addHandler(handler)
    # ログレベルの設定
    logger.setLevel(logging.INFO)

    print(f"\n===Start Training===\n")
    results = train(cfg)

    print(f"\n===Save Training Results===\n")
    visualize(results, cfg.epochs)
