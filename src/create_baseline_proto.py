# import
import configparser
import os
import warnings
from pathlib import Path
from typing import List, Tuple

import ignite
import mlflow
import numpy as np
import seaborn as sns
import torch
from albumentations import (Compose, Flip, GaussNoise, Normalize, RandomCrop,
                            Resize, Rotate)
from albumentations.pytorch import ToTensorV2
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset, random_split
from torchsummary import summary
from tqdm import tqdm as tqdm

from metrics import DiceBCELoss, FBetaScore
from models import UNet
from utils.utils import flatten_dict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.simplefilter("ignore", category=DeprecationWarning)
config = configparser.ConfigParser()
config.read("../configs/config.ini")


def get_train_transform():
    """データオーグメンテーション関数

    Returns:
        albumentations.Compose()
    """
    return Compose(
        [
            # --aug ver1--
            # リサイズ
            Resize(config.getint("AUGMENTATION", "image_size"),
                   config.getint("AUGMENTATION", "image_size")),
            # ランダム回転
            Rotate(p=1),
            # 水平、垂直、水平垂直のいずれかにランダム反転
            Flip(p=1),
            # 正規化(予め平均と標準偏差は計算しておく？)
            # albumentationsは値域が0~255のunit8を返すので
            # std=1, mean=0で正規化を施してfloat32型にしておく
            Normalize(mean=config.getfloat("AUGMENTATION", "mean"),
                      std=config.getfloat("AUGMENTATION", "std")),
            # テンソル化
            ToTensorV2()
        ],
        additional_targets={
            "image0": "image",
            "image1": "image",
            "image2": "image"
        })


class LoadDataSet(Dataset):
    """データセットアクセスクラス
    """

    def __init__(self, path, transforms=None) -> None:
        """
        Args:
            path (str): データセットへのパス
            transforms (_type_, optional): _description_. Defaults to None.
        """
        self.path = path
        self.folders = os.listdir(os.path.join(path, "train_images/"))
        self.transforms = get_train_transform()

    def __len__(self) -> int:
        """len()を使用する時に呼び出される関数

        Returns:
            int: データサイズ
        """
        return len(self.folders)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """データを取り出す時に呼び出される関数

        Args:
            idx (int): インデックス

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 教師用データとアノテーションマスクによるタプル
        """
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
        raw_mask = io.imread(mask_path).astype("float32")

        augmented = self.transforms(
            image=img_0_vh, image0=img_0_vv, image1=img_1_vh, image2=img_1_vv, mask=raw_mask)

        imgs = [augmented["image"], augmented["image0"],
                augmented["image1"], augmented["image2"]]
        _mask = augmented["mask"]
        mask = torch.unsqueeze(_mask, 0)

        img = torch.cat(imgs, dim=0)

        return (img, mask)


def train():
    """
    学習コードはpytorch igniteを利用してコンパクトに書く
    ->あまりに隠蔽されすぎてエラー発生時対処できないので一端forLoop
    """
    # データセットの取得、分割
    dataset = LoadDataSet(
        "../dataset/", transforms=get_train_transform())

    train_size = int(np.round(dataset.__len__()
                     * (1 - config.getfloat("LEARNING", "split_ratio")), 0))
    valid_size = dataset.__len__() - train_size

    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size])
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config.getint("LEARNING", "batch_size"), shuffle=True)
    val_loader = DataLoader(
        dataset=valid_dataset, batch_size=config.getint("LEARNING", "batch_size"), shuffle=False)

    # モデル、デバイス、最適化手法、損失関数、ロガーの定義
    model = UNet(4, 1).cuda()
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.getfloat("LEARNING", "lr"))
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.getfloat(
    # "LEARNING", "lr"), momentum=config.getfloat("LEARNING", "momentum"))
    # optimizer = torch.optim.AdamW(
    #     model.parameters(), lr=config.getfloat("LEARNING", "lr"))
    criterion = DiceBCELoss()
    accuracy_metric = FBetaScore(
        beta=0.5, threshold=config.getfloat("LEARNING", "binary_threshold"))

    valid_loss_min = np.inf
    checkpoint_path = "../model/chkpoint_"
    best_model_path = "../model/bestmodel.pt"

    mlflow.start_run()

    # param記入
    params_dict = flatten_dict({s: dict(config.items(s))
                                for s in config.sections()})
    params_dict["model"] = model.__class__.__name__
    params_dict["optimizer"] = optimizer.__class__.__name__
    params_dict["train size"] = train_size
    params_dict["device name"] = f"{torch.cuda.get_device_name()}" if torch.cuda.is_available(
    ) else device
    params_dict["pytorch version"] = torch.__version__
    params_dict["ignite version"] = ignite.__version__
    params_dict["cuda version"] = torch.version.cuda

    mlflow.log_params(params_dict)

    total_train_loss = []
    total_train_score = []
    total_valid_loss = []
    total_valid_score = []

    losses_value = 0

    for epoch in range(config.getint("LEARNING", "epochs")):
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

    # 自分で計算したメトリックを記録する
    for epoch in range(0, config.getint("LEARNING", "epochs")):
        mlflow.log_metrics(
            {
                "training loss": total_train_loss[epoch],
                "validation loss": total_valid_loss[epoch],
                "training f_score_05": total_train_score[epoch],
                "validation f_score_05": total_valid_score[epoch],
            },
            step=epoch+1)

    # Modelの保存
    mlflow.pytorch.log_model(model, model.__class__.__name__)
    model_info = str(summary(model, verbose=0))
    with open("../model/model_schema.txt", "w") as f:
        f.write(model_info)
    mlflow.log_artifact("../model/model_schema.txt")


if __name__ == "__main__":
    """メイン関数
    """

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

    # ロガーの定義 : mlflowを用いる
    mlflow.set_tracking_uri(config.get("GENERAL", "mlrun_path"))
    mlflow.set_experiment(config.get("GENERAL", "experiment_name"))

    print(f"\n===Start Training===\n")
    results = train()
