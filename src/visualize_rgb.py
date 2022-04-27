# import
import os
import random
import warnings
from pathlib import Path

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from skimage import io, transform
from torch.utils.data import Dataset
from tqdm import tqdm as tqdm

# constants
DATASET_SIZE = 28
CHANNEL_LIST = ["0_VH", "0_VV", "1_VH", "1_VV"]


def get_train_transform():
    """
    画像データオーグメンテーション
    可視化しかしないのでいじらない
    """
    return A.Compose(
        [
            # リサイズ
            A.Resize(256, 256),
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
    dataloaderの練習
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
        mask = np.zeros((256, 256, 1), dtype=np.bool)
        mask_ = io.imread(mask_path)
        mask_ = transform.resize(mask_, (256, 256))
        mask_ = np.expand_dims(mask_, axis=-1)
        mask = np.maximum(mask, mask_)

        augmented = self.transforms(
            image=img_0_vh, image0=img_0_vv, image1=img_1_vh, image2=img_1_vv, mask=mask)

        imgs = [augmented["image"], augmented["image0"],
                augmented["image1"], augmented["image2"]]

        img = torch.cat(imgs, dim=0)
        mask = augmented["mask"]

        return (img, mask)


def format_image(img):
    img = img.permute(1, 2, 0).cpu().detach().numpy()
    return img


def format_mask(img):
    img = img.cpu().detach().numpy()
    img = img.astype(np.uint8)
    return img


def visualize_dataset(dest_dir, n_images):
    """
    各チャンネル画像とマスクを可視化する
    """

    image_idx_list = random.sample(range(0, DATASET_SIZE), n_images)
    figure, ax = plt.subplots(nrows=n_images, ncols=len(CHANNEL_LIST) + 1, figsize=(
        10, 10), gridspec_kw={"wspace": 0.1, "hspace": 0.5})

    img_name = "comparision.png"
    output_path = os.path.join(dest_dir, img_name)

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

    plt.savefig(output_path, dpi=figure.dpi)
    # plt.show()


def create_imgcolor(dest_dir):

    image_idx_list = list(range(DATASET_SIZE))

    for idx in image_idx_list:

        image, mask = train_dataset.__getitem__(idx)
        images = torch.tensor_split(image, len(CHANNEL_LIST), dim=0)

        img_name = f"color_{str(idx).zfill(2)}.png"
        output_path = os.path.join(dest_dir, img_name)

        train_0_vh = format_image(images[0])
        train_0_vv = format_image(images[1])
        train_1_vh = format_image(images[2])
        train_1_vv = format_image(images[3])

        mask = format_mask(mask)

        # vh,vvの差分をカラーとして合成した写真に対してアノテーションを表示させる。
        train_vh_diff = ((train_1_vh - train_0_vh) /
                         (train_1_vh - train_0_vh).max()*255)
        train_vv_diff = ((train_1_vv - train_0_vv) /
                         (train_1_vv - train_0_vv).max()*255)
        mask = np.where(mask == 1, np.float32(255), np.float32(0))

        # vv:b anno:g vh:r
        img_bgr = cv2.merge((train_vv_diff, train_vv_diff, train_vh_diff))
        # float画像はimshowでしか表示できない
        img_bgr = np.clip(img_bgr * 255, 0, 255).astype(np.uint8)

        cv2.imwrite(output_path, img_bgr)
        # cv2.imshow("image", img_bgr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindo()
        # break


if __name__ == "__main__":

    ROOT_DIR = Path.cwd().parent.resolve()
    print("root path:", ROOT_DIR)

    target_path = os.path.join(ROOT_DIR, "dataset/tmp/")
    if not os.path.exists(target_path):
        os.mkdir(target_path)

    train_dataset = LoadDataSet(
        "../dataset/", transforms=get_train_transform())

    visualize_dataset(target_path, 5)

    create_imgcolor(target_path)
