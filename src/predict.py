import configparser
import os

import mlflow
import numpy as np
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from skimage import img_as_uint, io
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

config = configparser.ConfigParser()
config.read("../configs/config.ini")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

run_id = "fd371da79c0746a18f0f0985d7964fc2"
model_name = "UNet"


def get_train_transform():
    """
    画像データオーグメンテーション
    """
    return Compose(
        [
            # リサイズ
            Resize(config.getint("AUGMENTATION", "IMAGE_SIZE"),
                   config.getint("AUGMENTATION", "IMAGE_SIZE")),
            # albumentationsはピクセル値域が0~255のunit8を返すので
            # std=1, mean=0で正規化を施してfloat32型にしておく
            Normalize(mean=config.getfloat("AUGMENTATION", "MEAN"),
                      std=config.getfloat("AUGMENTATION", "STD")),
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
        self.folders = os.listdir(os.path.join(path, "test_images/"))
        self.transforms = get_train_transform()

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        prefix = f"test_{str(idx).zfill(2)}"

        image_folder = os.path.join(
            self.path, "test_images/", prefix + "/")
        image_0_vh_path = os.path.join(
            image_folder, sorted(os.listdir(image_folder))[0])
        image_0_vv_path = os.path.join(
            image_folder, sorted(os.listdir(image_folder))[1])
        image_1_vh_path = os.path.join(
            image_folder, sorted(os.listdir(image_folder))[2])
        image_1_vv_path = os.path.join(
            image_folder, sorted(os.listdir(image_folder))[3])

        # 画像データの取得
        img_0_vh = io.imread(image_0_vh_path, 0).astype("float32")
        img_0_vv = io.imread(image_0_vv_path, 0).astype("float32")
        img_1_vh = io.imread(image_1_vh_path, 0).astype("float32")
        img_1_vv = io.imread(image_1_vv_path, 0).astype("float32")

        h, w = img_0_vh.shape
        size = (h, w)

        augmented = self.transforms(
            image=img_0_vh, image0=img_0_vv, image1=img_1_vh, image2=img_1_vv)

        imgs = [augmented["image"], augmented["image0"],
                augmented["image1"], augmented["image2"]]

        img = torch.cat(imgs, dim=0)

        return img, size


# 可視化用の関数
def visualize(idx, images, size):
    """PLot images in one row."""
    n = len(images)

    resized_images = []
    for image in images:
        resized_image = resize(image, (size[0], size[1]),
                               anti_aliasing=True)
        resized_images.append(resized_image)

    plt.figure(figsize=(16, 5))
    for i, image in enumerate(resized_images):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.gray()
        plt.imshow(image, cmap='gray', vmin=0, vmax=1)

    io.imsave(
        f'../submit/test_predictions/pred_{str(idx).zfill(2)}.png', img_as_uint(resized_images[-1]))
    plt.savefig(
        f"../dataset/tmp/prediction_comparision/pred_comp_{str(idx).zfill(2)}.png")
   # plt.show()


def predict():
    test_dataset = LoadDataSet(
        "../dataset/")

    mlflow.set_tracking_uri(config.get("GENERAL", "MLRUN_PATH"))
    mlflow.set_experiment(config.get("GENERAL", "EXPERIMENT_NAME"))
    model_uri = f"runs:/{run_id}/{model_name}"
    loaded_model = mlflow.pytorch.load_model(model_uri)

    for idx in range(config.getint("PREDICTION", "TEST_BATCH_SIZE")):

        images, size = test_dataset[idx]
        images = images.to(device).unsqueeze(0)

        pr_mask = loaded_model(images)
        pr_mask = (pr_mask.squeeze(0)
                   .permute(1, 2, 0).cpu().detach().numpy().round())

        image_tensor = list(torch.tensor_split(images.squeeze(0), 4, dim=0))
        image_list = [image.permute(1, 2, 0).cpu().detach().numpy().copy() * 255
                      for image in image_tensor]
        image_list.append(pr_mask)

        visualize(idx, image_list, size)


if __name__ == "__main__":
    print("\n===Start Prediction===\n")
    predict()
