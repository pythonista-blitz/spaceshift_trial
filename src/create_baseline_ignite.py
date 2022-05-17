# import
import configparser
import logging
import os
import random
import warnings
from pathlib import Path

import ignite
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from albumentations import Compose, Flip, Normalize, RandomCrop, Resize, Rotate
from albumentations.pytorch import ToTensorV2
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.mlflow_logger import *
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import (Engine, Events, create_supervised_evaluator,
                           create_supervised_trainer, supervised_training_step)
from ignite.handlers import EarlyStopping, global_step_from_engine
from ignite.metrics import (Accuracy, Fbeta, Loss, Precision, Recall,
                            RunningAverage)
from skimage import io
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchsummary import summary
from tqdm import tqdm as tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

warnings.simplefilter("ignore", category=DeprecationWarning)
config = configparser.ConfigParser()
config.read("../configs/config.ini")
CHANNEL_LIST = ["0_VH", "0_VV", "1_VH", "1_VV"]
enable_cuda_flag = torch.cuda.is_available()


def seed_worker(worker_id):
    """
    dataloader内での挙動に再現性を持たせる
    https://qiita.com/north_redwing/items/1e153139125d37829d2d#dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_train_transform():
    """
    画像データオーグメンテーション
    """
    return Compose(
        [
            # リサイズ
            # Resize(config.getint("AUGMENTATION", "IMAGE_SIZE"),
            #        config.getint("AUGMENTATION", "IMAGE_SIZE")),
            # ランダム回転
            Rotate(p=1),
            # 水平、垂直、水平垂直のいずれかにランダム反転
            Flip(p=0.5),
            # ランダム切り取り
            RandomCrop(p=1, height=config.getint(
                "AUGMENTATION", "IMAGE_SIZE"), width=config.getint("AUGMENTATION", "IMAGE_SIZE")),
            # 正規化(予め平均と標準偏差は計算しておく？)

            # albumentationsはピクセル値域が0~255のunit8を返すので
            # std=1, mean=0で正規化を施してfloat32型にしておく
            Normalize(mean=config.getfloat("AUGMENTATION", "MEAN"),
                      std=config.getfloat("AUGMENTATION", "STD")),
            # テンソル化
            ToTensorV2()
        ],
        # 可視光のRGBと異なりvh,vvの前後計4チャンネルを同時に加工するのでターゲットの追加
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
        raw_mask = io.imread(mask_path).astype("float32")

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


def fscore(beta):
    p = Precision(average=False)
    r = Recall(average=False)
    fbeta = Fbeta(beta=beta, precision=p, recall=r)
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


def train(generator):
    """
    学習コードはpytorch igniteを利用してコンパクトに書く
    """
    # データセットの取得、分割
    dataset = LoadDataSet(
        "../dataset/", transforms=get_train_transform())

    train_size = int(np.round(dataset.__len__()
                     * (1 - config.getfloat("LEARNING", "split_ratio")), 0))
    valid_size = dataset.__len__() - train_size

    train_dataset, valid_dataset = random_split(
        dataset, [train_size, valid_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.getint("LEARNING", "BATCH_SIZE"),
                              shuffle=True, worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=config.getint("LEARNING", "BATCH_SIZE"),
                            shuffle=False, worker_init_fn=seed_worker, generator=generator)

    # モデル生成、デバイス割り当て、最適化手法、損失関数、ロガーの定義
    model = UNet(4, 1)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.getfloat("LEARNING", "lr"), eps=0.1)
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=config.getfloat("LEARNING", "lr"), momentum=config.getfloat("LEARNING", "momentum"))

    scheduler = StepLR(optimizer, step_size=config.getfloat(
        "LEARNING", "step_size"), gamma=config.getfloat("LEARNING", "gamma"))
    # scheduler = ExponentialLR(
    #     optimizer, gamma=config.getfloat("LEARNING", "gamma"))
    lr_scheduler = LRScheduler(scheduler)

    # 画像内の正解ピクセル数に対して背景ピクセル数が大.つまり
    # 正解不正解の頻度差が大きいのでCE Loss系は学習が上手くいかない
    # criterion = nn.CrossEntropyLoss()
    criterion = DiceBCELoss()
    fbeta = fscore(beta=0.5)
    # fbeta = FBetaScore(beta=0.5)

    # 学習器の設定
    # update_fn = supervised_training_step(
    #     model, optimizer, criterion, gradient_accumulation_steps=2)
    # trainer = Engine(update_fn)
    trainer = create_supervised_trainer(
        model=model, optimizer=optimizer, loss_fn=criterion, device=device)

    # 評価指標の定義 : f-beta and loss to compute on val dataset
    metrics = {
        # 精度評価、コンペのルール参照
        "f_score_05": fbeta,
        "loss": ignite.metrics.Loss(criterion)  # 損失関数
    }

    # train_evaluator = create_supervised_evaluator(
    #     model, metrics=metrics, device=device, output_transform=thresholded_output_transform)
    # valid_evaluator = create_supervised_evaluator(
    #     model, metrics=metrics, device=device, output_transform=thresholded_output_transform)
    train_evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device)
    valid_evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device)

    # プログレスバーの実装
    pbar = ProgressBar(persist=False)
    pbar.attach(trainer, metric_names="all")

   # ロガーの定義 : mlflowを用いる
    mlflow.set_tracking_uri(config.get("GENERAL", "MLRUN_PATH"))
    mlflow.set_experiment(config.get("GENERAL", "EXPERIMENT_NAME"))
    mlflow_logger = MLflowLogger(
        tracking_uri=config.get("GENERAL", "MLRUN_PATH"))
    # param記入
    mlflow_logger.log_params({
        "random_seed": config.getint("GENERAL", "RANDOM_SEED"),
        "image size": config.getint("AUGMENTATION", "IMAGE_SIZE"),
        "batch size": config.getint("LEARNING", "BATCH_SIZE"),
        "model": model.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "train size": train_size,
        "early stopping patience": config.getint("LEARNING", "PATIENCE"),
        "device name": f"{torch.cuda.get_device_name()}" if torch.cuda.is_available() else device,
        "pytorch version": torch.__version__,
        "ignite version": ignite.__version__,
        "cuda version": torch.version.cuda,
    })

    # イテレーションごとに損失関数を記録する
    mlflow_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="training",
        output_transform=lambda loss: {"batchloss": loss}
    )

    @ trainer.on(Events.EPOCH_COMPLETED)
    def log_training_result(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        avg_fscore = metrics["f_score_05"]
        avg_loss = metrics["loss"]
        pbar.log_message(
            f"Training Results - Epoch: {engine.state.epoch} Avg f-score: {avg_fscore:.2f} Avg loss: {avg_loss:.2f}")

    @ trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_result(engine):
        valid_evaluator.run(val_loader)
        metrics = valid_evaluator.state.metrics
        avg_fscore = metrics["f_score_05"]
        avg_loss = metrics["loss"]
        pbar.log_message(
            f"Validation Results - Epoch: {engine.state.epoch} Avg f-score: {avg_fscore:.2f} Avg loss: {avg_loss:.2f}")
        pbar.n = pbar.last_print_n = 0

    # エポック終了時損失関数と精度を記録する
    for tag, evaluator in [("training", train_evaluator), ("validation", valid_evaluator)]:
        mlflow_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=["loss", "f_score_05"],
            global_step_transform=global_step_from_engine(trainer),
        )

    # Optimizerのパラメータを記録する
    mlflow_logger.attach_opt_params_handler(
        trainer, event_name=Events.ITERATION_COMPLETED(every=config.getint("LOG", "interval")), optimizer=optimizer
    )

    # lrの更新
    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    # Early Stoppingの実装
    handler = EarlyStopping(
        patience=config.getint("LEARNING", "PATIENCE"), score_function=score_function, trainer=trainer)
    valid_evaluator.add_event_handler(Events.COMPLETED, handler)

    # Checkpointの保存
    # 計算時間が長いならコメントアウト
    # handler = ModelCheckpoint(
    #     "../model/", "", n_saved=2, create_dir=True, save_as_state_dict=True)
    # trainer.add_event_handler(Events.EPOCH_COMPLETED(
    #     every=2), handler, {"mymodel": model})

    # 学習
    trainer.run(train_loader, max_epochs=config.getint("LEARNING", "epochs"))

    # Modelの保存
    mlflow.pytorch.log_model(model, model.__class__.__name__)
    mlflow.log_artifact("../configs/config.ini")
    model_info = str(summary(model, verbose=0))
    with open("../model/unet_schema.txt", "w") as f:
        f.write(model_info)

    mlflow_logger.close()


if __name__ == "__main__":

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
    if enable_cuda_flag:
        print(f"RUNNING ON GPU - {torch.cuda.get_device_name()}")
    else:
        print(f"RUNNING ON CPU")

    print("\n===Set Seed===\n")
    random.seed(config.getint("GENERAL", "RANDOM_SEED"))
    np.random.seed(config.getint("GENERAL", "RANDOM_SEED"))
    torch.manual_seed(config.getint("GENERAL", "RANDOM_SEED"))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(config.getint("GENERAL", "RANDOM_SEED"))
    print(f"Seed value: {config.getint('GENERAL', 'RANDOM_SEED')}")

    print("\n===Create Loggers===\n")
    logger = logging.getLogger("ignite.engine.engine.Engine")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    print(f"\n===Start Training===\n")
    t_history, v_history = train(g)

    print(f"\n===Training Finished===\n")
