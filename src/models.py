import torch
from torch import nn


# 以下URLを写経
# https://obgynai.com/unet-semantic-segmentation/
class UNet(nn.Module):
    """Unetクラス
    """

    def __init__(self, input_channels, num_classes):
        """

        Args:
            input_channels (int): 入力チャンネル数
            num_classes (int): 分類クラス数
        """
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
        """順伝播の処理

        Args:
            x (torch.Tensor): 入力層

        Returns:
            torch.Tensor: 0~1からなるテンソル
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

    Args:
        in_channels (int): 入力チャネル数
        out_channels (int): 出力チャネル数
        kernel_size (int, optional): カーネルサイズ. Defaults to 3.
        stride (int, optional): ストライド数. Defaults to 1.
        padding (int, optional): パディング数. Defaults to 1.

    Returns:
        torch.Tensor: 適用後テンソル
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
    """ダウンサンプリング層
    """
    return nn.MaxPool2d(2)


def up_pooling(in_channels, out_channels, kernel_size=2, stride=2):
    """アップサンプリング層

    Args:
        in_channels (int): 入力チャネル数
        out_channels (int): 出力チャネル数
        kernel_size (int, optional): カーネルサイズ. Defaults to 2.
        stride (int, optional): ストライド数. Defaults to 2.

    Returns:
        torch.Tensor
    """
    return nn.Sequential(
        # 転置畳み込み
        nn.ConvTranspose2d(in_channels, out_channels,
                           kernel_size=kernel_size, stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
