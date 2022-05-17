import torch
import torch.nn.functional as F
from torch import nn


class DiceBCELoss(nn.Module):
    """DiceBCELoss導出関数
    """

    def __init__(self, weight=None, size_average=True):
        """
        Args:
            weight (_type_, optional): _description_. Defaults to None.
            size_average (bool, optional): _description_. Defaults to True.
        """
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """順伝播処理関数

        Args:
            inputs (torch.Tensor): 推論結果テンソル
            targets (torch.Tensor): 正解テンソル
            smooth (int, optional): 逆伝播を行う際の関数平滑化用定数. Defaults to 1.

        Returns:
            float: loss値
        """

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
        """_summary_

        Args:
            beta (float, optional): F-scoreにおけるbeta値. Defaults to 1..
            threshold (float, optional): 確率をバイナリに変換する際の閾値. Defaults to 0.5.
        """
        super(FBetaScore, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, inputs, targets, const=1e-7):
        """順伝播処理関数

        Args:
            inputs (torch.Tensor): 推論結果テンソル
            targets (torch.Tensor): 正解テンソル
            const (float, optional): ゼロ除算対策定数. Defaults to 1e-7.

        Returns:
            float: F-score
        """
        # Binarize probablities
        inputs = torch.where(inputs < self.threshold, 0, 1)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        fbeta = ((1 + self.beta**2)*intersection) / \
            (inputs.sum() + self.beta**2 * targets.sum() + const)

        return fbeta
