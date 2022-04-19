import pathlib

import cv2
import numpy as np


def create_imgcolor(train_0_vh_path, train_0_vv_path, train_1_vh_path, train_1_vv_path, anno_path, color_path):

    # tifを表示するにはcv2.IMREAD_ANYDEPTHが必要
    train_0_vh = cv2.imread(train_0_vh_path, cv2.IMREAD_ANYDEPTH)
    train_0_vv = cv2.imread(train_0_vv_path, cv2.IMREAD_ANYDEPTH)
    train_1_vh = cv2.imread(train_1_vh_path, cv2.IMREAD_ANYDEPTH)
    train_1_vv = cv2.imread(train_1_vv_path, cv2.IMREAD_ANYDEPTH)

    # 一覧画像
    # train_0_merge = np.hstack((train_0_vh, train_0_vv))
    # train_1_merge = np.hstack((train_1_vh, train_1_vv))
    # train_merge = np.vstack((train_0_merge, train_1_merge))
    # train_merge_resize = cv2.resize(train_merge, dsize=None, fx=0.5, fy=0.5)

    # vh,vvの差分をカラーとして合成した写真に対してアノテーションを表示させる。
    train_vh_diff = (train_1_vh - train_0_vh) / \
        (train_1_vh - train_0_vh).max()*255
    train_vv_diff = (train_1_vv - train_0_vv) / \
        (train_1_vv - train_0_vv).max()*255

    anno = cv2.imread(anno_path, cv2.IMREAD_GRAYSCALE)
    anno_converted = np.where(anno == 1, np.float32(255), np.float32(0))

    # vv:b anno:g vh:r
    img_bgr = cv2.merge((train_vv_diff, anno_converted, train_vh_diff))
    # float画像はimshowでしか表示できない
    img_bgr = np.clip(img_bgr * 255, 0, 255).astype(np.uint8)
    # cv2.imwrite(color_path, img_bgr)

    # visualization
    img_sobel = cv2.Sobel(anno, cv2.CV_32F, 1, 0, ksize=3)
    cv2.imshow('image', img_sobel)
    # cv2.imshow('image', train_vh_diff)
    # cv2.imshow('image', train_vv_diff)
    # cv2.imshow('image', anno_converted)
    # cv2.imshow('image', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindo()


if __name__ == "__main__":
    for i in range(0, 28):
        file_suffix = str(i).zfill(2)
        train_0_vh_path = f"../dataset/train_images/train_{file_suffix}/0_VH.tif"
        train_0_vv_path = f"../dataset/train_images/train_{file_suffix}/0_VV.tif"
        train_1_vh_path = f"../dataset/train_images/train_{file_suffix}/1_VH.tif"
        train_1_vv_path = f"../dataset/train_images/train_{file_suffix}/1_VV.tif"
        anno_name_path = f"../dataset/train_annotations/train_{file_suffix}.png"
        color_img_path = f"../dataset/color/color_{file_suffix}.png"

        create_imgcolor(train_0_vh_path, train_0_vv_path, train_1_vh_path,
                        train_1_vv_path, anno_name_path, color_img_path)
        break
