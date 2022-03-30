"""
    KHPA classification dataset.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet.gluon.data import Dataset
from imgaug import augmenters as iaa
from imgaug import parameters as iap


class KHPA(Dataset):
    """
    Load the KHPA classification dataset.

    Parameters:
    ----------
    root : str, default '~/.mxnet/datasets/imagenet'
        Path to the folder stored the dataset.
    train : bool, default True
        Whether to load the training or validation set.
    """
    def __init__(self,
                 root=os.path.join("~", ".mxnet", "datasets", "khpa"),
                 split_file_path=os.path.join("~", ".mxnet", "datasets", "khpa", "split.csv"),
                 generate_split=False,
                 num_split_folders=10,
                 working_split_folder_ind1=1,
                 stats_file_path=os.path.join("~", ".mxnet", "datasets", "khpa", "stats.json"),
                 generate_stats=False,
                 num_classes=28,
                 preproc_resize_image_size=(256, 256),
                 model_input_image_size=(224, 224),
                 train=True):
        super(KHPA, self).__init__()
        self.suffices = ("red", "green", "blue", "yellow")

        root_dir_path = os.path.expanduser(root)
        assert os.path.exists(root_dir_path)

        train_file_name = "train.csv"
        train_file_path = os.path.join(root_dir_path, train_file_name)
        if not os.path.exists(train_file_path):
            raise Exception("Train file doesn't exist: {}".format(train_file_path))

        images_dir_path = os.path.join(root_dir_path, "train")
        if not os.path.exists(images_dir_path):
            raise Exception("Train image directory doesn't exist: {}".format(images_dir_path))

        train_df = pd.read_csv(
            train_file_path,
            sep=",",
            index_col=False,
            dtype={"Id": np.unicode, "Target": np.unicode})
        train_file_ids = train_df["Id"].values.astype(np.unicode)
        train_file_labels = train_df["Target"].values.astype(np.unicode)

        image_count = len(train_file_ids)

        if os.path.exists(split_file_path):
            if generate_split:
                logging.info("Split file already exists: {}".format(split_file_path))

            slice_df = pd.read_csv(
                split_file_path,
                sep=",",
                index_col=False,
            )
            categories = slice_df["Folder{}".format(working_split_folder_ind1)].values.astype(np.uint8)
        else:
            if not generate_split:
                raise Exception("Split file doesn't exist: {}".format(split_file_path))

            label_position_lists, label_counts = self.calc_label_position_lists(
                train_file_labels=train_file_labels,
                num_classes=num_classes)
            assert (num_split_folders <= label_counts.min())
            unique_label_position_lists, unique_label_counts = self.calc_unique_label_position_lists(
                label_position_lists=label_position_lists,
                label_counts=label_counts)
            assert (image_count == unique_label_counts.sum())
            dataset_folder_table = self.create_dataset_folder_table(
                num_samples=image_count,
                num_folders=num_split_folders,
                unique_label_position_lists=unique_label_position_lists)
            assert (image_count == dataset_folder_table.sum())

            slice_df_dict = {"Id": train_file_ids}
            slice_df_dict.update({"Folder{}".format(i + 1): dataset_folder_table[i]
                                  for i in range(num_split_folders)})

            slice_df = pd.DataFrame(slice_df_dict)

            slice_df.to_csv(
                split_file_path,
                sep=',',
                index=False)

            categories = slice_df["Folder{}".format(working_split_folder_ind1)].values.astype(np.uint8)

        if os.path.exists(stats_file_path):
            if generate_stats:
                logging.info("Stats file already exists: {}".format(stats_file_path))

            with open(stats_file_path, "r") as f:
                stats_dict = json.load(f)

            mean_rgby = np.array(stats_dict["mean_rgby"], np.float32)
            std_rgby = np.array(stats_dict["std_rgby"], np.float32)
            label_counts = np.array(stats_dict["label_counts"], np.int32)
        else:
            if not generate_split:
                raise Exception("Stats file doesn't exist: {}".format(stats_file_path))

            label_counts = self.calc_label_counts(train_file_labels, num_classes)
            mean_rgby, std_rgby = self.calc_image_widths(train_file_ids, self.suffices, images_dir_path)
            stats_dict = {
                "mean_rgby": [float(x) for x in mean_rgby],
                "std_rgby": [float(x) for x in std_rgby],
                "label_counts": [int(x) for x in label_counts],
            }
            with open(stats_file_path, 'w') as f:
                json.dump(stats_dict, f)

        self.label_widths = self.calc_label_widths(label_counts, num_classes)

        self.mean_rgby = mean_rgby
        self.std_rgby = std_rgby

        mask = (categories == (0 if train else 1))
        self.train_file_ids = train_file_ids[mask]
        list_labels = train_file_labels[mask]

        self.images_dir_path = images_dir_path
        self.num_classes = num_classes
        self.train = train
        self.onehot_labels = self.calc_onehot_labels(
            num_classes=num_classes,
            list_labels=list_labels)

        if train:
            self._transform = KHPATrainTransform(
                mean=self.mean_rgby,
                std=self.std_rgby,
                crop_image_size=model_input_image_size)
            self.sample_weights = self.calc_sample_weights(
                label_widths=self.label_widths,
                list_labels=list_labels)
        else:
            self._transform = KHPAValTransform(
                mean=self.mean_rgby,
                std=self.std_rgby,
                resize_image_size=preproc_resize_image_size,
                crop_image_size=model_input_image_size)

    def __str__(self):
        return self.__class__.__name__ + "({})".format(len(self.train_file_ids))

    def __len__(self):
        return len(self.train_file_ids)

    def __getitem__(self, idx):
        image_prefix = self.train_file_ids[idx]
        image_prefix_path = os.path.join(self.images_dir_path, image_prefix)

        imgs = []
        for suffix in self.suffices:
            image_file_path = "{}_{}.png".format(image_prefix_path, suffix)
            img = mx.image.imread(image_file_path, flag=0)
            imgs += [img]
        img = mx.nd.concat(*imgs, dim=2)

        label = mx.nd.array(self.onehot_labels[idx])

        if self._transform is not None:
            img, label = self._transform(img, label)
        return img, label

    @staticmethod
    def calc_onehot_labels(num_classes, list_labels):
        num_samples = len(list_labels)
        onehot_labels = np.zeros((num_samples, num_classes), np.int32)
        for i, train_file_label in enumerate(list_labels):
            label_str_list = train_file_label.split()
            for label_str in label_str_list:
                label_int = int(label_str)
                onehot_labels[i, label_int] = 1
        return onehot_labels

    @staticmethod
    def calc_sample_weights(label_widths, list_labels):
        label_widths1 = label_widths / label_widths.sum()
        num_samples = len(list_labels)
        sample_weights = np.zeros((num_samples, ), np.float64)
        for i, train_file_label in enumerate(list_labels):
            label_str_list = train_file_label.split()
            for label_str in label_str_list:
                label_int = int(label_str)
                # sample_weights[i] += label_widths1[label_int]
                sample_weights[i] = max(sample_weights[i], label_widths1[label_int])
        assert (sample_weights.min() > 0.0)
        sample_weights /= sample_weights.sum()
        sample_weights = sample_weights.astype(np.float32)
        return sample_weights

    @staticmethod
    def calc_label_position_lists(train_file_labels, num_classes):
        label_counts = np.zeros((num_classes, ), np.int32)
        label_position_lists = [[] for _ in range(num_classes)]
        for sample_ind, train_file_label in enumerate(train_file_labels):
            label_str_list = train_file_label.split()
            for label_str in label_str_list:
                label_int = int(label_str)
                assert (0 <= label_int < num_classes)
                label_counts[label_int] += 1
                label_position_lists[label_int] += [sample_ind]
        assert ([len(x) for x in label_position_lists] == list(label_counts))
        return label_position_lists, label_counts

    @staticmethod
    def calc_unique_label_position_lists(label_position_lists, label_counts):
        unique_label_position_lists = label_position_lists.copy()
        unique_label_counts = label_counts.copy()
        order_inds = np.argsort(label_counts)
        for i, class_ind_i in enumerate(order_inds):
            for sample_ind in unique_label_position_lists[class_ind_i]:
                for class_ind_k in order_inds[(i + 1):]:
                    if sample_ind in unique_label_position_lists[class_ind_k]:
                        unique_label_position_lists[class_ind_k].remove(sample_ind)
                        unique_label_counts[class_ind_k] -= 1
        assert ([len(x) for x in unique_label_position_lists] == list(unique_label_counts))
        return unique_label_position_lists, unique_label_counts

    @staticmethod
    def create_dataset_folder_table(num_samples, num_folders, unique_label_position_lists):
        dataset_folder_table = np.zeros((num_folders, num_samples), np.uint8)
        for label_position_list in unique_label_position_lists:
            label_positions = np.array(label_position_list)
            np.random.shuffle(label_positions)
            split_list = np.array_split(label_positions, indices_or_sections=num_folders)
            for folder_ind, folder_split_list in enumerate(split_list):
                dataset_folder_table[folder_ind, folder_split_list] = 1
        return dataset_folder_table

    @staticmethod
    def calc_label_counts(train_file_labels, num_classes):
        label_counts = np.zeros((num_classes, ), np.int32)
        for train_file_label in train_file_labels:
            label_str_list = train_file_label.split()
            for label_str in label_str_list:
                label_int = int(label_str)
                assert (0 <= label_int < num_classes)
                label_counts[label_int] += 1
        return label_counts

    @staticmethod
    def calc_label_widths(label_counts, num_classes):
        total_label_count = label_counts.sum()
        label_widths = (1.0 / label_counts) / num_classes * total_label_count
        return label_widths

    @staticmethod
    def calc_image_widths(train_file_ids, suffices, images_dir_path):
        logging.info("Calculating image widths...")
        mean_rgby = np.zeros((len(suffices),), np.float32)
        std_rgby = np.zeros((len(suffices),), np.float32)
        for i, suffix in enumerate(suffices):
            logging.info("Processing suffix: {}".format(suffix))
            imgs = []
            for image_prefix in train_file_ids:
                image_prefix_path = os.path.join(images_dir_path, image_prefix)
                image_file_path = "{}_{}.png".format(image_prefix_path, suffix)
                img = mx.image.imread(image_file_path, flag=0).asnumpy()
                imgs += [img]
            imgs = np.concatenate(tuple(imgs), axis=2).flatten()
            mean_rgby[i] = imgs.mean()
            imgs = imgs.astype(np.float32, copy=False)
            imgs -= mean_rgby[i]
            imgs **= 2
            std = np.sqrt(imgs.mean() * len(imgs) / (len(imgs) - 1))
            std_rgby[i] = std
            logging.info("i={}, mean={}, std={}".format(i, mean_rgby[i], std_rgby[i]))
        return mean_rgby, std_rgby


class KHPATrainTransform(object):
    def __init__(self,
                 mean=(0.0, 0.0, 0.0, 0.0),
                 std=(1.0, 1.0, 1.0, 1.0),
                 crop_image_size=(224, 224)):
        if isinstance(crop_image_size, int):
            crop_image_size = (crop_image_size, crop_image_size)
        self._mean = mean
        self._std = std
        self.crop_image_size = crop_image_size

        self.seq = iaa.Sequential(
            children=[
                iaa.Sequential(
                    children=[
                        iaa.Fliplr(
                            p=0.5,
                            name="Fliplr"),
                        iaa.Flipud(
                            p=0.5,
                            name="Flipud"),
                        iaa.Sequential(
                            children=[
                                iaa.Affine(
                                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                                    rotate=(-45, 45),
                                    shear=(-16, 16),
                                    order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                                    mode="reflect",
                                    name="Affine"),
                                iaa.Sometimes(
                                    p=0.01,
                                    then_list=iaa.PiecewiseAffine(
                                        scale=(0.0, 0.01),
                                        nb_rows=(4, 20),
                                        nb_cols=(4, 20),
                                        order=iap.Choice([0, 1, 3], p=[0.15, 0.80, 0.05]),
                                        mode="reflect",
                                        name="PiecewiseAffine"))],
                            random_order=True,
                            name="GeomTransform"),
                        iaa.Sequential(
                            children=[
                                iaa.Sometimes(
                                    p=0.75,
                                    then_list=iaa.Add(
                                        value=(-10, 10),
                                        per_channel=0.5,
                                        name="Brightness")),
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.Emboss(
                                        alpha=(0.0, 0.5),
                                        strength=(0.5, 1.2),
                                        name="Emboss")),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.Sharpen(
                                        alpha=(0.0, 0.5),
                                        lightness=(0.5, 1.2),
                                        name="Sharpen")),
                                iaa.Sometimes(
                                    p=0.25,
                                    then_list=iaa.ContrastNormalization(
                                        alpha=(0.5, 1.5),
                                        per_channel=0.5,
                                        name="ContrastNormalization"))
                            ],
                            random_order=True,
                            name="ColorTransform"),
                        iaa.Sequential(
                            children=[
                                iaa.Sometimes(
                                    p=0.5,
                                    then_list=iaa.AdditiveGaussianNoise(
                                        loc=0,
                                        scale=(0.0, 10.0),
                                        per_channel=0.5,
                                        name="AdditiveGaussianNoise")),
                                iaa.Sometimes(
                                    p=0.1,
                                    then_list=iaa.SaltAndPepper(
                                        p=(0, 0.001),
                                        per_channel=0.5,
                                        name="SaltAndPepper"))],
                            random_order=True,
                            name="Noise"),
                        iaa.OneOf(
                            children=[
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.MedianBlur(
                                        k=3,
                                        name="MedianBlur")),
                                iaa.Sometimes(
                                    p=0.05,
                                    then_list=iaa.AverageBlur(
                                        k=(2, 4),
                                        name="AverageBlur")),
                                iaa.Sometimes(
                                    p=0.5,
                                    then_list=iaa.GaussianBlur(
                                        sigma=(0.0, 2.0),
                                        name="GaussianBlur"))],
                            name="Blur"),
                    ],
                    random_order=True,
                    name="MainProcess")])

    def __call__(self, img, label):

        # import cv2
        # cv2.imshow(winname="src_img1", mat=img.asnumpy()[:, :, :3])
        # cv2.imshow(winname="src_img2", mat=img.asnumpy()[:, :, 1:])

        seq_det = self.seq.to_deterministic()
        imgs_aug = img.asnumpy().copy()
        # imgs_aug = seq_det.augment_images(img.asnumpy().transpose((2, 0, 1)))
        imgs_aug[:, :, :3] = seq_det.augment_image(imgs_aug[:, :, :3])
        imgs_aug[:, :, 3:] = seq_det.augment_image(imgs_aug[:, :, 3:])
        # img_np = imgs_aug.transpose((1, 2, 0))
        img_np = imgs_aug

        # cv2.imshow(winname="dst_img1", mat=img_np[:, :, :3])
        # cv2.imshow(winname="dst_img2", mat=img_np[:, :, 1:])
        # cv2.waitKey(0)

        img_np = img_np.astype(np.float32)
        img_np = (img_np - self._mean) / self._std
        img = mx.nd.array(img_np, ctx=img.context)
        img = mx.image.random_size_crop(
            src=img,
            size=self.crop_image_size,
            area=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interp=1)[0]
        img = img.transpose((2, 0, 1))
        return img, label


class KHPAValTransform(object):
    def __init__(self,
                 mean=(0.0, 0.0, 0.0, 0.0),
                 std=(1.0, 1.0, 1.0, 1.0),
                 resize_image_size=(256, 256),
                 crop_image_size=(224, 224)):
        if isinstance(crop_image_size, int):
            crop_image_size = (crop_image_size, crop_image_size)
        self._mean = mean
        self._std = std
        self.resize_image_size = resize_image_size
        self.crop_image_size = crop_image_size

    def __call__(self, img, label):
        h, w, _ = img.shape
        if h > w:
            wsize = self.resize_image_size
            hsize = int(h * wsize / w)
        else:
            hsize = self.resize_image_size
            wsize = int(w * hsize / h)
        img = mx.image.imresize(
            src=img,
            w=wsize,
            h=hsize,
            interp=1)
        img = mx.image.center_crop(
            src=img,
            size=self.crop_image_size,
            interp=1)[0]
        img = img.astype(np.float32)
        img = (img - mx.nd.array(self._mean, ctx=img.context)) / mx.nd.array(self._std, ctx=img.context)
        img = img.transpose((2, 0, 1))
        return img, label


class KHPAMetaInfo(object):
    label = "KHPA"
    root_dir_name = "khpa"
    dataset_class = KHPA
    num_training_samples = None
    in_channels = 4
    num_classes = 56
    input_image_size = (224, 224)
