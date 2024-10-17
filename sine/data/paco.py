import logging
import os
import pickle
import copy
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BitMasks, Boxes, Instances, BoxMode
from detectron2.data import detection_utils as utils, transforms as T

from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

import pycocotools.mask as mask_util

logger = logging.getLogger(__name__)

ATTR_TYPE_END_IDXS = [0, 30, 41, 55, 59]
ATTR_TYPE_BG_IDXS = [29, 38, 54, 58]

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = mask_util.frPyObjects(polygons, height, width)
        mask = mask_util.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def transform_instance_annotations(
    annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Borrowed from https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/detection_utils.py#L257
    with support for attributes
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(
        annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS
    )
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray]"
                ", COCO-style RLE as a dict.".format(type(segm))
            )

    # support for attributes
    if "attr_labels" in annotation:
        attr_label_tensor = np.zeros(ATTR_TYPE_END_IDXS[-1])
        attr_ignore_tensor = np.zeros(len(ATTR_TYPE_END_IDXS) - 1)
        for _a in annotation["attr_labels"]:
            attr_label_tensor[_a] = 1.0

        for _aid, _a in enumerate(annotation["attr_ignores"]):
            attr_ignore_tensor[_aid] = _a

        for attr_type_id in range(1, len(ATTR_TYPE_END_IDXS)):
            st_idx = ATTR_TYPE_END_IDXS[attr_type_id - 1]
            end_idx = ATTR_TYPE_END_IDXS[attr_type_id]
            bg_idx = ATTR_TYPE_BG_IDXS[attr_type_id - 1]
            attr_label_tensor[st_idx:end_idx] /= max(
                attr_label_tensor[st_idx:end_idx].sum(), 1.0
            )
            attr_label_tensor[bg_idx] = 1.0 - (
                attr_label_tensor[st_idx:end_idx].sum() - attr_label_tensor[bg_idx]
            )

        annotation["attr_label_tensor"] = attr_label_tensor
        annotation["attr_ignore_tensor"] = attr_ignore_tensor

    return annotation

class PACODataset(torch.utils.data.Dataset):

    def __init__(
        self,
        image_size,
        root='datasets',
        dataset_name='paco_lvis_v1_train',
        is_train=True,
        crop_ratio=1.0,
        tfm_gens_crop_pair=None,
        tfm_gens_sel_pair=None,
        dino_transform=None,
        img_format='RGB',
        serialize=True,
    ):

        self.is_train = is_train
        assert is_train, "PACODataset only used in training"
        self._serialize = serialize

        self.root_ = root
        self.dataset_name = dataset_name
        coco_ins_meta = MetadataCatalog.get(dataset_name)
        coco_ins_data = {item['file_name']: item for item in DatasetCatalog.get(dataset_name)}

        self.catid2img = self.load_catid2img(coco_ins_data)
        self.class_ids = list(self.catid2img.keys())

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        if self._serialize:
            coco_ins_data = {k: _serialize(v) for k, v in coco_ins_data.items()}

        self.coco_ins_data = coco_ins_data

        self.ignore_label = coco_ins_meta.get('ignore_label')

        self.crop_ratio = crop_ratio
        self.tfm_gens_crop_pair = tfm_gens_crop_pair
        self.tfm_gens_sel_pair = tfm_gens_sel_pair
        self.dino_transform = dino_transform

        self.img_format = img_format

        self.image_size = image_size


    def load_catid2img(self, coco_ins_data):
        if not os.path.exists(os.path.join(self.root_, 'sine_pkls', f'{self.dataset_name}_catid2img.pkl')):
            if not os.path.exists(os.path.join(self.root_, 'sine_pkls')) :os.makedirs(os.path.join(self.root_, 'sine_pkls'))
            catid2img = {}
            for item in coco_ins_data.values():
                for anno in item['annotations']:
                    if anno['category_id'] not in catid2img:
                        catid2img[anno['category_id']] = []
                    if item['file_name'] not in catid2img[anno['category_id']]:
                        catid2img[anno['category_id']].append(item['file_name'])
            with open(os.path.join(self.root_, 'sine_pkls', f'{self.dataset_name}_catid2img.pkl'), 'wb') as file:
                pickle.dump(catid2img, file)
        else:
            with open(os.path.join(self.root_, 'sine_pkls', f'{self.dataset_name}_catid2img.pkl'), 'rb') as file:
                catid2img = pickle.load(file)
        return catid2img


    def __len__(self):
        return len(self.coco_ins_data)

    def _rand_range(self, low=1.0, high=None, size=None):
        """
        Uniform float random number between low and high.
        """
        if high is None:
            low, high = 0, low
        if size is None:
            size = []
        return np.random.uniform(low, high, size)

    def get_ref_tar_dict(self, class_sample, crop_pair_flag=True):

        def load_dict(file):
            dict = self.coco_ins_data[file]
            if self._serialize:
                dict = memoryview(dict)
                dict = pickle.loads(dict)
            return dict

        if len(self.catid2img[class_sample]) < 2: crop_pair_flag = True
        ref_file = np.random.choice(self.catid2img[class_sample], 1, replace=False)[0]
        ref_dict = load_dict(ref_file)
        if crop_pair_flag:
            tar_dict = copy.deepcopy(ref_dict)
        else:
            while True:
                tar_file = np.random.choice(self.catid2img[class_sample], 1, replace=False)[0]
                if ref_file != tar_file: break
            tar_dict = load_dict(tar_file)

        return ref_dict, tar_dict

    def pad_img(self, x, pad_size):

        assert isinstance(x, torch.Tensor)
        # Pad
        h, w = x.shape[-2:]
        padh = pad_size - h
        padw = pad_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def process_img_dict(self, dataset_dict, crop_pair_flag, keep_cat_ids=None):

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        while True:
            try:
                image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(3)
            else:
                break

        utils.check_image_size(dataset_dict, image)

        if crop_pair_flag:
            tfm_gens = self.tfm_gens_crop_pair
        else:
            tfm_gens = self.tfm_gens_sel_pair

        image, transforms = T.apply_transform_gens(tfm_gens, image)
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        if self.dino_transform is None:
            dino_image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        else:
            dino_image = self.dino_transform(image)
        dataset_dict["image"] = self.pad_img(dino_image, self.image_size)

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annotations = dataset_dict.pop("annotations")
            # sort
            cat_ids = [info['category_id'] for info in annotations]
            annotations = [annotations[idx] for idx in sorted(range(len(cat_ids)), key=lambda k: cat_ids[k])]

            for anno in annotations:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                transform_instance_annotations(obj, transforms, image_shape)
                for obj in annotations if obj.get("iscrowd", 0) == 0 and obj['category_id'] in keep_cat_ids
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape, mask_format='bitmask')
            if len(annos) and "attr_label_tensor" in annos[0]:
                attr_label_tensor = torch.tensor(
                    [obj["attr_label_tensor"] for obj in annos]
                )
                attr_ignore_tensor = torch.tensor(
                    [obj["attr_ignore_tensor"] for obj in annos], dtype=torch.int64
                )
                instances.gt_attr_label_tensor = attr_label_tensor
                instances.gt_attr_ignore_tensor = attr_ignore_tensor
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            try:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            except:
                dataset_dict["instances"] = instances
                return dataset_dict

            ins_ids = [anno['id'] for anno in annos]
            ins_ids = np.array(ins_ids)
            instances.ins_ids = torch.tensor(ins_ids, dtype=torch.int64)
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks.tensor
                gt_masks = self.pad_img(gt_masks, self.image_size)
                instances.gt_masks = gt_masks

            dataset_dict["instances"] = instances

        return dataset_dict

    def __getitem__(self, idx):

        ref_mask_num = 0
        tar_mask_num = 0

        while ref_mask_num == 0 or tar_mask_num == 0:

            # sample category
            class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]

            #sample reference and target dataset dict
            crop_pair_flag = self._rand_range() < self.crop_ratio
            ref_dict, tar_dict = self.get_ref_tar_dict(class_sample, crop_pair_flag)
            ref_cat_ids = list(set([anno['category_id'] for anno in ref_dict['annotations']]))
            ref_dict = self.process_img_dict(ref_dict, crop_pair_flag, keep_cat_ids=ref_cat_ids)
            ref_cat_ids = list(set(ref_dict['instances'].gt_classes.tolist()))
            tar_dict = self.process_img_dict(tar_dict, crop_pair_flag, keep_cat_ids=ref_cat_ids)

            ref_mask_num = len(ref_dict['instances'])
            tar_mask_num = len(tar_dict['instances'])

        return {"ref_dict": ref_dict, "tar_dict": tar_dict}


if __name__ == '__main__':

    import argparse
    from detectron2.data import transforms as T
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import matcherv2.data

    parser = argparse.ArgumentParser('paco dataset', add_help=False)
    parser.add_argument('--random_flip', default="horizontal", type=str)
    parser.add_argument('--min_scale', default=0.1, type=float)
    parser.add_argument('--max_scale', default=2.0, type=float)
    parser.add_argument('--image_size', default=896, type=int)
    parser.add_argument('--sam_image_size', default=1024, type=int)
    args = parser.parse_args()

    # LSJ aug
    augmentation = []
    if args.random_flip != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=args.random_flip == "horizontal",
                vertical=args.random_flip == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=args.min_scale, max_scale=args.max_scale, target_height=args.image_size, target_width=args.image_size
        ),
        T.FixedSizeCrop(crop_size=(args.image_size, args.image_size), pad=False),
    ])

    dino_transform = transforms.Compose([
            MaybeToTensor(),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    dino_pixel_mean = [i*255 for i in [0.485, 0.456, 0.406]]
    dino_pixel_std = [i*255 for i in [0.229, 0.224, 0.225]]

    sam_transform = ResizeLongestSide(args.sam_image_size)
    sam_pixel_mean = [123.675, 116.28, 103.53]
    sam_pixel_std = [58.395, 57.12, 57.375]

    dataset = PACODataset(
        image_size=args.image_size,
        sam_image_size=args.sam_image_size,
        crop_ratio=0.5,
        tfm_gens_crop_pair=augmentation,
        tfm_gens_sel_pair=augmentation,
        dino_transform=dino_transform,
        sam_pixel_mean=sam_pixel_mean,
        sam_pixel_std=sam_pixel_std
    )

    show_size = (224, 224)

    for id in range(len(dataset)):

        if id > 100:
            break

        ref_dict, tar_dict = list(dataset[id].values())

        ref_dino_img = (ref_dict['image'] * torch.Tensor(dino_pixel_std).view(-1, 1, 1)) + torch.Tensor(
            dino_pixel_mean).view(-1, 1, 1)
        tar_dino_img = (tar_dict['image'] * torch.Tensor(dino_pixel_std).view(-1, 1, 1)) + torch.Tensor(
            dino_pixel_mean).view(-1, 1, 1)
        tar_sam_img = (tar_dict['sam_image'] * torch.Tensor(sam_pixel_std).view(-1, 1, 1)) + torch.Tensor(
            sam_pixel_mean).view(-1, 1, 1)

        ref_dino_img = F.interpolate(
            ref_dino_img[None, ...], show_size, mode="bilinear", align_corners=False, antialias=True
        )[0]
        tar_dino_img = F.interpolate(
            tar_dino_img[None, ...], show_size, mode="bilinear", align_corners=False, antialias=True
        )[0]
        tar_sam_img = F.interpolate(
            tar_sam_img[None, ...], show_size, mode="bilinear", align_corners=False, antialias=True
        )[0]

        ref_masks = ref_dict['instances'].gt_masks
        tar_masks = tar_dict['instances'].gt_masks
        ref_masks = F.interpolate(
            ref_masks[None, ...].float(), show_size
        )[0] > 0
        tar_masks = F.interpolate(
            tar_masks[None, ...].float(), show_size
        )[0] > 0

        ref_dino_img_np = ref_dino_img.permute(1,2,0).numpy()
        tar_dino_img_np = tar_dino_img.permute(1, 2, 0).numpy()
        tar_sam_img_np = tar_sam_img.permute(1, 2, 0).numpy()

        show_imgs = torch.cat([ref_dino_img, tar_dino_img, tar_sam_img], dim=-1).permute(1,2,0).numpy() /255.

        ref_dino_mask = []
        for mask in ref_masks:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            mask = mask.numpy()
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ref_dino_mask.append(mask_image)
        ref_dino_mask = sum(ref_dino_mask)

        tar_dino_mask = []
        for mask in tar_masks:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            mask = mask.numpy()
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
            tar_dino_mask.append(mask_image)
        tar_dino_mask = sum(tar_dino_mask)

        show_masks = np.concatenate([ref_dino_mask, tar_dino_mask, tar_dino_mask], axis=1)

        if not os.path.exists(f'shows_ins/paco'):
            os.makedirs(f'shows_ins/paco')

        save_path = f'shows_ins/paco/img{id}.jpg'
        plt.figure(figsize=(10, 10))
        plt.imshow(show_imgs)
        ax = plt.gca()
        ax.imshow(show_masks)
        plt.axis('off')
        plt.savefig(save_path)

