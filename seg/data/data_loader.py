import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import json
import cv2

from .transforms import get_image_resize

def read_mask(path):
    """Reads class segmentation mask from an image file."""
    mask = np.array(Image.open(path))

    # Masks stored in RGB channels or as class ids
    if mask.ndim == 3:
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = np.stack([mask==0, mask==1, mask==2], axis=-1).astype(np.float32)

    return mask

def read_image_list(path):
    """Reads image list from a file"""
    with open(path, 'r') as file:
        images = [line.strip() for line in file]
    return images

def get_image_list(image_dir):
    """Returns the list of images in the dir."""
    image_list = [os.path.splitext(img)[0] for img in os.listdir(image_dir)]
    return image_list

def refine_yolo_preds(yolo_preds, yolo_thres):
    len_yolo_preds = 0
    yolo_preds_dict = {}
    for yolo_pred in yolo_preds:
        image_id = yolo_pred['image_id']
        category_id = yolo_pred['category_id']
        bbox = yolo_pred['bbox']
        score = yolo_pred['score']
        if score >= yolo_thres:
            len_yolo_preds += 1
            if image_id in yolo_preds_dict:
                yolo_preds_dict[image_id].append([category_id, *bbox, score])
            else: yolo_preds_dict[image_id] = [[category_id, *bbox, score]]
    return yolo_preds_dict, len_yolo_preds

def seg_preprocessing(img, img_size=(640, 640), stride=32):
    # Image Resize
    resize_shape = letterbox_shape(img.shape[:2], new_shape=img_size, stride=stride)  # letterboxed shape
    data = get_image_resize(*resize_shape)({'image': img})
    img = data['image']

    return T.ToTensor()(img)

def letterbox_shape(ori_shape, new_shape=(640, 640), auto=True, scaleFill=False, scaleup=True, stride=32):
    # ori_shape: (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / ori_shape[0], new_shape[1] / ori_shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
    
    # Compute padding
    resize_shape = int(round(ori_shape[0] * r)), int(round(ori_shape[1] * r))
    dh, dw = new_shape[0] - resize_shape[0], new_shape[1] - resize_shape[1]  # hw padding
    if auto:  # minimum rectangle
        dh, dw = np.mod(dh, stride), np.mod(dw, stride)  # hw padding
    elif scaleFill:  # stretch
        dh, dw = 0.0, 0.0
        resize_shape = new_shape
    
    resize_shape = resize_shape[0] + dh, resize_shape[1] + dw  # (h, w)
    return resize_shape


class MaSTr1325Dataset(torch.utils.data.Dataset):
    """MaSTr1325 dataset wrapper

    Args:
        dataset_file (str): Path to the dataset configuration file.
        transform (optional): Tranform to apply to image and masks
        normalize_t (optional): Transform that normalizes the input image
        include_original (optional): Include original (non-normalized) version of the image in the features
    """
    def __init__(self, dataset_dir, yolo_preds_dir=None, transform=None, normalize_t=None, normalize_b = None, shift_b = None, include_original=False, sort=False, yolo_resize=(None, None), yolo_thres=0.05):

        # Set data directories
        self.image_dir = Path(dataset_dir).resolve()
        self.mask_dir = self.image_dir.parent / 'masks'
        if not self.mask_dir.exists():
            self.mask_dir = None 
        self.yolo_preds_dir = Path(yolo_preds_dir).resolve() if yolo_preds_dir is not None else None
        self.images = os.listdir(self.image_dir)
        
        # Load yolo predictions
        self.yolo_preds, self.len_yolo_preds = None, None
        if self.yolo_preds_dir is not None: 
            with open((self.yolo_preds_dir / 'yolov7_preds/yolov7_predictions.json').resolve(), "r") as json_file:
                self.yolo_preds = json.load(json_file)
            self.yolo_preds, self.len_yolo_preds = refine_yolo_preds(self.yolo_preds, yolo_thres)
            self.images = [fname for fname in self.images if os.path.splitext(fname)[0] in self.yolo_preds]
        
        if sort:
            self.images.sort()
            
        self.transform = transform
        self.normalize_t = normalize_t
        self.include_original = include_original
        self.img_size, self.stride = yolo_resize
        self.normalize_b = normalize_b
        self.shift_b = shift_b

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        img_path = str(self.image_dir / img_name)
        ext = os.path.splitext(img_name)[-1]
        mask_filename = img_name.replace(ext, 'm.png') if 'lars' not in str(self.image_dir).lower() else img_name.replace(ext, '.png')

        img = np.array(Image.open(img_path))
        img_original = img

        data = {'image': img} # shape: (h, w, ch)

        if self.mask_dir is not None:
            mask_path = str(self.mask_dir / mask_filename)
            mask = read_mask(mask_path)
            data['segmentation'] = mask
        
        # Transform images if transform is provided
        if self.transform is not None:
            data = self.transform(data)
            img = data['image']
        
        if self.img_size is not None:
            resize_shape = self.img_size
            # resize_shape = letterbox_shape(img.shape[:2], new_shape=self.img_size, stride=self.stride)  # letterboxed shape
            data = get_image_resize(*resize_shape)(data)
            img = data['image']

        if self.shift_b is not None:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            img_v_mean = np.mean(hsv_img[:,:,2])
            hsv_img[:,:,2] = np.clip(hsv_img[:,:,2].astype(np.uint64) - (img_v_mean - 157.566) ,0,255) # mastr+aihub mean = 157.5663538901869
            img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

        if self.normalize_b is not None:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv_img = self.normalize_b(hsv_img)
            hsv_img = np.transpose(hsv_img.numpy(), (1, 2, 0))
            img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB) 

        if self.normalize_t is not None:
            img = self.normalize_t(img)
        else:
            # Default: divide by 255
            img = TF.to_tensor(img)


        features = {
            'image': img,
            'img_name': img_name,
            'original_shape': img_original.shape[:2],
            }

        if self.include_original:
            features['image_original'] = torch.from_numpy(img_original.transpose(2,0,1))

        if 'segmentation' in data:
            features['segmentation'] = torch.from_numpy(data['segmentation'].transpose(2,0,1))
            features['mask_filename'] = mask_filename
        else:
            features['mask_filename'] = img_name

        if self.yolo_preds is not None:
            features['yolo_preds'] = np.array(self.yolo_preds[os.path.splitext(img_name)[0]])

        return features
