
import torch
from torchvision import transforms

from torch.utils.data import Dataset
from torchvision.io import read_image

import os
import json

class YoloOutputDataset(Dataset):
    def __init__(self, data_dir_path, transform=None, threshold=0.05):
        json_path = os.path.join(data_dir_path + f'/yolov7_preds/yolov7_preds_refined.json')
        print('dataset path: ', json_path)

        with open(json_path, "r") as json_file:
            self.bbox_infos = json.load(json_file)
        
        self.bbox_infos = [ bbox_info for bbox_info in self.bbox_infos if bbox_info['score']>= threshold and int(bbox_info['image_id'])]
        self.img_dir = os.path.join(data_dir_path, 'images')
        self.transform = transform

    def __len__(self):
        return len(self.bbox_infos)

    def __getitem__(self, idx):
        img_id = self.bbox_infos[idx]["image_id"]
        img_name = img_id+'.jpg'
        bbox = self.bbox_infos[idx]["bbox"]
        label = self.bbox_infos[idx]["category_id"]

        img_path = os.path.join(self.img_dir, img_id+'.jpg')
        image = read_image(img_path)

        cutout = image[:, int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        if self.transform:
            cutout = self.transform(cutout)

        return cutout, img_name, label, bbox

def get_train_loader(data_dir_paths, batch_size, patch_size):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([patch_size, patch_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])  
    
    datasets = []
    for data_dir_path in data_dir_paths:
        dataset = YoloOutputDataset(data_dir_path, transform=transform)
        datasets.append(dataset)
    
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    data_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    return data_loader