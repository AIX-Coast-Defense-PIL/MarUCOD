import cv2
import numpy as np
import torch

from yolov7.yoloUtils.general import xyxy2xywh, xywh2xyxy, scale_coords
from ood.ood_scores import calc_distance_score

def yolo_ood_classifier(yolo_preds, seg_model, filter_masks, fe_model, train_distribution, ood_thres, 
                        img, im0, score_metric, filter_thres=0.7, patch_size=16):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0

    for b_idx, (yolo_pred, filter_mask) in enumerate(zip(yolo_preds, filter_masks)):  # per image
        if yolo_pred is not None and len(yolo_pred):
            device = yolo_pred.device
            yolo_pred = yolo_pred.clone()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], yolo_pred[:, :4], im0[b_idx].shape)
            
            # Classes
            ims = []
            unfiltered_idx = []
            for idx, bbox in enumerate(yolo_pred):  # per item
                if seg_model != 'None': # yolo predictions(bbox) Filtering
                    bbox_sum = filter_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])].sum()
                    bbox_area = (int(bbox[3])-int(bbox[1])) * (int(bbox[2])-int(bbox[0])) + 1e-15
                    if bbox_sum/bbox_area >= filter_thres:
                        continue

                # Reshape and pad cutouts
                b = xyxy2xywh(bbox)  # boxes
                b[2:] = max(b[2:]) * 1.3 + 30  # pad
                bbox = xywh2xyxy(b).long()
                bbox[0::2] = torch.clamp(bbox[0::2], 0, im0[b_idx].shape[1])
                bbox[1::2] = torch.clamp(bbox[1::2], 0, im0[b_idx].shape[0])

                cutout = im0[b_idx][int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                im = cv2.resize(cutout, (patch_size, patch_size))  # BGR

                im = im[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                im = np.ascontiguousarray(im, dtype=np.float32)  # uint8 to float32
                im /= 255.0  # 0 - 255 to 0.0 - 1.0
                ims.append(im)
                unfiltered_idx.append(idx)
            
            ood_scr = torch.Tensor([-1]*len(yolo_preds[b_idx]))
            ood_cls = torch.Tensor([2]*len(yolo_preds[b_idx]))
            if len(ims):
                # ood classify
                feats = fe_model(torch.Tensor(ims).to(device))  # classifier prediction
                feats = feats.data.cpu().numpy()

                ood_scores = calc_distance_score(train_distribution, feats, score_metric)
                ood_scr[unfiltered_idx] = torch.Tensor(ood_scores)
                ood_cls[unfiltered_idx] = torch.Tensor([0 if ood_score > ood_thres else 1 for ood_score in ood_scores])
            
            ood_scr = torch.unsqueeze(ood_scr, 1).to(device)
            ood_cls = torch.unsqueeze(ood_cls, 1).to(device)
            yolo_preds[b_idx] = torch.cat([yolo_preds[b_idx], ood_scr, ood_cls], dim=1)

    return yolo_preds