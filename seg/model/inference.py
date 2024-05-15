from PIL import Image
import numpy as np

import torch
import torchvision.transforms.functional as TF

from .utils import tensor_map
from .metrics import PixelAccuracy, ClassIoU
from .processing import ground_sky_filtering

from seg.data.data_loader import seg_preprocessing

class Predictor():
    def __init__(self, model, half_precision=False, eval_mode=False, model_name='wodis'):
        self.model = model
        self.half_precision = half_precision
        self.resize = False if 'wodis' in model_name.lower() else True

        use_gpu = torch.cuda.is_available()
        self.device = torch.device('cuda:0') if use_gpu else torch.device('cpu')

        if self.half_precision:
            self.model = self.model.half()
        self.model = self.model.eval().to(self.device)
        
        if eval_mode:
            # Metrics
            num_classes = 3
            self.accuracy = PixelAccuracy(num_classes)
            self.iou_0 = ClassIoU(0, num_classes)
            self.iou_1 = ClassIoU(1, num_classes)
            self.iou_2 = ClassIoU(2, num_classes)

    def predict_batch(self, batch, ood_mode=False):

        map_fn = lambda t: t.to(self.device)
        batch = tensor_map(batch, map_fn)

        with torch.no_grad():
            if self.half_precision:
                with torch.cuda.amp.autocast():
                    res = self.model(batch)
            else:
                res = self.model(batch)

        out = res['out'].detach()

        if self.resize:
            size = (batch['image'].size(2), batch['image'].size(3))
            out = TF.resize(out, size, interpolation=Image.BILINEAR)  
        out_class = out.argmax(1)
        out_class = out_class.byte().cpu().numpy()

        if ood_mode:
            aux_feats = res['aux'].detach()
            return out_class, aux_feats
        else:
            return out_class

    def evaluate_batch(self, features):

        map_fn = lambda t: t.to(self.device)
        features = tensor_map(features, map_fn)

        with torch.no_grad():
            if self.half_precision:
                with torch.cuda.amp.autocast():
                    res = self.model(features)
            else:
                res = self.model(features)

        out = res['out'].detach()
        
        # Metrics
        if self.resize:
            labels_size = (features['segmentation'].size(2), features['segmentation'].size(3))
            out = TF.resize(out, labels_size, interpolation=Image.BILINEAR)
        preds = out.argmax(1).cpu()
        
        # Create hard labels from soft
        labels_hard = features['segmentation'].argmax(1)
        ignore_mask = features['segmentation'].sum(1) < 0.9
        labels_hard = labels_hard * ~ignore_mask + 4 * ignore_mask
        labels_hard = labels_hard.cpu()

        self.accuracy(preds, labels_hard)
        self.iou_0(preds, labels_hard)
        self.iou_1(preds, labels_hard)
        self.iou_2(preds, labels_hard)

        return {'accuracy':self.accuracy.compute(), 'iou_obstacle':self.iou_0.compute(), 
                'iou_water':self.iou_1.compute(), 'iou_sky':self.iou_2.compute()}

    def bbox_filtering(self, im0s, img_size=(640, 640), stride=32):
        if isinstance(im0s, list):
            if len(im0s[0].shape) == 3:
                im0_shape = im0s[0].shape[:2]
            else:
                im0_shape = im0s[0].shape[1:3]
        else:
            if len(im0s.shape) == 3:
                im0s = np.expand_dims(im0s, axis=0)
            im0_shape = im0s.shape[1:3]

        # Preprocessing for Segmentation
        img = []
        for im0 in im0s:
            img.append(seg_preprocessing(im0, img_size=img_size, stride=stride))

        # Segmentation Inference
        seg_preds = self.predict_batch({'image': torch.stack(img, dim=0)})

        # Binary Mask (Ground&Sky pixels = 1 / other pixels = 0)
        ground_sky_bin = []
        for seg_pred in seg_preds:
            ground_sky_bin.append(ground_sky_filtering(seg_pred, im0_shape))
        
        return ground_sky_bin