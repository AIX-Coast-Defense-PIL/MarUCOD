import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm
import json
import time

from seg.data.data_loader import MaSTr1325Dataset
from seg.data.transforms import PytorchHubNormalization, Normalization_B
from seg.model.wodis import WODIS_model
from seg.model.utils import load_weights
from seg.model.inference import Predictor


def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

DATASET_DIR = './data/images'
ADD_DATASET_DIR = './data2/images'
SEG_WEIGHT = './seg/weights/wodis_csol.ckpt'
BATCH_SIZE = 1
MODE = 'eval'
IMG_SIZE = [1024, 1280]


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WODIS Network Inference")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR, help="Dataset directory.")
    parser.add_argument("--add_dataset", type=str, default=None, help="Path to the test dataset directory.")
    parser.add_argument("--seg_weights", type=str, default=SEG_WEIGHT, help="Path to the model weights or a model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--fp16", action='store_true', help="Use half precision for inference.")
    parser.add_argument("--mode", type=str, default=MODE, help="Inference mode (pred: qualitative analysis, eval: qualitative & quantitative analysis)")
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=IMG_SIZE, help='inference size h,w')
    parser.add_argument("--normalize_t", action="store_true", help="ImageNet Normalization")
    parser.add_argument("--normalize_b", action="store_true", help="Brightness Normalization")
    parser.add_argument("--shift_b", action="store_true", help="Brightness Shift")
    return parser.parse_args()


def predict(args):
    
    normalize_t = PytorchHubNormalization() if args.normalize_t else None
    NORMALIZE_B = Normalization_B(mean_=0.519, std_=0.244) if args.normalize_b else None 
    ''' mastr1325 mean: 0.681, mastr1478 mean: 0.661, LaRS mean: 0.5244, 
        mastr1325 std: 0.2042, mastr1478 std: 0.226, LaRS std: 0.241 '''
    SHIFT_B = True if args.shift_b else None

    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    dataset = MaSTr1325Dataset(args.dataset_dir, normalize_t=normalize_t, normalize_b = NORMALIZE_B, shift_b = SHIFT_B, yolo_resize=(args.imgsz, 32))
    
    if args.add_dataset:
        add_ds = MaSTr1325Dataset(args.add_dataset, normalize_t=normalize_t, normalize_b = NORMALIZE_B, shift_b = SHIFT_B, yolo_resize=(args.imgsz, 32))
        dataset = ConcatDataset([dataset, add_ds])
    
    dl = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)
    print('The number of test datasets: ', len(dl))

    # Prepare model
    model = WODIS_model(num_classes=args.num_classes)
    state_dict = load_weights(args.seg_weights)
    model.load_state_dict(state_dict)
    predictor = Predictor(model, args.fp16, eval_mode=True) if args.mode=='eval' else Predictor(model, args.fp16)

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir / 'images'):
        os.makedirs(output_dir / 'images')

    seen, dt = 0, [0.0]
    for features in tqdm(iter(dl), total=len(dl)):
        t1 = time_sync()
        pred_masks = predictor.predict_batch(features)
        dt[0] += time_sync() - t1

        for i, pred_mask in enumerate(pred_masks):
            pred_mask = SEGMENTATION_COLORS[pred_mask]
            mask_img = Image.fromarray(pred_mask)

            out_file = output_dir / 'images' / features['mask_filename'][i]

            mask_img.save(out_file)
            seen += 1
            
        if args.mode=='eval':
            eval_results = predictor.evaluate_batch(features)
    
    with open(output_dir / 'result.txt', 'w') as f:
        for key, value in vars(args).items(): 
            f.write('%s:%s\n' % (key, value))
        
        f.write('the number of data : %d\n\n' % (len(dataset)))
    
        if args.mode=='eval':
            for key, value in eval_results.items(): 
                f.write('%s:%s\n' % (key, round(value.item(),4)))
            f.write('mean_iou:%s\n' % (np.mean([eval_results['iou_obstacle'], 
                                                eval_results['iou_water'], eval_results['iou_sky']])))

        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        f.write(f'Speed: %.1fms inference per image at shape {(1, 3, 384, 512)}' % t)


def main():
    args = get_arguments()
    print(args)

    predict(args)


if __name__ == '__main__':
    main()
