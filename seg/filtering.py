import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision.transforms as T
import json

from seg.model.wodis import WODIS_model
from seg.data.data_loader import MaSTr1325Dataset
from seg.data.transforms import PytorchHubNormalization
from seg.model.utils import load_weights
from seg.model.inference import Predictor
from seg.model.processing import ground_sky_filtering

from yolov7.yoloUtils.general import scale_coords
from yolov7.yoloUtils.torch_utils import time_synchronized

import warnings
warnings.filterwarnings('ignore')

# Colors corresponding to each segmentation class, (0 = obstacles, 1 = water, 2 = sky)
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

DATASET_DIR = './data/images'
YOLO_PREDS_DIR = './data'
SEG_WEIGHT = './seg/weights/wodis_csol.ckpt'
OUTPUT_DIR = './seg/output'
BATCH_SIZE = 1
MODEL = 'wodis'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WODIS Network BBOX Filtering")
    parser.add_argument("--dataset_dir", type=str, default=DATASET_DIR, help="Dataset directory.")
    parser.add_argument("--yolo_preds_dir", type=str, default=YOLO_PREDS_DIR, help="yolov7_predictions.json directory.")
    parser.add_argument("--seg_weights", type=str, default=SEG_WEIGHT, help="Path to the model weights or a model checkpoint.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Minibatch size (number of samples) used on each device.")
    parser.add_argument("--fp16", action='store_true', help="Use half precision for inference.")
    parser.add_argument('--save_results', default=False, action='store_true', help='save segmentation results(images)')
    return parser.parse_args()


def predict(args):
    
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand
    dataset = MaSTr1325Dataset(args.dataset_dir, args.yolo_preds_dir, 
                               normalize_t=PytorchHubNormalization(), yolo_resize=(args.imgsz, 32),
                               include_original=args.save_results)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)
    print('The number of test datasets: ', len(dataloader))

    # Prepare model
    model = WODIS_model(num_classes=args.num_classes)
    state_dict = load_weights(args.seg_weights)
    model.load_state_dict(state_dict)
    predictor = Predictor(model, args.fp16)

    if args.save_results:
        dirs = ['seg_pred', 'ground_sky_bin', 'pred_images']
        output_dir = Path(args.output_dir)
        for dname in dirs:
            os.makedirs(output_dir/dname, exist_ok=True)

    seen, dt, jdict, len_filtered = 0, [0.0, 0.0], [], 0
    for ep_idx, features in enumerate(dataloader):
        t1 = time_synchronized()
        seg_preds = predictor.predict_batch(features)
        dt[0] += time_synchronized() - t1

        yolo_preds = features['yolo_preds'].numpy()
        ori_shape = list(map(int, features['original_shape']))

        # Output image draw
        if args.save_results:
            transform = T.ToPILImage()
            pred_img = transform(features['image_original'][0,:,:,:])
            pred_draw = ImageDraw.Draw(pred_img)


        for b_idx, (seg_pred, yolo_pred) in enumerate(zip(seg_preds, yolo_preds)):
            seen += 1
            t2 = time_synchronized()
            img_name = features['img_name'][b_idx]
            # print(img_name)

            # Rescale boxes from original image shape to seg_pred size
            yolo_pred_cp = yolo_pred.copy()
            yolo_pred_cp[:, 1:5] = scale_coords(ori_shape, yolo_pred_cp[:, 1:5], seg_pred.shape).round()

            ## yolo predictions(bbox) Filtering
            ground_sky_bin = ground_sky_filtering(seg_pred)
            for y_idx, yolo_info in enumerate(yolo_pred_cp):
                if not ground_sky_bin[int(yolo_info[2]):int(yolo_info[4]), int(yolo_info[1]):int(yolo_info[3])].all():
                    jdict.append({'image_id': os.path.splitext(img_name)[0],
                                  'category_id': int(yolo_info[0]),
                                  'bbox': yolo_pred[y_idx, 1:5].tolist(),
                                  'score': yolo_info[5]})
                    if args.save_results:
                        pred_draw.rectangle(yolo_pred[y_idx, 1:5].tolist(), outline=(255,0,0), width = 3)
                else: 
                    # print(img_name)
                    len_filtered += 1
                    if args.save_results:
                        pred_draw.rectangle(yolo_pred[y_idx, 1:5].tolist(), outline=(0,0,255), width = 3)
            dt[1] += time_synchronized() - t2
            
            # save results
            if args.save_results:
                seg_pred = SEGMENTATION_COLORS[seg_pred] # (0 = obstacles, 1 = water, 2 = sky)
                mask_img = Image.fromarray(seg_pred)
                mask_img.save(output_dir / 'seg_pred' / img_name)
                obs_img = Image.fromarray(np.uint8(ground_sky_bin)*255)
                obs_img.save(output_dir / 'ground_sky_bin' / img_name)
                pred_img.save(output_dir / 'pred_images' / img_name)

        
        ep_dt = time_synchronized() - t1          
        print(f'{ep_idx}/{len(dataset)} [{np.round(1/ep_dt, 2)}it/s]')
    
    t = [x / seen * 1E3 for x in dt]  # speeds per image
    tot_ms = np.round(np.sum(t), 2)
    tot_fps = np.round(1/(np.sum(t)/1E3), 2)
    print(f'Speed: {tot_ms}ms, {tot_fps}fps total / %.1fms inference, %.1fms bbox filtering per image at shape {ori_shape}' % tuple(t))
    print(f'{len_filtered} objects filtered out from {dataset.len_yolo_preds} yolo prediction objects in {len(dataset)} images.')

    # Save JSON
    if len(jdict):
        json_path = (Path(args.yolo_preds_dir) / 'yolov7_preds/yolov7_preds_filtered.json').resolve()
        with open(json_path, 'w') as f:
            json.dump(jdict, f)
            
    # Save results
    if args.save_results:
        with open(output_dir / 'result.txt', 'w') as f:
            for key, value in vars(args).items(): 
                f.write('%s:%s\n' % (key, value))
            
            f.write('the number of data : %d\n\n' % (len(dataset)))
            f.write(f'Speed: %.1fms inference, %.1fms bbox filtering per image at shape {ori_shape}' % tuple(t))
            f.write(f'{len_filtered} objects filtered out from {dataset.len_yolo_preds} yolo prediction objects in {len(dataset)} images.')


def main():
    print("Start filtering YOLO-v7 predictions! \n")
    args = get_arguments()
    print(args)

    predict(args)
    print("\nYOLO-v7 prediction filtering Done! \n")

if __name__ == '__main__':
    main()
