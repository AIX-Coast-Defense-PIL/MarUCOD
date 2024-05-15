import argparse
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os
import json
import platform

from models.experimental import attempt_load
from yoloUtils.datasets import LoadStreams, LoadImages
from yoloUtils.general import check_img_size, check_imshow, non_max_suppression, \
    scale_coords, strip_optimizer, set_logging, increment_path
from yoloUtils.plots import plot_one_box
from yoloUtils.torch_utils import select_device, time_synchronized, TracedModel

from utils.converter import convert_txt_to_json
from utils.file_processing import load_file, save_file
from utils.metrics import calc_iou_performance

from seg.model.wodis import WODIS_model
from seg.model.utils import load_weights
from seg.model.inference import Predictor

import warnings
warnings.filterwarnings(action='ignore')

from PIL import Image
import platform
import copy
import shutil

# Colors corresponding to each segmentation class # (0 = water, 1 = sky & ground)
BIN_COLORS = np.array([
    [0, 0, 0],
    [255, 105, 180],
    [247, 195, 37],
], np.uint8)


def detect(opt, second_classifier):
    save_img = not opt.no_save and not opt.source.endswith('.txt')  # save inference images
    webcam = opt.source.isnumeric() or opt.source.endswith('.txt') or opt.source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    if opt.save_txt or opt.calc_performance or save_img:
        (save_dir / 'labels' if opt.save_txt or opt.calc_performance else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    bnd_dir = Path('./ood/datasets/boundary_data')
    bnd_json = str(bnd_dir / 'yolov7_preds/yolov7_preds_filtered.json')
    if opt.save_boundary_data:
        if os.path.exists(bnd_json):
            with open(bnd_json, 'rb') as file:
                jdict = json.load(file)
        else:
            (bnd_dir / 'images').mkdir(parents=True, exist_ok=True)
            (bnd_dir / 'yolov7_preds').mkdir(parents=True, exist_ok=True)
            jdict = []

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(opt.weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(opt.img_size, s=stride)  # check img_size

    if not opt.no_trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    ## Segmentation
    if opt.seg_model != 'None':
        seg_model = WODIS_model(num_classes=3)
        state_dict = load_weights(opt.seg_weights)
        seg_model.load_state_dict(state_dict)
        seg_predictor = Predictor(seg_model, half)

    ## OOD
    fe_model = second_classifier['backbone_model']
    fe_model.to(device).eval()
    train_distribution = second_classifier['train_distribution']
    second_classify = second_classifier['pred_func']
    ood_thres_dict = second_classifier['thresholds']
    
    # Set Dataloader
    view_img = opt.view_img
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(opt.source, img_size=imgsz, stride=stride)
        print('The number of test datasets: ', len(dataset))

    # Get names and colors
    names = ['unknown', 'known', 'filtered']
    colors = [[0,0,255], [0, 0, 0], [0, 255, 0]]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    t0 = time_synchronized()
    time_records, windows = [], []
    break_point = False

    if (not webcam) and dataset.cap:
        fps = int(dataset.cap.get(cv2.CAP_PROP_FPS))
        w = int(dataset.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(dataset.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:  # stream
        fps, w, h = 20, imgsz, imgsz
    
    conf_thres = opt.conf_thres
    filter_thres = opt.filter_thres if not opt.no_filter else 2
    ood_thres = opt.ood_thres
    
    def on_conf_thres_change(value):
        nonlocal conf_thres
        conf_thres = value / 100.0

    def on_filter_thres_change(value):
        nonlocal filter_thres
        filter_thres = value / 100.0

    def on_ood_thres_change(value):
        nonlocal ood_thres
        ood_thres = value
    
    for img_i, [path, img, im0s, vid_cap] in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        # (prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,)
        pred = non_max_suppression(pred, conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        ground_sky_bin = seg_predictor.bbox_filtering(im0s, img_size=imgsz, stride=stride) if opt.seg_model != 'None' else [None for i in range(len(im0s))]
        t4 = time_synchronized()
        pred = second_classify(pred, opt.seg_model, ground_sky_bin, fe_model, train_distribution, ood_thres_dict[str(ood_thres)+'%'], 
                               img, im0s, opt.score_metric, filter_thres=filter_thres, patch_size=opt.patch_size)
        # print("pred: ", pred) # pred=[x1,x2,y1,y2,conf_score,label]
        t5 = time_synchronized()

        # Time calculation
        totalT, objT, nmsT, segT, oodT = t5-t1, t2-t1, t3-t2, t4-t3, t5-t4
        time_records.append([totalT, objT, nmsT, segT, oodT])
        

        # Process detections
        bnd_min = ood_thres_dict[f"{max(1, int(ood_thres) - opt.boundary_thres)}%"]
        bnd_max = ood_thres_dict[f"{min(100, int(ood_thres) + opt.boundary_thres)}%"]
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, f'[{img_i}/{len(dataset)}] ', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            img_dir = os.path.join(save_dir, 'images')
            if save_img:
                os.makedirs(img_dir, exist_ok=True)
            save_path = os.path.join(img_dir, p.name.split('.')[0]+'.jpg') # img.jpg
            video_path = str(save_dir / 'video')
            txt_path = str(save_dir / 'labels' / p.stem)  # img.txt
            if dataset.mode == 'image':
                bnd_fname = str(Path(bnd_dir) / 'images' / f"{opt.source.split('/')[-3]}_{opt.source.split('/')[-2]}_{opt.source.split('/')[-1]}_{p.name.split('.')[0]}")
            else:
                bnd_fname = str(Path(bnd_dir) / 'images' / f"{p.name.split('.')[0]}_{img_i}")
            bnd_fname = bnd_fname.split('/')[-1]
            bnd_img_path = os.path.join(bnd_dir, 'images', bnd_fname+'.jpg') # img.jpg
                
            if opt.draw_interm:
                im0 = Image.blend(Image.fromarray(BIN_COLORS[ground_sky_bin[0]]), Image.fromarray(im0), 0.6)
                im0 = np.asarray(im0)
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf_scr, cat_id, ood_scr, ood_cls in reversed(det):
                    xyxy = (torch.tensor(xyxy).view(1, 4).to(torch.int32)).view(-1).tolist()
                    if opt.save_txt or opt.calc_performance:  # Write to file
                        line = (ood_cls, *xyxy, ood_scr, cat_id)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if opt.save_boundary_data and (bnd_min < ood_scr < bnd_max):
                        if (dataset.mode == 'image') or (img_i % 5 == 0):
                            jdict.append({'image_id': bnd_fname,
                                        'category_id': int(cat_id.item()),
                                        'bbox': [round(x, 3) for x in xyxy],
                                        'score': round(conf_scr.item(), 5),
                                        'is_known': -1})
                            try:
                                Image.fromarray(im0s[:,:,[2,1,0]]).save(bnd_img_path)
                            except:
                                Image.fromarray(im0s[i][:,:,[2,1,0]]).save(bnd_img_path)
                    
                    # im0 = tracked_frame
                    if save_img or view_img:  # Add bbox to image
                        if (not opt.draw_interm) and (int(ood_cls) != 0):
                            continue
                        label = f'{names[int(ood_cls)]} {ood_scr:.2f}'
                        plot_one_box(xyxy, im0, color=colors[int(ood_cls)], line_thickness=2)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * totalT):.1f}ms, {(1/totalT):.1f} fps) Total, ({(1E3 * objT):.1f}ms) Object Recognize, ({(1E3 *nmsT):.1f}ms) NMS, ({(1E3 *segT):.1f}ms) BBox filtering, ({(1E3*oodT):.1f}ms) OOD')
            
            # Stream results
            if view_img:
                im0 = np.asarray(im0)
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.createTrackbar("YOLO Threshold", str(p), int(conf_thres*100), 100, on_conf_thres_change)
                    if not opt.no_filter:
                        cv2.createTrackbar("Filter Threshold", str(p), int(filter_thres*100), 100, on_filter_thres_change)
                    cv2.createTrackbar("OOD Threshold", str(p), int(ood_thres), 100, on_ood_thres_change)
                cv2.putText(im0, "exit: 'q', parameter reset: 'r'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if opt.no_filter:
                    text = f"YOLO Threshold: {conf_thres:.2f}, OOD Threshold: {ood_thres}" 
                else:
                    text = f"YOLO Threshold: {conf_thres:.2f}, Filter Threshold: {filter_thres:.1f}, OOD Threshold: {ood_thres}" 
                cv2.putText(im0, text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.imshow(str(p), im0)
                key = cv2.waitKey(1)

                # 'q' 키를 누르면 종료
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    break_point=True
                    break

                # 'r' 키를 누르면 파라미터 초기화
                if key == ord('r'):
                    conf_thres = opt.conf_thres
                    filter_thres = opt.filter_thres if not opt.no_filter else 2
                    ood_thres = opt.ood_thres
                    cv2.setTrackbarPos("YOLO Threshold", str(p), int(conf_thres*100))
                    if not opt.no_filter:
                        cv2.setTrackbarPos("Filter Threshold", str(p), int(filter_thres*100))
                    cv2.setTrackbarPos("OOD Threshold", str(p), int(ood_thres))

            # Save results (image with detections)
            if save_img:
                if dataset.mode in ['video', 'stream']:
                    im0 = np.asarray(im0)
                    if vid_path != video_path:  # new video
                        vid_path = video_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 20, im0.shape[1], im0.shape[0]
                        video_path += '.mp4'
                        vid_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                elif dataset.mode == 'image':
                    im0 = Image.fromarray(im0[:,:,[2,1,0]]) if isinstance(im0, np.ndarray) else im0
                    im0.save(save_path)
                    print(f" The image with the result is saved in: {save_path}")

        if break_point:
            print("The window has closed.")
            break

    if opt.save_boundary_data and len(jdict):
        print('saving %s...' % bnd_json)
        with open(bnd_json, 'w') as f:
            json.dump(jdict, f)
    
    all_performances = {}
    if opt.calc_performance:
        # save json
        txt_dir = str(save_dir / 'labels')
        json_dir = str(save_dir / 'json')
        os.makedirs(json_dir, exist_ok=True)
        convert_txt_to_json(txt_dir, json_dir)
            
        # call and set format of pred_infos & ann_infos
        ood_infos = load_file(os.path.join(json_dir, 'preds.json'))
        ann_infos = load_file(os.path.join(opt.source, '../annotations/all.json'))
        ood_infos_cp = copy.deepcopy(ood_infos)
        
        # calc precision and recall
        main_performance = calc_iou_performance(ood_infos, ann_infos)
        print(main_performance)
        
        for ood_key, ood_value in ood_thres_dict.items():
            all_performances[ood_key] = calc_iou_performance(ood_infos_cp, ann_infos, ood_threshold=ood_value)
        
        if not opt.save_txt:
            shutil.rmtree(txt_dir)

        print(f'Done. ({time_synchronized() - t0:.3f}s)')
        mTotalT, mObjT, mNmsT, mSegT, mOodT = np.mean(time_records, axis=0)
        obj_nums = [len(yolo_pred) for yolo_pred in pred]
        mObjN = np.mean(obj_nums)
        time_performance = f'mean time per frame : ({(1E3 * mTotalT):.1f}ms, {(1/mTotalT):.1f} fps) Total, ({(1E3 * mObjT):.1f}ms) Object Recognize, ({(1E3 *mNmsT):.1f}ms) NMS, ({(1E3 *mSegT):.1f}ms) BBox Filtering, ({(1E3*mOodT):.1f}ms) OOD'
        print(time_performance)
        all_performances['time'] = {'fps':1/mTotalT, 'total':time_performance, 'obj_num':mObjN}
        save_file(all_performances, os.path.join(save_dir, 'performance.json'))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov7/yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default=f'./data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1024, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.05, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.001, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-boundary-data', action='store_true', help='save boundary data')
    parser.add_argument('--boundary-thres', type=int, default=1, help='boundary data threshold')
    parser.add_argument('--calc-performance', action='store_true', help='save results to preds.json')
    parser.add_argument('--no-save', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', default=True, help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='./runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='marucod_test', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--draw-interm', action='store_true', help='save intermediate results')
    parser.add_argument('--patch-size', default=114, type=int, help='patch size')

    # Segmentation
    parser.add_argument("--seg_model", default='wodis', type=str, choices=['wodis','None'], help="Model architecture.")
    parser.add_argument("--seg_weights", default='./seg/weights/wodis_csol.ckpt',
                        type=str, help="Path to the model weights or a model checkpoint.")
    parser.add_argument('--filter-thres', type=float, default=0.8, help='filtering threshold')
    parser.add_argument('--no-filter', action='store_true', help='do not filter objects')
    
    # Feature extractor
    parser.add_argument('--backbone_arch', default='resnet50', choices=['resnet50', 'resnet50_tune', 'resnet101', 'efficientnet_v2', 'mobilenet_v3'], type=str, help='')
    parser.add_argument('--backbone_weight', default='./ood/backbone/resnet_funed_e100.pth', type=str, help='Path to backbone weight')

    # OOD
    parser.add_argument('--ood-thres', type=str, default='95', help='OOD threshold')
    parser.add_argument('--score_metric', default='cosineSim', type=str, choices=['euclidean', 'mahalanobis', 'cosineSim'])
    parser.add_argument('--threshold_path', default='./ood/cache/threshold/BisectingKMeans_k30_cosineSim_resnet50_s128_SeaShips.json', type=str, help='Path to threshold')
    parser.add_argument('--distribution_path', default='./ood/cache/distribution/BisectingKMeans_k30_resnet50_s128_SeaShips.pkl', type=str, help='Path to cluster model')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = get_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)