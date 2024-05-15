import json

def compute_iou(bbox1, bbox2):
    intersection_x_length = max(min(bbox1['xbr'], bbox2['xbr']) - max(bbox1['xtl'], bbox2['xtl']), 0)
    intersection_y_length = max(min(bbox1['ybr'], bbox2['ybr']) - max(bbox1['ytl'], bbox2['ytl']), 0)

    area1 = (bbox1['xbr'] - bbox1['xtl'])*(bbox1['ybr'] - bbox1['ytl'])
    area2 = (bbox2['xbr'] - bbox2['xtl'])*(bbox2['ybr'] - bbox2['ytl'])
    intersect_area = intersection_x_length*intersection_y_length
    whole_area = area1 + area2 - intersect_area
    
    iou = intersect_area / whole_area
    return iou

def calc_iou_performance(pred_infos, ann_infos, ood_threshold=None):
    img_ids = set(ann_infos.keys()).union(pred_infos.keys())
    TP_per_imgs = [0] * len(img_ids)
    True_per_imgs = [0] * len(img_ids) # num of annotation true
    Pos_per_imgs = [0] * len(img_ids) # num of prediction true

    for img_idx, img_id in enumerate(img_ids):
        if img_id in pred_infos.keys():
            if ood_threshold:
                pred_bboxes = [bbox for bbox in pred_infos[img_id]['bboxes'] if (bbox['label'] != 2) and (bbox['ood_score'] > ood_threshold)]
            else:
                pred_bboxes = [bbox for bbox in pred_infos[img_id]['bboxes'] if bbox['label'] == 0]
        else:
            pred_bboxes = []
        
        ann_bboxes = [bbox for bbox in ann_infos[img_id]['bboxes'] if bbox['label'] == 0] if img_id in ann_infos.keys() else []

        True_per_imgs[img_idx] = len(ann_bboxes) # if img_id in ann_infos.keys() else 0
        Pos_per_imgs[img_idx] = len(pred_bboxes) # if img_id in pred_infos.keys() else 0

        if True_per_imgs[img_idx] != 0 and Pos_per_imgs[img_idx] != 0:
            for ann_bbox in ann_bboxes:
                for pred_bbox in pred_bboxes:
                    iou = compute_iou(ann_bbox, pred_bbox)
                    if iou >= 0.5:
                        TP_per_imgs[img_idx] += 1
                        break

    precision = round(sum(TP_per_imgs) / (sum(Pos_per_imgs) + 1e-15), 4)
    recall = round(sum(TP_per_imgs) / (sum(True_per_imgs) + 1e-15), 4)
    f1_score = round((2 * precision * recall) / (precision + recall + 1E-15), 4)

    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


if __name__=='__main__':
    ann_path = './data/annotations/all.json'
    pred_path = './data/yolov7_preds/yolov7_predictions.json'

    with open(ann_path, "r") as json_file:
        ann_info = json.load(json_file)
    with open(pred_path, "r") as json_file:
        pred_info = json.load(json_file)
    
    print(calc_iou_performance(pred_info, ann_info, conf_threshold=0.05))