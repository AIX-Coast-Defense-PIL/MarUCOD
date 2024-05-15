import xmltodict
import json
import os
from PIL import Image
from scipy import io

def convert_xml_to_json(origin_file_path, destination_dir_path, custom_format=None):
    # convert to json format
    with open(origin_file_path, encoding='utf-8') as file:
        xml_file = xmltodict.parse(file.read())
                          
    json_data = json.loads(json.dumps(xml_file))

    if custom_format == None:
        save_format = json_data
    else:  # change the json format as you want(default : custom102)
        save_format = {}
        for item in json_data['annotations']['image']:
            img_info = {'img_name':None, 'width':None, 'height':None, 'bboxes':[]}
            img_id = item['@name'].split('.')[0]
            img_info['img_name'] = img_id+'.jpg' # item['@name']
            img_info['width'] = int(item['@width'])
            img_info['height'] = int(item['@height'])
            if 'box' in item.keys():
                if not isinstance(item['box'], list):
                    item['box'] = [item['box']]
                for box in item['box']:
                    box_label = 0 if box['@label'] == 'Unknown' else box['@label']
                    box_info = {'label':box_label, 'xtl':float(box['@xtl']), 'ytl':float(box['@ytl']), 'xbr':float(box['@xbr']), 'ybr':float(box['@ybr'])}
                    img_info['bboxes'].append(box_info)
                    
            save_format[img_id] = img_info
    
    # save file
    save_file_name = origin_file_path.split('/')[-1].split('.')[0]+'.json'
    save_path = os.path.join(destination_dir_path, save_file_name)
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(save_format, file, indent="\t")
    
    return save_format

def convert_txt_to_json(origin_dir_path, destination_dir_path):
    # convert txt file(yolo output) to json format(custom102)
    txt_list = os.listdir(origin_dir_path)
    
    # change the txt format as you want(default : custom102)
    save_format = {}
    for txt_path in txt_list:
        img_info = {'img_name':None, 'bboxes':[]}
        img_id = txt_path.split('.')[0]
        img_info['img_name'] = img_id + '.jpg'
        with open(os.path.join(origin_dir_path, txt_path), encoding='utf-8') as file:
            txt_files = file.read().split('\n')[:-1]
            for txt_file in txt_files:
                bboxes = txt_file.split(' ')
                box_info = {'label':int(bboxes[0]), 'xtl':min(int(bboxes[1]), int(bboxes[3])), 'ytl':min(int(bboxes[2]), int(bboxes[4])), 'xbr':max(int(bboxes[1]), int(bboxes[3])), 'ybr':max(int(bboxes[2]), int(bboxes[4])), 'ood_score': float(bboxes[5]), 'category_id': int(bboxes[6])}
                img_info['bboxes'].append(box_info)
                    
        save_format[img_id] = img_info
    save_format = dict(sorted(save_format.items(), key=lambda x:x[0]))
    # save_format = sorted(save_format, key=lambda k: k["img_name"], reverse=False)
    
    # save file
    save_file_name = 'preds.json'
    save_path = os.path.join(destination_dir_path, save_file_name)
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(save_format, file, indent="\t")
    
    return save_format

def convert_mat_to_json(origin_file_path, destination_file_path):
    mat_file = io.loadmat(origin_file_path)
    print(mat_file.keys())
    # print(mat_file['masks'])

def combine_json_list(origin_dir_path):
    file_list = os.listdir(origin_dir_path)
    all_data = {}
    for file in file_list:
        with open(os.path.join(origin_dir_path, file), "r") as json_file:
            loaded_file = json.load(json_file)
        all_data.update(loaded_file)

    all_data = dict(sorted(all_data.items(), key=lambda x:x[0]))

    # save file
    save_path = os.path.join(origin_dir_path, 'all.json')
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(all_data, file, indent="\t")
    return all_data

def convert_imgs_format(origin_dir_path, destination_dir_path, file_format = 'jpg'):
    file_list = os.listdir(origin_dir_path)
    for img_name in file_list:
        origin_img_path = os.path.join(origin_dir_path, img_name)
        new_img_path = os.path.join(destination_dir_path, img_name.split('.')[0]+'.'+file_format)
        img = Image.open(origin_img_path).convert('RGB')
        if file_format == 'jpg':
            img.save(new_img_path, 'jpeg')
        else:
            img.save(new_img_path, file_format)

def convert_annForm_bbox2img(bbox_infos):
    img_infos = {}

    for bbox_info in bbox_infos:
        if 'image_id' in bbox_info.keys():
            img_id = bbox_info['image_id']
            img_name = img_id+'.jpg'
        else:
            img_name = bbox_info['img_name']
            img_id = img_name.split('.')[0]
        bbox = {'xtl':bbox_info['bbox'][0], 'ytl':bbox_info['bbox'][1], 'xbr':bbox_info['bbox'][2], 'ybr':bbox_info['bbox'][3]}
        for bbox_key in bbox_info.keys():
            if bbox_key != 'bbox' and bbox_key != 'img_name' and bbox_key != 'image_id':
                bbox[bbox_key] = bbox_info[bbox_key]

        if img_id in img_infos.keys():
            img_infos[img_id]['bboxes'].append(bbox)
        else:
            img_infos[img_id] = {'img_name':img_name, 'bboxes' : [bbox]}

    return img_infos

def convert_annForm_img2bbox(img_infos):
    # {"image_id": "img097", "category_id": 0, "bbox": [89.981, 142.143, 101.862, 160.4], "score": 0.53613}
    bbox_infos = []
    # img_infos = list(img_infos.items())
    for img_id in img_infos.keys():
        for bbox in img_infos[img_id]['bboxes']:
            bbox_info = {"image_id": img_id, "category_id": None, "bbox": [], "score": None, "is_known": -1}
            bbox_info['category_id'] = bbox['category_id']
            bbox_info['score'] = bbox['score']
            if 'is_known' in bbox.keys():
                bbox_info['is_known'] = bbox['is_known']
            bbox_info['bbox'] = [bbox['xtl'], bbox['ytl'], bbox['xbr'], bbox['ybr']]
            bbox_infos.append(bbox_info)
    return bbox_infos


if __name__ == '__main__':
    ## load original file names
    origin_file_path = 'data2/images/01/gt.mat'
    destination_dir_path = 'data2/images/01'
    if not os.path.exists(destination_dir_path):
        os.makedirs(destination_dir_path)

    ## use what you want
    # convert_xml_to_json(origin_file_path, destination_dir_path, 'custom102')  # xml to json
    # convert_txt_to_json(origin_dir_path, destination_dir_path) # txt to json
    convert_mat_to_json(origin_file_path, destination_dir_path) # mat to json
    # combine_json_list(origin_dir_path)   # integrate json files
    # convert_imgs_format(origin_dir_path, destination_dir_path)
    # convert_annForm_bbox2img(bbox_infos)
    # convert_annForm_img2bbox(img_infos)