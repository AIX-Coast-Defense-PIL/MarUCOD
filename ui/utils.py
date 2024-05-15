import os
import json

WINDOW_TITLE = {'seg': 'Train Segmentation Model',
                'ood': 'Train OOD Classifier',
                'test': 'Inference',
                'label': 'Labeling Tool'}

DESCRIPTIONS = {
'seg': 
"- This is the space for training the Segmentation model (WODIS).\n\
- It filters out unconcerned foregrounds detected in the coastal land area.",

'seg_data':
"1. Select the training dataset for the Segmentation model.\n\
    - Select the parent folder containing the images.",

'seg_params':
"2. You can adjust the model's parameter values through Hyperparameter Setting.\n\
    - Epochs: Number of training iterations for the Segmentation model.\n\
    - Batch Size: Size of mini-batch divisions in the training dataset.",

'ood': 
"- This is the space for training the OOD Classifier model (ResNet-50).\n\
- It discriminates known or unknown objects among concerned objects detected in the sea area.\n\
- You can train the model using labeled data through the labeling tool.",

'ood_params':
"1. You can change the model's parameter values through Hyperparameter Setting.\n\
    - The number of clusters: Number of clusters in bisecting K-means.\n\
    - Input image size: Size of input images for the feature extractor (ResNet-50).",

'ood_error': 
"- Before training the OOD Classifier model, follow these steps:\n\
     1. Run Inference first. (This will automatically save data for OOD Classifier training.)\n\
     2. Use the labeling tool to label the data saved during the Test execution.\n\
     3. Train the OOD Classifier using the finally saved data through the labeling tool.\n\
     4. Proceed with OOD Classifier training again.",

'test': 
"- This is the space to view the inference results of the MarUCOD framework.\n\
- You can view the results for three types of data (Video, Image, RTSP server).\n\
- After running the Test, the data required for OOD Classifier training is automatically saved and can be used in the labeling tool.",

'test_data':
"1. Select one of Image/Video/RTSP for the test dataset.\n\
    - Image: Select the parent folder containing the images.\n\
    - Video: Select the video file.\n\
    - RTSP: Enter the RTSP server address. Press enter to apply after inputting.",

'test_params':
"2. You can adjust the model's parameter values through Hyperparameter Setting.\n\
    - YOLO Threshold: Threshold for finding 'object-like things' (candidates for unknown objects).\n\
            Setting it higher will only detect objects certain to be objects, \n\
            but there's a higher chance of only detecting 'known' objects.\n\
            Setting it lower increases the chance of detecting 'unknown' objects outside of 'known' objects, \n\
            but may also detect areas like the background that are not objects.\n\
    - Filter Threshold: Threshold for filtering objects between land & sky and sea.\n\
            Setting it higher will detect all objects floating above the sea, but are also largely spread over land & sky,\n\
            and setting it lower will filter out objects that spread over land & sky and only detect objects floating above the sea.\n\
    - OOD Threshold: Threshold that divides 'known' objects and 'unknown' objects.\n\
            Setting it higher broadens the range of 'known' objects, making it more difficult to define 'unknown' objects.",

'test_weights':
"3. You can apply the previously trained OOD Classifier using Weight Setting.\n\
    - Weight is saved with the name 'BisectingKMeans_k##_resnet50_s&&_@@'.\n\
            ## is the number of clusters in bisecting K-means, && is the input image size for ResNet-50, \n\
            and @@ is the time (YYYYMMDDHHMMSS) when OOD Classifier was trained.",

'label':
"- This is the space where you can label the data for which the classification confidence during inference was low.\n\
- The re-labeled data through the labeling tool will be used to train the OOD Classifier model.",

'labeling': 
"  - The '[ Enlarged Image of the Box Area ]' is an image that enlarges the red box area of the raw image.\n\
  - You can select whether the corresponding box area is an unknown object, a known object, or noise.\n\
  - Press the save button only once after processing multiple images, and be sure to press it before exiting.\n\
  - Keyboard controls - 1: Known,  2: Unknown,  3: Noise,  9: Previous Image,  0: Next Image,  S: Save"
}

CB_OPTIONS = {'seg': [{'name': 'Epochs', 'options': ['100 (Default)', '200', '10']},
                    {'name': 'Batch Size', 'options': ['2', '4', '8', '16 (Default)', '32', '64']}],
            'ood': [{'name': 'The number of clusters', 'options': ['10', '20', '30 (Default)', '40', '50']},
                    {'name': 'Input image size', 'options': ['32', '64', '128 (Default)', '256', '512']}],
            'test': [{'name': 'YOLO Threshold', 'options': ['0.05 (Default)', '0.1', '0.2', '0.5']},
                    {'name': 'Filter Threshold', 'options': ['0.5', '0.6', '0.7', '0.8 (Default)', '0.9']},
                    {'name': 'OOD Threshold', 'options': ['87', '89', '91', '93', '95 (Default)', '97', '99']}]}

SCRIPT_PATH = {'seg': './shell/train_seg.sh',
               'ood': './shell/train_ood_cluster.sh',
               'test': './shell/infer_marucod.sh',
               'refine': './shell/refine_yolo_preds.sh'}


def edit_script(shell_path, arg_dict=None):
    with open(shell_path, 'r') as f:
        content = f.read().replace('\n', '').replace('\\', '')
    content = content.split(' ')
    
    if arg_dict:
        for arg, value in arg_dict.items() :
            if (value) and ('None' not in value):
                content[content.index(f'--{arg}') + 1] = value

    return content

def load_json(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'yolov7_preds'), exist_ok=True)
        return [], ''

    json_path = os.path.join(dir_path, 'yolov7_preds/yolov7_preds_filtered_refined.json')
    if not os.path.exists(json_path):
        json_path = os.path.join(dir_path, 'yolov7_preds/yolov7_preds_filtered.json')

    if os.path.exists(json_path):
        with open(json_path, 'rb') as file:
            json_dict = json.load(file)
    else: json_dict = []

    return json_dict, json_path

def save_json(contents, dir_path, json_path=None):
    if json_path is None:
        json_path = os.path.join(dir_path, "yolov7_preds/yolov7_preds_filtered.json")
    with open(json_path, "w") as file:
        json.dump(contents, file)