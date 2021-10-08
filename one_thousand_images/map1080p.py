import json
import os
import pickle
import numpy as np
from mean_average_precision import MetricBuilder
from sklearn.preprocessing import normalize


# read pickle
def gt_data(gt_file):
    with open(gt_file, 'rb') as handle:
        boxes = pickle.load(handle)["boxes"]
    #for i in range(len(boxes)):
     #   boxes[i] = list(boxes[i][0] + boxes[i][1])
    boxes = np.array(boxes.detach().cpu().numpy())
    add = np.zeros((boxes.shape[0], 3))
    gt = np.hstack((boxes, add))
    gt = normalize(gt,axis=0,norm='max')

    return gt


# json way
def gt_data_json(json_file):

    with open(json_file, 'r') as f:
        data = json.load(f)
    
    boxes = []
    for index, value in enumerate(data[0]["annotations"]):
        left = (value["coordinates"]["x"], value["coordinates"]["y"])
        height = value["coordinates"]["height"]
        width = value["coordinates"]["width"]
        right = (left[0]+width, left[1]+height)
        coordinates = [left, right]
        boxes.append(coordinates)
    for i in range(len(boxes)):
        boxes[i] = list(boxes[i][0] + boxes[i][1])
    boxes = np.array(boxes)
    add = np.zeros((boxes.shape[0], 3))
    gt = np.hstack((boxes, add))
    gt = normalize(gt, axis=0, norm='max')
    print(gt)
    return gt


def gt_json(json_file):

    with open(json_file, 'r') as f:
        data = json.load(f)
    key = list(data.keys())[0]
    boxes = []

    for detection_target in data[key].keys():
        box = data[key][detection_target]['box']
        boxes.append(box)
    boxes = np.array(boxes)
    add = np.zeros((boxes.shape[0], 3))
    gt = np.hstack((boxes, add))

    return gt

def pred_data(pred_file):

    with open(pred_file, 'rb') as handle:
        info = pickle.load(handle)

    boxes = info["boxes"]

    score = info["scores"]

    boxes = np.array(boxes.detach().cpu().numpy())
    score = np.array(score.detach().cpu().numpy())
    score = score.reshape(score.shape[0], 1)
    
    add = np.zeros((boxes.shape[0], 1))
    boxes_add = np.hstack((boxes, add))
    pred = np.hstack((boxes_add, score))                                                                                    
    pred=normalize(pred, axis=0, norm='max')

    return pred


def get_map(gt, pred):
                                                                                                                            
    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=1)
                                                                                                                            
    for i in range(10):
        metric_fn.add(pred, gt)

    print(f"COCO mAP: {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}")
    return {metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')['mAP']}


#for gt_file in os.listdir(gt_folder):
    
   # basement = gt_file.split(".")[0]
   # print(basement)
   # json_file = basement + '.json'
   # gt = gt_data_json(os.path.join(gt_folder, json_file))
   # pickle_file = basement + '.pickle'
   # pred = pred_data(os.path.join(pred_folder, pickle_file))
   # mapp =  list(get_map(gt, pred))
   # maps.append(mapp[0])


gt_folder = r"/home/wxz/Documents/pickle/fastrcnn_retain"

pred_folder = r"/home/wxz/Documents/pickle/fastrcnn/1080p"


maps = []
count = 0
# two pickle file
for gt_file in os.listdir(gt_folder):

    basement = gt_file.split(".")[0]
    pred_file = basement + ".pickle"
    print(os.path.join(pred_folder, pred_file))
    gt = gt_data(os.path.join(gt_folder, gt_file))       
    #print(gt)
    pred = pred_data(os.path.join(pred_folder, pred_file))
    count += 1
    print(count)
    mapp = list(get_map(gt, pred))
    maps.append(mapp[0])

print("average COCO mAP:" + str(sum(maps)/ len(maps)))



