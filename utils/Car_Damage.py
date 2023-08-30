import requests
import string
import random
import cv2
import os
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import numpy as np

cfg=get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "model/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
cfg.MODEL.DEVICE = "cuda"


classes=['scratch','crack','big_dent','small_dent','spot','car']
colors=[(229,204,255),(255,102,0),(204,255,255),(255,229,204),(153,204,255),(0,0,255)]

predictor = DefaultPredictor(cfg)

def find_car_indexes(lst):
    indices = []
    for i, elem in enumerate(lst):
        if elem == 5:##### 5 is for car label
            indices.append(i)
    return indices

def calculate_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def find_biggest_car_bbox(bboxes):
    biggest_bbox = None
    max_area = 0
    for bbox in bboxes:
        area = calculate_area(bbox)
        if area > max_area:
            max_area = area
            biggest_bbox = bbox
    return biggest_bbox


def random_name_generator(length=20):
    letters_and_digits = string.ascii_letters + string.digits
    return ''.join(random.choice(letters_and_digits) for i in range(length))


def s3_file_downloader(link,local_file_name,dir_name="s3_files/"):
    status=False
    response=requests.get(link)
    if response.status_code==200:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        with open(dir_name+local_file_name, 'wb') as f:
            f.write(response.content)
        status=True
    return status

def damage_predictor(frame):
    alpha=0.3
    frame_height,frame_width=frame.shape[0],frame.shape[1]
    frame=cv2.resize(frame,(int(frame_width/2),int(frame_height/2)))
    overlay=frame.copy()

    outputs = predictor(frame)
    bboxes = outputs['instances'].pred_boxes.tensor
    pred_classes=outputs['instances'].pred_classes
    pred_scores=outputs['instances'].scores

    bboxes=bboxes.cpu().numpy()
    pred_classes=pred_classes.cpu().numpy()
    pred_scores=pred_scores.cpu().numpy()

    car_indexes=find_car_indexes(pred_classes)
    if len(car_indexes)>0:
        car_bboxes=[]
        for index in car_indexes:
            if pred_scores[index]>=0.70:
                car_bboxes.append(bboxes[index])

        bboxes=np.delete(bboxes,car_indexes,axis=0)
        pred_classes=np.delete(pred_classes,car_indexes,axis=0)
        pred_scores=np.delete(pred_scores,car_indexes,axis=0)

        if len(car_bboxes)>0:
            if len(car_bboxes)>1:
                car_bbox=find_biggest_car_bbox(car_bboxes)
            else:
                car_bbox=car_bboxes[0]
            
            car_x1,car_y1,car_x2,car_y2=car_bbox
            car_x1,car_y1,car_x2,car_y2=int(car_x1),int(car_y1),int(car_x2),int(car_y2)
            frame=cv2.rectangle(frame, (car_x1,car_y1), (car_x2, car_y2), (0,0,255), 3)
            frame=cv2.putText(frame,"car",(car_x1+4,car_y1+4),0,0.5,(0,0,0),3)

            for box,class_ in zip(bboxes,pred_classes):
                x1,y1,x2,y2=box
                if x1 > car_x1 and x2 < car_x2 and y1 > car_y1 and y2 < car_y2:
                    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                    overlay=cv2.rectangle(overlay, (x1,y1), (x2, y2), colors[int(class_)], -1)
                    frame=cv2.putText(frame,classes[int(class_)],(x1+4,y1+4),0,0.6,(0,0,0),3)

    frame=cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame
    
def damage_detection_in_image(dir_name,file_name,extension):
    img_path=dir_name+file_name+extension
    image=cv2.imread(img_path)
    processed_image=damage_predictor(image)
    processed_image_path=dir_name+file_name+"_processed"+extension
    cv2.imwrite(processed_image_path,processed_image)
    return processed_image_path

def damage_detection_in_video(dir_name,file_name,extension):
    video_path=dir_name+file_name+extension
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    processed_video_path=dir_name+file_name+"_processed"+extension
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(processed_video_path, fourcc, fps, (int(width//2), int(height//2)))

    counter=0
    while (cap.isOpened() == True):
        ret, frame = cap.read()
        if ret == True:
            counter+=1
            if counter%2==0:
                frame=damage_predictor(frame)
                out.write(frame)
            else:
                out.write(frame)
                pass
            
        else:
            out.release()
            cap.release()
            break
    
    return processed_video_path
# cv2.destroyAllWindows()
