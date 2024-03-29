import requests
import string
import random
import cv2
import os
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
import numpy as np
import imageio

# cfg=get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.WEIGHTS = "model/model_final.pth"
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 6
# cfg.MODEL.DEVICE = "cuda"


# classes=['scratch','crack','big_dent','small_dent','spot','car']
# colors=[(229,204,255),(255,102,0),(204,255,255),(255,229,204),(153,204,255),(0,0,255)]

# predictor = DefaultPredictor(cfg)

import cv2
import torch
import numpy as np

from utils.general import non_max_suppression, scale_coords
from custom_utils.utils import img_preprocessing, load_model
from utils.torch_utils import select_device


imgsz = 640
conf_thres = 0.10
iou_thres = 0.45

weights = "model/car_damage_detection_v3.pt"
imgsz = 640
conf_thres = 0.1
iou_thres = 0.45
device = select_device('0')
print(device)
### load yolov7 model ###
stride, model, names, half, old_img_w,old_img_h, old_img_b = load_model(device,weights,imgsz)

classes=['check_for_dent','check_for_dent','check_for_scratch','check_for_spot','vehicle_body']
colors=[(0,255,255),(255,255,0),(255,51,255),(51,255,51),(0,0,255)]

def find_car_indexes(lst):
    indices = []
    for i, elem in enumerate(lst):
        if elem == 4:##### 4 is for car label
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
    overlay=frame.copy()
    global old_img_b, old_img_h, old_img_w
    tl = 3 or round(0.002 * (img.shape[0] + frame.shape[1]) / 2) + 1  # line/font thickness
    big_dent_count,scratch_cont,spot_count=0,0,0

    ### This block is for detectron2 model
    # outputs = predictor(frame)
    # bboxes = outputs['instances'].pred_boxes.tensor
    # pred_classes=outputs['instances'].pred_classes
    # pred_scores=outputs['instances'].scores

    # bboxes=bboxes.cpu().numpy()
    # pred_classes=pred_classes.cpu().numpy()
    # pred_scores=pred_scores.cpu().numpy()


    img = img_preprocessing(frame, imgsz, stride, device, half)

    ### Prediction
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
        for i in range(3):
            model(img)[0]

    with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
        pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False) #Apply NMS
    # Process detections
    for i, det in enumerate(pred):
        if len(det):
            bboxes=[]
            pred_classes=[]
            pred_scores=[]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                bboxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
                pred_classes.append(int(cls))
                pred_scores.append(round(float(conf), 2))
            
            car_indexes=find_car_indexes(pred_classes)
            if len(car_indexes)>0:
                car_bboxes=[]
                for index in car_indexes:
                    if pred_scores[index]>=0.70:
                        car_bboxes.append(bboxes[index])

                bboxes=np.array(bboxes)
                pred_classes=np.array(pred_classes)
                pred_scores=np.array(pred_scores)
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
                    cv2.rectangle(frame, (car_x1,car_y1), (car_x2, car_y2), (0,0,255), thickness=tl, lineType=cv2.LINE_AA)
                    tf = max(tl - 1, 1)  # font thickness
                    t_size = cv2.getTextSize("vehicle_body", 0, fontScale=tl / 3, thickness=tf)[0]
                    c2 = car_x1 + t_size[0], car_y1 - t_size[1] - 3
                    cv2.rectangle(frame, (car_x1,car_y1), c2, colors[-1], -1, lineType=cv2.LINE_AA)
                    cv2.putText(frame,"vehicle_body",(car_x1,car_y1-2),0,tl / 3,(255,255,255),thickness=tf, lineType=cv2.LINE_AA)

                    for box,class_ in zip(bboxes,pred_classes):
                        tf = max(tl - 1, 1)  # font thickness
                        t_size = cv2.getTextSize(classes[int(class_)], 0, fontScale=tl / 3, thickness=tf)[0]
                        
                        x1,y1,x2,y2=box
                        c2 = x1 + t_size[0], y1 - t_size[1] - 3
                        if x1 > car_x1 and x2 < car_x2 and y1 > car_y1 and y2 < car_y2:
                            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                            cv2.rectangle(frame, (x1,y1), (x2, y2), colors[int(class_)], thickness=tl, lineType=cv2.LINE_AA)
                            cv2.rectangle(frame, (x1,y1), c2, colors[int(class_)], -1, lineType=cv2.LINE_AA)
                            cv2.putText(frame,classes[int(class_)],(x1,y1-2),0,tl / 3,(255,255,255),thickness=tf, lineType=cv2.LINE_AA)

                            if classes[int(class_)]=="check_for_dent":
                                big_dent_count+=1
                            elif classes[int(class_)]=="check_for_scratch":
                                scratch_cont+=1
                            elif classes[int(class_)]=="check_for_spot":
                                spot_count+=1
    message=""                        
    if big_dent_count==0 and scratch_cont==0 and spot_count==0:
        message=""
    else:
        list_counter=[big_dent_count,scratch_cont,spot_count]
        for i,counter in enumerate(list_counter):
            print(i,": ",counter)
            if counter>0 and i==0:
                message+=str(counter)+" dent detected"
            elif counter>0 and i==1:
                message+=" "+str(counter)+" scratch detected"
            elif counter>0 and i==2:
                message+=" "+str(counter)+" spot detected"
    
    frame=cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame,message



    # car_indexes=find_car_indexes(pred_classes)
    # if len(car_indexes)>0:
    #     car_bboxes=[]
    #     for index in car_indexes:
    #         if pred_scores[index]>=0.70:
    #             car_bboxes.append(bboxes[index])

    #     bboxes=np.delete(bboxes,car_indexes,axis=0)
    #     pred_classes=np.delete(pred_classes,car_indexes,axis=0)
    #     pred_scores=np.delete(pred_scores,car_indexes,axis=0)

    #     if len(car_bboxes)>0:
    #         if len(car_bboxes)>1:
    #             car_bbox=find_biggest_car_bbox(car_bboxes)
    #         else:
    #             car_bbox=car_bboxes[0]
            
    #         car_x1,car_y1,car_x2,car_y2=car_bbox
    #         car_x1,car_y1,car_x2,car_y2=int(car_x1),int(car_y1),int(car_x2),int(car_y2)
    #         frame=cv2.rectangle(frame, (car_x1,car_y1), (car_x2, car_y2), (0,0,255), 3)
    #         frame=cv2.putText(frame,"car",(car_x1+4,car_y1+4),0,0.5,(0,0,0),3)

    #         for box,class_ in zip(bboxes,pred_classes):
    #             x1,y1,x2,y2=box
    #             if x1 > car_x1 and x2 < car_x2 and y1 > car_y1 and y2 < car_y2:
    #                 x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
    #                 overlay=cv2.rectangle(overlay, (x1,y1), (x2, y2), colors[int(class_)], -1)
    #                 frame=cv2.putText(frame,classes[int(class_)],(x1+4,y1+4),0,0.6,(0,0,0),3)

    # frame=cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    # return frame
    
def damage_detection_in_image(dir_name,file_name,extension):
    img_path=dir_name+file_name+extension
    image=cv2.imread(img_path)
    processed_image,message=damage_predictor(image)
    processed_image_path=dir_name+file_name+"_processed"+extension
    cv2.imwrite(processed_image_path,processed_image)
    if os.path.exists(img_path):
        os.remove(img_path)
    return processed_image_path,message

def damage_detection_in_video(dir_name,file_name,extension):
    video_path=dir_name+file_name+extension
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    processed_video_path=dir_name+file_name+"_processed"+extension
    writer = imageio.get_writer(processed_video_path, fps=fps)

    counter=0
    while (cap.isOpened() == True):
        ret, frame = cap.read()
        if ret == True:
            frame=cv2.resize(frame,(int(width/2), int(height/2)))
            counter+=1
            # frame=damage_predictor(frame)
            if counter%2==0:
                # print("processed frame number: ",counter)
                frame,message=damage_predictor(frame)
            else:
                pass
            frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            writer.append_data(frame)
            
        else:
            cap.release()
            writer.close()
            if os.path.exists(video_path):
                os.remove(video_path)
            break
    
    message=""
    return processed_video_path,message
# cv2.destroyAllWindows()
