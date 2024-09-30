# import requests
# import string
# import random
# import cv2
# import os
# import numpy as np
# import imageio
# import cv2
# import torch
# import numpy as np
# from collections import Counter
# from utils.general import non_max_suppression, scale_coords
# from custom_utils.utils import img_preprocessing, load_model
# from utils.torch_utils import select_device

# from byte_tracker import BYTETracker
# from byte_tracker_utils.tracker_update import Update_Detections


import cv2
from ultralytics import YOLO
from bytetracker import BYTETracker
from Utils.util import *
import numpy as np
import os

##### initilize bytetracker #####
tracker = BYTETracker()


# weights = "model/car_damage_detection_v3.pt"
base_dir = "AiModels/"
model=YOLO(base_dir+os.environ['DAMAGE_MODEL_NAME'])
conf_thres = 0.15

classes=['check for dent','check for scratch','vehicle body']

colors=[(0,255,255),(255,255,0),(0,0,255)]


###### This function is responsible for Ai model predections #########
def model_prediction(frame):
    pred_bboxes=[]
    pred_classes=[]
    pred_scores=[]

    results=model.predict(frame,conf=0.15,imgsz=(640, 640))
    result = results[0]
    
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    # Iterate through detections and draw bounding boxes
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        pred_bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        pred_classes.append(int(class_id))
        pred_scores.append(round(float(score), 2))

    return pred_bboxes,pred_classes,pred_scores


def damage_predictor_for_image(frame):
    alpha=0.3
    damage_rectangle = []
    overlay=frame.copy()
    tl = 3 or round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1  # line/font thickness
    big_dent_count,scratch_cont,spot_count=0,0,0

    ##### Model Ai prediction ######
    bboxes, pred_classes, pred_scores = model_prediction(frame)
    if pred_classes:
        car_indexes=find_car_indexes(pred_classes) ### First we need to check is there any car in the image/frame or not
        if car_indexes:
            car_bboxes=[]
            for index in car_indexes:
                if pred_scores[index]>=0.70:
                    car_bboxes.append({"bbox":bboxes[index],"conf_score":pred_scores[index]}) #### dictionary
            
            bboxes=np.array(bboxes)
            pred_classes=np.array(pred_classes)
            pred_scores=np.array(pred_scores)
            bboxes=np.delete(bboxes,car_indexes,axis=0)
            pred_classes=np.delete(pred_classes,car_indexes,axis=0)
            pred_scores=np.delete(pred_scores,car_indexes,axis=0)

            if car_bboxes:
                if len(car_bboxes)>1:
                    car_obj=find_biggest_car_bbox(car_bboxes)
                else:
                    car_obj=car_bboxes[0]
                
                car_x1,car_y1,car_x2,car_y2=car_obj['bbox']
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
                        # My Changes
                        damage_rectangle.append({
                            
                            "damageRectangle": {
                            "x": (x1), 
                            "y": (y1),
                            "height": (y2 - y1),
                            "width": (x2 - x1),
                            "label": classes[int(class_)],
                            "byAI": True,
                            "deleted": False,
                            "accuracyMatrix": {
                                "tp": 1,
                                "fp": 0,
                                "fn": 0
                            }

                            }
                        })
                        if classes[int(class_)]=="check for dent":
                            big_dent_count+=1
                        elif classes[int(class_)]=="check for scratch":
                            scratch_cont+=1
                        elif classes[int(class_)]=="check for spot":
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
    return frame,message,damage_rectangle
    


# ############## Below  two functions control detection in image and video #########    

def damage_detection_in_image2(dir_name,file_name,extension):
    img_path=dir_name+file_name+extension
    image=cv2.imread(img_path)
    height,width,_=image.shape
    original_image_info = {"OrgImgHeight":height,"OrgImgWidth":width}
    processed_image,message,damage_rectangle=damage_predictor_for_image(image)
    processed_image_path=dir_name+file_name+"_processed"+extension
    cv2.imwrite(processed_image_path,processed_image)
    if os.path.exists(img_path):
        os.remove(img_path)
    return processed_image_path,message,damage_rectangle,original_image_info



# ##### for videos ######
# def damage_predictor_for_video(frame,frame_count,threshold,fps):
#     alpha=0.3
#     damage_rectangle = []
#     overlay=frame.copy()
#     global old_img_b, old_img_h, old_img_w
#     tl = 3 or round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1  # line/font thickness
#     big_dent_count,scratch_cont,spot_count=0,0,0

#     ##### Model Ai prediction ######
#     pred, preprocessed_img = model_prediction(frame)

#     raw_detection = np.empty((0,6), float) #### for tracker define raw detection

#     # Process detections
#     for i, det in enumerate(pred):
#         if len(det):
#             bboxes=[]
#             pred_classes=[]
#             pred_scores=[]
#             det[:, :4] = scale_coords(preprocessed_img.shape[2:], det[:, :4], frame.shape).round()

#             for *xyxy, conf, cls in reversed(det):
#                 bboxes.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])])
#                 pred_classes.append(int(cls))
#                 pred_scores.append(round(float(conf), 2))
            
#             car_indexes=find_car_indexes(pred_classes) ### First we need to check is there any car in the image/frame or not
#             if len(car_indexes)>0:
#                 car_bboxes=[]
#                 for index in car_indexes:
#                     if pred_scores[index]>=0.70:
#                         car_bboxes.append({"bbox":bboxes[index],"conf_score":pred_scores[index]}) #### dictionary

#                 bboxes=np.array(bboxes)
#                 pred_classes=np.array(pred_classes)
#                 pred_scores=np.array(pred_scores)
#                 bboxes=np.delete(bboxes,car_indexes,axis=0)
#                 pred_classes=np.delete(pred_classes,car_indexes,axis=0)
#                 pred_scores=np.delete(pred_scores,car_indexes,axis=0)
#                 if len(car_bboxes)>0:
#                     if len(car_bboxes)>1:
#                         car_obj=find_biggest_car_bbox(car_bboxes)
#                     else:
#                         car_obj=car_bboxes[0]
                    
#                     car_x1,car_y1,car_x2,car_y2=car_obj['bbox']
#                     car_x1,car_y1,car_x2,car_y2=int(car_x1),int(car_y1),int(car_x2),int(car_y2)

#                     ## update value in raw detection if object detected for tracker
#                     raw_detection = np.concatenate((raw_detection, [[int(car_x1), int(car_y1), int(car_x2), int(car_y2), car_obj['conf_score'], 3]]))
                    
#                     # cv2.rectangle(frame, (car_x1,car_y1), (car_x2, car_y2), (0,0,255), thickness=tl, lineType=cv2.LINE_AA)
#                     tf = max(tl - 1, 1)  # font thickness
#                     t_size = cv2.getTextSize("vehicle_body", 0, fontScale=tl / 3, thickness=tf)[0]
#                     c2 = car_x1 + t_size[0], car_y1 - t_size[1] - 3
#                     # cv2.rectangle(frame, (car_x1,car_y1), c2, colors[-1], -1, lineType=cv2.LINE_AA)
#                     # cv2.putText(frame,"vehicle_body",(car_x1,car_y1-2),0,tl / 3,(255,255,255),thickness=tf, lineType=cv2.LINE_AA)

#                     # print("pred_classes: ",pred_classes)
#                     for box,class_,conf_score in zip(bboxes,pred_classes,pred_scores):
#                         tf = max(tl - 1, 1)  # font thickness
#                         t_size = cv2.getTextSize(classes[int(class_)], 0, fontScale=tl / 3, thickness=tf)[0]
                        
#                         x1,y1,x2,y2=box
#                         c2 = x1 + t_size[0], y1 - t_size[1] - 3
#                         if x1 > car_x1 and x2 < car_x2 and y1 > car_y1 and y2 < car_y2:
#                             x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
#                             # print(classes[int(class_)])
#                             raw_detection = np.concatenate((raw_detection, [[int(x1), int(y1), int(x2), int(y2), conf_score, int(class_)]]))
#                     raw_detection = tracker.update(raw_detection)
#                     detections = Update_Detections(raw_detection, classes, tracking="True").to_dict()

#                     current_frame_ids = set()
#                     for box in detections: ##iterate detected object one by one
#                         width = box['width']
#                         height = box['height']
#                         id = box['id']
#                         detected_class=box['class']
#                         # print(detected_class)

#                         ###### calculate start and ending points of the detected object ####
#                         x,y,w,h = int(box['x']), int(box['y']),int(box['x'] + width), int(box['y'] + height)
                        

#                         current_frame_ids.add(id)

#                         if id not in recent_detections:
#                             # New ID detected, start tracking it
#                             recent_detections[id] = {'count': 1, 'first_frame': frame_count}
#                         else:
#                             detection_info = recent_detections[id]
#                             if frame_count <= detection_info['first_frame'] + fps:
#                                 # Increment count if within the next fps frames
#                                 detection_info['count'] += 1
#                         ####  Ploting ###
#                         cv2.rectangle(frame, (x - 1, y), (w, h),(0,0,255), tl)
#                         cv2.putText(frame,("id: " + str(id)),(x+2,y-4),0,0.8, (0,0,255),2)
#                         cv2.putText(frame,detected_class,(w-2,y-4),0,tl/3, (0,0,255),tf)

#                     # Check and add to main_detected_objects after the fps frame window
#                     for id, detection_info in list(recent_detections.items()):
#                         if frame_count > detection_info['first_frame'] + fps:
#                             # Retrieve the class for the ID, or set a default value if not found
#                             detected_class = next((box['class'] for box in detections if box['id'] == id), None)
                            
#                             if detected_class is not None and detection_info['count'] >= threshold:
#                                 id_class_combo = (id, detected_class)
#                                 if id_class_combo not in added_objects:
#                                     main_detected_objects.append({"id": id, "class": detected_class})
#                                     added_objects.add(id_class_combo)
                            
#                             # Remove ID from recent_detections as it's outside the fps window
#                             del recent_detections[id]

#     frame=cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
#     return frame







# def damage_detection_in_video2(dir_name,file_name,extension):
#     global recent_detections, main_detected_objects, added_objects
#     recent_detections={}
#     main_detected_objects = []  # Main detected objects list
#     added_objects = set()
    
#     video_path=dir_name+file_name+extension
#     cap = cv2.VideoCapture(video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     processed_video_path=dir_name+file_name+"_processed"+extension
#     writer = imageio.get_writer(processed_video_path, fps=fps)
#     frame_count=0
#     threshold = int(fps * 0.1)

#     while (cap.isOpened() == True):
#         ret, frame = cap.read()
#         if ret == True:
#             # frame=cv2.resize(frame,(int(width/2), int(height/2)))
#             frame_count+=1
#             # frame=damage_predictor_for_video(frame,frame_count,threshold,fps)
#             if frame_count%2==0:
#                 # print("processed frame number: ",counter)
#                 frame=damage_predictor_for_video(frame,frame_count,threshold,fps)
#                 frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
#                 writer.append_data(frame)
#             else:
#                 pass
            
            
#         else:
#             cap.release()
#             writer.close()
#             if os.path.exists(video_path):
#                 os.remove(video_path)
#             break
#     class_counts = Counter(obj['class'] for obj in main_detected_objects if obj['class'] != 'vehicle_body')

#     # Construct the message
#     messages = []
#     for class_type, count in class_counts.items():
#         if count > 0:
#             item = f"{count} {class_type.replace('check_for_', '')}{'s' if count > 1 else ''} detected"
#             messages.append(item)

#     final_message = ', '.join(messages)

#     if not final_message:
#         final_message = "No damages detected."
#     return processed_video_path,final_message
