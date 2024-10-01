import cv2
from ultralytics import YOLO
import numpy as np
import os



base_dir = "AiModels/"
model = YOLO(base_dir + os.environ['DAMAGE_MODEL_NAME'])
conf_thres = 0.15

# Define the class labels
classes = ['dent', 'scratch', 'mud']
colors = [(0, 255, 255), (255, 255, 0), (0, 0, 255)]





def model_prediction(frame):
    pred_bboxes = []
    pred_classes = []
    pred_scores = []









    results = model.predict(frame, conf=conf_thres, imgsz=(640, 640))

    result = results[0]
    
    
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)


    for box, score, class_id in zip(boxes, scores, class_ids):
        pred_bboxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
        pred_classes.append(int(class_id))
        pred_scores.append(round(float(score), 2))

    return pred_bboxes, pred_classes, pred_scores


def damage_predictor_for_image(frame):
    alpha = 0.3
    damage_rectangle = []
    overlay = frame.copy()
    tl = 3 or round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1
    dent_count, scratch_count, mud_count = 0, 0, 0


    bboxes, pred_classes, pred_scores = model_prediction(frame)

    if pred_classes:
        for box, class_id, score in zip(bboxes, pred_classes, pred_scores):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_id], thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)
            t_size = cv2.getTextSize(classes[class_id], 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(frame, (x1, y1), c2, colors[class_id], -1, lineType=cv2.LINE_AA)
            cv2.putText(frame, classes[class_id], (x1, y1 - 2), 0, tl / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

            
            damage_rectangle.append({
                "damageRectangle": {
                    "x": x1,
                    "y": y1,
                    "height": y2 - y1,
                    "width": x2 - x1,
                    "label": classes[class_id],
                    "byAI": True,
                    "deleted": False,
                    "accuracyMatrix": {
                        "tp": 1,
                        "fp": 0,
                        "fn": 0
                    }
                }
            })

            
            if class_id == 0:  # fro dent
                dent_count += 1
            elif class_id == 1:  # for scratch
                scratch_count += 1
            elif class_id == 2:  #   formud
                mud_count += 1

    
    message = ""
    if dent_count > 0:
        message += f"{dent_count} dent(s) detected. "
    if scratch_count > 0:
        message += f"{scratch_count} scratch(es) detected. "
    if mud_count > 0:
        message += f"{mud_count} mud(s) detected."

    
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame, message, damage_rectangle






def damage_detection_in_image2(dir_name, file_name, extension):
    img_path = dir_name + file_name + extension
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    original_image_info = {"OrgImgHeight": height, "OrgImgWidth": width}
    processed_image, message, damage_rectangle = damage_predictor_for_image(image)
    processed_image_path = dir_name + file_name + "_processed" + extension
    cv2.imwrite(processed_image_path, processed_image)
    
    if os.path.exists(img_path):
        os.remove(img_path)
    
    return processed_image_path, message, damage_rectangle, original_image_info
