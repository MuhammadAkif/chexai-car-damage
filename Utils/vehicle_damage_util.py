import cv2
from ultralytics import YOLO
from Utils.util import *
import numpy as np
import os


base_dir = "AiModels/"
model = YOLO(base_dir + os.environ['DAMAGE_MODEL_NAME'])
vehicle_model=YOLO(base_dir + os.environ['VEHICLE_MODEL_NAME'])
conf_thres = 0.17

# Define the class labels
classes = ['dent', 'scratch', 'mud']
colors = [(0, 255, 255), (255, 255, 0), (0, 0, 255)]


def vehicle_detection(frame):
    results=vehicle_model.predict(frame ,conf=0.7)
    bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    if len(bboxes):
        if len(bboxes)>1:
            car_bbox=find_biggest_car_bbox(bboxes)
            print("Multi car_bbox: ",car_bbox)

        elif len(bboxes)==1:
            car_bbox=bboxes[0]
            print("single car_bbox: ",car_bbox)
        x1,y1,x2,y2 = car_bbox
        cropped_image=frame[y1:y2, x1:x2]
        return cropped_image ,True, car_bbox
    else:
        return None, False, False


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

    vehicle_image,vehicle_status,vehicle_box=vehicle_detection(frame)
    if vehicle_status:
        vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = vehicle_box

        bboxes, pred_classes, pred_scores = model_prediction(vehicle_image)

        if pred_classes:
            for box, class_id, score in zip(bboxes, pred_classes, pred_scores):
                x1, y1, x2, y2 = box
                original_x1 = int(vehicle_x1 + x1)
                original_y1 = int(vehicle_y1 + y1)
                original_x2 = int(vehicle_x1 + x2)
                original_y2 = int(vehicle_y1 + y2)

                cv2.rectangle(frame, (original_x1, original_y1), (original_x2, original_y2), colors[class_id], thickness=tl, lineType=cv2.LINE_AA)
                tf = max(tl - 1, 1)
                t_size = cv2.getTextSize(classes[class_id], 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = original_x1 + t_size[0], original_y1 - t_size[1] - 3
                cv2.rectangle(frame, (original_x1, original_y1), c2, colors[class_id], -1, lineType=cv2.LINE_AA)
                cv2.putText(frame, classes[class_id], (original_x1, original_y1 - 2), 0, tl / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

                # Save damage rectangle info with the adjusted coordinates
                damage_rectangle.append({
                    "damageRectangle": {
                        "x": original_x1,
                        "y": original_y1,
                        "height": original_y2 - original_y1,
                        "width": original_x2 - original_x1,
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



def damage_detection_in_video(dir_name, file_name, extension):
    video_path = dir_name + file_name + extension
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = dir_name + file_name + "_processed" + extension
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
    tracked_ids = {'dent': set(), 'scratch': set(), 'mud': set()}
    total_dent_count, total_scratch_count, total_mud_count = 0, 0, 0
    message = "Processing complete"
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Call the damage predictor for the current frame
        damage_rectangle, tracked_ids, total_dent_count, total_scratch_count, total_mud_count = damage_predictor_for_frame(frame, tracked_ids, total_dent_count, total_scratch_count, total_mud_count)
        out.write(frame)  # Write the processed frame to the output video
    cap.release()
    out.release()
    # Remove the original video to save space
    if os.path.exists(video_path):
        os.remove(video_path)
    unique_counts = {
        "dent": total_dent_count,
        "scratch": total_scratch_count,
        "mud": total_mud_count,
    }
    return output_path, message, unique_counts
def damage_predictor_for_frame(frame, tracked_ids, total_dent_count, total_scratch_count, total_mud_count):
    alpha = 0.15
    damage_rectangle = []
    overlay = frame.copy()
    tl = 3 or round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1  # line/font thickness
    ##### Model AI prediction ######
    bboxes, pred_classes, pred_scores = model_prediction(frame)
    # Debugging: Print detected classes and scores
    print(f"Detected Classes: {pred_classes}, Scores: {pred_scores}")
    if pred_classes:
        car_indexes = find_car_indexes(pred_classes)
        if car_indexes:
            car_bboxes = []
            for index in car_indexes:
                if pred_scores[index] >= 0.70:
                    car_bboxes.append({"bbox": bboxes[index], "conf_score": pred_scores[index]})
            bboxes = np.array(bboxes)
            pred_classes = np.array(pred_classes)
            pred_scores = np.array(pred_scores)
            bboxes = np.delete(bboxes, car_indexes, axis=0)
            pred_classes = np.delete(pred_classes, car_indexes, axis=0)
            pred_scores = np.delete(pred_scores, car_indexes, axis=0)
            if car_bboxes:
                if len(car_bboxes) > 1:
                    car_obj = find_biggest_car_bbox(car_bboxes)
                else:
                    car_obj = car_bboxes[0]
                car_x1, car_y1, car_x2, car_y2 = car_obj['bbox']
                cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 0, 255), thickness=tl, lineType=cv2.LINE_AA)
                for box, class_ in zip(bboxes, pred_classes):
                    x1, y1, x2, y2 = box
                    if x1 > car_x1 and x2 < car_x2 and y1 > car_y1 and y2 < car_y2:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), colors[int(class_)], thickness=tl, lineType=cv2.LINE_AA)
                        damage_rectangle.append({
                            "damageRectangle": {
                                "x": x1,
                                "y": y1,
                                "height": y2 - y1,
                                "width": x2 - x1,
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
                        # Update tracked_ids based on the detected class
                        track_id = ...  # Implement your track ID logic here
                        if class_ == 0 and track_id not in tracked_ids['dent']:
                            tracked_ids['dent'].add(track_id)
                            total_dent_count += 1
                        elif class_ == 1 and track_id not in tracked_ids['scratch']:
                            tracked_ids['scratch'].add(track_id)
                            total_scratch_count += 1
                        elif class_ == 2 and track_id not in tracked_ids['mud']:
                            tracked_ids['mud'].add(track_id)
                            total_mud_count += 1
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return damage_rectangle, tracked_ids, total_dent_count, total_scratch_count, total_mud_count
