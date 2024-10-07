import cv2
import torch
from bytetracker import BYTETracker
from ultralytics import YOLO
from Utils.util import *
import numpy as np
import os


base_dir = "AiModels/"
model = YOLO(base_dir + os.environ['DAMAGE_MODEL_NAME'])
vehicle_model=YOLO(base_dir + os.environ['VEHICLE_MODEL_NAME'])
conf_thres = 0.15

# Define the class labels
classes = ['dent', 'scratch', 'mud']
colors = [(0, 255, 255), (255, 255, 0), (0, 0, 255)]



def vehicle_detection(frame):
    results = vehicle_model.predict(frame, conf=0.7)
    bboxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

    if len(bboxes):
        if len(bboxes) > 1:
            car_bbox = find_biggest_car_bbox(bboxes)
        elif len(bboxes) == 1:
            car_bbox = bboxes[0]
        x1, y1, x2, y2 = car_bbox
        cropped_image = frame[y1:y2, x1:x2]
        return cropped_image, True, car_bbox
    else:
        return None, False, None


def model_prediction(frame):
    pred_bboxes = []
    pred_classes = []
    pred_scores = []

    results = model.predict(frame, conf=conf_thres)

    result = results[0]
    
    
    
    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)



    for box, score, class_id in zip(boxes, scores, class_ids):
        pred_bboxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])
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
        
        return frame, message, damage_rectangle, True
    else:
        return frame, message, damage_rectangle, False

########################################### New Function with Non Vehicle condition ########################################################


def damage_detection_in_image2(dir_name, file_name, extension):
    img_path = dir_name + file_name + extension
    image = cv2.imread(img_path)
    height, width, _ = image.shape
    original_image_info = {"OrgImgHeight": height, "OrgImgWidth": width}
    
    # First, detect the vehicle
    # vehicle_image, vehicle_status, vehicle_box = vehicle_detection(image)
    
    # if not vehicle_status or vehicle_box is None:
    #     message = "Vehicle Not detected"
    #     return None, message, [], original_image_info
    
    processed_image, message, damage_rectangle, vehicle_det_status= damage_predictor_for_image(image)
    if not vehicle_det_status:
        return None, message, [], original_image_info
    processed_image_path = dir_name + file_name + "_processed" + extension
    cv2.imwrite(processed_image_path, processed_image)
    
    if os.path.exists(img_path):
        os.remove(img_path)
    
    return processed_image_path, message, damage_rectangle, original_image_info



####################################################### Video Tracking Function ###########################################

def damage_detection_in_video(dir_name, file_name, extension):
    video_path = dir_name + file_name + extension
    cap = cv2.VideoCapture(video_path)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = dir_name + file_name + "_processed" + extension
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    tracker = BYTETracker()

    unique_objects = {class_name: [] for class_name in classes}

    def update_unique_objects(class_name, x, y, frame_count, track_id):
        for obj in unique_objects[class_name]:
            dx, dy = x - obj['x'], y - obj['y']
            distance = np.sqrt(dx*dx + dy*dy)
            if distance < 50 or track_id == obj['track_id']:  # Check both distance and track_id
                obj['last_seen'] = frame_count
                obj['x'], obj['y'] = x, y  # Update position
                obj['track_id'] = track_id  # Update track_id
                return
        unique_objects[class_name].append({
            'x': x, 'y': y, 
            'first_seen': frame_count, 
            'last_seen': frame_count,
            'track_id': track_id
        })

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # YOLO inference
        results = model(frame, conf=conf_thres, imgsz=(640, 640))
        result = results[0]
        bboxes = result.boxes.xyxy.cpu().numpy().astype(int)
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        # Prepare detections for ByteTrack
        detections = []
        for box, score, class_id in zip(bboxes, scores, class_ids):
            if score > conf_thres:
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, score, class_id])

        if detections:
            detections = torch.tensor(detections)
            online_targets = tracker.update(detections, frame)

            for target in online_targets:
                x1, y1, x2, y2, score, class_id, track_id = target[:7]
                class_id, track_id = int(class_id), int(track_id)
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_id], 2)
                label = f"{classes[class_id]}: {score:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

                # Update unique objects
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                update_unique_objects(classes[class_id], center_x, center_y, frame_count, track_id)

        # Write processed frame
        out.write(frame)

    cap.release()
    out.release()

    # Remove original video to save space
    if os.path.exists(video_path):
        os.remove(video_path)

    # Count objects, considering both spatial proximity and tracking consistency
    unique_counts = {class_name: 0 for class_name in classes}
    for class_name, objects in unique_objects.items():
        tracked_ids = set()
        for obj in objects:
            if obj['track_id'] not in tracked_ids:
                unique_counts[class_name] += 1
                tracked_ids.add(obj['track_id'])

    message = f"{unique_counts['dent']} dents, {unique_counts['scratch']} scratches, and {unique_counts['mud']} mud detected."
    return output_path, message, unique_counts