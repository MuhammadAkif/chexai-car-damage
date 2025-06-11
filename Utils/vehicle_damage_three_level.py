import torch
import numpy as np
import cv2

from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes)
from utils.segment.general import process_mask
from utils.torch_utils import select_device
from utils.augmentations import letterbox


# --- Global configuration and model loading ---
vehilce_weights = 'AiModels/vehicle_segmentation.pt'      # update to your weights path
body_part_weights = "AiModels/vehicle_body_part_segmentation.pt"
damage_detection_weights = "AiModels/3rd_level_best_model.pt"
data = 'data/coco.yaml'
imgsz = (640, 640)                             
conf_thres = 0.35
iou_thres = 0.45
max_det = 1000
device = select_device('')                         # use CPU or GPU if available

# Load models once on startup
veh_seg_model = DetectMultiBackend(vehilce_weights, device=device, dnn=False, data=data, fp16=False)
veh_seg_model.warmup(imgsz=(1 if veh_seg_model.pt else 1, 3, *imgsz))

body_parts_seg_model = DetectMultiBackend(body_part_weights, device=device, dnn=False, data=data, fp16=False)
body_parts_seg_model.warmup(imgsz=(1 if body_parts_seg_model.pt else 1, 3, *imgsz))

damage_seg_model = DetectMultiBackend(damage_detection_weights, device=device, dnn=False, data=data, fp16=False)
damage_stride, damage_names, damage_pt = damage_seg_model.stride, damage_seg_model.names, damage_seg_model.pt
damage_seg_model.warmup(imgsz=(1 if damage_seg_model.pt else 1, 3, *imgsz))

print(body_parts_seg_model.names)


# -------------------------------
# 1. Vehicle Segmentation Module
# -------------------------------
def vehicle_segmentation(img, save_path=None):
    original = img.copy()
    im_resized = cv2.resize(img, imgsz)
    im_tensor = torch.from_numpy(im_resized).to(veh_seg_model.device)
    im_tensor = im_tensor.half() if veh_seg_model.fp16 else im_tensor.float()
    im_tensor /= 255.0
    if im_tensor.ndim == 3:
        im_tensor = im_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred, proto = veh_seg_model(im_tensor, augment=False)[:2]
        pred = non_max_suppression(pred, conf_thres, iou_thres,
                                   classes=None, agnostic=False, max_det=max_det, nm=32)

    if len(pred[0]) > 0:
        masks = process_mask(proto[-1][0], pred[0][:, 6:], pred[0][:, :4],
                             im_tensor.shape[2:], upsample=True)
        pred[0][:, :4] = scale_boxes(im_tensor.shape[2:], pred[0][:, :4], original.shape).round()

        largest_area = 0
        largest_idx = None
        for idx, mask in enumerate(masks):
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
            mask_resized = cv2.resize(mask_np, (original.shape[1], original.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            mask_bin = mask_resized > 0.5
            area = np.sum(mask_bin)
            if area > largest_area:
                largest_area = area
                largest_idx = idx

        if largest_idx is not None:
            best_mask = masks[largest_idx]
            best_mask_np = best_mask.cpu().numpy() if hasattr(best_mask, 'cpu') else best_mask
            best_mask_resized = cv2.resize(best_mask_np, (original.shape[1], original.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
            best_mask_bin = best_mask_resized > 0.5
            coords = np.column_stack(np.where(best_mask_bin))
            if coords.shape[0] > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                segmented_obj = np.zeros_like(original)
                segmented_obj[best_mask_bin] = original[best_mask_bin]
                cropped_obj = segmented_obj[y_min:y_max+1, x_min:x_max+1]
                if save_path:
                    cv2.imwrite(save_path, cropped_obj)
                return cropped_obj, True, (x_min, y_min, x_max, y_max)
    return None, False, None

# -------------------------------
# 2. Body-Part Segmentation Module (Updated with left/right distinction)
# -------------------------------
def body_parts_segmentation(img, img_type):
    # Updated rule dictionaries with left/right variants
    front_rule = {
        "bumper": {"status": False, "cropped": []},
        "hood": {"status": False, "cropped": []},
        "windshield": {"status": False, "cropped": []},
        "grill": {"status": False, "cropped": []},
        "left_headlight": {"status": False, "cropped": []},
        "right_headlight": {"status": False, "cropped": []},
        "roof": {"status": False, "cropped": []},
        "left_side_mirror": {"status": False, "cropped": []},
        "right_side_mirror": {"status": False, "cropped": []},
        "fender": {"status": False, "cropped": []}
    }
    left_side_rule = {
        "fender": {"status": False, "cropped": []},
        "door_windshield": {"status": False, "cropped": []},
        "front_tire": {"status": False, "cropped": []},
        "back_tire": {"status": False, "cropped": []},
        "door": {"status": False, "cropped": []},
        "left_side_mirror": {"status": False, "cropped": []},
        "side_body": {"status": False, "cropped": []},
        "slide_door": {"status": False, "cropped": []}
    }
    right_side_rule = {
        "side_door": {"status": False, "cropped": []},
        "fender": {"status": False, "cropped": []},
        "door_windshield": {"status": False, "cropped": []},
        "front_tire": {"status": False, "cropped": []},
        "back_tire": {"status": False, "cropped": []},
        "door": {"status": False, "cropped": []},
        "right_side_mirror": {"status": False, "cropped": []},
        "side_body": {"status": False, "cropped": []},
        "slide_door": {"status": False, "cropped": []}
    }
    back_rule = {
        "back": {"status": False, "cropped": []},
        "left_back_light": {"status": False, "cropped": []},
        "right_back_light": {"status": False, "cropped": []},
    }
    front_left_rule = {
        "bumper": {"status": False, "cropped": []},
        "hood": {"status": False, "cropped": []},
        "windshield": {"status": False, "cropped": []},
        "grill": {"status": False, "cropped": []},
        "left_headlight": {"status": False, "cropped": []},
        "roof": {"status": False, "cropped": []},
        "left_side_mirror": {"status": False, "cropped": []},
        "fender": {"status": False, "cropped": []},
        "door_windshield": {"status": False, "cropped": []},
        "front_tire": {"status": False, "cropped": []},
        "back_tire": {"status": False, "cropped": []},
        "door": {"status": False, "cropped": []},
        "side_body": {"status": False, "cropped": []},
        "slide_door": {"status": False, "cropped": []}
    }
    front_right_rule = {
        "bumper": {"status": False, "cropped": []},
        "hood": {"status": False, "cropped": []},
        "windshield": {"status": False, "cropped": []},
        "grill": {"status": False, "cropped": []},
        "right_headlight": {"status": False, "cropped": []},
        "roof": {"status": False, "cropped": []},
        "right_side_mirror": {"status": False, "cropped": []},
        "fender": {"status": False, "cropped": []},
        "door_windshield": {"status": False, "cropped": []},
        "front_tire": {"status": False, "cropped": []},
        "back_tire": {"status": False, "cropped": []},
        "door": {"status": False, "cropped": []},
        "side_body": {"status": False, "cropped": []},
        "slide_door": {"status": False, "cropped": []}
    }
    rear_right_rule = {
        "side_door": {"status": False, "cropped": []},
        "fender": {"status": False, "cropped": []},
        "door_windshield": {"status": False, "cropped": []},
        "front_tire": {"status": False, "cropped": []},
        "back_tire": {"status": False, "cropped": []},
        "door": {"status": False, "cropped": []},
        "right_side_mirror": {"status": False, "cropped": []},
        "side_body": {"status": False, "cropped": []},
        "slide_door": {"status": False, "cropped": []},
        "back": {"status": False, "cropped": []},
        "right_back_light": {"status": False, "cropped": []},
    }
    rear_left_rule = {
        "side_door": {"status": False, "cropped": []},
        "fender": {"status": False, "cropped": []},
        "door_windshield": {"status": False, "cropped": []},
        "front_tire": {"status": False, "cropped": []},
        "back_tire": {"status": False, "cropped": []},
        "door": {"status": False, "cropped": []},
        "left_side_mirror": {"status": False, "cropped": []},
        "side_body": {"status": False, "cropped": []},
        "slide_door": {"status": False, "cropped": []},
        "back": {"status": False, "cropped": []},
        "left_back_light": {"status": False, "cropped": []},
    }

    original = img.copy()
    im_resized = cv2.resize(img, imgsz)
    im_tensor = torch.from_numpy(im_resized).to(body_parts_seg_model.device)
    im_tensor = im_tensor.half() if body_parts_seg_model.fp16 else im_tensor.float()
    im_tensor /= 255.0
    if im_tensor.ndim == 3:
        im_tensor = im_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred, proto = body_parts_seg_model(im_tensor, augment=False)[:2]
        pred = non_max_suppression(pred, 0.3, iou_thres,
                                   classes=None, agnostic=False, max_det=max_det, nm=32)

    if len(pred[0]) > 0:
        masks = process_mask(proto[-1][0], pred[0][:, 6:], pred[0][:, :4],
                             im_tensor.shape[2:], upsample=True)
        pred[0][:, :4] = scale_boxes(im_tensor.shape[2:], pred[0][:, :4], original.shape).round()
        img_center_x = original.shape[1] / 2  # Image center for left/right determination
        
        for i, det in enumerate(pred[0]):
            mask = masks[i]
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
            mask_resized = cv2.resize(mask_np, (original.shape[1], original.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
            mask_bin = mask_resized > 0.5
            segmented_obj = np.zeros_like(original)
            segmented_obj[mask_bin] = original[mask_bin]
            coords = np.column_stack(np.where(mask_bin))
            if coords.shape[0] == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            cropped_obj = segmented_obj[y_min:y_max+1, x_min:x_max+1]
            class_id = int(det[5]) if det.shape[0] > 5 else 0
            part_name = (body_parts_seg_model.names[class_id]
                         if hasattr(body_parts_seg_model, "names")
                         else f"part_{class_id}")
            
            # Calculate part center for left/right determination
            part_center_x = (x_min + x_max) / 2
            side = "left" if part_center_x < img_center_x else "right"
            
            # Update part names for symmetric components
            if part_name == "headlight":
                part_name = f"{side}_headlight"
            elif part_name == "side_mirro":  # Note: Original model uses 'side_mirro'
                part_name = f"{side}_side_mirror"  # Use consistent naming
            elif part_name == "back_light":
                part_name = f"{side}_back_light"

            rule_dict = None

            # Determine which rule set to apply
            if img_type == "exterior_front" and part_name in front_rule:
                rule_dict = front_rule
            elif img_type == "exterior_driver_side" and part_name in left_side_rule:
                rule_dict = left_side_rule
            elif img_type == "exterior_passenger_side" and part_name in right_side_rule:
                rule_dict = right_side_rule
            elif img_type == "exterior_rear" and part_name in back_rule:
                rule_dict = back_rule
            elif img_type in ["front_driver_side_corner","front_left_corner"] and part_name in front_left_rule:
                rule_dict = front_left_rule
            elif img_type in ["front_right_corner", "front_passenger_side_corner"] and part_name in front_right_rule:
                rule_dict = front_right_rule
            elif img_type in ["rear_driver_side_corner", "rear_left_corner"] and part_name in rear_left_rule:
                rule_dict = rear_left_rule
            elif img_type in ["rear_right_corner", "rear_passenger_side_corner"] and part_name in rear_right_rule:
                rule_dict = rear_right_rule

            if rule_dict is not None:
                rule_dict[part_name]["status"] = True
                rule_dict[part_name]["cropped"].append((cropped_obj, (x_min, y_min, x_max, y_max)))

        def finalize_rule(rule):
            for key, value in rule.items():
                if not value["cropped"]:
                    value["cropped"] = None
            return rule

        # Return the appropriate rule set
        if img_type == "exterior_front":
            return finalize_rule(front_rule)
        elif img_type == "exterior_rear":
            return finalize_rule(back_rule)
        elif img_type == "exterior_driver_side":
            return finalize_rule(left_side_rule)
        elif img_type == "exterior_passenger_side":
            return finalize_rule(right_side_rule)
        elif img_type in ["front_driver_side_corner","front_left_corner"]:
            return finalize_rule(front_left_rule)
        elif img_type in ["front_right_corner", "front_passenger_side_corner"]:
            return finalize_rule(front_right_rule)
        elif img_type in ["rear_driver_side_corner", "rear_left_corner"]:
            return finalize_rule(rear_left_rule)
        elif img_type in ["rear_right_corner", "rear_passenger_side_corner"]:
            return finalize_rule(rear_right_rule)
    else:
        return None

# -------------------------------
# 3. Damage Segmentation Module (with dynamic severity, distinct damage colors, overlay, and rectangle)
# -------------------------------
def damage_segmentation(img, part_key=None, save_path=None):
    """
    Processes damage segmentation on a body-part crop.
    The severity is computed based on the maximum damage dimension (relative to the crop size).
    Each damage class receives a distinct candidate color.
    A colored rectangle and overlay are drawn on the damage, and the label (with severity) is drawn
    at the top-left corner of the damage box.
    Returns the damage-overlay image and a list of damage detection info.
    """
    original = img.copy()
    im_resized = letterbox(original, imgsz, stride=damage_stride, auto=damage_pt)[0]
    # Convert
    im_resized = im_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im_resized = np.ascontiguousarray(im_resized)
    im_tensor = torch.from_numpy(im_resized).to(damage_seg_model.device)
    im_tensor = im_tensor.half() if damage_seg_model.fp16 else im_tensor.float()
    im_tensor /= 255.0
    im_tensor = im_tensor.unsqueeze(0)
    # if im_tensor.ndim == 3:
    #     im_tensor = im_tensor.permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        pred, proto = damage_seg_model(im_tensor, augment=False)[:2]
        # Adjust threshold if needed (here 0.15)
        pred = non_max_suppression(pred, 0.15, iou_thres,
                                   classes=None, agnostic=False, max_det=max_det, nm=32)
    result_img = original.copy()
    damage_detections = []
    candidate_colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),
        (128, 128, 0),
        (0, 128, 128)
    ]
    damage_class_colors = {}
    candidate_index = 0
    crop_h, crop_w = img.shape[:2]

    if len(pred[0]) > 0:
        masks = process_mask(proto[-1][0], pred[0][:, 6:], pred[0][:, :4],
                             im_tensor.shape[2:], upsample=True)
        pred[0][:, :4] = scale_boxes(im_tensor.shape[2:], pred[0][:, :4], original.shape).round()
        h, w = original.shape[:2]
        thickness = max(1, int(min(h, w) * 0.005))
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, det in enumerate(pred[0]):
            mask = masks[i]
            mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else mask
            mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bin = mask_resized > 0.5
            coords = np.column_stack(np.where(mask_bin))
            if coords.shape[0] == 0:
                continue
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            damage_width = x_max - x_min
            damage_height = y_max - y_min
            ratio = max(damage_width/crop_w, damage_height/crop_h)
            if ratio < 0.1:
                severity = "minor"
            elif ratio < 0.3:
                severity = "medium"
            else:
                severity = "major"
            if hasattr(damage_seg_model, "names") and len(det) > 5:
                class_id = int(det[5])
                label = damage_seg_model.names[class_id]
            else:
                label = "broken"
            if label.lower() == "missing":
                label = "broken"
            base_label = label.split()[0].lower()
            if base_label not in damage_class_colors:
                damage_class_colors[base_label] = candidate_colors[candidate_index % len(candidate_colors)]
                candidate_index += 1
            color = damage_class_colors[base_label]
            dynamic_font_scale = 0.5 + 1.0 * ratio  
            final_label = f"{base_label}"
            cv2.rectangle(result_img, (x_min, y_min), (x_max, y_max), color, thickness)
            (text_w, text_h), _ = cv2.getTextSize(final_label, font, dynamic_font_scale, thickness)
            text_bg_y = max(y_min - text_h - 4, 0)
            cv2.rectangle(result_img, (x_min, text_bg_y), (x_min + text_w, y_min), color, -1)
            cv2.putText(result_img, final_label, (x_min, y_min - 2), font, dynamic_font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            colored_mask = np.zeros_like(result_img, dtype=np.uint8)
            colored_mask[mask_bin] = color
            result_img = cv2.addWeighted(result_img, 1.0, colored_mask, 0.7, 0)
            segmented_damage = cv2.bitwise_and(original, original, mask=(mask_bin.astype(np.uint8)*255))
            damage_crop = segmented_damage[y_min:y_max, x_min:x_max]
            if damage_crop.size > 0:
                thumb = cv2.resize(damage_crop, (50, 50))
                thumb_y = max(y_min - 60, 0)
                thumb_x = x_min
                thumb_h, thumb_w = thumb.shape[:2]
                if thumb_x + thumb_w <= w and thumb_y + thumb_h <= h:
                    result_img[thumb_y:thumb_y+thumb_h, thumb_x:thumb_x+thumb_w] = thumb
            damage_detections.append({
                "bbox": (x_min, y_min, x_max, y_max),
                "label": final_label,
                "severity": severity,
                "color": color
            })
            print(f"Damage detection {i}: bbox=({x_min}, {y_min}, {x_max}, {y_max}), label={final_label}, ratio={ratio:.2f}")
        if save_path:
            cv2.imwrite(save_path, result_img)
        return result_img, damage_detections
    else:
        print("No damage detections found.")
        return original, []

# -------------------------------
# 4. Full Pipeline: Damage Detection on Original Image (Updated with new part names)
# -------------------------------
def full_damage_detection(dir_name, file_name, extension, img_type):
    img_path = dir_name + file_name + extension
    original_img = cv2.imread(img_path)
    height, width, _ = original_img.shape
    original_img_info = {"OrgImgHeight": height, "OrgImgWidth": width}
    # Step 1: Vehicle segmentation
    veh_img, veh_status, veh_bbox = vehicle_segmentation(original_img)
    if not veh_status:
        return original_img, {"error": "Vehicle segmentation failed.","org_img_info":original_img_info}
    # Step 2: Body-part segmentation on the vehicle crop
    body_parts = body_parts_segmentation(veh_img, img_type)
    annotated_img = original_img.copy()
    damage_counts = {}
    body_parts_status = {}
    missing_body_parts = []
    damage_rectangles = []
    thickness = 1
    small_font_scale = 0.3
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Updated ignore list with new directional part names
    ignore_parts = [
        "front_tire", "back_tire", "windshield", "door_windshield", "grill",
        "left_headlight", "right_headlight", 
        "left_side_mirror", "right_side_mirror",
        "left_back_light", "right_back_light"
    ]
    
    if body_parts is not None:
        for part, info in body_parts.items():
            print(f"info about {part} _ {info}")
            body_parts_status[part] = info["status"]
            if not info["status"]:
                missing_body_parts.append(part)
                continue
            damage_counts[part] = {}
            if info["cropped"] is None:
                continue
            for (crop_img, bp_bbox) in info["cropped"]:
                if part in ignore_parts:
                    continue
                _, detections = damage_segmentation(crop_img, part_key=part)
                for det in detections:
                    d_x_min, d_y_min, d_x_max, d_y_max = det["bbox"]
                    bp_x_min, bp_y_min, _, _ = bp_bbox
                    v_x_min, v_y_min, _, _ = veh_bbox
                    final_x_min = v_x_min + bp_x_min + d_x_min
                    final_y_min = v_y_min + bp_y_min + d_y_min
                    final_x_max = v_x_min + bp_x_min + d_x_max
                    final_y_max = v_y_min + bp_y_min + d_y_max
                    # Draw a small rectangle (optional; the annotations are minimal)
                    color = det.get("color", (0, 255, 0))
                    cv2.rectangle(annotated_img, (final_x_min, final_y_min), (final_x_max, final_y_max), color, thickness)
                    (text_w, text_h), _ = cv2.getTextSize(det["label"], font, small_font_scale, thickness)
                    text_x = final_x_min + 2
                    text_y = final_y_min - 2
                    if text_y - text_h < 0:
                        text_y = final_y_min + text_h + 2
                    cv2.putText(annotated_img, det["label"], (text_x, text_y), font, small_font_scale, color, thickness, cv2.LINE_AA)
                    damage_counts[part][det["label"]] = damage_counts[part].get(det["label"], 0) + 1
                    damage_info = {
                        "damageRectangle": {
                            "x": final_x_min,
                            "y": final_y_min,
                            "width": final_x_max - final_x_min,
                            "height": final_y_max - final_y_min,
                            "label": det["label"],
                            "severity": det["severity"],
                            "damage_location":part,
                            "byAI": True,
                            "deleted": False,
                            "accuracyMatrix": {"tp": 1, "fp": 0, "fn": 0}
                        }
                    }
                    damage_rectangles.append(damage_info)
        print("missing body parts: ", missing_body_parts)
        print("body parts status: ", body_parts_status)
    final_status = "fail" if damage_rectangles else "pass"
    # Construct a simple message from damage counts (e.g., total damages detected)
    total_damages = sum(sum(d.values()) for d in damage_counts.values())
    message = f"{total_damages} damage(s) detected." if total_damages else "No damage detected."
    report = {
        "image_s3_link": None,  # to be set in endpoint
        "processed_img_s3_link": None,  # to be set in endpoin
        "extension": extension,
        "message": message,
        "final_status": final_status,
        "org_img_info": original_img_info,
        "damages": damage_rectangles,
        "missing_body_parts": missing_body_parts,
        "damage_counts": damage_counts,
        "body_parts_status": body_parts_status
    }
    return annotated_img, report