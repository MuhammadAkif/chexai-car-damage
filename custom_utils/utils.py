
import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import letterbox


def load_model(device,weights,imgsz):
        # device= 'cuda:0' #for cpu-> 'cpu'
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride

    if half:
        model.half()  # to FP16

    names = model.module.names if hasattr(model, 'module') else model.names

    ###### infrence ####
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

    return stride, model, names, half ,old_img_w,old_img_h, old_img_b

def img_preprocessing(frame, imgsz, stride, device, half):
    img = letterbox(frame, imgsz, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    return img