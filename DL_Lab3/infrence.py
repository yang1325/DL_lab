from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T
import os
import torch
import pandas as pd
import json
import cv2
import numpy as np

from RLE import encode_mask


val_transform = T.Compose([T.ToTensor(),])

def prepare_image(image_path, target_size=(512, 512)):
    image = cv2.imread(image_path)
    orig_w, orig_h = image.shape[:2]
    image = torch.tensor(np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0)
    return image, (orig_w, orig_h)
    image = Image.fromarray(image.astype(np.uint8))
    image = image.resize(target_size, Image.BILINEAR)
    image = torch.tensor(np.array(image).transpose(2, 0, 1).astype(np.float32) / 255.0)
    return image, (orig_w, orig_h)


def rescale_bbox(box, orig_size, resized_size=(512, 512)):
    return box
    x_scale = orig_size[0] / resized_size[0]
    y_scale = orig_size[1] / resized_size[1]

    x, y, w, h = box
    x *= x_scale
    y *= y_scale
    w *= x_scale
    h *= y_scale

    return [x, y, w, h]

def generate_predictions(model, image_folder, json_path, score_thresh=0.3):
    model.eval()

    coco_results = []
    with open(json_path, 'r') as file:
        ann_json = json.load(file)

    for pic_dict in ann_json:
        image_name = pic_dict["file_name"]
        image_id = pic_dict["id"]
        image_path = os.path.join(image_folder, image_name)

        image_tensor, orig_size = prepare_image(image_path)
        image_tensor = image_tensor.to("cuda")

        with torch.no_grad():
            output = model([image_tensor])[0]

        boxes = output["boxes"].cpu()
        labels = output["labels"].cpu()
        scores = output["scores"].cpu()
        masks = output["masks"].cpu()

        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]
            score = scores[i]
            mask = masks[i][0]>score_thresh
            # mask_uint8 = (mask.numpy().astype(np.uint8)) * 255
            # mask_img = Image.fromarray(mask_uint8, mode="L")
            # resized_mask = mask_img.resize(orig_size, Image.NEAREST)
            # mask = np.array(resized_mask) > 127

            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1

            rescaled_box = rescale_bbox([x1, y1, x2, y2], orig_size)

            coco_results.append({
                "image_id": int(image_id),
                "bbox": rescaled_box,
                "score": round(score.item(), 4),
                "category_id": label.item(),
                "segmentation":encode_mask(binary_mask=mask)
            })


    return coco_results

def output_prediction(coco_results):
    with open("test-results.json", "w") as f:
        json.dump(coco_results, f, indent=2)
