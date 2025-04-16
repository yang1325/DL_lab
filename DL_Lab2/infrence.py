from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T
import os
import torch
import pandas as pd
import json

val_transform = T.Compose([T.ToTensor(),])

def prepare_image(image_path, target_size=(256, 128)):
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    new_w, new_h = target_size
    image = F.resize(image, (new_h, new_w))
    return val_transform(image), (orig_w, orig_h)


def rescale_bbox(box, orig_size, resized_size=(256, 128)):
    x_scale = orig_size[0] / resized_size[0]
    y_scale = orig_size[1] / resized_size[1]

    x, y, w, h = box
    x *= x_scale
    y *= y_scale
    w *= x_scale
    h *= y_scale

    return [x, y, w, h]

def detect_and_recognize(model, image_folder, score_thresh=0.6):
    model.eval()

    coco_results = []
    recog_results = []

    image_list = sorted(os.listdir(image_folder))

    for image_name in image_list:
        image_id = os.path.splitext(image_name)[0]
        image_path = os.path.join(image_folder, image_name)

        image_tensor, orig_size = prepare_image(image_path)
        image_tensor = image_tensor.to("cuda")

        with torch.no_grad():
            output = model([image_tensor])[0]

        boxes = output["boxes"].cpu()
        labels = output["labels"].cpu()
        scores = output["scores"].cpu()

        digits_in_image = []

        for box, label, score in zip(boxes, labels, scores):

            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1

            rescaled_box = rescale_bbox([x1, y1, w, h], orig_size)

            coco_results.append({
                "image_id": int(image_id),
                "bbox": rescaled_box,
                "score": round(score.item(), 4),
                "category_id": label.item()
            })

            if score < score_thresh:
                continue

            digits_in_image.append((rescaled_box[0], label.item()))

        if digits_in_image:
            sorted_digits = sorted(digits_in_image, key=lambda x: x[0])
            pred_label = ''.join(str(d[1] - 1) for d in sorted_digits)
        else:
            pred_label = "-1"

        recog_results.append({
            "image_id": int(image_id),
            "pred_label": pred_label
        })

    return coco_results, recog_results

def output_prediction(coco_results, recog_results):
    with open("pred.json", "w") as f:
        json.dump(coco_results, f, indent=2)
    
    pd.DataFrame(recog_results).to_csv("pred.csv", index=False)
