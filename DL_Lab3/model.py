import torch
import torchvision

def get_model(num_classes = 5):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
    model.train()

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, num_classes)

    model.to(device)
    return model, device
