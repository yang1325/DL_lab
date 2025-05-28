# infer.py

import os
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.transforms as T
import numpy as np

from net.model import PromptIR

class PromptIRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = torch.nn.L1Loss()

    def forward(self, degrad_patch):
        return self.net(degrad_patch)

    def compute_loss(self, degrad_patch, clean_patch):
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        return loss

def load_model(model_path, device):
    model = PromptIRModel().to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=True)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model

def process_images(model, input_dir, output_dir, device):
    os.makedirs(output_dir, exist_ok=True)

    transform = T.Compose([T.ToTensor()])
    to_pil = T.ToPILImage()

    image_list = [f for f in os.listdir(input_dir)
                  if f.lower().endswith((".png"))]
    # print(os.listdir(input_dir))

    for img_name in tqdm(image_list, desc="infer"):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        image = Image.open(input_path).convert("RGB")
        input_tensor = transform(np.array(image)).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            output = torch.clamp(output, 0, 1)

        restored_img = to_pil(output.squeeze(0).cpu())
        restored_img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="PromptIR Inference Script")
    parser.add_argument('--model', type=str, default="train_ckpt/epoch_140.pth", help="Path to model checkpoint (e.g. best.pth)")
    parser.add_argument('--input_dir', type=str, default="data/hw4_realse_dataset/test/degraded", help="Path to input folder with degraded images")
    parser.add_argument('--output_dir', type=str, default="output", help="Path to save restored images")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    process_images(model, args.input_dir, args.output_dir, device)

if __name__ == '__main__':
    main()
