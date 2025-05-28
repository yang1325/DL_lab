import csv
import os
from math import log10


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from dataset import DegrationDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
from net.model import PromptIR
from options import options as opt


class PromptIRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = nn.L1Loss()

    def forward(self, degrad_patch):
        return self.net(degrad_patch)

    def compute_loss(self, degrad_patch, clean_patch):
        restored = self.net(degrad_patch)
        loss = self.loss_fn(restored, clean_patch)
        return loss
    
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def calculate_psnr(pred, target, max_pixel=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100
    return 20 * log10(max_pixel) - 10 * log10(mse.item())

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PromptIRModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=15, max_epochs=opt.epochs)

    trainset = DegrationDataset(opt)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    validset = DegrationDataset(opt, train=False)
    validloader = DataLoader(validset, batch_size=1, shuffle=True, num_workers=0)
    os.makedirs(opt.ckpt_dir, exist_ok=True)

    csv_path = os.path.join(opt.ckpt_dir, "training_log.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Validation PSNR"])

    best_psnr = 0.0
    for epoch in range(1, opt.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{opt.epochs}")

        for batch in pbar:
            (clean_name, de_id), degrad_patch, clean_patch = batch
            degrad_patch = degrad_patch.to(device)
            clean_patch = clean_patch.to(device)

            optimizer.zero_grad()
            loss = model.compute_loss(degrad_patch, clean_patch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        scheduler.step()

        avg_loss = total_loss / len(trainloader)
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

        model.eval()
        total_psnr = 0
        with torch.no_grad():
            pbar = tqdm(validloader, desc=f"[Valid] Epoch {epoch}")
            for batch in pbar:
                (clean_name, de_id), degrad_patch, clean_patch = batch
                degrad_patch = degrad_patch.to(device)
                clean_patch = clean_patch.to(device)

                restored = model(degrad_patch)
                restored = torch.clamp(restored, 0, 1)

                psnr = calculate_psnr(restored, clean_patch)
                pbar.set_postfix(psnr=psnr)
                total_psnr += psnr

        avg_psnr = total_psnr / len(validloader)
        print(f"[Epoch {epoch}] Validation PSNR: {avg_psnr:.2f} dB")

        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_loss, avg_psnr])

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            best_ckpt_path = os.path.join(opt.ckpt_dir, "best.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"Best model updated! PSNR = {best_psnr:.2f} dB")

        ckpt_path = os.path.join(opt.ckpt_dir, "checkpoint.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss
        }, ckpt_path)

        if(epoch%20 == 0):
            ckpt_path = os.path.join(opt.ckpt_dir, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)

if __name__ == '__main__':
    main()