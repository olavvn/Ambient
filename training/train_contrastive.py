"""대조 학습으로 SegmentEncoder 훈련"""
import torch
from torch.utils.data import DataLoader
from src.theme.contrastive import SegmentEncoder, NTXentLoss
from training.dataset import ContrastiveDataset
from src.utils.config_loader import load_yaml
import pathlib


def train(config_path: str = "configs/training_config.yaml"):
    cfg = load_yaml(config_path)["contrastive"]

    dataset = ContrastiveDataset(cfg["data_dir"], segment_len=cfg["segment_len"])
    loader = DataLoader(dataset, batch_size=cfg["batch_size"],
                        shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SegmentEncoder(vocab_size=238, d_model=256, proj_dim=128).to(device)
    loss_fn = NTXentLoss(temperature=cfg["temperature"])
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=cfg["lr"],
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    ckpt_dir = pathlib.Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        total_loss = 0.0
        for seg1, seg2 in loader:
            seg1, seg2 = seg1.to(device), seg2.to(device)
            z1 = encoder(seg1)
            z2 = encoder(seg2)
            loss = loss_fn(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch:3d} | Loss: {avg:.4f}")

        if (epoch + 1) % cfg["save_every"] == 0:
            torch.save(encoder.state_dict(),
                       ckpt_dir / f"encoder_ep{epoch}.pt")

    torch.save(encoder.state_dict(), ckpt_dir / "best_encoder.pt")
    print("대조 학습 완료")


if __name__ == "__main__":
    train()
