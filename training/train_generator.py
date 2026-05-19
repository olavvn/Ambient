"""AmbientFlowModel 훈련 스크립트"""
import torch
from torch.utils.data import DataLoader
from src.model.transformer import AmbientFlowModel
from src.model.config import ModelConfig
from training.dataset import ThemeGenerationDataset
from training.losses import ThemeAwareCrossEntropy
from src.utils.config_loader import load_yaml
import pathlib


def train(model_config: str = "configs/model_config.yaml",
          training_config: str = "configs/training_config.yaml"):
    model_cfg_dict = load_yaml(model_config)
    train_cfg = load_yaml(training_config)["generation"]

    cfg = ModelConfig(**model_cfg_dict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AmbientFlowModel(cfg).to(device)

    dataset = ThemeGenerationDataset(
        train_cfg["data_dir"],
        max_theme_len=cfg.max_theme_len,
        max_target_len=cfg.max_seq_len,
    )
    loader = DataLoader(
        dataset, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=0, pin_memory=(device == "cuda"),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=train_cfg["lr"],
        total_steps=train_cfg["epochs"] * max(len(loader), 1),
    )
    loss_fn = ThemeAwareCrossEntropy(cfg.vocab_size, cfg.label_smoothing)

    ckpt_dir = pathlib.Path(train_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    for epoch in range(train_cfg["epochs"]):
        model.train()
        total_loss = 0.0

        for batch in loader:
            theme = batch["theme_tokens"].to(device)
            target = batch["target_tokens"].to(device)
            theme_mask = batch["theme_pad_mask"].to(device)

            logits = model(theme, target[:, :-1], theme_mask)
            loss = loss_fn(logits, target[:, 1:])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           train_cfg["gradient_clip"])
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / max(len(loader), 1)
        print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": model_cfg_dict,
            }, ckpt_dir / "best_model.pt")

    print("생성 모델 훈련 완료")


if __name__ == "__main__":
    train()
