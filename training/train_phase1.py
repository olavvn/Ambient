"""
Phase 1: 임베딩 적응 학습.
트랜스포머 블록을 동결하고 토큰 임베딩 + LM head만 학습한다.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model.anchorflow import AnchorFlowModel, resize_and_reinit_embeddings
from src.tokenizer import TSDTokenizer
from training.dataset import AmbientMIDIDataset
from training.losses import make_loss_fn
from src.utils.config_loader import load_yaml


def freeze_backbone(model):
    for name, param in model.named_parameters():
        if "wte" in name or "lm_head" in name or "wpe" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def train_phase1(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = TSDTokenizer()

    # 데이터셋
    train_ds = AmbientMIDIDataset(
        cfg["data"]["train_dir"],
        seq_len=cfg["common"]["seq_len"],
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
    )
    val_ds = AmbientMIDIDataset(
        cfg["data"]["val_dir"],
        seq_len=cfg["common"]["seq_len"],
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["phase1"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["phase1"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
    )

    # 모델 로드 + 임베딩 교체
    model = AnchorFlowModel.from_pretrained(
        model_id=cfg["pretrained"]["base_model"],
        vocab_size=tokenizer.vocab_size,
        reinit_embeddings=cfg["pretrained"]["reinit_embeddings"],
    )
    model = model.to(device)
    freeze_backbone(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Phase 1 — 학습 파라미터: {trainable:,} / {total:,}")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["phase1"]["lr"],
        weight_decay=cfg["common"]["weight_decay"],
        betas=tuple(cfg["common"]["betas"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["phase1"]["epochs"])
    loss_fn = make_loss_fn(tokenizer.pad_id, cfg["common"]["label_smoothing"])

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, cfg["phase1"]["epochs"] + 1):
        # Train
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # (B, L, V)
            B, L, V = logits.shape
            loss = loss_fn(logits.view(B * L, V), labels.view(B * L))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg["common"]["gradient_clip"]
            )
            optimizer.step()
            total_loss += loss.item()

            if (step + 1) % 100 == 0:
                print(f"  [Phase1 E{epoch} S{step+1}] loss={loss.item():.4f}")

        scheduler.step()
        avg_train = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels    = batch["labels"].to(device)
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
                B, L, V = logits.shape
                val_loss += loss_fn(logits.view(B * L, V), labels.view(B * L)).item()
        avg_val = val_loss / len(val_loader)

        print(f"[Phase1 Epoch {epoch}] train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], "phase1_best.pt")
            torch.save({
                "model_state_dict": model.model.state_dict(),
                "config": {
                    "pretrained_model_id": cfg["pretrained"]["base_model"],
                    "vocab_size": tokenizer.vocab_size,
                },
                "epoch": epoch,
                "val_loss": avg_val,
            }, ckpt_path)
            print(f"  → 체크포인트 저장: {ckpt_path}")

    print("Phase 1 완료")
    return os.path.join(cfg["output"]["checkpoint_dir"], "phase1_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_phase1(cfg)
