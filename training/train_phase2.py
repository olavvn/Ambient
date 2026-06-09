"""
Phase 2: 전체 파인튜닝.
임베딩/LM head와 트랜스포머 블록에 차별화된 학습률을 적용한다.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.model.anchorflow import AnchorFlowModel
from src.tokenizer import TSDTokenizer
from training.dataset import AmbientMIDIDataset
from training.losses import make_loss_fn
from src.utils.config_loader import load_yaml


def train_phase2(cfg: dict, phase1_ckpt: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = TSDTokenizer()

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

    grad_accum = cfg["phase2"].get("gradient_accumulation", 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["phase2"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"].get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["phase2"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 4),
    )

    # Phase 1 체크포인트 또는 사전학습 가중치에서 로드
    if phase1_ckpt and os.path.exists(phase1_ckpt):
        print(f"Phase 1 체크포인트에서 로드: {phase1_ckpt}")
        model = AnchorFlowModel.from_checkpoint(phase1_ckpt, device=device)
    else:
        print("사전학습 가중치에서 직접 로드 (Phase 1 스킵)")
        model = AnchorFlowModel.from_pretrained(
            model_id=cfg["pretrained"]["base_model"],
            vocab_size=tokenizer.vocab_size,
            reinit_embeddings=cfg["pretrained"]["reinit_embeddings"],
        )
    model = model.to(device)

    # 학습률 그룹 분리 (ULMFiT 원칙)
    embedding_params = [
        p for n, p in model.named_parameters()
        if "wte" in n or "lm_head" in n or "wpe" in n
    ]
    backbone_params = [
        p for n, p in model.named_parameters()
        if "wte" not in n and "lm_head" not in n and "wpe" not in n
    ]

    optimizer = AdamW(
        [
            {"params": embedding_params, "lr": cfg["phase2"]["lr_embedding"]},
            {"params": backbone_params,  "lr": cfg["phase2"]["lr_backbone"]},
        ],
        weight_decay=cfg["common"]["weight_decay"],
        betas=tuple(cfg["common"]["betas"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["phase2"]["epochs"])
    loss_fn = make_loss_fn(tokenizer.pad_id, cfg["common"]["label_smoothing"])

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, cfg["phase2"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits
            B, L, V = logits.shape
            loss = loss_fn(logits.view(B * L, V), labels.view(B * L))
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["common"]["gradient_clip"]
                )
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum

            if (step + 1) % 100 == 0:
                print(f"  [Phase2 E{epoch} S{step+1}] loss={loss.item() * grad_accum:.4f}")

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

        print(f"[Phase2 Epoch {epoch}] train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], "phase2_best.pt")
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

        # 주기적 저장
        if epoch % cfg["output"].get("save_every", 5) == 0:
            ckpt_path = os.path.join(
                cfg["output"]["checkpoint_dir"], f"phase2_epoch{epoch:03d}.pt"
            )
            torch.save({
                "model_state_dict": model.model.state_dict(),
                "config": {
                    "pretrained_model_id": cfg["pretrained"]["base_model"],
                    "vocab_size": tokenizer.vocab_size,
                },
                "epoch": epoch,
                "val_loss": avg_val,
            }, ckpt_path)

    print("Phase 2 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--phase1_ckpt", default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    train_phase2(cfg, phase1_ckpt=args.phase1_ckpt)
