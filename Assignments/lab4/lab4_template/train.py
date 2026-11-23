import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.my_student import MyStudentNet
from models.student import StudentNet
from torch.utils.data import DataLoader
from utils.dataset import get_voc
from utils.kd import feature_kd_loss, response_kd_loss
from utils.metrics import mean_iou, update_confusion
from utils.teacher import load_teacher, teacher_forward

parser = argparse.ArgumentParser()
parser.add_argument("--kd", choices=["none", "response", "feature"], default="none")
parser.add_argument("--model", choices=["student", "mine"], default="student")
parser.add_argument("--data", default="./data")
parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument("-b", "--batch", type=int, default=16)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--size", type=int, default=320)
parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=0.5)
parser.add_argument("--tau", type=float, default=4.0)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument(
    "--lr-patience",
    type=int,
    default=2,
    help="epochs without improvement before the learning rate is reduced",
)
parser.add_argument(
    "--lr-factor",
    type=float,
    default=0.5,
    help="factor the learning rate is multiplied by on a plateau",
)
parser.add_argument(
    "--patience",
    type=int,
    default=5,
    help="early stopping: epochs without improvement before training stops",
)
parser.add_argument(
    "--tol",
    type=float,
    default=0.1,
    help="minimum rise in validation mIoU (in mIoU points) that counts as an improvement",
)
parser.add_argument("--workers", type=int, default=2)
parser.add_argument(
    "--iters",
    type=int,
    default=0,
    help="kick-start: stop after this many iterations (0 = full training)",
)
args = parser.parse_args()

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
train_loader = DataLoader(
    get_voc(args.data, "train", args.size),
    batch_size=args.batch,
    shuffle=True,
    num_workers=args.workers,
    drop_last=True,
)
val_loader = DataLoader(
    get_voc(args.data, "val", args.size),
    batch_size=args.batch,
    shuffle=False,
    num_workers=args.workers,
)

student = (MyStudentNet() if args.model == "mine" else StudentNet()).to(device)
print(f"Model: {args.model}, {sum(p.numel() for p in student.parameters()):,} params")
params = list(student.parameters())
teacher = load_teacher(device) if args.kd != "none" else None

# 1x1 adapters map student features to the teacher's channel counts (training only)
adapters = None
if args.kd == "feature":
    adapters = nn.ModuleDict(
        {"low": nn.Conv2d(24, 512, 1), "high": nn.Conv2d(576, 2048, 1)}
    ).to(device)
    params += list(adapters.parameters())

optimizer = torch.optim.Adam(params, lr=args.lr)
# Reduce the learning rate when validation mIoU (maximised) stops improving by more than --tol points
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=args.lr_factor,
    patience=args.lr_patience,
    threshold=args.tol,
    threshold_mode="abs",
)

print(
    f"Training student with kd={args.kd} on {device} ({hardware})"
    + (
        f" for {args.iters} iterations"
        if args.iters
        else f" for up to {args.epochs} epochs"
    )
)
if not args.iters:
    print(
        f"Settings: batch {args.batch}, lr {args.lr:g}, size {args.size}, "
        f"alpha {args.alpha}, beta {args.beta}, tau {args.tau}, gamma {args.gamma}, "
        f"lr-patience {args.lr_patience}, lr-factor {args.lr_factor}, "
        f"patience {args.patience}, tol {args.tol}"
    )

os.makedirs("checkpoints", exist_ok=True)
prefix = "student" if args.model == "student" else "my_student"
ckpt_path = f"checkpoints/{prefix}_{args.kd}.pth"
plot_path = (
    f"plots/loss_{args.kd}.png"
    if args.model == "student"
    else f"plots/loss_my_student_{args.kd}.png"
)
# best mIoU so far: decides which weights are saved
best_miou, best_epoch = -1.0, 0
stop_ref, bad_epochs = (
    -1.0,
    0,
)
epoch_times = []

losses = []
start = time.time()
for epoch in range(1, (10**6 if args.iters else args.epochs) + 1):
    epoch_start = time.time()
    student.train()
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        out = student(img)
        if args.kd == "none":
            loss = F.cross_entropy(out["out"], mask, ignore_index=255)
        else:
            with torch.no_grad():
                z_t, f_t = teacher_forward(teacher, img)
            if args.kd == "response":
                loss = response_kd_loss(
                    out["out"], z_t, mask, args.alpha, args.beta, args.tau
                )
            else:
                loss = F.cross_entropy(
                    out["out"], mask, ignore_index=255
                ) + args.gamma * (
                    feature_kd_loss(adapters["low"](out["low"]), f_t["layer2"])
                    + feature_kd_loss(adapters["high"](out["high"]), f_t["layer4"])
                )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if args.iters:
            if len(losses) % 5 == 0:
                print(f"iter {len(losses):3d}/{args.iters}  loss {loss.item():.3f}")
            if len(losses) >= args.iters:
                break
    if args.iters:
        if len(losses) >= args.iters:
            break
        continue

    student.eval()
    conf = torch.zeros(21, 21, dtype=torch.long)
    with torch.no_grad():
        for img, mask in val_loader:
            pred = student(img.to(device))["out"].argmax(1).cpu()
            conf = update_confusion(conf, pred, mask)
    miou = 100 * mean_iou(conf)
    # learning rate used during this epoch
    lr_now = optimizer.param_groups[0]["lr"]
    epoch_times.append(time.time() - epoch_start)

    marks = []
    # keep the weights of the best epoch
    if miou > best_miou:
        best_miou, best_epoch = miou, epoch
        torch.save(student.state_dict(), ckpt_path)
        marks.append("saved best")

    if miou > stop_ref + args.tol:
        # counts as an improvement for early stopping
        stop_ref, bad_epochs = miou, 0
    else:
        bad_epochs += 1
    scheduler.step(miou)

    print(
        f"epoch {epoch}/{args.epochs}  loss {sum(losses[-len(train_loader) :]) / len(train_loader):.3f}"
        f"  val mIoU {miou:.1f}  lr {lr_now:.2e}  ({time.time() - start:.0f} s)"
        + (f"  [{', '.join(marks)}]" if marks else "")
    )

    if bad_epochs >= args.patience:
        print(
            f"Early stopping: no improvement of more than {args.tol} mIoU for {args.patience} epochs"
        )
        break

os.makedirs("plots", exist_ok=True)
if args.iters:
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    print(
        f"Kick-start finished: average loss {first:.3f} (first 5) -> {last:.3f} (last 5)"
    )
else:
    total = time.time() - start
    print(
        f"Summary kd={args.kd}: {len(epoch_times)} epochs, best val mIoU {best_miou:.1f} at epoch {best_epoch}, "
        f"final lr {optimizer.param_groups[0]['lr']:.2e}, training time {total:.0f} s "
        f"({total / 60:.1f} min, {total / len(epoch_times):.1f} s/epoch), hardware {hardware}"
    )
    print(f"Saved {ckpt_path} (weights from epoch {best_epoch})")
plt.figure(figsize=(6, 4))
plt.plot(losses)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.title(f"Training loss (kd={args.kd})")
plt.tight_layout()
plt.savefig(
    f"plots/loss_{args.kd}.png"
    if args.model == "student"
    else f"plots/loss_my_student_{args.kd}.png"
)
print(f"Saved plots/loss_{args.kd}.png")
