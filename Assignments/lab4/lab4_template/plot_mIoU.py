import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


# Usage: python plot_mIoU.py [log file]   (default: postlab_terminal_outputs.txt next to this script)
log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("postlab_terminal_outputs.txt")
runs = {"none": [], "response": [], "feature": []}
teacher_miou = None
current_run = None

run_pattern = re.compile(r"Training student with kd=(none|response|feature)")
# Matches both the old log format and the new one (which adds "lr ..." after the mIoU)
epoch_pattern = re.compile(r"epoch\s+(\d+)/\d+\b.*?val mIoU\s+([\d.]+)")
teacher_pattern = re.compile(r"^Teacher \(FCN-ResNet50\)\s+([\d.]+)")

for line in log_path.read_text(encoding="utf-8").splitlines():
    run_match = run_pattern.search(line)
    if run_match:
        current_run = run_match.group(1)
        continue

    epoch_match = epoch_pattern.search(line)
    if current_run and epoch_match:
        epoch = int(epoch_match.group(1))
        miou = float(epoch_match.group(2))
        runs[current_run].append((epoch, miou))
        continue

    teacher_match = teacher_pattern.match(line)
    if teacher_match:
        teacher_miou = float(teacher_match.group(1))

# Runs may stop early, so each run only needs sequential epochs starting at 1
for name, values in runs.items():
    epochs = [epoch for epoch, _ in values]
    if not epochs or epochs != list(range(1, len(epochs) + 1)):
        raise ValueError(f"Expected sequential epochs starting at 1 for {name}; found {len(epochs)}.")
if teacher_miou is None:
    raise ValueError("Teacher mIoU was not found in the log.")

fig, ax = plt.subplots(figsize=(10, 6))

styles = [
    ("feature", "Feature KD", "blue"),
    ("none", "No KD", "lightgray"),
    ("response", "Response KD", "lightblue"),
]
for key, label, color in styles:
    epochs, scores = zip(*runs[key])
    ax.plot(epochs, scores, label=label, color=color)
    best_epoch, best_score = max(runs[key], key=lambda item: item[1])
    ax.plot(best_epoch, best_score, marker="o", color=color)   # best epoch = saved checkpoint

ax.axhline(
    teacher_miou,
    label=f"Teacher ({teacher_miou:.1f}%)",
    color="gray",
    linestyle="--",
)

last_epoch = max(len(values) for values in runs.values())
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation mIoU (%)")
ax.set_xlim(1, last_epoch)
ax.set_title("Validation mIoU by Epoch")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

output_path = Path(__file__).with_name("plots") / "validation_miou.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
print(f"Saved {output_path}")
