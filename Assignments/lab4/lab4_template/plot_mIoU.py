import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


log_path = Path(__file__).with_name("postlab_terminal_outputs.txt")
runs = {"none": [], "response": [], "feature": []}
teacher_miou = None
current_run = None

run_pattern = re.compile(r"Training student with kd=(none|response|feature)")
epoch_pattern = re.compile(r"epoch\s+(\d+)/100\b.*?val mIoU\s+([\d.]+)")
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

expected_epochs = list(range(1, 101))
for name, values in runs.items():
    if [epoch for epoch, _ in values] != expected_epochs:
        raise ValueError(f"Expected 100 sequential epochs for {name}; found {len(values)}.")
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

ax.axhline(
    teacher_miou,
    label=f"Teacher ({teacher_miou:.1f}%)",
    color="gray",
    linestyle="--",
)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation mIoU (%)")
ax.set_xlim(1, 100)
ax.set_xticks(range(0, 101, 10))
ax.set_title("Validation mIoU by Epoch")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()

output_path = Path(__file__).with_name("plots") / "validation_miou.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150)
print(f"Saved {output_path}")