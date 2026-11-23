import argparse
import os

import torch
from models.my_student import MyStudentNet
from models.student import StudentNet
from torch.utils.data import DataLoader
from utils.dataset import get_voc
from utils.metrics import mean_iou, update_confusion
from utils.profiler import ms_per_image
from utils.teacher import load_teacher
from utils.visualization import save_comparison

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./data")
parser.add_argument("--size", type=int, default=320)
parser.add_argument("-b", "--batch", type=int, default=16)
args = parser.parse_args()
# Show for visualization
num_examples = 8
example_batch = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = DataLoader(
    get_voc(args.data, "val", args.size), batch_size=args.batch, shuffle=False
)

models = {"Teacher (FCN-ResNet50)": load_teacher(device)}
candidates = [
    ("student_none", "Student, no KD", StudentNet),
    ("student_response", "Student, response KD", StudentNet),
    ("student_feature", "Student, feature KD", StudentNet),
    ("my_student_none", "My student, no KD", MyStudentNet),
    ("my_student_response", "My student, response KD", MyStudentNet),
    ("my_student_feature", "My student, feature KD", MyStudentNet),
]
for stem, name, net in candidates:
    path = f"checkpoints/{stem}.pth"
    if os.path.exists(path):
        m = net(pretrained=False).to(device).eval()
        m.load_state_dict(torch.load(path, map_location=device))
        models[name] = m
    else:
        print(f"Skipping {name}: {path} not found")
rows, examples = [], {}
with torch.no_grad():
    for name, model in models.items():
        conf = torch.zeros(21, 21, dtype=torch.long)
        for i, (img, mask) in enumerate(loader):
            pred = model(img.to(device))["out"].argmax(1).cpu()
            conf = update_confusion(conf, pred, mask)
            if i == example_batch:
                examples[name] = pred[:num_examples]
                first_img, first_mask = img[:num_examples], mask[:num_examples]
        params = sum(p.numel() for p in model.parameters())
        rows.append(
            (name, 100 * mean_iou(conf), params, ms_per_image(model, device, args.size))
        )

print(f"\n{'Model':<26}{'mIoU':>8}{'Params':>14}{'ms/img':>10}")
for name, miou, params, ms in rows:
    print(f"{name:<26}{miou:>8.1f}{params:>14,}{ms:>10.1f}")

save_comparison(first_img, first_mask, examples, "plots/compare.png")
print("\nSaved plots/compare.png")
