import torch
import torchvision.transforms.functional as F
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. Dataset Preparation & Transforms
# ==========================================
class VOCSegmentationTransforms:
    def __call__(self, img, target):
        # Convert PIL Image to tensor and apply ImageNet normalization
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Convert target PIL Image (palette-based) to an integer tensor
        # PASCAL VOC masks use indices 0-20 for classes, and 255 for ignored border regions
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return img, target

# ==========================================
# 2. mIoU Calculation Metric
# ==========================================
class SegmentationMetric:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes))

    def update(self, label_true, label_pred):
        # Filter out ignored regions (index 255 in PASCAL VOC)
        mask = (label_true >= 0) & (label_true < self.num_classes)
        
        # Calculate confusion matrix using fast bincounting
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask], 
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += hist

    def get_miou(self):
        hist = self.confusion_matrix
        # Intersection over Union formula: True Positives / (True Positives + False Positives + False Negatives)
        # i.e., diagonal / (row sum + column sum - diagonal)
        iou_per_class = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        
        # Ignore NaNs (classes that do not appear in the dataset at all)
        valid_iou = iou_per_class[~np.isnan(iou_per_class)]
        return np.mean(valid_iou), iou_per_class

# ==========================================
# 3. Main Evaluation Function
# ==========================================
def evaluate_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset (Validation set)
    # Using batch_size=1 because VOC images have varying dimensions
    print("Loading PASCAL VOC 2012 Dataset...")
    dataset = VOCSegmentation(
        root='./data', 
        year='2012', 
        image_set='val', 
        download=True, 
        transforms=VOCSegmentationTransforms()
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Load Pretrained FCN-ResNet50
    print("Loading Pretrained FCN-ResNet50...")
    weights = FCN_ResNet50_Weights.DEFAULT
    model = fcn_resnet50(weights=weights)
    model = model.to(device)
    model.eval()

    # PASCAL VOC has 21 classes (20 objects + 1 background)
    metrics = SegmentationMetric(num_classes=21)

    print(f"Starting evaluation on {len(dataset)} images...")
    with torch.no_grad():
        for imgs, targets in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            # imgs shape: [1, 3, H, W], targets shape: [1, H, W]
            imgs = imgs.to(device)
            targets = targets.numpy()

            # Forward pass
            output = model(imgs)['out'] # output shape: [1, 21, H, W]
            
            # Get class predictions using argmax over the class dimension (dim=1)
            preds = output.argmax(dim=1).cpu().numpy() # preds shape: [1, H, W]

            # Update confusion matrix
            metrics.update(targets.flatten(), preds.flatten())

    # Calculate final mIoU
    mIoU, iou_per_class = metrics.get_miou()
    
    print("\n" + "="*30)
    print(f"Evaluation Complete!")
    print(f"Mean IoU (mIoU): {mIoU:.4f}")
    print("="*30)
    
    # Optional: Display per-class IoU
    voc_classes = [
        "__background__", "aeroplane", "bicycle", "bird", "boat", "bottle", 
        "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", 
        "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]
    for cls_name, iou in zip(voc_classes, iou_per_class):
        print(f"{cls_name.rjust(15)}: {iou:.4f}")

if __name__ == '__main__':
    evaluate_model()