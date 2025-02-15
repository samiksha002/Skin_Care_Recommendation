import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from pycocotools.coco import COCO
import os
from PIL import Image


class AcneDataset(Dataset):
    def __init__(self, annotation_file, image_dir, transforms=None):
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, index):
        image_id = self.ids[index]
        annotation = self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id))
        image_path = os.path.join(self.image_dir, self.coco.imgs[image_id]['file_name'])

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Extract boxes and labels
        boxes = []
        labels = []
        for ann in annotation:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([image_id])}

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.ids)


def get_data_loaders(train_json, valid_json, test_json, image_dir, batch_size=4):
    transform = T.Compose([T.ToTensor()])

    train_dataset = AcneDataset(train_json, image_dir, transforms=transform)
    valid_dataset = AcneDataset(valid_json, image_dir, transforms=transform)
    test_dataset = AcneDataset(test_json, image_dir, transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    return train_loader, valid_loader, test_loader


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    return model


def train_one_epoch(model, optimizer, data_loader, device, scaler):
    model.train()
    total_loss = 0
    print("Training started...")  # ✅ Check if training starts

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        
        # ✅ Use mixed precision training to save memory
        with torch.amp.autocast("cuda"):  
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()

    return total_loss / len(data_loader)

def validate(model, data_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(sum(loss.values()) for loss in loss_dict)  # ✅ Works if each item is a dictionary
            total_loss += losses.item()

    return total_loss / len(data_loader)


def train_faster_rcnn(train_json, valid_json, test_json, image_dir, num_classes=7, epochs=5, batch_size=2, lr=0.0001):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = get_data_loaders(train_json, valid_json, test_json, image_dir, batch_size)
    model = get_model(num_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ✅ Use GradScaler for mixed precision training
    scaler = torch.amp.GradScaler("cuda")  

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, scaler)
        val_loss = validate(model, valid_loader, DEVICE)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "faster_rcnn_acne.pth")
    print("Model saved!")
    
    return model, test_loader


def test_model(model, test_loader, device):
    model.eval()
    results = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for img_id, output in zip(targets, outputs):
                results.append({
                    "image_id": img_id["image_id"].item(),
                    "boxes": output["boxes"].cpu().numpy(),
                    "scores": output["scores"].cpu().numpy(),
                    "labels": output["labels"].cpu().numpy(),
                })
    
    return results


if __name__ == "__main__":
    # Set file paths
    TRAIN_JSON = r"D:\Acne_Project\train_coco.json"
    VALID_JSON = r"D:\Acne_Project\valid_coco.json"
    TEST_JSON = r"D:\Acne_Project\test_coco.json"
    IMAGE_DIR = r"D:\Acne_Project\images"

    # Train model
    model, test_loader = train_faster_rcnn(TRAIN_JSON, VALID_JSON, TEST_JSON, IMAGE_DIR, num_classes=7, epochs=5)

    # Load trained model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("faster_rcnn_acne.pth"))
    model.to(DEVICE)

    # Run inference
    test_results = test_model(model, test_loader, DEVICE)
    print("Testing completed!")
