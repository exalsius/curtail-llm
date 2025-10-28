"""Vision models, data loading, and training for image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from tqdm import tqdm

from pilot.data import ShardedDataset, DATASET_SIZES


# ============================================================================
# Models
# ============================================================================

class SimpleCNN(nn.Module):
    """Simple CNN with 2 conv layers and 3 fully connected layers."""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 model adapted for classification."""

    def __init__(self, num_classes=10, pretrained=False):
        super(EfficientNetB0, self).__init__()
        if pretrained:
            self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            self.model = efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_model(model_type: str, **kwargs):
    """Get vision model by type."""
    if model_type == "simple_cnn":
        return SimpleCNN(**kwargs)
    elif model_type == "efficientnet_b0":
        return EfficientNetB0(**kwargs)
    else:
        raise ValueError(f"Unknown vision model: {model_type}")


# ============================================================================
# Data Loading & Transforms
# ============================================================================

def get_transforms(dataset_name: str):
    """Get transforms for vision dataset."""
    if "imagenet" in dataset_name.lower():
        return Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]), "image"
    else:
        # CIFAR-10 and similar
        return Compose([
            ToTensor(),
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]), "img"


def apply_transforms(batch, dataset_name):
    """Apply transforms to batch."""
    transforms, image_key = get_transforms(dataset_name)
    if "imagenet" in dataset_name.lower():
        batch[image_key] = [transforms(img.convert("RGB")) for img in batch[image_key]]
    else:
        batch[image_key] = [transforms(img) for img in batch[image_key]]
    return batch


def get_train_loader(dataset_name, shard_id, num_shards, processed_batches, batch_size):
    """Get training dataloader for vision dataset."""
    from datasets import load_dataset

    # Determine streaming mode
    use_streaming = dataset_name in DATASET_SIZES and dataset_name != "uoft-cs/cifar10"

    # Load dataset
    dataset = load_dataset(dataset_name, split="train", streaming=use_streaming)

    # Apply transforms
    if not use_streaming:
        dataset = dataset.with_format("torch").with_transform(
            lambda batch: apply_transforms(batch, dataset_name)
        )

    # Create sharded dataset
    sharded_dataset = ShardedDataset(
        dataset=dataset,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
        dataset_name=dataset_name if use_streaming else None,
        streaming=use_streaming,
    )

    return DataLoader(sharded_dataset, batch_size=batch_size, shuffle=False)


def get_test_loader(dataset_name, batch_size):
    """Get test dataloader for vision dataset."""
    from datasets import load_dataset

    split = "validation" if "imagenet" in dataset_name.lower() else "test"
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.with_format("torch").with_transform(
        lambda batch: apply_transforms(batch, dataset_name)
    )
    return DataLoader(dataset, batch_size=batch_size)


# ============================================================================
# Training & Evaluation
# ============================================================================

def train(model, trainloader, num_batches, lr, device, weight_decay=0.01):
    """Train vision model for specified number of batches."""
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()

    running_loss = 0.0
    batches_processed = 0

    for batch in tqdm(trainloader, desc="Training", total=num_batches):
        images = batch["img"].to(device) if "img" in batch else batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches_processed += 1

        if batches_processed >= num_batches:
            break

    avg_loss = running_loss / batches_processed if batches_processed > 0 else 0.0
    return avg_loss, batches_processed


def test(model, testloader, device):
    """Evaluate vision model on test set."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(testloader, desc="Testing"):
            images = batch["img"].to(device) if "img" in batch else batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


# ============================================================================
# Federated Learning Client Handler
# ============================================================================

def train_client(msg, config, context):
    """Handle vision training on federated client.

    Returns:
        tuple: (model_state_dict, train_loss, batches_processed)
    """
    model_type = context.run_config["model_type"]
    dataset_name = config["dataset_name"]
    shard_id = config["shard_id"]
    num_shards = config["num_shards"]
    processed_batches = config["processed_batches"]
    batch_size = context.run_config["batch_size"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.01)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Determine number of batches
    import random
    num_batches = 80 + int(40 * random.random())

    # Load model
    model = get_model(model_type)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)

    # Get data
    trainloader = get_train_loader(
        dataset_name=dataset_name,
        shard_id=shard_id,
        num_shards=num_shards,
        processed_batches=processed_batches,
        batch_size=batch_size,
    )

    # Train
    train_loss, batches_processed = train(
        model=model,
        trainloader=trainloader,
        num_batches=num_batches,
        lr=lr,
        device=device,
        weight_decay=weight_decay,
    )

    return model.state_dict(), train_loss, batches_processed
