import torch
from tqdm import tqdm


def train(net, trainloader, num_batches, lr, device, weight_decay=0.01):
    """Train the model on the training set for a fixed number of batches.

    Args:
        net: Neural network model
        trainloader: DataLoader that yields batches (can be infinite)
        num_batches: Number of batches to process
        lr: Learning rate
        device: Device to train on (CPU/GPU)
        weight_decay: Weight decay (L2 regularization) for AdamW optimizer

    Returns:
        tuple: (avg_train_loss, batches_processed)
    """
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()

    running_loss = 0.0
    batches_processed = 0

    # Process exactly num_batches batches
    for batch in tqdm(trainloader, desc=f"Training", total=num_batches):
        images = batch["img"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches_processed += 1

        # Stop after processing the desired number of batches
        if batches_processed >= num_batches:
            break

    avg_trainloss = running_loss / batches_processed if batches_processed > 0 else 0.0
    return avg_trainloss, batches_processed


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    net.eval()
    with torch.no_grad():
        for batch in tqdm(testloader, desc="Testing"):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
