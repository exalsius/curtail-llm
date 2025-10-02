import torch
from tqdm import tqdm


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set with progress tracking.

    Returns:
        tuple: (avg_train_loss, batches_processed, epochs_completed)
    """
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()

    running_loss = 0.0
    batches_processed = 0
    epochs_completed = 0

    for epoch in range(epochs):
        epoch_batches = 0

        for batch in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches_processed += 1
            epoch_batches += 1

        # Only count epoch as completed if we processed at least one batch
        if epoch_batches > 0:
            epochs_completed += 1

    avg_trainloss = running_loss / batches_processed if batches_processed > 0 else 0.0
    return avg_trainloss, batches_processed, epochs_completed


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
