import torch
from tqdm import tqdm


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    total_batches = epochs * len(trainloader)

    with tqdm(total=total_batches, desc="Training") as pbar:
        for epoch in range(epochs):
            for batch in trainloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                loss = criterion(net(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

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
