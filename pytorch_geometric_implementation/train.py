import torch
from torch_geometric.data import DataLoader
from models import GraphChebNet


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


if __name__ == "__main__":
    from torch_geometric.datasets import TUDataset

    # Load dataset
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    dataset = dataset.shuffle()


    # Train/test split
    train_dataset = dataset[:len(dataset) // 10 * 8]
    test_dataset = dataset[len(dataset) // 10 * 8:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_node_features = dataset.num_node_features
    hidden_channels = 64
    num_classes = dataset.num_classes
    learning_rate = 0.0005
    num_epochs = 600
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GraphChebNet(num_node_features, hidden_channels, num_classes, K=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Entraînement et test
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_acc = test(model, test_loader, device)
        print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')
