import argparse, time
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from mnist_cnn import MNISTCNN

def get_loaders(quick: bool):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

    # QUICK mode: use a smaller subset so CPU training finishes fast
    if quick:
        # ~5,000 training samples and full test set
        train_ds = Subset(train_ds, range(0, 5000))

    # Windows tip: num_workers=0 avoids multiprocessing overhead
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, test_loader

def train_one_epoch(model, loader, device, opt, loss_fn, max_batches=None):
    model.train()
    total = 0.0
    for b, (x, y) in enumerate(loader, 1):
        if max_batches and b > max_batches:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += float(loss.item())
    return total / max(1, min(len(loader), max_batches or len(loader)))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Train on a small subset for a fast demo")
    args = parser.parse_args()

    # On some CPUs, fewer threads is faster/more stable
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_loaders(args.quick)
    model = MNISTCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # QUICK mode: 1 epoch, cap batches; FULL mode: 3 epochs, all batches
    epochs = 1 if args.quick else 3
    max_batches = 80 if args.quick else None  # ~80 * 128 ≈ 10k samples

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, device, opt, loss_fn, max_batches=max_batches)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch}/{epochs} - loss {loss:.4f} - test acc {acc:.4f}")

    torch.save(model.state_dict(), "mnist_cnn.pt")
    print(f"Saved model to mnist_cnn.pt in {time.time()-t0:.1f}s (quick={args.quick})")

if __name__ == "__main__":
    main()
