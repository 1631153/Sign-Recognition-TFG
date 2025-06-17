import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import videotransforms
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0) * 100


def calculate_metrics(preds, targets):
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    return acc, f1


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0

    for inputs, labels, _ in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # [B, C, T]
        outputs = torch.mean(outputs, dim=2)  # [B, C]

        labels = torch.argmax(torch.max(labels, dim=2)[0], dim=1)  # [B]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        num_batches += 1

    avg_loss = total_loss / num_batches
    acc, f1 = calculate_metrics(all_preds, all_labels)
    return avg_loss, acc, f1


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    num_batches = 0

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = torch.mean(outputs, dim=2)

            labels = torch.argmax(torch.max(labels, dim=2)[0], dim=1)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1

    avg_loss = total_loss / num_batches
    acc, f1 = calculate_metrics(all_preds, all_labels)
    return avg_loss, acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='rgb', help='rgb or flow')
    parser.add_argument('--save_dir', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/checkpoints/')
    parser.add_argument('--root', type=str, default='{"word": "/export/fhome/jcomes/WLASL1/data/WLASL2000"}')
    parser.add_argument('--train_split', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/preprocess/nslt_100.json')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--config', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/configfiles/asl100.ini')

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = Config(args.config)

    # Transforms
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip()
    ])
    test_transforms = transforms.Compose([
        videotransforms.CenterCrop(224)
    ])

    root = eval(args.root)
    train_dataset = Dataset(args.train_split, 'train', root, args.mode, train_transforms)
    val_dataset = Dataset(args.train_split, 'test', root, args.mode, test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=4)

    if args.mode == 'flow':
        model = InceptionI3d(400, in_channels=2)
        model.load_state_dict(torch.load('/export/fhome/jcomes/WLASL1/WLASL/I3D/weights/flow_imagenet.pt'))
    else:
        model = InceptionI3d(400, in_channels=3)
        model.load_state_dict(torch.load('/export/fhome/jcomes/WLASL1/WLASL/I3D/weights/rgb_imagenet.pt'))

    model.replace_logits(train_dataset.num_classes)
    if args.weights:
        model.load_state_dict(torch.load(args.weights))

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.init_lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, 101):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch:03d} - Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | F1: {train_f1:.4f}")

        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:03d} - Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            path = os.path.join(args.save_dir, f"best_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), path)
            print(f"\nBest model saved at epoch {epoch} with Acc: {val_acc:.2%}, F1: {val_f1:.4f}\n")

        scheduler.step()

    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2%}")


if __name__ == '__main__':
    main()
