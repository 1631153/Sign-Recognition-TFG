import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
import videotransforms
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from configs import Config
from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import NSLT as Dataset

# Configuraciones
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('--save_dir', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/checkpoints/')
parser.add_argument('--root', type=str, default='{"word": "/export/fhome/jcomes/WLASL1/data/WLASL2000"}')
parser.add_argument('--train_split', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/preprocess/nslt_100.json')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--config', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/configfiles/asl100.ini')


def calculate_metrics(preds, targets):
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    return acc, f1


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_loc_loss = 0
    all_preds = []
    all_targets = []

    for count, (inputs, labels, _) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Preparar etiquetas: [B] entero por clase
        label_indices = torch.argmax(torch.max(labels, dim=2)[0], dim=1)

        optimizer.zero_grad()
        outputs = model(inputs)

        # logits de video
        video_logits = outputs.mean(dim=2)  # [B, C]
        cls_loss = criterion(video_logits, label_indices)

        # logits por frame
        frame_logits = outputs.permute(0, 2, 1)  # [B, T, C]
        expanded_labels = label_indices.unsqueeze(1).expand(-1, frame_logits.size(1))
        loc_loss = criterion(frame_logits.reshape(-1, frame_logits.size(-1)), expanded_labels.reshape(-1))

        loss = 0.5 * cls_loss + 0.5 * loc_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_loc_loss += loc_loss.item()

        preds = video_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(label_indices.cpu().numpy())

        if (count + 1) % 10 == 0:
            print(f"  [Batch {count+1}/{len(dataloader)}] Loss: {loss.item():.4f} | Cls: {cls_loss.item():.4f} | Loc: {loc_loss.item():.4f}")

    acc, f1 = calculate_metrics(all_preds, all_targets)
    return total_loss / len(dataloader), total_cls_loss / len(dataloader), total_loc_loss / len(dataloader), acc, f1


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_cls_loss = 0
    total_loc_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            label_indices = torch.argmax(torch.max(labels, dim=2)[0], dim=1)
            outputs = model(inputs)
            video_logits = outputs.mean(dim=2)
            cls_loss = criterion(video_logits, label_indices)
            frame_logits = outputs.permute(0, 2, 1)
            expanded_labels = label_indices.unsqueeze(1).expand(-1, frame_logits.size(1))
            loc_loss = criterion(frame_logits.reshape(-1, frame_logits.size(-1)), expanded_labels.reshape(-1))
            loss = 0.5 * cls_loss + 0.5 * loc_loss

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_loc_loss += loc_loss.item()

            preds = video_logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(label_indices.cpu().numpy())

    acc, f1 = calculate_metrics(all_preds, all_targets)
    return total_loss / len(dataloader), total_cls_loss / len(dataloader), total_loc_loss / len(dataloader), acc, f1


def main():
    args = parser.parse_args()
    configs = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    train_dataset = Dataset(args.train_split, 'train', eval(args.root), args.mode, train_transforms)
    val_dataset = Dataset(args.train_split, 'test', eval(args.root), args.mode, test_transforms)
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

    # Freeze feature extractor if pretrained
    for name, param in model.named_parameters():
        if 'Mixed_5c' not in name:
            param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=configs.init_lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, 301):
        train_loss, train_cls, train_loc, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch} Train - Loc Loss: {train_loc:.4f} | Cls Loss: {train_cls:.4f} | Tot Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | F1: {train_f1:.4f}")

        val_loss, val_cls, val_loc, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch} Val - Loc Loss: {val_loc:.4f} | Cls Loss: {val_cls:.4f} | Tot Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.4f}")

        scheduler.step()

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'best_model_epoch_{epoch}.pth'))
            print(f"\nModel saved at epoch {epoch} with Acc: {val_acc:.2%}, F1: {val_f1:.4f}\n")

if __name__ == '__main__':
    main()
