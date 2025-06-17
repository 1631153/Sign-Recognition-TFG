import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import transforms
import videotransforms

import numpy as np

from configs import Config
from pytorch_i3d import InceptionI3d

# from datasets.nslt_dataset import NSLT as Dataset
from datasets.nslt_dataset import NSLT as Dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('--num_class', type=int)

args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sklearn.metrics import accuracy_score, f1_score

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (inputs, labels, _) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.mean(outputs, dim=2)  # [B, C]

        label_indices = torch.argmax(torch.max(labels, dim=2)[0], dim=1)
        loss = criterion(outputs, label_indices)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label_indices.cpu().numpy())

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            acc_so_far = accuracy_score(all_labels, all_preds)
            f1_so_far = f1_score(all_labels, all_preds, average='weighted')
            print(f"  [Batch {batch_idx+1}/{len(dataloader)}] Avg Loss: {running_loss / (batch_idx+1):.4f} | Acc so far: {acc_so_far:.2%} | F1 so far: {f1_so_far:.4f}")

    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, acc, f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            outputs = torch.mean(outputs, dim=2)

            label_indices = torch.argmax(torch.max(labels, dim=2)[0], dim=1)
            loss = criterion(outputs, label_indices)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_indices.cpu().numpy())

            running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, acc, f1


def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None):
    
    print(configs)

    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    train_dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if mode == 'flow':
        model = InceptionI3d(400, in_channels=2)
        model.load_state_dict(torch.load('/export/fhome/jcomes/WLASL1/WLASL/I3D/weights/flow_imagenet.pt'))
    else:
        model = InceptionI3d(400, in_channels=3)
        model.load_state_dict(torch.load('/export/fhome/jcomes/WLASL1/WLASL/I3D/weights/rgb_imagenet.pt'))

    model.replace_logits(train_dataset.num_classes)

    if weights:
        print(f'Loading pretrained weights: {weights}')
        model.load_state_dict(torch.load(weights))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=configs.init_lr, weight_decay=configs.adam_weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience = 20
    no_improve = 0

    os.makedirs(save_model, exist_ok=True)

    for epoch in range(1, 201):
        print(f"\n=== Epoch {epoch} ===")

        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.2%} | F1: {train_f1:.4f}")

        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.2%} | F1: {val_f1:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            model_path = os.path.join(save_model, f'best_model_epoch_{epoch}_v6.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved: {model_path}")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs")

        if no_improve >= patience:
            print("Early stopping!!")
            break

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")



if __name__ == '__main__':
    # WLASL setting
    mode = 'rgb'
    root = {'word': '/export/fhome/jcomes/WLASL1/data/WLASL2000'}

    save_model = '/export/fhome/jcomes/WLASL1/WLASL/I3D/checkpoints/'
    train_split = '/export/fhome/jcomes/WLASL1/WLASL/I3D/preprocess/nslt_100.json'

    # weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    weights = None
    config_file = '/export/fhome/jcomes/WLASL1/WLASL/I3D/configfiles/asl100.ini'

    configs = Config(config_file)
    print(root, train_split)
    run(configs=configs, mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)
