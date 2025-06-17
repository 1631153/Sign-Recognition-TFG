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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('--save_dir', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/checkpoints/')
parser.add_argument('--root', type=str, default='{"word": "/export/fhome/jcomes/WLASL1/data/WLASL2000"}')
parser.add_argument('--train_split', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/preprocess/nslt_100.json')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--config', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/configfiles/asl100.ini')
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_metrics(outputs, labels):
    _, preds = torch.max(outputs, 1)
    acc = accuracy_score(labels.cpu(), preds.cpu())
    f1 = f1_score(labels.cpu(), preds.cpu(), average='weighted')
    return acc * 100, f1

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for inputs, labels, _ in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = torch.argmax(torch.max(labels, dim=2)[0], dim=1)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.mean(outputs, dim=2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc, f1 = accuracy_score(all_labels, all_preds)*100, f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = torch.argmax(torch.max(labels, dim=2)[0], dim=1)
            outputs = model(inputs)
            outputs = torch.mean(outputs, dim=2)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    acc, f1 = accuracy_score(all_labels, all_preds)*100, f1_score(all_labels, all_preds, average='weighted')
    return avg_loss, acc, f1

def run(configs, mode='rgb', root='/export/fhome/jcomes/WLASL1/data/WLASL2000', train_split='/export/fhome/jcomes/WLASL1/WLASL/I3D/preprocess/nslt_100.json', save_model='/export/fhome/jcomes/WLASL1/WLASL/I3D/checkpoints/', weights=None):
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224), videotransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)

    dataloaders = {
        'train': torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0, pin_memory=True),
        'test': torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=2, pin_memory=False)
    }

    i3d = InceptionI3d(400, in_channels=2 if mode == 'flow' else 3)
    i3d.load_state_dict(torch.load(f'/export/fhome/jcomes/WLASL1/WLASL/I3D/weights/{mode}_imagenet.pt'))
    i3d.replace_logits(dataset.num_classes)
    if weights:
        i3d.load_state_dict(torch.load(weights))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i3d = i3d.to(device)
    i3d = nn.DataParallel(i3d) if torch.cuda.device_count() > 1 else i3d

    # Class weights
    all_labels = [torch.argmax(torch.max(label, dim=1)[0]).item() for _, label, _ in dataset]
    class_counts = np.bincount(np.array(all_labels), minlength=dataset.num_classes)
    class_weights = 1 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(i3d.parameters(), lr=configs.init_lr, weight_decay=configs.adam_weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_acc = 0
    patience, no_improve = 5, 0
    for epoch in range(1, 301):
        print(f"Epoch {epoch}")
        train_loss, train_acc, train_f1 = train_one_epoch(i3d, dataloaders['train'], optimizer, criterion, device)
        print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.4f}")

        if epoch % 10 == 0:
            val_loss, val_acc, val_f1 = validate(i3d, dataloaders['test'], criterion, device)
            print(f"Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.4f}")
            
            scheduler.step()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                os.makedirs(save_model, exist_ok=True)
                torch.save(i3d.state_dict(), os.path.join(save_model, f"best_model_epoch{epoch}_{val_acc:.2f}.pth"))
                print("Validation accuracy improved, model saved.")
            else:
                no_improve += 1
                print(f"No improvement for {no_improve} epoch(s).")
                if no_improve >= patience:
                    print("Early stopping triggered.")
                    break

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
