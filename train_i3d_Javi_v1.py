import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

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

def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    correct = (preds == labels).sum().item()
    accuracy = correct / labels.size(0) * 100
    return accuracy

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0

    for count, (inputs, labels, _) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = torch.mean(outputs, dim=2) 
        labels = torch.argmax(torch.max(labels, dim=2)[0], dim=1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += calculate_accuracy(outputs, labels)
        num_batches += 1

        if (count + 1) % 10 == 0:
            print(f"Batch [{count+1}/{len(dataloader)}], Loss: {loss.item():.4f}, Accuracy: {calculate_accuracy(outputs, labels):.2f}%")

    epoch_loss = total_loss / num_batches
    epoch_accuracy = total_accuracy / num_batches

    return epoch_loss, epoch_accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_accuracy  = 0.0
    num_batches  = 0

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            outputs = torch.mean(outputs, dim=2) 
            labels = torch.argmax(torch.max(labels, dim=2)[0], dim=1)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_accuracy += calculate_accuracy(outputs, labels)
            num_batches += 1

    epoch_loss = total_loss / num_batches
    epoch_accuracy = total_accuracy / num_batches

    return epoch_loss, epoch_accuracy


def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None):
    print(configs)

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}


    all_labels = [torch.argmax(torch.max(label, dim=1)[0]).item() for _, label, _ in dataset]
    all_labels = np.array(all_labels)

    class_counts = np.bincount(all_labels, minlength=dataset.num_classes) #contar cuantas veces aparece cada clase
    class_weights = 1 / (class_counts + 1e-6)
    class_weights *= len(class_counts) / class_weights.sum() #normalizar los pesos para que sumen la cantidad de clases
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('/export/fhome/jcomes/WLASL1/WLASL/I3D/weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('/export/fhome/jcomes/WLASL1/WLASL/I3D/weights/rgb_imagenet.pt'))

    num_classes = dataset.num_classes
    i3d.replace_logits(num_classes)

    if weights:
        print('loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    i3d = i3d.to(device)
    i3d = nn.DataParallel(i3d) if torch.cuda.device_count() > 1 else i3d
    
    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay 
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    num_epochs = 100
    epochs_no_improve = 0
    patience = 5  # For early stopping
    best_val_accuracy = -1
    early_stop = False
    
    # train it
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(i3d, dataloaders['train'], optimizer, criterion, device)
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        if (epoch % 10 == 0):
            val_loss, val_acc = validate(i3d, dataloaders['test'], criterion, device)
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            scheduler.step()

            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                epochs_no_improve = 0
                os.makedirs(os.path.join(save_model, 'checkpoints'), exist_ok=True)
                checkpoint_path = os.path.join(save_model, f"best_model_{epoch}_{val_acc:.0f}.pth")
                torch.save(i3d.state_dict(), checkpoint_path)
                print(f"Model saved: {checkpoint_path}\n")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("Early stopping triggered!")
                    early_stop = True
                    break
    
    if not early_stop:
        final_model_path = os.path.join(save_model, 'final_model.pth')
        torch.save(i3d.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")



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
