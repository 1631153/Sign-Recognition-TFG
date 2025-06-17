
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
import videotransforms #archivo externo que contiene las transformaciones especificas para los videos

import numpy as np

from pytorch_i3d import InceptionI3d #modelo I3D preentrenado
from datasets.nslt_dataset import NSLT as Dataset #dataset personalizado. No lo he tocado. Va bastante bien.

from sklearn.metrics import accuracy_score, f1_score

#En teoria esto es la configuracion de la GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#argumentos que se pueden pasar por linea de comandos. Se puede cambiar, decidir!!
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

def calculate_metrics(outputs, labels):
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    targets = torch.argmax(labels, dim=1).cpu().numpy()

    acc = accuracy_score(targets, preds) * 100.0
    f1 = f1_score(targets, preds, average='weighted') * 100.0
    return acc, f1


def train_one_epoch(model, dataloader, optimizer, device, criterion):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0
    running_f1 = 0.0
    total_batches = 0

    for inputs, labels, _ in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        t = inputs.size(2) #longitud temporal. Relacionado con los epochs

        optimizer.zero_grad()
        outputs = model(inputs, pretrained=False)
        outputs = F.interpolate(outputs, size=t, mode='linear') # interpolar al tamaño de entrada. 

        #calculo de perdida y optimización.
        loss = criterion(outputs.max(dim=2)[0], labels.max(dim=2)[0])
        loss.backward()
        optimizer.step()

        #calculo de metricas
        acc, f1 = calculate_metrics(outputs.max(dim=2)[0], labels.max(dim=2)[0])
        running_loss += loss.item()
        running_accuracy += acc
        running_f1 += f1
        total_batches += 1

    epoch_loss = running_loss / total_batches
    epoch_accuracy = running_accuracy / total_batches
    epoch_f1 = running_f1 / total_batches
    return epoch_loss, epoch_accuracy, epoch_f1


def validate_one_epoch(model, dataloader, device, criterion):
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    running_f1 = 0.0
    total_batches = 0

    with torch.no_grad(): #no se calculan gradientes
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            t = inputs.size(2)

            outputs = model(inputs, pretrained=False)
            outputs = F.interpolate(outputs, size=t, mode='linear')
            
            #calculo de perdida y optimización.
            loss = criterion(outputs.max(dim=2)[0], labels.max(dim=2)[0])
            acc, f1 = calculate_metrics(outputs.max(dim=2)[0], labels.max(dim=2)[0])

            running_loss += loss.item()
            running_accuracy += acc
            running_f1 += f1
            total_batches += 1

    epoch_loss = running_loss / total_batches
    epoch_accuracy = running_accuracy / total_batches
    epoch_f1 = running_f1 / total_batches
    return epoch_loss, epoch_accuracy, epoch_f1


def run(mode='rgb', root='', train_split='', save_model='', weights=None):
    #principio igual que el original
    #transformaciones para entrenamiento y validación
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224), videotransforms.RandomHorizontalFlip()])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    train_dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)

    Batch_size = 3
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=Batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #necesario por si el ordenador no cuenta con gpu nvidia, ahorra cambiar manualmente

    #carga del modelo preentrenado según el modo, modificar??
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('/export/fhome/jcomes/WLASL1/WLASL/I3D/weights/flow_imagenet.pt'))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('/export/fhome/jcomes/WLASL1/WLASL/I3D/weights/rgb_imagenet.pt'))

    i3d.replace_logits(train_dataset.num_classes)
    if weights:
        i3d.load_state_dict(torch.load(weights))

    i3d = i3d.to(device)
    i3d = nn.DataParallel(i3d) #para usar múltiples GPUs si están disponibles

    init_lr = 0.001
    adam_weight_decay = 1e-7
    criterion = nn.BCEWithLogitsLoss() #si quieres definir la función una vez y reutilizarla
    optimizer = optim.Adam(i3d.parameters(), lr=init_lr, weight_decay=adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)

    best_val_acc = 0.0
    num_epochs = 50
    #eliminado patience para early stopping

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc, train_f1 = train_one_epoch(i3d, train_loader, optimizer, device, criterion)
        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}% | F1-score: {train_f1:.2f}%")
        
        val_loss, val_acc, val_f1 = validate_one_epoch(i3d, val_loader, device, criterion)
        print(f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.2f}% | F1-score: {val_f1:.2f}%")

        scheduler.step(val_loss) #ajustar el learning rate (ReduceLR)

        if val_acc > best_val_acc: #guardar el mejor modelo según la validación
            best_val_acc = val_acc
            model_path = f"{save_model}/best_model_epoch{epoch+1}_acc{val_acc:.2f}_v7.pt"
            torch.save(i3d.module.state_dict(), model_path)
            print(f"Model saved: {model_path}")


if __name__ == '__main__':
    mode = 'rgb'
    root = {'word': '/export/fhome/jcomes/WLASL1/data/WLASL2000'}
    save_model = '/export/fhome/jcomes/WLASL1/WLASL/I3D/checkpoints/'
    train_split = '/export/fhome/jcomes/WLASL1/WLASL/I3D/preprocess/nslt_100.json'
    weights = None

    run(mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights)