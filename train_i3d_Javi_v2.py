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

# Configuración global
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True

# Parser global
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('--save_dir', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/checkpoints/')
parser.add_argument('--root', type=str, default='{"word": "/export/fhome/jcomes/WLASL1/data/WLASL2000"}')
parser.add_argument('--train_split', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/preprocess/nslt_100.json')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--config', type=str, default='/export/fhome/jcomes/WLASL1/WLASL/I3D/configfiles/asl100.ini')

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    loc_loss = 0.0
    cls_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch_idx, (inputs, labels, _) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Convertir labels one-hot a clase (forma [batch_size])
        label_indices = torch.argmax(torch.max(labels, dim=2)[0], dim=1)
        
        optimizer.zero_grad()
        
        # Forward pass - outputs tendrá forma [batch_size, num_classes, temporal_dim]
        outputs = model(inputs)
        
        # Calcular pérdida de clasificación (promedio temporal)
        video_logits = torch.mean(outputs, dim=2)  # [batch_size, num_classes]
        cls_loss_val = criterion(video_logits, label_indices)
        
        # Calcular pérdida temporal (por frame)
        # Necesitamos permutar outputs a [batch_size, temporal_dim, num_classes]
        frame_logits = outputs.permute(0, 2, 1)  # [batch_size, temporal_dim, num_classes]
        # Y expandir labels a [batch_size, temporal_dim]
        expanded_labels = label_indices.unsqueeze(1).expand(-1, frame_logits.size(1))
        loc_loss_val = criterion(frame_logits.reshape(-1, frame_logits.size(-1)), 
                               expanded_labels.reshape(-1))
        
        loss = 0.5 * loc_loss_val + 0.5 * cls_loss_val
        
        loss.backward()
        optimizer.step()
        
        # Acumular métricas
        total_loss += loss.item()
        loc_loss += loc_loss_val.item()
        cls_loss += cls_loss_val.item()
        
        preds = torch.argmax(video_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label_indices.cpu().numpy())

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
            print(f"  [Batch {batch_idx + 1}/{len(loader)}] Loss: {loss.item():.4f} | "
                f"Cls: {cls_loss_val.item():.4f} | Loc: {loc_loss_val.item():.4f}")
    
    # Calcular métricas finales
    avg_loss = total_loss / len(loader)
    avg_loc_loss = loc_loss / len(loader)
    avg_cls_loss = cls_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, avg_loc_loss, avg_cls_loss, acc, f1

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    loc_loss = 0.0
    cls_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels, _) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            label_indices = torch.argmax(torch.max(labels, dim=2)[0], dim=1)
            
            outputs = model(inputs)
            
            # Pérdida de clasificación
            video_logits = torch.mean(outputs, dim=2)
            cls_loss_val = criterion(video_logits, label_indices)
            
            # Pérdida temporal
            frame_logits = outputs.permute(0, 2, 1)
            expanded_labels = label_indices.unsqueeze(1).expand(-1, frame_logits.size(1))
            loc_loss_val = criterion(frame_logits.reshape(-1, frame_logits.size(-1)), 
                           expanded_labels.reshape(-1))
            
            loss = 0.5 * loc_loss_val + 0.5 * cls_loss_val
            
            # Acumular métricas
            total_loss += loss.item()
            loc_loss += loc_loss_val.item()
            cls_loss += cls_loss_val.item()
            
            preds = torch.argmax(video_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label_indices.cpu().numpy())

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"  [Batch {batch_idx + 1}/{len(loader)}] Loss: {loss.item():.4f} | "
                    f"Cls: {cls_loss_val.item():.4f} | Loc: {loc_loss_val.item():.4f}")
            
    # Calcular métricas finales
    avg_loss = total_loss / len(loader)
    avg_loc_loss = loc_loss / len(loader)
    avg_cls_loss = cls_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, avg_loc_loss, avg_cls_loss, acc, f1

def print_metrics(epoch, phase, loc_loss, cls_loss, tot_loss, acc, f1):
    print(f'Epoch {epoch} {phase} - '
          f'Loc Loss: {loc_loss:.4f} | '
          f'Cls Loss: {cls_loss:.4f} | '
          f'Tot Loss: {tot_loss:.4f} | '
          f'Acc: {acc:.2%} | '
          f'F1: {f1:.4f}')

def main():
    args = parser.parse_args()
    configs = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformaciones
    train_transforms = transforms.Compose([
        videotransforms.RandomCrop(224),
        videotransforms.RandomHorizontalFlip(),
    ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    # Datasets
    train_dataset = Dataset(args.train_split, 'train', eval(args.root), args.mode, train_transforms)
    val_dataset = Dataset(args.train_split, 'test', eval(args.root), args.mode, test_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=configs.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Modelo
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

    # Optimizador y pérdida
    optimizer = optim.Adam(model.parameters(), lr=configs.init_lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # Entrenamiento
    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    num_epochs = 100
    
    for epoch in range(num_epochs):
        # Fase de entrenamiento
        train_loss, train_loc, train_cls, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        print_metrics(epoch+1, 'Train', train_loc, train_cls, train_loss, train_acc, train_f1)
        
        # Fase de validación
        val_loss, val_loc, val_cls, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )
        print_metrics(epoch+1, 'Val', val_loc, val_cls, val_loss, val_acc, val_f1)
        
        # Guardar mejor modelo
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f'¡Mejor modelo guardado con Acc: {val_acc:.2%} y F1: {val_f1:.4f}!')
        
        scheduler.step()
        print('-' * 80)

    print(f'\nEntrenamiento completo. Mejor precisión: {best_acc:.2%}')

if __name__ == '__main__':
    main()