import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import videotransforms
from torch.utils.data import DataLoader

from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset_all import NSLT as Dataset

from sklearn.metrics import accuracy_score, f1_score

#extrae la clase en función multiclase (en teoria)
def extract_true_label(labels):
    if labels.ndim == 1:
        return labels[0].item()
    elif labels.ndim == 2:
        return labels[0].argmax().item()
    elif labels.ndim == 3:
        return torch.argmax(torch.max(labels, dim=2)[0], dim=1)[0].item()
    else:
        raise ValueError(f"Formato de etiqueta no soportado: shape={labels.shape}")

def evaluate_model(model, dataloader, num_classes):
    model.eval()
    top1_tp, top5_tp, top10_tp = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)
    top1_fp, top5_fp, top10_fp = np.zeros(num_classes), np.zeros(num_classes), np.zeros(num_classes)

    all_preds = []
    all_labels = []

    #calculo manual de top-1, top-5, top-10
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()

            logits = model(inputs)                     # [B, C, T]
            preds = torch.mean(logits, dim=2)[0]       # [C]
            pred_indices = preds.argsort(descending=True).cpu().numpy()
            true_label = extract_true_label(labels)

            all_preds.append(pred_indices[0])
            all_labels.append(true_label)

            if true_label == pred_indices[0]:
                top1_tp[true_label] += 1
            else:
                top1_fp[true_label] += 1

            if true_label in pred_indices[:5]:
                top5_tp[true_label] += 1
            else:
                top5_fp[true_label] += 1

            if true_label in pred_indices[:10]:
                top10_tp[true_label] += 1
            else:
                top10_fp[true_label] += 1

    def safe_divide(tp, fp): return tp / (tp + fp + 1e-8)
    top1_acc = safe_divide(top1_tp, top1_fp).mean()
    top5_acc = safe_divide(top5_tp, top5_fp).mean()
    top10_acc = safe_divide(top10_tp, top10_fp).mean()

    overall_acc = accuracy_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"\n=== Resultados ===")
    print(f"Top-1 Accuracy per class:  {top1_acc:.4f}")
    print(f"Top-5 Accuracy per class:  {top5_acc:.4f}")
    print(f"Top-10 Accuracy per class: {top10_acc:.4f}")
    print(f"Global Accuracy:           {overall_acc:.4f}")
    print(f"Weighted F1-score:         {overall_f1:.4f}")

def load_model(mode, num_classes, weights_path):
    model = InceptionI3d(400, in_channels=2 if mode == 'flow' else 3)
    imagenet_weights = f'/export/fhome/jcomes/WLASL1/WLASL/I3D/weights/{mode}_imagenet.pt'
    model.load_state_dict(torch.load(imagenet_weights))
    model.replace_logits(num_classes)
    model.load_state_dict(torch.load(weights_path))
    return nn.DataParallel(model.cuda())

def main():
    mode = 'rgb'
    root = '/export/fhome/jcomes/WLASL1/data/WLASL2000'
    num_classes = 100
    train_split = f'/export/fhome/jcomes/WLASL1/WLASL/I3D/preprocess/nslt_{num_classes}.json'
    weights = '/export/fhome/jcomes/WLASL1/WLASL/I3D/modelos/[v7]best_model_epoch43_acc77.91_v7-2.pt' #cambiar por el modelo entrenado

    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    test_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    model = load_model(mode, num_classes, weights)
    evaluate_model(model, test_loader, num_classes)

if __name__ == '__main__':
    main()
