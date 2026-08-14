import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

def evaluate_model(student_model, test_loader, device):
    """Calculates top-1 validation accuracy and average cross-entropy loss."""
    student_model.eval()
    v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out = student_model(imgs)
            l = F.cross_entropy(out, lbls)
            v_loss += l.item()
            _, pred = torch.max(out, 1)
            v_correct += (pred == lbls).sum().item()
            v_total += lbls.size(0)

    val_acc = 100.0 * v_correct / v_total
    avg_v_loss = v_loss / len(test_loader)
    return val_acc, avg_v_loss

def compute_advanced_metrics(student_model, test_loader, device):
    """Computes Weighted Precision, Recall, and F1 Score."""
    student_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = student_model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    print("\n--- Test Set Advanced Metrics ---")
    print(f"Precision : {precision * 100.0:.2f}%")
    print(f"Recall    : {recall * 100.0:.2f}%")
    print(f"F1 Score  : {f1 * 100.0:.2f}%\n")
    return precision, recall, f1

def get_confusion_matrix(student_model, test_loader, device):
    student_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = student_model(imgs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
    return confusion_matrix(all_labels, all_preds)