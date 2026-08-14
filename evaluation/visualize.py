import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from models.student import get_student_model
from evaluation.metrics import get_confusion_matrix

def unnormalize_image(img_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)

def plot_training_curves(history, save_path="training_curves.png"):
    epochs = range(1, len(history['train_acc']) + 1)
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    plt.title('VL2Lite Accuracy over Epochs', fontsize=13)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.title('Loss over Epochs', fontsize=13)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_confusion_matrix(student_model, test_loader, class_names, device, save_path="confusion_matrix.png"):
    cm = get_confusion_matrix(student_model, test_loader, device)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title('Confusion Matrix: Predicted vs True Classes', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def test_on_20_images(student_model, test_loader, class_names, device, save_path="sample_predictions.png"):
    student_model.eval()
    imgs, labels = next(iter(test_loader))
    imgs, labels = imgs[:20], labels[:20]
    imgs_device = imgs.to(device)

    with torch.no_grad():
        outputs = student_model(imgs_device)
        _, preds = torch.max(outputs, 1)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('VL2Lite Student Model: 20 Image Inference Grid', fontsize=16, y=0.98)

    for i in range(20):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(unnormalize_image(imgs[i]))
        true_name = class_names[labels[i]][:18]
        pred_name = class_names[preds[i]][:18]
        color = 'darkgreen' if preds[i] == labels[i] else 'darkred'
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def compare_models_25_images(student_model, test_loader, class_names, arch, device, save_path="comparison_25.png"):
    baseline_model, _ = get_student_model(arch, len(class_names), device)
    baseline_model.eval()
    student_model.eval()

    imgs, labels = next(iter(test_loader))
    imgs, labels = imgs[:25], labels[:25]
    imgs_device = imgs.to(device)
    labels_device = labels.to(device)

    with torch.no_grad():
        student_out = student_model(imgs_device)
        _, student_preds = torch.max(student_out, 1)

        baseline_out = baseline_model(imgs_device)
        _, baseline_preds = torch.max(baseline_out, 1)

    s_score = (student_preds == labels_device).sum().item()
    b_score = (baseline_preds == labels_device).sum().item()

    print(f"\n====================================")
    print(f"      SCORE on 25-Sample Batch      ")
    print(f"====================================")
    print(f"Distilled Model : {s_score}/25 ({(s_score / 25) * 100:.1f}%)")
    print(f"Baseline Model  : {b_score}/25 ({(b_score / 25) * 100:.1f}%)")
    improvement = ((s_score - b_score) / 25) * 100
    print(f"Improvement     : {improvement:+.1f}%")
    print(f"====================================\n")

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(f'Comparison: VL2Lite {arch.upper()} vs Baseline {arch.upper()}', fontsize=18, y=0.98)

    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(unnormalize_image(imgs[i]))
        true_name = class_names[labels[i]][:15]
        s_pred = class_names[student_preds[i]][:15]
        b_pred = class_names[baseline_preds[i]][:15]
        s_mark = "✔" if student_preds[i] == labels[i] else "✘"
        b_mark = "✔" if baseline_preds[i] == labels[i] else "✘"

        ax.set_title(
            f"True: {true_name}\nKD: {s_pred} {s_mark}\nBase: {b_pred} {b_mark}",
            fontsize=9, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1)
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()