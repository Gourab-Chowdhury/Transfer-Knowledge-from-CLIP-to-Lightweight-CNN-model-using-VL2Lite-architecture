import os
import torch
from configs.config import get_config
from dataset.dataloader import get_dataloaders
from models.student import get_student_model
from training.trainer import train_vl2lite
from evaluation.metrics import compute_advanced_metrics
from evaluation.visualize import (
    plot_training_curves,
    plot_confusion_matrix,
    test_on_20_images,
    compare_models_25_images
)

def load_checkpoint(checkpoint_path, student_arch, num_classes, device):
    student, _ = get_student_model(student_arch, num_classes, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    student.load_state_dict(checkpoint['model_state_dict'])
    student.eval()
    print(f"Loaded checkpoint from Epoch {checkpoint['best_epoch']} with Best Acc: {checkpoint['best_acc']:.2f}%")
    return student, checkpoint.get('history', {})

def main():
    config = get_config()
    train_loader, test_loader, class_names = get_dataloaders(config)

    if config.mode == "train":
        history, trained_student = train_vl2lite(config, train_loader, test_loader, class_names)
        compute_advanced_metrics(trained_student, test_loader, config.device)
        plot_training_curves(history)
        plot_confusion_matrix(trained_student, test_loader, class_names, config.device)
        test_on_20_images(trained_student, test_loader, class_names, config.device)
        compare_models_25_images(trained_student, test_loader, class_names, config.student_arch, config.device)

    elif config.mode in ["eval", "visualize"]:
        if not os.path.exists(config.save_path):
            raise FileNotFoundError(f"Checkpoint not found at {config.save_path}")
        
        student, history = load_checkpoint(config.save_path, config.student_arch, len(class_names), config.device)
        compute_advanced_metrics(student, test_loader, config.device)
        
        if history:
            plot_training_curves(history)
        plot_confusion_matrix(student, test_loader, class_names, config.device)
        test_on_20_images(student, test_loader, class_names, config.device)
        compare_models_25_images(student, test_loader, class_names, config.student_arch, config.device)

if __name__ == "__main__":
    main()