import os
import torch
import torch.optim as optim
from models.teacher import get_teacher_model, extract_teacher_image_features
from models.student import get_student_model, forward_student_features
from models.connector import get_connector
from utils.text_anchors import generate_text_anchors
from loss.vl2lite_loss import VL2LiteLoss
from evaluation.metrics import evaluate_model

def train_vl2lite(config, train_loader, test_loader, class_names):
    device = config.device
    num_classes = len(class_names)
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    # Instantiate Models & Components
    teacher = get_teacher_model(config.clip_model, device)
    student, feat_dim = get_student_model(config.student_arch, num_classes, device)
    connector = get_connector(input_dim=feat_dim, output_dim=config.embed_dim, device=device)
    text_anchors = generate_text_anchors(class_names, config.clip_model, device)
    
    criterion = VL2LiteLoss(temperature=config.temperature, ramp_ratio=config.ramp_ratio)
    optimizer = optim.AdamW(list(student.parameters()) + list(connector.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\n--- Starting VL2Lite Training ({config.student_arch.upper()} on {config.dataset}) ---")
    for epoch in range(config.epochs):
        student.train()
        connector.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                t_feats = extract_teacher_image_features(teacher, imgs)
                logits, feat_proj = forward_student_features(student, connector, imgs)
                loss, _, _, _ = criterion(
                    logits, feat_proj, t_feats, text_anchors, lbls,
                    current_epoch=epoch, total_epochs=config.epochs
                )

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            t_loss += loss.item()
            _, pred = torch.max(logits, 1)
            t_correct += (pred == lbls).sum().item()
            t_total += lbls.size(0)

        scheduler.step()

        train_acc = 100.0 * t_correct / t_total
        avg_t_loss = t_loss / len(train_loader)
        val_acc, avg_v_loss = evaluate_model(student, test_loader, device)

        history['train_loss'].append(avg_t_loss)
        history['val_loss'].append(avg_v_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1

            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': student.state_dict(),
                'connector_state_dict': connector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'best_acc': best_val_acc,
                'best_epoch': best_epoch,
                'config': {
                    'dataset': config.dataset,
                    'student_arch': config.student_arch,
                    'batch_size': config.batch_size,
                    'learning_rate': config.lr,
                    'num_classes': num_classes,
                    'device': config.device
                }
            }
            torch.save(checkpoint, config.save_path)
            print(f" --> Best checkpoint saved: {val_acc:.2f}% Val Accuracy")

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:02d}/{config.epochs:02d}: Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}% | Loss {avg_t_loss:.4f} | LR: {current_lr:.6f}")

    print(f"\nTraining Complete. Checkpoint saved to {config.save_path}")
    return history, student