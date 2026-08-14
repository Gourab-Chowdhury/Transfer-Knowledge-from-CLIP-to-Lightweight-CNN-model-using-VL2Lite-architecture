import argparse
import torch

def get_config():
    parser = argparse.ArgumentParser(description="VL2Lite PyTorch Knowledge Distillation")
    
    # Dataset & Model
    parser.add_argument("--dataset", type=str, default="fgvc_aircraft", choices=["fgvc_aircraft", "cifar10"], help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--student_arch", type=str, default="resnet50", choices=["resnet18", "resnet34", "resnet50"], help="Student backbone")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="Pretrained CLIP teacher model name")
    
    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=40, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--temperature", type=float, default=0.07, help="Logit temperature for KL Divergence")
    parser.add_argument("--ramp_ratio", type=float, default=0.125, help="Ramp schedule fraction (default: 12.5% of total epochs)")
    parser.add_argument("--embed_dim", type=int, default=512, help="CLIP feature dimension")
    
    # Runtime & Checkpoints
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--save_path", type=str, default="checkpoints/vl2lite_best_checkpoint.pt", help="Checkpoint save destination")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "visualize"], help="Execution mode")
    
    args = parser.parse_args()
    return args