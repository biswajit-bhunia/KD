import torch
import os

def save_checkpoint(model, optimizer, epoch, path):
    dir_name = os.path.dirname(path)
    if dir_name:  # os.path.dirname returns '' for bare filenames — makedirs("") raises FileNotFoundError
        os.makedirs(dir_name, exist_ok=True)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)

def load_checkpoint(model, optimizer, path, device="cpu"):
    # Bug B: weights_only=True avoids arbitrary pickle deserialization (security
    # risk) and suppresses the FutureWarning raised by PyTorch 2.0+.
    # Use weights_only=False only if the checkpoint stores non-tensor objects.
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["epoch"]