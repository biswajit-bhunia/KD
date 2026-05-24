import torch
import numpy as np
from torch.utils.data import DataLoader
from models.teacher import TeacherModel
from training.validate import evaluate
from data.loader import load_samples
from data.split import split_samples
from data.dataset import DeepfakeDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading data...")
samples = load_samples("data")
train_s, val_s, test_s, split_info = split_samples(samples, test_size=0.2, mode="generator_holdout_3way", random_state=42)

val_dataset = DeepfakeDataset(val_s, augment=False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Loading teacher model...")
model = TeacherModel(pretrained=False).to(device)
state_dict = torch.load("checkpoints/teacher_federated.pth", map_location=device)
model.load_state_dict(state_dict["model_state_dict"])
model.eval()

print("Evaluating...")
all_probs = []
all_labels = []
with torch.no_grad():
    for batch in val_loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        from features.forensic import build_forensic_stack
        x_for = build_forensic_stack(x)
        out = model(x, x_for)
        probs = torch.softmax(out["logits"], dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
print("Labels[:20]:", all_labels[:20])
print("Probs[:20]:", all_probs[:20])

import collections
print("Label counts:", collections.Counter(all_labels))
print("Prob > 0.5 counts:", collections.Counter([p > 0.5 for p in all_probs]))

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
print("Min prob:", np.min(all_probs), "Max prob:", np.max(all_probs))
