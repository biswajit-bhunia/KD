import torch
import torchvision.transforms.functional as TF

x = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 1, 3)
x_aug = TF.adjust_brightness(x, 1.1)
print(x_aug)
