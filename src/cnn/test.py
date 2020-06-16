import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import net

transform = transforms.Compose(
    [transforms.CenterCrop(360),
     transforms.Resize(100),
     transforms.ToTensor()]
)

# dataloader
dataset_tr = ImageFolder('./data/modified/', transform=transform)
dataloader_tr = DataLoader(dataset_tr, batch_size=8, shuffle=False, num_workers=4)

dataset_te = ImageFolder('./data/modified/', transform=transforms.ToTensor)
dataloader_te = DataLoader(dataset_tr, batch_size=8, shuffle=False, num_workers=4)
# show image
for i, (img, target) in enumerate(dataloader_tr):
    npimgs = torchvision.utils.make_grid(img).numpy()
    plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.show()
    break
for i, (img, target) in enumerate(dataloader_te):
    npimgs = torchvision.utils.make_grid(img).numpy()
    plt.imshow(np.transpose(npimgs, (1, 2, 0)))
    plt.show()
    break


model = net.Net()
model.eval()

for i, (img, targets) in enumerate(dataloader_tr):
    out = model(img)
    print(torch.argmax(out, dim=1))
