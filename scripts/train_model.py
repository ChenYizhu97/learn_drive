import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

tsfm = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ])

class MyTDataset(Dataset):

    def __init__(self, images_dir='images', labels_dir='labels'):
        images = os.listdir(images_dir) 
        self.image_paths = [os.path.join(images_dir, image) for image in images]
        self.T_labels = np.loadtxt(os.path.join(labels_dir, 'T_labels.csv'), delimiter='\n', dtype=np.float32)
        self.transform = tsfm
        super().__init__()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = self.transform(image)
        label = self.T_labels[index]
        return image, label

model = nn.Sequential(
        nn.Conv2d(3, 6, (6, 6)),
        nn.Conv2d(6, 12, (6, 6)),
        nn.MaxPool2d((5, 5)),
        nn.Conv2d(12, 24, (4, 4)),
        nn.Conv2d(24, 32, (4, 4)),
        nn.MaxPool2d((3, 3)),
        nn.Conv2d(32, 64, (3, 3)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.MaxPool2d((2, 2)),
        nn.Conv2d(128, 128, (2, 2)),
        nn.Conv2d(128, 256, (1, 1)),
        nn.Flatten(),
        nn.Linear(256, 1),
        nn.Tanh(),
        )
    
if __name__ == "__main__":
       
    dataset = MyTDataset()
    dataloader = DataLoader(dataset, 128, shuffle=True)
    
    L1_loss = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    losses = []

    for epoch in range(0, 10):
        loss = 0
        for batch_id, (images, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            predicts = model(images).reshape(-1)
            batch_loss = L1_loss(predicts, labels)
            batch_loss.backward()
            optimizer.step()
            
            loss += batch_loss.detach().numpy().item()       
        print('epoch {}, loss is {:.10f}'.format(epoch, loss))
        losses.append(loss)    
    torch.save(model.state_dict(), 'model_state')

    print(losses)