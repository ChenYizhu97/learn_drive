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

    def __init__(self, images_dir='dataset/images', labels_dir='dataset/labels'):
        self.images_dir = images_dir
        self.T_labels = np.loadtxt(os.path.join(labels_dir, 'T_labels.csv'), delimiter='\n', dtype=np.float32)
        self.C_labels = np.loadtxt(os.path.join(labels_dir, 'C_labels.csv'), delimiter='\n', dtype=np.float32)
        self.transform = tsfm
        super().__init__()

    def __len__(self):
        return len(os.listdir(self.images_dir))

    def __getitem__(self, index):
        image_name = 'image' + str(index) +'.png'
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path)
        image = self.transform(image)
        label = (self.T_labels[index], self.C_labels[index])
        return image, label

model = nn.Sequential(
        nn.Conv2d(3, 6, (6, 6)),
        nn.Conv2d(6, 12, (6, 6)),
        nn.MaxPool2d((5, 5)),
        nn.ReLU(),
        nn.Conv2d(12, 24, (4, 4)),
        nn.Conv2d(24, 32, (4, 4)),
        nn.MaxPool2d((3, 3)),
        nn.ReLU(),
        nn.Conv2d(32, 64, (3, 3)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.MaxPool2d((2, 2)),
        nn.ReLU(),
        nn.Conv2d(128, 128, (2, 2)),
        nn.Conv2d(128, 256, (1, 1)),
        nn.Flatten(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 2),
        nn.Tanh(),
        )
    
if __name__ == "__main__":
       
    dataset = MyTDataset()
    dataloader = DataLoader(dataset, 128, shuffle=True)
    
    L1_loss = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    losses_T = []
    losses_C = []

    for epoch in range(0, 100):
        loss_T = 0
        loss_C = 0
        for batch_id, (images, (label_T, label_C)) in enumerate(dataloader):
            optimizer.zero_grad()
            predicts = model(images).reshape(-1, 2)
            predicts_T = predicts[:,0]
            predicts_C = predicts[:,1]
            batch_loss_T = L1_loss(predicts_T, label_T)
            batch_loss_C = L1_loss(predicts_C, label_C)
            batch_loss_C.backward(retain_graph=True)
            batch_loss_T.backward()
            optimizer.step()            
            loss_T += batch_loss_T.detach().numpy().item()       
            loss_C += batch_loss_C.detach().numpy().item()  
        print('epoch {}, T loss is {:.5f}, C loss is {:.5f}'.format(epoch, loss_T, loss_C))
        losses_T.append(loss_T)
        losses_C.append(loss_C)    
    torch.save(model.state_dict(), 'model_state')

    print(losses_T)
    print(losses_C)