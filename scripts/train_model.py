import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

tsfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]) 

class MyTDataset(Dataset):

    def __init__(self, images_dir=os.path.join('dataset', 'images'), labels_dir=os.path.join('dataset', 'labels'), transform=tsfm):
        self.images_dir = images_dir
        self.T_labels = np.loadtxt(os.path.join(labels_dir, 'T_labels.csv'), delimiter='\n', dtype=np.float32)
        self.C_labels = np.loadtxt(os.path.join(labels_dir, 'C_labels.csv'), delimiter='\n', dtype=np.float32)
        self.transform = transform
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

model =  nn.Sequential(
        torchvision.models.resnet18(pretrained=True),
        nn.BatchNorm1d(1000),
        nn.ReLU(),
        nn.Linear(1000, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 2),
        nn.Tanh(),
        ).cuda()   
    
if __name__ == "__main__":
       
    #dataset = MyTDataset()
    dataset = MyTDataset(transform=tsfm)
    dataloader = DataLoader(dataset, 64, shuffle=True, num_workers=4)
    
    L1_loss = nn.L1Loss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    losses_T = []
    losses_C = []

    for epoch in range(0, 100):
        loss_T = torch.Tensor([0.0]).cuda()
        loss_C = torch.Tensor([0.0]).cuda()
        for batch_id, (images, (labels_T, labels_C)) in enumerate(dataloader):
            batch_size = images.shape[0]
            optimizer.zero_grad()
            images = images.cuda()
            labels_C = labels_C.cuda()
            labels_T = labels_T.cuda()
            predicts = model(images).reshape(-1, 2)
            predicts_T = predicts[:,0]
            predicts_C = predicts[:,1]
            batch_loss_T = L1_loss(predicts_T, labels_T)
            batch_loss_C = L1_loss(predicts_C, labels_C)
            batch_loss_C.backward(retain_graph=True)
            batch_loss_T.backward()
            optimizer.step()            
            loss_T += batch_loss_T.detach() * batch_size             
            loss_C += batch_loss_C.detach() * batch_size            
            print('epoch {}, batch {}...\r'.format(epoch, batch_id), end='')
        print('epoch {}, loss_T is {}, loss_C is {}'.format(epoch, loss_T, loss_C))
        losses_T.append(loss_T)
        losses_C.append(loss_C)    
    torch.save(model.state_dict(), 'model_state')
    print(losses_T)
    print(losses_C)