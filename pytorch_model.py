
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pytorch_datagenerator import DataGenerator
from dataset import Dataset

class NNModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        #VGG starts from 64 filter channel!
        #stage1
        layers += [nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32 ,momentum=0.99),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32 ,momentum=0.99),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)]
        #stage2
        layers += [nn.Conv2d(32, 64, kernel_size=3, padding=1),      
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)]
        #stage3
        layers += [nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2)]
        #stage4
        layers += [nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),  
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(), 
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)]
        #vgg has stage5!
        self.stages = nn.Sequential(*layers)
        self.dense_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 4)
        )
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.stages(x)
        x = torch.flatten(x, 1)
        x = self.dense_layers(x)
        return x

class PytorchModel():
    def __init__(self, dataset):
        self.dataset = dataset
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nnmodel = NNModel()
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.nnmodel.parameters(), lr=1e-3)

    def train(self, logdir):
        epochs = 40
        cp_path = os.path.join(logdir, 'models')
        os.makedirs(cp_path, 0o700)
        cp_path = os.path.join(cp_path, 'best.pt')

        train_loader = DataLoader(DataGenerator(self.dataset, Dataset.TRAIN), batch_size=32,
            pin_memory=True, shuffle=True)
        val_loader = DataLoader(DataGenerator(self.dataset, Dataset.VAL), batch_size=32,
            pin_memory=True, shuffle=True)
        self.nnmodel.to(self.device)
        val_loss_min = np.Inf
        val_accuracy_max = 0
        for epoch in range(1, epochs + 1):
            train_loss, val_loss = 0, 0
            val_accuracy, count = 0, 0
            self.nnmodel.train()
            for inputs, target in train_loader:
                self.optimizer.zero_grad()
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                outputs = self.nnmodel(inputs)
                loss = self.loss_func(outputs, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss
            self.nnmodel.eval()
            with torch.no_grad():
                for inputs, target in val_loader:
                    inputs = inputs.to(self.device)
                    target = target.to(self.device)
                    outputs = self.nnmodel(inputs)
                    loss = self.loss_func(outputs, target)
                    val_loss += loss
                    pred = np.argmax(outputs.cpu().numpy(), axis=1)
                    val_accuracy += np.sum(pred==target.cpu().numpy())
                    count += len(pred)
            if val_loss < val_loss_min:
               val_loss_min = val_loss
               torch.save(self.nnmodel.state_dict(), cp_path)
            val_accuracy /= count
            # if val_accuracy > val_accuracy_max:
            #     val_accuracy_max = val_accuracy
            #     torch.save(self.nnmodel.state_dict(), cp_path) 
            print(f'epoch {epoch:02d}: train loss = {train_loss:.4f}\tval loss = {val_loss:.4f}\tval acc={val_accuracy:.4f}')
        self.nnmodel.load_state_dict(torch.load(cp_path))

    def predict_dataset(self):
        for subset, images in self.dataset.images_dict.items():
            labels = self.dataset.labels_dict[subset]
            norm_images = (images - self.dataset.mean) / self.dataset.stddev
            self.nnmodel.eval()
            with torch.no_grad():
                norm_images = torch.as_tensor(norm_images, device=self.device)
                pred = self.nnmodel(norm_images)
                pred = pred.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            acc = np.mean(pred==labels)
            print(f'Predict: {subset} acc = {acc}')
            #Train = 0.93, Val = 0.93, Test = 0.89