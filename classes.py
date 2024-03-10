import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
class Net(nn.Module):
    def __init__(self): #Constructor
        super(Net, self).__init__()
        self.main = nn.Sequential(
            #784=28*28為將一開始的pixels拉成一維數列
            nn.Linear(in_features=784, out_features=128),
            #activation function使用ReLU()
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            #最後輸出為10個節點, 表示數字0~9
            nn.Linear(in_features=64, out_features=10),
            # nn.LogSoftmax(dim=1)
        )
    def forward(self, input):
        return self.main(input)
    def validation(net , dataloader, loss_function, device):
        loss = 0.0
        accuracy = 0.0
        with torch.no_grad():
            for i, data in enumerate(dataloader):  
                inputs, labels = data[0].to(device), data[1].to(device)
                inputs = inputs.view(inputs.shape[0], -1)

                output = net(inputs)

                loss += loss_function(output, labels).item()

                ps = torch.exp(output) # get the class probabilities from log-softmax
                equality = (labels.data == ps.max(dim=1)[1])
                accuracy += equality.type(torch.FloatTensor).mean()
    
        return loss, accuracy