import torch,os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
from classes import Net
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using {device}")

transform = transforms.Compose( 
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),
    #transforms.Normalize((0.1307,), (0.3081,))
    ]
)

epochs = 50
trainSet = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform) 
testSet = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)

trainLoader = dset.DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = dset.DataLoader(testSet, batch_size=64, shuffle=False) #testSet不用打散 vali
dataiter = iter(trainLoader)                    #設定迭代器
images, labels = next(dataiter)             #獲得每批訓練數據
print(images.shape, labels.shape, images.min(), images.max())

class graph():
    global trainSet,x,y,testLoader,prob,classes
    def show6img():
        plt.figure(figsize=(10,10))
        random_inds = np.random.choice(60000,36)
        for i in range(36):
            plt.subplot(6,6,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            image_ind = random_inds[i]
            plt.imshow(np.squeeze(trainSet[image_ind][0]), cmap=plt.cm.binary)   
            plt.xlabel(trainSet[image_ind][1])
            
    def showLoss(figpath = 'F:\\AIresult' , text_data = 'None.' ):
        fig = plt.figure()
        plt.plot(x, y, 'r-', label=u'Training Loss')
        plt.legend()
        plt.xlabel(u'epochs')
        plt.ylabel(u'loss')
        plt.title('Compare loss')
        number = 1
        while True:
            if os.path.isdir(f'{figpath}\\{number}.png') == False:
                plt.savefig(f'{figpath}\\{number}.png')
                f = open(f'{figpath}\\{number}.txt','a')
                f.write(text_data)
                f.close()
                break
            else:
                number += 1
        
    def randresult():
        max_index = np.argmax(prob)
        max_probability = prob[max_index]
        label = classes[max_index]
        fig = plt.figure(figsize=(8,8))
        ax1 = plt.subplot2grid((20,10), (0,0), colspan=9, rowspan=9)
        ax2 = plt.subplot2grid((20,10), (10,2), colspan=5, rowspan=9)

        All= next(iter(testLoader)) 
        inputs=All[0][62]
        labels=All[1][62]
        ax1.axis('off')
        ax1.set_title(labels.tolist())
        ax1.imshow(inputs.numpy().squeeze(), cmap='gray_r')

        labels = classes
    
        y_pos = np.arange(5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(labels)
        ax2.set_xlabel('Probability')
        ax2.invert_yaxis()
        ax2.barh(y_pos, prob, xerr=0, align='center', color='blue')

        plt.show()

net = Net().to(device)
print(net)

optimizer = optim.Adam(net.parameters(), lr=0.0001)
loss_function = nn.CrossEntropyLoss()


steps = 0.0
running_loss = 0.0
train_accuracy = 0.0
start = time.time()
x=[]
y=[]
x1=[]
y1=[]

for e in range(epochs):
    print(f'\r\r epochs = {e+1}/{epochs}                                                    ')
    for i, data in enumerate(tqdm(trainLoader)):
        inputs, labels = data[0].to(device), data[1].to(device) # Move to GPU
        inputs = inputs.view(inputs.shape[0], -1)
        steps += 1
        optimizer.zero_grad()#清零梯度
        # Forward
        output = net(inputs)#將inputs丟入神經元進行運算得到Output
        loss = loss_function(output, labels)#誤差為交叉熵(Cross entropy)，越小越好
        # Backward
        loss.backward()#微分
        optimizer.step()#更新
        running_loss += loss.item()
        # Get the class probabilities from log-softmax
        ps = torch.exp(output) 
        equality = (labels.data == ps.max(dim=1)[1])
        train_accuracy += equality.type(torch.FloatTensor).mean()

        if steps % len(trainLoader) == 0:
            # model.eval() # Validate in each epoch
            with torch.no_grad():
                valid_running_loss, valid_accuracy = Net.validation(net , testLoader, loss_function, device)#validation
                
            print(f'\x1b[1K\r Train Loss:{running_loss/len(trainLoader)}    Train Accu:{train_accuracy/len(trainLoader)}    Val Loss:{valid_running_loss/len(testLoader)}   Vali Accu:{valid_accuracy/len(testLoader)}')
                
            x.append(e+1)
            y.append(running_loss/len(trainLoader))
            x1.append(e+1)
            y1.append(valid_running_loss/len(testLoader))
            running_loss = 0
            train_accuracy = 0
            # model.train() # Make sure training is back on
            

time_elapsed = time.time() - start         
print(f'Total time: {time_elapsed//60}m {time_elapsed % 60}s')


with torch.no_grad():
    test_loss, test_accuracy = Net.validation(net, testLoader, loss_function, device)#testing
    
print(f'Testing Accuracy: {100 * test_accuracy/len(testLoader)}%')
graph.showLoss('F:\\AIresult',f'Average Accuracy: {100 * test_accuracy/len(testLoader)}%')

def single_result(topk=5):
    with torch.no_grad():
        All= next(iter(testLoader)) 
        data=All[0][62],All[1][62]
        inputs, labels = data[0].to(device), data[1].to(device)

        inputs = inputs.view(inputs.shape[0], -1)
        output = net(inputs)
        softmax = nn.Softmax(dim=1)
        ps = softmax(output)
        probs, indices = torch.topk(ps, topk)
        probs = [float(prob) for prob in probs[0]]                       #將每個內容轉為float
        invert = {v: k for k, v in testSet.class_to_idx.items()}    #字典內容與名稱對調
        classes = [invert[int(index)] for index in indices[0]]      #對齊
    return probs, classes

prob, classes= single_result(topk=5)
print("prob: ", prob)
print("classes: ", classes)

#graph.randresult()