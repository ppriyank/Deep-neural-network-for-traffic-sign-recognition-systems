from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


### Data Initialization and Loading
from data import initialize_data, data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/val_images',
                         transform=data_transforms),
    batch_size=args.batch_size, shuffle=False, num_workers=1)

### Neural Network and Optimizer
# We define neural net in model.py so that it can be reused by the evaluate.py script
from model import Net
model = Net()

cuda = True 
resume = False

batch_size =50
epochs = 1500
seed = 1
log_interval=400
data = "data"
torch.manual_seed(1)

if cuda : 
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.to(device)


if resume :
    state_dict = torch.load("model_28.pth")
    model.load_state_dict(state_dict) 
    

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


           
def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data =data.to(device)
        target =target.to(device)
        output = model(data)
        validation_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return validation_loss


def train(epoch , train_loader):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss = F.nll_loss(output, target).cuda()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    return 100. * correct / len(train_loader.dataset)



rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer = torch.optim.Adam(model.parameters(), lr=rate)
optimizer = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0.0001)
optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)



step = 10
s1 = []
s2 = []
temp = 999
for epoch in range(1, epochs):
    tran  = train(epoch, train_loader1)
    val = validation()
    if epoch % step :
        print("train: " , tran)
        print("val:" , val)
        s1 += [tran]
        s2 += [val] 
        # scheduler.step()
    if val < temp : 
        temp = val
        model_file = 'model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('\nSaved model to ' + model_file + '. You can run `python evaluate.py ' + model_file + '` to generate the Kaggle formatted csv file') 
