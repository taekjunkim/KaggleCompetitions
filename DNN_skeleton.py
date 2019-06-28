# basic import for pytorch deep neural network
import torch
import torch.nn as nn
import torch.nn.Fucntional as F
from torch.utils.data import Dataset, DataLoader

# Make dataset
class myDataSet(Dataset):
    def __init__(self):
        # download, read data, etc
        self.len = ...
        self.xdata = torch.from_numpy(...)
        self.ydata = torch.from_numpy(...)
    def __getitem__(self,index):
        # return one item on the index
        return self.xdata[index], self.ydata[index]
    def __len__(self):        
        # return the data length
        return self.len

dataset = myDataSet();

# Make dataloader
myDataLoader = DataLoader(dataset=dataset,batch_size=...,shuffle=True);

# define myModel
class myModel(nn.module):
    def __init__(self):
        ...
    def forward(self,x):
        ...
        return y_pred

model = myModel()

loss_fun = ...
optimizer = ...

"""
### training part ###
# 1 epoch: one forward pass, and one backward pass of all the training examples
# batch size: the number of training examples in one forward/backward pass
# number of iteration: number of passes, each pass uses the batch size number of examples
# 1000 training examples: if batch size = 500, 2 iterations are required to complete 1 epoch
"""
for epoch in range(10):
    # loop over all batches
    for i,data in enumerate(myDataLoader,0):
        xNow, yNow = data
        
        # torch --> Variable for autograd
        xNow = Variable(xNow)
        yNow = Variable(yNow)        
        
        # forward pass
        y_pred = model(xNow)
        
        # compute loss
        loss = loss_fun(y_pred,yNow);
        
        # zero graidents, backward pass, update weights
        optimizer.zero_grad();
        loss.backward()
        optimizer.step();
