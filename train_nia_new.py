import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.cuda.comm
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# from models.ST_Former import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader.dataset_NIA import train_data_loader, test_data_loader
import wandb

from runner_helper import *

class Pseudoarg():
    def __init__(self):
        self.workers = 1
        self.epochs = 100
        self.start_epoch = 0
        self.batch_size = 32
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.print_freq = 10
        self.resume = None
        self.data_set = 0
        
args = Pseudoarg()


# In[45]:


now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H:%M]-")
project_path = '/media/di/data2/nia/Former-DFER/nia/data/'
log_txt_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.txt'
log_curve_path = project_path + 'log/' + time_str + 'set' + str(args.data_set) + '-log.png'
checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model.pth'
best_checkpoint_path = project_path + 'checkpoint/' + time_str + 'set' + str(args.data_set) + '-model_best.pth'


# In[51]:


#def main():
best_acc = 0
recorder = RecorderMeter(args.epochs)
print('The training time: ' + now.strftime("%m-%d %H:%M"))
print('The training set: set ' + str(args.data_set))
os.makedirs("/media/di/data2/nia/Former-DFER/nia/data/log",exist_ok = True)
with open(log_txt_path, 'a') as f:
    f.write('The training set: set ' + str(args.data_set) + '\n')

# create model and load pre_trained parameters
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )
    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
device = "cuda" if torch.cuda.is_available() else "cpu"    
model = NeuralNetwork().to(device)
# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


# In[47]:


if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        recorder = checkpoint['recorder']
        best_acc = best_acc.cuda()
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
cudnn.benchmark = True


# In[48]:


# Data loading code
train_data = train_data_loader(project_dir=project_path, 
                               data_set=args.data_set)
test_data = test_data_loader(project_dir=project_path,
                             data_set=args.data_set)

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.workers,
                                           pin_memory=True,
                                           drop_last=True)
val_loader = torch.utils.data.DataLoader(test_data,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.workers,
                                         pin_memory=True)


# In[50]:


wandb.init(project="nia")

for epoch in range(args.start_epoch, args.epochs):
    inf = '********************' + str(epoch) + '********************'
    start_time = time.time()
    current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

    with open(log_txt_path, 'a') as f:
        f.write(inf + '\n')
        f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

    print(inf)
    print('Current learning rate: ', current_learning_rate)

    # train for one epoch
    train_acc, train_los = train(train_loader, model, criterion, optimizer, epoch, args)

    # evaluate on validation set
    val_acc, val_los = validate(val_loader, model, criterion, args)

    scheduler.step()

    # remember best acc and save checkpoint
    is_best = val_acc > best_acc
    best_acc = max(val_acc, best_acc)
    save_checkpoint({'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'best_acc': best_acc,
                     'optimizer': optimizer.state_dict(),
                     'recorder': recorder}, is_best)

    # print and save log
    epoch_time = time.time() - start_time
    recorder.update(epoch, train_los, train_acc, val_los, val_acc)
    recorder.plot_curve(log_curve_path)
    
    wandb.log({"Train_Loss" : train_los, "Accuracy": train_acc}, step = epoch)
    print('The best accuracy: {:.3f}'.format(best_acc.item()))
    print('An epoch time: {:.1f}s'.format(epoch_time))
    with open(log_txt_path, 'a') as f:
        f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
        f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')


# In[35]:


def train(train_loader, model, criterion, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(train_loader):

        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i, log_txt_path)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i, log_txt_path)

        # TODO: this should also be done with the ProgressMeter
        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg


def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


# In[ ]:




