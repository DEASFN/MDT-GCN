import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import glob
import hdf5storage
from random import shuffle
import time
import torch.nn.functional as F
import os
from models.MDTGCN import MDTGCN, ResidualBlock
from models import graph

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

num_epochs = 50
num_frame = 5
kk = 2
batch_sequence_size = 128  #batch_sequence_size = batch_size / num_frame
learning_rate = 0.001
load_model = False
model_path = 'weights/MDTGCN.pkl'

def save_checkpoint(state, filename=target):
    torch.save(state, filename)


def getSequenceMinibatch(file_names):
    sequence_num = len(file_names)
    csi_data = torch.zeros(sequence_num, num_frame, 150, 3, 3)
    heatmaps = torch.zeros(sequence_num, 57, 46, 62)
    for i in range(sequence_num):
        for j in range(num_frame):
            data = hdf5storage.loadmat(file_names[i][j], variable_names={'csi_serial', 'heatmaps'})
            csi_data[i, j, :, :, :] = torch.from_numpy(data['csi_serial']).type(torch.FloatTensor).view(-1, 3, 3)
            if j == num_frame-1:
                heatmaps[i, :, :, :] = torch.from_numpy(data['heatmaps']).type(torch.FloatTensor)
    return csi_data, heatmaps


def takeSecond(elem):
    return int(elem.split("/")[-1].split(".")[0])


mat = []
mat.append(glob.glob('/data/*.mat'))

#split training data
for i in range(len(mat)) :
    mat[i].sort(key = takeSecond)


mats_sequence = []
for i in range(len(mat)) :
    for j in range(int(len(mat[i])*0.9)) :
        per_sequence = []
        for k in range(num_frame):
            per_sequence.append(mat[i][j+k])
        mats_sequence.append(per_sequence)


mats_sequence_num = len(mats_sequence)
batch_sequence_num = int(np.floor(mats_sequence_num/batch_sequence_size))
print('num_frame = ', num_frame)
print('mats_sequence_num = ', mats_sequence_num)



adj1 = Variable(torch.from_numpy(graph.preprocess_adj_5part(graph.JH_graph_center)).float())
adj2 = Variable(torch.from_numpy(graph.preprocess_adj_5part(graph.JH_graph_closer)).float())
adj3 = Variable(torch.from_numpy(graph.preprocess_adj_5part(graph.JH_graph_farther)).float())
adj4 = Variable(torch.from_numpy(graph.preprocess_adj_5part(graph.JH_graph_symmetric)).float())
adj5 = Variable(torch.from_numpy(graph.preprocess_adj_5part(graph.JH_graph_background)).float())

adj = []
adj.append(adj1.unsqueeze(0).unsqueeze(0).expand(1, 1, 19, 19).cuda())
adj.append(adj2.unsqueeze(0).unsqueeze(0).expand(1, 1, 19, 19).cuda())
adj.append(adj3.unsqueeze(0).unsqueeze(0).expand(1, 1, 19, 19).cuda())
adj.append(adj4.unsqueeze(0).unsqueeze(0).expand(1, 1, 19, 19).cuda())
adj.append(adj5.unsqueeze(0).unsqueeze(0).expand(1, 1, 19, 19).cuda())


MDTGCN = MDTGCN(ResidualBlock, [2, 2], num_frame, batch_sequence_size, adj, 3)
optimizer = torch.optim.Adam(MDTGCN.cuda().parameters(), lr=learning_rate)


if load_model:
    if os.path.isfile(model_path):
        print("=> loading checkpoint '{}'".format(target))
        checkpoint = torch.load(target)
        start_epoch = checkpoint['epoch']
        MDTGCN.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(target, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(model_path))
else:
    start_epoch = 0
    print("=> Not loading a checkpoint")


MDTGCN = MDTGCN.cuda()
criterion_L2 = nn.MSELoss(reduction='none').cuda()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15, 20, 25, 30, 35, 40, 45, 50], gamma=0.5)

MDTGCN.train()
for epoch_index in range(start_epoch,num_epochs):

    print('epoch_index=', epoch_index)

    start = time.time()

    # shuffling dataset
    shuffle(mats_sequence)

    # in each minibatch
    for batch_index in range(batch_sequence_num):
        if batch_index < batch_sequence_num:
            file_names = mats_sequence[batch_index*batch_sequence_size:(batch_index+1)*batch_sequence_size]
        else:
            file_names = mats_sequence[batch_sequence_num*batch_sequence_size:]

        csi_data, heatmaps = getSequenceMinibatch(file_names)

        csi_data = Variable(csi_data.cuda())
        heatmaps = Variable(heatmaps.cuda())

        mask = torch.ones(batch_sequence_size, 57, 46, 62).cuda()
        mask = k * torch.abs(heatmaps) + mask

        pred_JH, pred_PAF = MDTGCN(csi_data)
        prediction = torch.cat((pred_JH, pred_PAF), axis=1)

        loss = torch.sum(torch.mul(mask, criterion_L2(heatmaps, prediction)))
        print("total " + str(loss.item() / batch_sequence_size * 32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    scheduler.step()

    endl = time.time()
    print('Costing time:', (endl-start)/60)
    save_checkpoint({
        'epoch': epoch_index + 1,
        # 'arch': args.arch,
        'state_dict': MDTGCN.state_dict(),
        # 'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    })
