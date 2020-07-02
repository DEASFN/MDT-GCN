import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from models import directed_gcn as gcn


# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1,  bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# MDTGCN Module
class MDTGCN(nn.Module):
    def __init__(self, block, layers, num_frame, batch_size, adj, gcn_layer):
        super(MDTGCN, self).__init__()

        self.num_frame = num_frame
        self.batch_size = batch_size
        self.adj = adj
        self.in_channels = self.num_frame * 30
        self.gcn_layer = gcn_layer
        self.layer1 = self.make_layer(block, 38, layers[0])
        self.layer2 = self.make_layer(block, 19, layers[1])


        self.decoder_JH = nn.Sequential(

            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(19),
            nn.ReLU(inplace=True),

            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(19),
            nn.ReLU(inplace=True),

            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.decoder_PAF = nn.Sequential(

            nn.Conv2d(19, 38, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(38),
            nn.ReLU(inplace=True),

            nn.Conv2d(38, 38, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(38),
            nn.ReLU(inplace=True),

            nn.Conv2d(38, 38, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.encoder = nn.Sequential(

            nn.Conv2d(19, 19, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(19),
            nn.ReLU(inplace=True),

            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(19),
            nn.ReLU(inplace=True),

            nn.Conv2d(19, 19, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.featuremap_2_graph = nn.ModuleList([])
        self.gru_cell = nn.ModuleList([])
        self.graph_conv = nn.ModuleList([])

        for f in range(self.num_frame):
            self.featuremap_2_graph.append(gcn.Featuremaps_to_Graph(input_channels=19, hidden_layers=62, nodes=19))
            self.gru_cell.append(torch.nn.GRUCell(19*62, 19*32))
            self.graph_conv.append(nn.ModuleList([]))
            for _ in range(self.gcn_layer):
                self.graph_conv[f].append(gcn.GraphConvolution_5part(in_features=62, out_features=62))

        self.graph_2_featuremap = gcn.Graph_to_Featuremaps_savemem(input_channels=19, output_channels=19,
                                                                     hidden_layers=32, nodes=19)


    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels*block.expansion):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels*block.expansion, stride=stride),
                nn.BatchNorm2d(out_channels*block.expansion))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)


    def map_2_graph(self, feature_map):

        graph = []
        for f in range(self.num_frame):
            graph.append(self.featuremap_2_graph[f](feature_map[f*self.batch_size: (f+1)*self.batch_size]))

        return graph


    def graph_convolution(self, graph):

        for f in range(self.num_frame):
            for l in range(self.gcn_layer):
                graph[f] = self.graph_conv[f][l].forward(graph[f], self.adj, relu=True)

        return graph


    def gru(self, graph):
        h = torch.randn(self.batch_size, 19*32).cuda()
        for f in range(self.num_frame):
            h = self.gru_cell[f](graph[f], h)
        h = h.view(self.batch_size, 19, -1)
        h = h.unsqueeze(0)
        return h


    def graph_transform(self, graph):
        for f in range(self.num_frame):
            graph[f] = torch.squeeze(graph[f])
            graph[f] = graph[f].view(self.batch_size, -1)
        return graph


    def forward(self, input):

        # input size : batch_sequence_size * num_frame * 150 * 3 * 3
        x = torch.zeros(self.batch_size * self.num_frame, 150, 3, 3).cuda()
        count = 0
        for i in range(self.num_frame):
            for j in range(self.batch_size):
                x[count] = input[j][i]
                count += 1

        # Feature Extractor
        x = self.layer1(x)
        x = self.layer2(x)

        # temporal encoder
        graph = self.map_2_graph(x)
        graph = self.graph_convolution(graph)
        graph = self.graph_transform(graph)
        h = self.gru(graph)
        

        # spatial encoder
        x = F.interpolate(x[(self.num_frame-1)*self.batch_size: self.num_frame*self.batch_size], size=[92, 124],
                             mode='bilinear', align_corners=False)
        x = self.encoder(x)

        # decoder
        fea_map = self.graph_2_featuremap(h, x)
        fea_map = fea_map + x
        JH = self.decoder_JH(fea_map)
        PAF = self.decoder_PAF(fea_map)

        return JH, PAF