# This is modelling part

# This is where to put models

#LIBS
import torch
from torch import nn
import torch.nn.functional as F
from model_configs import *

def flatten(x, xyz_together=True): 
    if xyz_together: 
        batch_num, lm_num, dim_num = x.size()
        return x.view(batch_num, -1)

    else: 
        x_values = x[:, :, 0]
        y_values = x[:, :, 1]
        z_values = x[:, :, 2]
        # Concatenate the grouped values
        return torch.cat((x_values, y_values, z_values), dim=1)

# MODELS
class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.lin1 = nn.Linear(n_chans, n_chans)
        self.lin2 = nn.Linear(n_chans, n_chans)
        # self.batch_norm = nn.BatchNorm1d(num_features=n_chans)  # <5>
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.lin1(x)
        # out = self.batch_norm(out)
        out = self.relu(out)
        out = self.lin2(out)
        # out = self.batch_norm(out)
        out = self.relu(out)
        return out + x


class LinPack(nn.Module):
    def __init__(self, n_in, n_out):
        super(LinPack, self).__init__()
        self.lin = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()
        # self.batch_norm = nn.BatchNorm1d(num_features=n_out)
        # self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        x = self.lin(x)
        # x = self.batch_norm(x)
        x = self.relu(x)
        # x = self.dropout(x)
        return x
    
class ConvPack(nn.Module): 
    def __init__(self, length, in_channels, out_channels, kernal_size):
        super(ConvPack, self).__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernal_size)
        self.act = nn.ReLU()
        self.maxpooling = nn.MaxPool1d(kernel_size=length - kernal_size + 1)

    def forward(self, x): 
        x = self.conv(x)
        x = self.act(x)
        x = self.maxpooling(x)
        return x


class CNNHandshapePredictor(nn.Module):
    def __init__(self, input_dim=in_dim, 
                 enc_lat_dims=enc_lat_dims, 
                 hid_dim=hid_dim, 
                 dec_lat_dims=dec_lat_dims, 
                 output_dim=out_dim, 
                 window_sizes=window_sizes):
        super(CNNHandshapePredictor, self).__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=input_dim, 
                    out_channels=hid_dim, 
                    kernel_size=ws
                ), 
                nn.MaxPool1d(kernel_size=length - ws + 1), 
                nn.ReLU(), 
                # nn.BatchNorm1d(num_features=hid_dim)
            ) for ws in window_sizes
        ])

        self.encoder = nn.Sequential(
            # LinPack(input_dim, enc_lat_dims[0]), 
            # ResBlock(enc_lat_dims[0]), 
            # LinPack(enc_lat_dims[0], enc_lat_dims[1]), 
            # ResBlock(enc_lat_dims[1]), 
            # LinPack(enc_lat_dims[1], enc_lat_dims[2]),
            # ResBlock(enc_lat_dims[2]), 
            nn.Linear(hid_dim * len(window_sizes), hid_dim), 
            nn.Dropout()
        )

        self.decoder =  nn.Sequential(
            LinPack(hid_dim, dec_lat_dims[1]), 
            # ResBlock(dec_lat_dims[0]), 
            # LinPack(dec_lat_dims[0], dec_lat_dims[1]), 
            # ResBlock(dec_lat_dims[1]), 
            # nn.Linear(dec_lat_dims[1], output_dim),
            nn.Linear(dec_lat_dims[1], out_dim)
        )

        

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, L, D) -> (B, D, L), as required by Conv1d
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.squeeze()
        h = self.encoder(x)
        pred_probs = self.decoder(h)
        return pred_probs
    

    def predict(self, x, handshapeDict): 
        x = x.permute(0, 2, 1)  # (B, L, D) -> (B, D, L), as required by Conv1d
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.squeeze()
        h = self.encoder(x)
        pred_probs = self.decoder(h)
        pred_probs =F.softmax(pred_probs, dim=1)
        class_pred = torch.argmax(pred_probs, dim=1)
        pred_tag = handshapeDict.batch_map(class_pred)
        return h, pred_tag
    
    def encode(self, x): 
        x = x.permute(0, 2, 1)  # (B, L, D) -> (B, D, L), as required by Conv1d
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.squeeze()
        h = self.encoder(x)
        return h
    

class LinearHandshapePredictor(nn.Module):
    def __init__(self, input_dim=in_dim, 
                 enc_lat_dims=enc_lat_dims, 
                 hid_dim=hid_dim, 
                 dec_lat_dims=dec_lat_dims, 
                 output_dim=out_dim):
        super(LinearHandshapePredictor, self).__init__()

        self.encoder = nn.Sequential(
            LinPack(input_dim, enc_lat_dims[0]), 
            ResBlock(enc_lat_dims[0]), 
            LinPack(enc_lat_dims[0], enc_lat_dims[1]), 
            ResBlock(enc_lat_dims[1]), 
            # LinPack(enc_lat_dims[1], enc_lat_dims[2]),
            # ResBlock(enc_lat_dims[2]), 
            nn.Linear(enc_lat_dims[1], hid_dim), 
        )

        self.decoder =  nn.Sequential(
            LinPack(hid_dim, dec_lat_dims[0]), 
            # ResBlock(dec_lat_dims[0]), 
            LinPack(dec_lat_dims[0], dec_lat_dims[1]), 
            # ResBlock(dec_lat_dims[1]), 
            nn.Linear(dec_lat_dims[1], output_dim),
        )

        

    def forward(self, x):
        x = flatten(x, xyz_together=False)

        h = self.encoder(x)
        pred_probs = self.decoder(h)
        # .view(size=y_size)
        # class_pred = nn.Softmax(class_pred)

        return pred_probs

    def predict(self, x, handshapeDict): 
        x = flatten(x, xyz_together=False)

        h = self.encoder(x)
        pred_probs = self.decoder(h)
        pred_probs =F.softmax(pred_probs, dim=1)
        class_pred = torch.argmax(pred_probs, dim=1)
        pred_tag = handshapeDict.batch_map(class_pred)
        return h, pred_tag
    
    def encode(self, x): 
        x = flatten(x, xyz_together=False)

        h = self.encoder(x)
        return h

#######################################################################################
#Here we put old items. #
"""
"""