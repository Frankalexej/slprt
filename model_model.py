# This is modelling part

# This is where to put models

#LIBS
import torch
from torch import nn
import torch.nn.functional as F
from model_configs import in_dim, out_dim, hid_dim, enc_lat_dims, dec_lat_dims, dropout

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
        # self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.lin(x)
        x = self.relu(x)
        # x = self.batch_norm(x)
        # x = self.dropout(x)
        return x


class HandshapePredictor(nn.Module):
    def __init__(self, input_dim=in_dim, 
                 enc_lat_dims=enc_lat_dims, 
                 hid_dim=hid_dim, 
                 dec_lat_dims=dec_lat_dims, 
                 output_dim=out_dim):
        super(HandshapePredictor, self).__init__()

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
        batch_num, lm_num, dim_num = x.size()
        x = x.view(batch_num, -1)   # pack the matrix into a vector

        h = self.encoder(x)
        pred_probs = self.decoder(h)
        # .view(size=y_size)
        # class_pred = nn.Softmax(class_pred)

        return pred_probs
    

    def predict(self, x, handshapeDict): 
        batch_num, lm_num, dim_num = x.size()
        x = x.view(batch_num, -1)   # pack the matrix into a vector

        h = self.encoder(x)
        pred_probs = self.decoder(h)
        pred_probs =F.softmax(pred_probs, dim=1)
        class_pred = torch.argmax(pred_probs, dim=1)
        pred_tag = handshapeDict.batch_map(class_pred)
        return h, pred_tag
    
    def encode(self, x): 
        batch_num, lm_num, dim_num = x.size()
        x = x.view(batch_num, -1)   # pack the matrix into a vector

        h = self.encoder(x)
        return h