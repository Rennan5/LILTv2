import torch
import torch.nn as nn
from pre_training_classes import *

class PretrainingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, grid_size=7, num_directions=8):
        super(PretrainingModel, self).__init__()
        self.mvlm = MVLMTask(vocab_size, embed_dim)
        self.kpl = KPLTask(grid_size)
        self.rpc = RPCTask(num_directions)
        self.wpa = WPATask(embed_dim)
    
    def forward(self, inputs, masks, key_points, relative_positions, word_embeds, patch_embeds):
        mvlm_loss = self.mvlm(inputs, masks)
        kpl_loss = self.kpl(key_points)
        rpc_loss = self.rpc(relative_positions)
        wpa_loss = self.wpa(word_embeds, patch_embeds)
        
        total_loss = mvlm_loss + kpl_loss + rpc_loss + wpa_loss
        return total_loss
