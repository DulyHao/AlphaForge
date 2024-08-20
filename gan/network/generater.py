import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gan.utils.builder import Builders
from gan.utils import save_blds

import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import variable
from gan.network.loss import get_losses
import copy

class NetG_DCGAN(nn.Module):
    def __init__(
            self, 
            n_chars:int,
            latent_size: int, 
            seq_len: int,
            hidden: int,

        ):
        super().__init__()
        assert seq_len == 20
        use_bias=True
        self.linear=nn.Linear(latent_size,6*384)
        self.deconv = nn.Sequential(
                    nn.ConvTranspose2d(384,256,(6,1),(1,1),bias=use_bias),#[5, 256, 47, 1]
                    nn.BatchNorm2d(256),nn.ReLU(),
                    nn.ConvTranspose2d(256,192,(5,1),(1,1),bias=use_bias),#[5, 192, 100, 1]
                    nn.BatchNorm2d(192),nn.ReLU(),
                    nn.ConvTranspose2d(192,128,(6,1),(1,1),bias=use_bias),#[5, 128, 205, 1]
                    nn.BatchNorm2d(128),nn.ReLU(),
                )
        self.conv=nn.Sequential(
                    nn.ZeroPad2d((0,0,4,3)),#[5, 128, 212, 1]
                    nn.Conv2d(128,128,(8,1),(1,1),0,bias=use_bias),#[5, 128, 205, 1]
                    nn.BatchNorm2d(128),nn.ReLU(),
                    nn.ZeroPad2d((0,0,4,3)),#[5, 128, 212, 1]
                    nn.Conv2d(128,64,(8,1),(1,1),0,bias=use_bias),#[5, 64, 205, 1]
                    nn.BatchNorm2d(64),nn.ReLU(),
                    nn.ZeroPad2d((0,0,4,3)),#[5, 64, 212, 1]
                    nn.Conv2d(64,n_chars,(8,1),(1,1),0,bias=use_bias),#[5, 4, 205, 1]
        #             nn.BatchNorm2d(4)
        )
        
    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0],384,6,1)
        x = self.deconv(x)
        x = self.conv(x)
        x = x.view(x.shape[0],x.shape[2],x.shape[1])
        return x#(bs, seq_len, 48)
    
class NetG_Lstm(nn.Module):
    def __init__(
        self,
        n_chars:int,
        n_layers: int,
        d_model: int,
        dropout: float,
        seq_len: int,
        potential_size: int,
    ):
        super().__init__()
        self.n_chars = n_chars
        self.max_len = seq_len
        self.n_layers = n_layers
        self.d_model = d_model
        
        
        self.fc_h = nn.Sequential(
            nn.Linear(potential_size,n_layers*d_model),nn.ReLU()
        )
        self.fc_c = nn.Sequential(
            nn.Linear(potential_size,n_layers*d_model),nn.ReLU()
        )
        self.emb = nn.Embedding(n_chars+1,d_model,0)
        self.rnn = nn.LSTM(
            input_size = d_model,
            hidden_size = d_model,
            num_layers = n_layers,
            batch_first = True,
            dropout = dropout
        )
        self.fc = nn.Linear(d_model,n_chars)
    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        # for name, param in self.named_parameters():
        #     nn.init.xavier_normal_(param.data)  # Initialize weight matrices using Xavier initialization
    
    def forward(self,z):
        # z: (batch_size, potential_size)
        # h,c:(n_layers, batch_size, hidden_size)
        device,bs = z.device,z.shape[0]
        h = self.fc_h(z).view(bs,self.n_layers,self.d_model).permute(1,0,2)
        c = self.fc_c(z).view(bs,self.n_layers,self.d_model).permute(1,0,2)
        builders = Builders(bs)
        result = []
        
        input_step = torch.full((bs,),fill_value=self.n_chars,dtype=torch.long)
        
        # onehot_s = torch.zeros(bs,self.max_len,  dtype=torch.long) 
        onehot_s = np.zeros([bs,self.max_len])
        logit_s =torch.zeros(bs,self.max_len,self.n_chars,device=device) 
        # logit_s = []
        mask_s = torch.zeros(bs,self.max_len,self.n_chars,dtype=torch.bool,device=device)
        for t in range(self.max_len):
            
            embedded = self.emb(input_step)[:,None]#[5000, 1, 128]
            if h is None:
                output,(h,c) = self.rnn(embedded)
            else:
                output,(h,c) = self.rnn(embedded,(h,c))#[2(n_layer*n_direction), bs, hidden_size]
            
            
            mask = builders.get_valid_op()# (bs, n_action)
            mask_tensor = torch.from_numpy(mask).to(device)
            logit = self.fc(output).squeeze(1) #(bs, n_chars)
            
            
            # 更新builders
            tmp = logit.detach().cpu().numpy().copy()
            tmp[~mask]=-1e8
            onehot = tmp.argmax(1)
            assert (mask[:,onehot]*1.).mean()
            builders.add_token(onehot)
            
            # 记录当前样本
            onehot_s[:,t] = onehot.flatten()
            logit_s[:,t] = logit
            # logit_s.append(self.fc(output))
            mask_s[:,t] = mask_tensor
            
            # 下一时间步输入
            input_step = torch.torch.from_numpy(onehot_s[:,t].astype(np.compat.long)).to(device)
            
            
        return (onehot_s,logit_s,mask_s),builders

class ResBlock(nn.Module):
    def __init__(self, hidden):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, 5, padding=2),#nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3*output)

class NetG_CNN(nn.Module):
    def __init__(self, n_chars, latent_size,seq_len , hidden):
        super( ).__init__()
        self.fc1 = nn.Linear(latent_size, hidden*seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
            # ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden, n_chars, 1)
        self.n_chars = n_chars
        self.seq_len = seq_len
        self.hidden = hidden

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape)>1:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, noise):
        batch_size = noise.size(0)
        output = self.fc1(noise)
        output = output.view(-1, self.hidden, self.seq_len) # (BATCH_SIZE, DIM, SEQ_LEN)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(batch_size*self.seq_len, -1)
        return output.view(shape) # (BATCH_SIZE, SEQ_LEN, len(charmap))
    
def train_network_generator(netG, netM, netP, cfg, data, target,current_round,random_method,metric,lr,n_actions):
    opt = torch.optim.Adam(netG.parameters(),lr=lr)
    best_weights = None
    best_score = -float('inf')
    patience_counter = 0
    z1 = torch.zeros([cfg.batch_size,cfg.potential_size]).to(cfg.device)
    z2 = torch.zeros([cfg.batch_size,cfg.potential_size]).to(cfg.device)

    netM.eval()
    netP.eval()
    
    empty_blds = None
    best_str_to_print = ''

    for epoch in range(cfg.num_epochs_g):
        netG.train()
        opt.zero_grad()
        z1 = random_method(z1)
        z2 = random_method(z2)
        logit_raw_1 =netG(z1)#（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
        logit_raw_2 =netG(z2)

        masked_x_1,masks_1,blds_1= netM(logit_raw_1)
        masked_x_2,masks_2,blds_2= netM(logit_raw_2)

            
        onehot_tensor_1 = F.gumbel_softmax(masked_x_1,hard=True)
        pred_1,latent_1 = netP(onehot_tensor_1,latent=True)

        onehot_tensor_2 = F.gumbel_softmax(masked_x_2,hard=True)
        pred_2,latent_2 = netP(onehot_tensor_2,latent=True)

        loss_inputs = {
            'logit_raw_1':logit_raw_1,#（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
            'logit_raw_2':logit_raw_2,
            'masked_x_1':masked_x_1,#（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
            'masked_x_2':masked_x_2,
            'masks_1':masks_1,#（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
            'masks_2':masks_2,
            'blds_1':blds_1,
            'blds_2':blds_2,
            'z1':z1,#（batch_size,latent_size）
            'z2':z2,
            'onehot_tensor_1':onehot_tensor_1,#（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
            'onehot_tensor_2':onehot_tensor_2,
            'pred_1':pred_1,#（batch_size,1）
            'pred_2':pred_2,
            'latent_1':latent_1,#（batch_size,256）
            'latent_2':latent_2,
        }
        loss = get_losses(loss_inputs,cfg)
        
        blds:Builders = blds_1+blds_2
        idx = [i for i in range(blds.batch_size) if blds.builders[i].is_valid()]
        blds.drop_invalid()
        blds.evaluate(data,target,metric)
        
        str_to_print = f"##{epoch}/{cfg.num_epochs_g} : n_valid_train:{len(idx)}, n_valid:{len(blds.scores)}, loss:{loss:.4f}"
        mean_score = np.mean(blds.scores)
        max_score = np.max(blds.scores) if len(blds.scores)>0 else 0
        std_score = np.std(blds.scores) if len(blds.scores)>0 else 0
        str_to_print += f", max_score:{max_score:.4f},   mean_score:{mean_score:.4f}, std_score:{std_score:.4f}"
        blds.drop_duplicated()
        str_to_print += f",unique:{blds.batch_size}"
        print(str_to_print)
        if max_score>0:
            exprs = blds.exprs_str[np.argmax(blds.scores)]
            print(f"Max score {max_score} expr: {exprs}")
        # save_blds(blds,f"out/{cfg.name}/train/{current_round}",epoch)

        if empty_blds is None:
            empty_blds = blds
        else:
            empty_blds = empty_blds + blds

        
        if cfg.g_es_score == 'mean':
            es_score = mean_score
        elif cfg.g_es_score == 'max':
            es_score = max_score
        elif cfg.g_es_score == 'combined':
            es_score = max_score + 2. *  std_score
        else:
            raise NotImplementedError
        
        if es_score > best_score:
            best_score = es_score
            best_weights = copy.deepcopy(netG.state_dict())
            best_str_to_print = str_to_print
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > cfg.g_es:
                print(f'Early stopping triggered at epoch {epoch}, {best_score} !')
                
                break
        
        if epoch>0:
            loss.backward()
            opt.step()

    if best_weights is not None:
        print('load_best_weights')
        netG.load_state_dict(best_weights)
        print(best_str_to_print)

    empty_blds.drop_duplicated()
    return empty_blds