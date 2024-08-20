import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr,batch_ret
from alphagen.utils.pytorch_utils import masked_mean_std

class FactorNet(nn.Module):
    def __init__(self,n_feat,n_fct,n_tgt,hidden=16,dr_ratio=0.3):
        super().__init__()
        self.n_feat = n_feat
        self.n_fct = n_fct
        self.n_tgt = n_tgt
        self.fc = nn.Sequential(
            nn.Linear(n_feat, hidden),
            nn.ReLU(),
            nn.Dropout(dr_ratio),
            # nn.Linear(hidden, hidden),
            # nn.ReLU(),
            # nn.Dropout(dr_ratio),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat,tgt,fct):
        x = feat
        # x:(n_fct,n_feat)
        x = self.fc(x)
        return x#(n_fct,1)
    
class FactorNetTrain:
    def __init__(self,
                 train_data,test_data,
                 n_feat,n_fct,n_tgt,device,tgt_idx = -1,
                 lr = 1e-3,weight_decay = 1e-4,dr_ratio=0.3,
                 hidden = 16,
                 loss_cfg = {'ic':1.0},
                 verbose=True,
                 ):
        self.train_data = train_data
        self.test_data = test_data
        self.n_feat,self.n_fct,self.n_tgt = n_feat,n_fct,n_tgt
        self.lr,self.weight_decay,self.dr_ratio = lr,weight_decay,dr_ratio
        self.device = device
        self.loss_cfg = loss_cfg
        self.tgt_idx = tgt_idx
        self.verbose = verbose
        self.hidden = hidden


    def build_model(self):
        self.model = FactorNet(self.n_feat,self.n_fct,self.n_tgt,hidden=self.hidden,dr_ratio=self.dr_ratio)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def get_pred_value(self,pred,tgt,fct):
        pred = pred.flatten()
        pred_value = (pred[None] * fct).sum(1)
        return pred_value
    
    def get_metrics_pred_value(self,pred,tgt,fct):
        return self.get_pred_value(pred,tgt,fct)
    
    def get_loss(self,pred,tgt,fct):
        pred = pred.flatten()
        tgt = tgt.flatten()
        # pred:(n_fct,1)
        # tgt:(n_stk)
        # fct:(n_stk,n_fct)

        def sub_corr(pred,label):
            mean_pred = torch.mean(pred)
            std_pred = torch.std(pred)
            mean_label = torch.mean(label)
            std_label = torch.std(label)

            cov = torch.mean((pred - mean_pred) * (label - mean_label))
            corr = cov / (std_pred * std_label)
            return corr
        
        def sub_ret(pred,label):
            mean_pred = torch.mean(pred)
            var_pred = torch.var(pred)
            mean_label = torch.mean(label)

            cov = torch.mean((pred - mean_pred) * (label - mean_label))
            ret = cov / var_pred
            return ret

        def l_ic(pred,tgt,fct):
            pred_value = self.get_pred_value(pred,tgt,fct) # (n_stk)
            ic =  -sub_corr(pred_value,tgt)
            return ic
        
        def l_ret(pred,tgt,fct):
            pred_value = self.get_pred_value(pred,tgt,fct) # (n_stk)
            ret = -sub_ret(pred_value,tgt)
            return ret
        
        def l_mse(pred,tgt,fct):
            pred_value = self.get_pred_value(pred,tgt,fct) # (n_stk)
            mse = F.mse_loss(pred_value,tgt)
            return mse
        def l_msenorm(pred,tgt,fct):
            pred_value = self.get_pred_value(pred,tgt,fct) # (n_stk)
            tgt_norm = (tgt-tgt.mean())/tgt.std()
            mse = F.mse_loss(pred_value,tgt_norm)
            return mse
        
        def l_l2weight(pred,tgt,fct):
            # 惩罚因子权重
            pred_value = pred/pred.sum()
            l2 = torch.mean(pred_value**2)
            return l2
        
        def l_l2fct(pred,tgt,fct):
            # 惩罚合成的大因子值的权重
            pred_value = self.get_pred_value(pred,tgt,fct) # (n_stk)
            l2 = torch.mean(pred_value**2)
            return l2

        final_loss = 0.
        for loss_name,loss_weight in self.loss_cfg.items():
            loss_func = locals()[f'l_{loss_name}']
            final_loss += loss_weight*loss_func(pred,tgt,fct)
        
        return final_loss

    def train_epoch(self,data):
        self.model.train()
        feat_s,tgt_s,fct_s = data
        # feat:(n_days,n_fct,n_feat)
        # tgt:(n_days,n_stk,n_tgt)
        # fct:(n_days,n_stk,n_fct)

        to_iter = list(range(feat_s.shape[0]))
        np.random.shuffle(to_iter)
        if self.verbose:
            to_iter = tqdm(to_iter)


        for j in tqdm(to_iter):

            feat,tgt,fct = feat_s[j],tgt_s[j],fct_s[j]
            # feat:(n_fct,n_feat)
            # tgt:(n_stk,n_tgt)
            # fct:(n_stk,n_fct)
            feat,tgt,fct = self.process_data(feat,tgt,fct)

            self.optimizer.zero_grad()
            pred = self.model(feat,tgt,fct)#(n_fct,1)
            loss = self.get_loss(pred,tgt,fct)
            if torch.isfinite(loss):
                loss.backward()
                self.optimizer.step()
            else:
                print(f"{j} loss invalid")

    def predict(self,data):
        self.model.eval()
        feat_s,tgt_s,fct_s = data
        # feat:(n_days,n_fct,n_feat)
        # tgt:(n_days,n_stk,n_tgt)
        # fct:(n_days,n_stk,n_fct)

        to_iter = range(feat_s.shape[0])
        if self.verbose:
            to_iter = tqdm(to_iter)


        fct_weight_list = []
        pred_list = []

        for j in to_iter:

            feat,tgt,fct = feat_s[j],tgt_s[j],fct_s[j]
            feat,fct = self.process_data_infer(feat,fct)
            feat = feat.to(self.device)
            fct_weight = self.model(feat,tgt,fct)
            pred = self.get_pred_value(fct_weight,tgt,fct)
            fct_weight = fct_weight.flatten()
            if j==1: print(fct_weight.shape,fct.shape,pred.shape)
            fct_weight_list.append(fct_weight)
            pred_list.append(pred)
        return torch.stack(fct_weight_list,dim=0),torch.stack(pred_list,dim=0)

    def get_metrics(self,pred,tgt,fct):
        
        tgt = tgt.flatten()
        # pred:(n_fct,1)
        # tgt:(n_stk)
        # fct:(n_stk,n_fct)

        def sub_corr(pred,label):
            mean_pred = torch.mean(pred)
            std_pred = torch.std(pred)
            mean_label = torch.mean(label)
            std_label = torch.std(label)

            cov = torch.mean((pred - mean_pred) * (label - mean_label))
            corr = cov / (std_pred * std_label)
            return corr
        
        def sub_ret(pred,label):
            mean_pred = torch.mean(pred)
            var_pred = torch.var(pred)
            mean_label = torch.mean(label)

            cov = torch.mean((pred - mean_pred) * (label - mean_label))
            ret = cov / var_pred
            return ret

        def l_ic(pred,tgt,fct):
            pred_value = self.get_metrics_pred_value(pred,tgt,fct) # (n_stk)
            ic =  sub_corr(pred_value,tgt)
            return ic
        def l_ric(pred,tgt,fct):
            pred_value = self.get_metrics_pred_value(pred,tgt,fct) # (n_stk)
            return batch_spearmanr(pred_value[None],tgt[None]).flatten().mean()
        
        def l_ret(pred,tgt,fct):
            pred_value = self.get_metrics_pred_value(pred,tgt,fct) # (n_stk)
            ret = sub_ret(pred_value,tgt)
            return ret
        
        result = {}
        for metric in ['ic','ric']:
            loss_func = locals()[f'l_{metric}']
            result[metric] = loss_func(pred,tgt,fct).item()
        return result

    @torch.no_grad()
    def test_epoch(self,data):
        self.model.eval()
        feat_s,tgt_s,fct_s = data
        # feat:(n_days,n_fct,n_feat)
        # tgt:(n_days,n_stk,n_tgt)
        # fct:(n_days,n_stk,n_fct)

        to_iter = range(feat_s.shape[0])
        if self.verbose:
            to_iter = tqdm(to_iter)


        metrics = []
        for j in to_iter:

            feat,tgt,fct = feat_s[j],tgt_s[j],fct_s[j]
            feat,tgt,fct = self.process_data(feat,tgt,fct)

            pred = self.model(feat,tgt,fct)#(n_fct,1)
            # loss = self.get_loss(pred,tgt,fct)
            cur_metric = self.get_metrics(pred,tgt,fct)
            metrics.append(cur_metric)
        print([np.round(i,4) for i in pred.detach().cpu().numpy().flatten()[:5]])    
        df = pd.DataFrame(metrics)
        # print(df.head())

        # raise Exception('stop')
        df = df.mean()
        return df.to_dict()

    def process_data_infer(self,feat,fct):
        # feat:(n_fct,n_feat)
        # fct:(n_stk,n_fct)
        feat,fct = feat.clone(),fct.clone()
        feat,fct = feat.to(self.device),fct.to(self.device)
        
        # fill nan in fct with 0
        fct[~torch.isfinite(fct)] = 0.
        feat[~torch.isfinite(feat)] = 0.
        # fct = (fct - fct.mean(dim=0)[None] ) / fct.std(dim=0)[None]
        fct[~torch.isfinite(fct)] = 0.

        return feat,fct

    def process_data(self,feat,tgt,fct):
        # feat:(n_fct,n_feat)
        # tgt:(n_stk,n_tgt)
        # fct:(n_stk,n_fct)
        feat,tgt,fct = feat.clone(),tgt.clone(),fct.clone()
        feat,tgt,fct = feat.to(self.device),tgt.to(self.device),fct.to(self.device)
        
        tgt = tgt[:,self.tgt_idx]
        nan_mask = torch.isfinite(tgt)
        tgt,fct = tgt[nan_mask],fct[nan_mask]
        tgt *= 100
        

        # fill nan in fct with 0
        fct[~torch.isfinite(fct)] = 0.
        feat[~torch.isfinite(feat)] = 0.
        # fct = (fct - fct.mean(dim=0)[None] ) / fct.std(dim=0)[None]
        fct[~torch.isfinite(fct)] = 0.

        return feat,tgt,fct

    def fit(self,train_data,valid_datas,epoch = 10,verbose = True):
        history = []
        for cur_epoche in range(epoch):
            if cur_epoche>0:
                self.train_epoch(train_data)
                
            losses = self.test_epoch(train_data)
            losses = {f"train_{k}":np.round(v,4) for k,v in losses.items() }
            for i,valid_data in enumerate(valid_datas):
                loss2 = {f"val{i}_{k}":np.round(v,4) for k,v in self.test_epoch(valid_data).items() }
                losses.update(loss2)
            to_print = [f'Epoch [ {cur_epoche}/{epoch}]']
            # to_print += [f'{k}:{v}' for k,v in losses.items()]
            
            to_print += [f'{k}:{v}' for k,v in losses.items()]
            # to_print +='\n'
            # to_print += [f'{v}' for k,v in losses.items()]
            history.append(losses)
            if verbose:
                print(' '.join(to_print))
        return history
    
    def fit_with_early_stopping(self,train_data,valid_datas,epoch = 10,verbose = True,patience = 5):
        history = []
        best_loss = -np.inf
        best_model = None
        best_epoch = -1
        cur_patience = 0
        best_to_print = None

        best_performance = None
        for cur_epoche in range(epoch):
            if cur_epoche>0:
                self.train_epoch(train_data)
            
            losses = self.test_epoch(train_data)
            losses = {f"train_{k}":np.round(v,4) for k,v in losses.items() }
            for i,valid_data in enumerate(valid_datas):
                loss2 = {f"val{i}_{k}":np.round(v,4) for k,v in self.test_epoch(valid_data).items() }
                losses.update(loss2)
            to_print = [f'Epoch [ {cur_epoche}/{epoch}]']
            # to_print += [f'{k}:{v}' for k,v in losses.items()]
            
            to_print += [f'{k}:{v}' for k,v in losses.items()]
            # to_print +='\n'
            # to_print += [f'{v}' for k,v in losses.items()]
            history.append(losses)
            if verbose:
                print(' '.join(to_print))
            cur_loss = losses['val0_ic']
            if cur_loss > best_loss and cur_epoche>0:
                best_loss = cur_loss
                best_model = self.model.state_dict().copy()
                best_epoch = cur_epoche
                cur_patience = 0
                best_to_print = to_print
                best_performance = losses
            else:
                cur_patience += 1
                if cur_patience >= patience:
                    print(f'early stopping at epoch {cur_epoche}')
                    break

        self.model.load_state_dict(best_model)
        print(f'best epoch:{best_epoch}')
        print(best_to_print)
        return history,best_epoch,best_performance

class DirectFactorNet(nn.Module):
    def __init__(self,n_feat,n_fct,n_tgt,hidden=16,dr_ratio=0.3):
        super().__init__()
        self.n_feat = n_feat
        self.n_fct = n_fct
        self.n_tgt = n_tgt
        self.fc = nn.Sequential(
            nn.Linear(n_fct, hidden),
            nn.ReLU(),
            nn.Dropout(dr_ratio),
            # nn.Linear(hidden, hidden),
            # nn.ReLU(),
            # nn.Dropout(dr_ratio),
            nn.Linear(hidden, 1),
        )

    def forward(self, feat,tgt,fct):
        x = fct
        # x:(batch,n_fct)
        x = self.fc(x)
        return x#(batch,1)
    
class DirectFactorNetTrain(FactorNetTrain):
    def __init__(self,
                 train_data,test_data,
                 n_feat,n_fct,n_tgt,device,tgt_idx = -1,
                 lr = 1e-3,weight_decay = 1e-4,dr_ratio=0.3,
                 hidden = 16,
                 loss_cfg = {'ic':1.0},
                 verbose=True,
                 ):
        super().__init__(train_data,test_data,
                 n_feat,n_fct,n_tgt,device,tgt_idx,
                 lr,weight_decay,dr_ratio,
                 hidden,
                 loss_cfg,
                 verbose,
                 )
    
    def build_model(self):
        self.model = DirectFactorNet(self.n_feat,self.n_fct,self.n_tgt,hidden=self.hidden,dr_ratio=self.dr_ratio)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def get_pred_value(self,pred,tgt,fct):
        # print(f'pred:{pred.shape},tgt:{tgt.shape},fct:{fct.shape}')
        pred = pred.flatten()
        # pred_value = (pred[None] * fct).sum(1)
        return pred


class LimitFactorNet(FactorNetTrain):
    '''
    输入有多个因子，但是选择可以有很多因子
    '''
    def __init__(self,
                 train_data,test_data,
                 n_feat,n_fct,n_tgt,device,tgt_idx = -1,
                 lr = 1e-3,weight_decay = 1e-4,dr_ratio=0.3,
                 hidden = 16,
                 loss_cfg = {'ic':1.0},
                 verbose=True,
                 factor_limit_loss = None,
                 factor_limit_metric = None,
                 ):
        super().__init__(train_data,test_data,
                 n_feat,n_fct,n_tgt,device,tgt_idx,
                 lr,weight_decay,dr_ratio,
                 hidden,
                 loss_cfg,
                 verbose,
                 )
        self.factor_limit_loss = factor_limit_loss
        self.factor_limit_metric = factor_limit_metric
    def get_pred_value(self, pred, tgt, fct):
        pred = pred.flatten()
        # keep TOP 20 as original value, and others to be zero
        if self.factor_limit_loss is not None:
            tail_idx = torch.argsort(pred.abs())[:-self.factor_limit_loss]
            # pred[tail_idx] *= 0.01
        pred_value = (pred[None] * fct).sum(1)
        return pred_value
    def get_metrics_pred_value(self, pred, tgt, fct):
        pred = pred.flatten()
        # keep TOP 20 as original value, and others to be zero
        if self.factor_limit_metric is not None:
            tail_idx = torch.argsort(pred.abs())[:-self.factor_limit_metric]
            pred[tail_idx] = 0.
        pred_value = (pred[None] * fct).sum(1)
        return pred_value


class TwoStepNet(nn.Module):
    def __init__(self,n_feat,n_fct,n_tgt,hidden=16,dr_ratio=0.3,n_limit=20):
        super().__init__()
        self.n_feat = n_feat
        self.n_fct = n_fct
        self.n_tgt = n_tgt
        self.n_limit = n_limit
        self.feat_weight_net = FactorNet(n_feat,n_fct,n_tgt,hidden=hidden,dr_ratio=dr_ratio) # input(n_fct,n_feat) output(n_fct,1)
        self.fct_net = DirectFactorNet(n_feat,n_fct,n_tgt,hidden=hidden,dr_ratio=dr_ratio) # input(n_stk,n_fct) output(n_stk,1)
    def forward(self, feat,tgt,fct):
        # feat:(n_fct,n_feat)
        # tgt:(n_stk,n_tgt)
        # fct:(n_stk,n_fct)
        weight = self.feat_weight_net(feat) # (n_fct,1)
        fct = fct*weight.flatten()[None] # (n_stk,n_fct)
        output = self.fct_net(fct)
        return weight ,output#(batch,1)
    
class LimitFactorNeuralNet(LimitFactorNet):
    def __init__(self,
                 train_data,test_data,
                 n_feat,n_fct,n_tgt,device,tgt_idx = -1,
                 lr = 1e-3,weight_decay = 1e-4,dr_ratio=0.3,
                 hidden = 16,
                 loss_cfg = {'ic':1.0},
                 verbose=True,
                 factor_limit_loss = None,
                 factor_limit_metric = None,
                 ):
        super().__init__(train_data,test_data,
                 n_feat,n_fct,n_tgt,device,tgt_idx,
                 lr,weight_decay,dr_ratio,
                 hidden,
                 loss_cfg,
                 verbose,
                 factor_limit_loss,
                 factor_limit_metric,
                 )
    
    def build_model(self):
        self.model = FactorNet(self.n_feat,self.n_fct,self.n_tgt,hidden=self.hidden,dr_ratio=self.dr_ratio)
        self.model.to(self.device)
        self.optim_weight = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.net = DirectFactorNet(self.n_feat,self.n_fct,self.n_tgt,hidden=self.hidden,dr_ratio=self.dr_ratio)
        self.net.to(self.device)
        self.optim_net = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        