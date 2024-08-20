import torch 
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
from gan.utils import load_pickle
from alphagen_generic.features import *
from alphagen.data.expression import *
from typing import Tuple
import json
from typing import Union
from gan.utils.data import get_data_by_year

def load_alpha_pool(raw) -> Tuple[List[Expression], List[float]]:
    exprs_raw = raw['exprs']
    exprs = [eval(expr_raw.replace('open', 'open_').replace('$', '')) for expr_raw in exprs_raw]
    weights = raw['weights']
    return exprs, weights

def load_alpha_pool_by_path(path: str) -> Tuple[List[Expression], List[float]]:
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
        return load_alpha_pool(raw)
    
import os
def load_ppo_path(path,name_prefix):
    
    files = os.listdir(path)
    folder = [i for i in files if name_prefix in i][0]
    names = [i for i in os.listdir(f"{path}/{folder}") if '.json' in i]
    name = sorted(names,key = lambda x:int(x.split('_')[0]))[-1]
    return f"{path}/{folder}/{name}"

from gan.utils import (
    load_pickle,get_blds_list_df)
import pandas as pd
from alphagen.utils.correlation import batch_pearsonr,batch_spearmanr,batch_ret

def get_feat_sign(feat,names):
    to_add = []
    for i,name in enumerate(names):
        if name.split('_')[-1]=='mean':
            to_add.append(feat[:,:,i:i+1].sign())
    return torch.cat(to_add,dim=-1)


def chunk_batch_spearmanr(x,y,chunk_size=100):
    n_days = len(x)
    spearmanr_list= []
    cur_fct = 0
    for i in range(0,n_days,chunk_size):
        spearmanr_list.append(batch_spearmanr(x[i:i+chunk_size],y[i:i+chunk_size]))
    spearmanr_list = torch.cat(spearmanr_list,dim=0)
    return spearmanr_list


def get_tensor_metrics(x,y):
    ic_s = batch_pearsonr(x,y)
    ric_s = chunk_batch_spearmanr(x,y,chunk_size=400)
    
    # ric_s = ic_s
    ret_s = batch_ret(x,y)

    ic_s = torch.nan_to_num(ic_s,nan=0)
    ric_s = torch.nan_to_num(ric_s,nan=0)
    ret_s = torch.nan_to_num(ret_s,nan=0)

    ic_s_mean = ic_s.mean().item()
    ic_s_std = ic_s.std().item()
    ric_s_mean = ric_s.mean().item()
    ric_s_std = ric_s.std().item()
    ret_s_mean = ret_s.mean().item()
    ret_s_std = ret_s.std().item()


    result = dict(
        ic = ic_s_mean,
        ic_std = ic_s_std,
        icir = ic_s_mean/ic_s_std,
        ric = ric_s_mean,
        ric_std = ric_s_std,
        ricir = ric_s_mean/ric_s_std,
        ret = ret_s_mean,
        ret_std = ret_s_std,
        retir = ret_s_mean/ret_s_std,

    )
    return result

def get_tensor_metrics_raw(x,y):
    ic_s = batch_pearsonr(x,y)
    ric_s = chunk_batch_spearmanr(x,y,chunk_size=400)
    
    ret_s = batch_ret(x,y)

    ic_s = torch.nan_to_num(ic_s,nan=0)
    ric_s = torch.nan_to_num(ric_s,nan=0)
    ret_s = torch.nan_to_num(ret_s,nan=0)

    return ic_s,ric_s,ret_s

import os

def main(
        instruments: str = "csi500",
        train_end_year:int = 2020,
        freq:str = 'day',
        seeds:str = '[0]',
        cuda:int = 0,
        save_name:str = 'test',
        n_factors:int = 10,
        window:Union[int,str] = 'inf',
):
    if isinstance(seeds,str):
        seeds = eval(seeds)
    assert isinstance(seeds,list)   

    if isinstance(window,str):
        assert window == 'inf'
        window = float('inf')

    os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda)
    train_end = train_end_year

    # read data
    returned = get_data_by_year(
        train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
        instruments=instruments, target=target,freq=freq,
    )
    data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned

    for seed in seeds:
        if isinstance(seeds,str):
            seeds = eval(seeds)
        assert isinstance(seeds,list)   
        path = f"out/{save_name}_{instruments}_{train_end}_{seed}/z_bld_zoo_final.pkl"
        tensor_save_path = f"out/{save_name}_{instruments}_{train_end}_{seed}/"
        name = f"{train_end}_{n_factors}_{window}_{seed}"
        zoo = load_pickle(path)
        
        df = get_blds_list_df([zoo]).sort_values('score',ascending=False,key=lambda x:abs(x))
        from gan.utils.builder import exprs2tensor
        fct_tensor = exprs2tensor(df['exprs'],data_all,normalize=True)
        tgt_tensor = exprs2tensor([target],data_all,normalize=False)


        ic_list = []
        ric_list = []
        ret_list = []
        from tqdm import tqdm
        for cur in tqdm(range(fct_tensor.shape[-1])):
            ic_s,ric_s,ret_s = get_tensor_metrics_raw(fct_tensor[...,cur],tgt_tensor[...,0])
            ic_list.append(ic_s)
            ric_list.append(ric_s)
            ret_list.append(ret_s)

        ic_s = torch.stack(ic_list,dim=-1)
        ric_s = torch.stack(ric_list,dim=-1)
        ret_s = torch.stack(ret_list,dim=-1)
        torch.cuda.empty_cache()

        shift = 21

        from tqdm import tqdm
        import numpy as np
        pred_list = []
        ics_list = []
        rics_list = []
        good_idx_list = []
        weights_list = []

        # evaluate from the first day of the valid set untill the last day of the test set
        pbar = tqdm(range(len(fct_tensor)-data_test.n_days-data_valid.n_days,len(fct_tensor)))
        for cur in pbar:

            # control the past window that we use to evaluate the factors in order to filter factors and generate the weights
            if np.isfinite(window):
                begin = cur-window-shift
            else:
                begin = 0

            cur_ic = ic_s[begin:cur-shift]
            cur_ric = ric_s[begin:cur-shift]
            cur_ret = ret_s[begin:cur-shift]

            ic_mean = cur_ic.mean(dim=0)
            ic_std = cur_ic.std(dim=0)
            ric_mean = cur_ric.mean(dim=0)
            ric_std = cur_ric.std(dim=0)
            ret_mean = cur_ret.mean(dim=0)
            ret_std = cur_ret.std(dim=0)

            icir = ic_mean/ic_std
            ricir = ric_mean/ric_std
            retir = ret_mean/ret_std

            metrics = dict(
                ic = ic_mean.detach().cpu().numpy(),
                ic_std = ic_std.detach().cpu().numpy(),
                icir = icir.detach().cpu().numpy(),
                ric = ric_mean.detach().cpu().numpy(),
                ric_std = ric_std.detach().cpu().numpy(),
                ricir = ricir.detach().cpu().numpy(),
                ret = ret_mean.detach().cpu().numpy(),
                ret_std = ret_std.detach().cpu().numpy(),
                retir = retir.detach().cpu().numpy(),
            )
            tmp = pd.DataFrame(metrics).sort_values('ricir',ascending=False,key=lambda x:abs(x))
            

            # filter the factors
            aaaa = tmp[(tmp['ric']>0.02)&(tmp['ricir']>0.2)] 
            if len(aaaa)<1:
                aaaa = tmp.iloc[:1]

            # select the best 'n_factors' alpha factors
            good_idx = aaaa.iloc[:n_factors].index.to_list()
            good_idx_list.append(good_idx)

            # prepare the linear regression data
            x = fct_tensor[begin:cur-shift,:,good_idx]
            y = tgt_tensor[begin:cur-shift,]#.flatten()
            to_pred = fct_tensor[cur,:,good_idx]
            y_true = tgt_tensor[cur,]
            y = y.reshape(-1,y.shape[-1])
            x = x.reshape(-1,x.shape[-1])

            to_select = torch.isfinite(y)[:,0]
            y = y[to_select]
            x = x[to_select]

            to_pred = torch.nan_to_num(to_pred,nan=0)

            # add the constant term
            ones = torch.ones_like(x[...,0:1])
            x = torch.cat([x,ones],dim=-1)
            ones = torch.ones_like(to_pred[...,0:1])
            to_pred = torch.cat([to_pred,ones],dim=-1)

            # train the linear regression model to get weights
            coef = torch.linalg.lstsq(x,y).solution

            # predict the target of the next day
            pred = to_pred @ coef
            weights_list.append(coef.detach().cpu().numpy())

            # calculate the metrics of the prediction
            cur_ic = batch_pearsonr(pred.T,y_true.T)[0]
            cur_ric = batch_spearmanr(pred.T,y_true.T)[0]
            ics_list.append(cur_ic.detach().cpu().numpy())
            rics_list.append(cur_ric.detach().cpu().numpy())

            pbar.set_description(
                f"ic:{np.nanmean(ics_list):.3f} ric:{np.nanmean(rics_list):.3f} n:{len(good_idx)}"
                )
            pred_list.append(pred[:,0])

        # infer the valid set and save the results
        num_1 = data_valid.n_days
        num_2 = data_test.n_days
        all_pred = torch.stack(pred_list,dim=0)
        all_pred = all_pred[-num_2-num_1:-num_1]
        torch.save(all_pred.detach().cpu(),f"{tensor_save_path}/pred_valid_{name}.pt")


        # infer the test set and save the results
        num_ = data_test.n_days
        all_pred = torch.stack(pred_list,dim=0)
        all_pred = all_pred[-num_:]
        torch.save(all_pred.detach().cpu(),f"{tensor_save_path}/pred_{name}.pt")

        # torch.cuda.empty_cache()

if __name__ == '__main__':
    import fire
    fire.Fire(main)