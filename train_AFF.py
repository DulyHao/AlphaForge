# Part of the of this repository refers to the following code:
# Shuo Yu et.al (2023).https://github.com/RL-MLDM/alphagen/tree/master
# Petersen et.al (2023)[https://github.com/dso-org/deep-symbolic-optimization]
# Microsoft (2024) [https://github.com/microsoft/qlib/tree/main/qlib]

import torch 
import os
from gan.dataset import Collector
from gan.network.masker import NetM
from gan.network.predictor import train_regression_model_with_weight
from alphagen.rl.env.wrapper import SIZE_ACTION
from gan.utils import Builders
from alphagen_generic.features import *
from alphagen.data.expression import *
from alphagen.utils.correlation import batch_ret,batch_pearsonr
import numpy as np
from alphagen.utils.random import reseed_everything
from gan.utils import filter_valid_blds,save_blds
from gan.network.generater import train_network_generator
import gc
from gan.utils.data import get_data_by_year

def pre_process_y(y):
    min_y = 0
    max_y = y.flatten().max()
    y = (y - min_y) / (max_y - min_y) * 100
    return y
def numpy2onehot(integer_matrix,max_num_categories=None,min_num_categories=None):
    if max_num_categories is None:
        max_num_categories = np.max(integer_matrix) + 1
    if min_num_categories is None:
        min_num_categories = np.min(integer_matrix)
    integer_matrix = integer_matrix - min_num_categories
    num_categories = max_num_categories - min_num_categories
    return np.eye(num_categories)[integer_matrix]

from typing import List

def blds_list_to_tensor(blds_list,weights_list:List[int]):
    assert len(blds_list) == len(weights_list)

    x_numpy_list = []
    y_numpy_list = []
    weights_numpy_list = []
    for blds,weight_int in zip(blds_list,weights_list):
        x_numpy = numpy2onehot(np.array(blds.builders_tokens),SIZE_ACTION,0).astype('float32')
        y_numpy = np.array(blds.scores).astype('float32')[:,None]
        weights_numpy = np.ones(x_numpy.shape[0]).astype('float32')[:,None] * weight_int
        x_numpy_list.append(x_numpy)
        y_numpy_list.append(y_numpy)
        weights_numpy_list.append(weights_numpy)
    x_numpy = np.concatenate(x_numpy_list,axis=0)
    y_numpy = np.concatenate(y_numpy_list,axis=0)
    weights_numpy = np.concatenate(weights_numpy_list,axis=0)
    x = torch.from_numpy(x_numpy)
    y = torch.from_numpy(y_numpy)
    weights = torch.from_numpy(weights_numpy)
    return x,y,weights

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def train_net_p_with_weight(cfg,net,x,y,weights,lr=0.001):
    
    x_train, x_valid, y_train, y_valid,weights_train,weights_valid = train_test_split(x, y,weights, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(x_train, y_train,weights_train),
                                batch_size=cfg.batch_size_p, shuffle=True,
                            )
    valid_loader = DataLoader(TensorDataset(x_valid, y_valid,weights_valid), 
                              batch_size=cfg.batch_size_p, shuffle=False)
    
    # loss with weight
    def weighted_mse_loss(input, target, weights):
        out = (input - target)**2
        out = out * weights.expand_as(out)
        loss = out.mean()
        return loss

    # Create your loss function, and optimizer
    # loss_fn = torch.nn.MSELoss()
    loss_fn = weighted_mse_loss
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)


    train_regression_model_with_weight(train_loader, valid_loader, net, 
                           loss_fn, optimizer, device=cfg.device,
                           num_epochs=cfg.num_epochs_p, use_tensorboard=False, 
                           tensorboard_path='logs', early_stopping_patience=cfg.es_p)

def get_metric(zoo_blds,device,corr_thresh=0.5,metric_target='ic'):

    n_blds = len(zoo_blds)
    if n_blds >0:
        n_days = len(zoo_blds.ret_list[0])
        existed = zoo_blds.ret_list # [blds1,blds2,...] ,blds1: (n_days,)
        existed = np.vstack(existed) # (n_blds,n_days)
        existed = torch.from_numpy(existed).to(device)
        assert existed.shape == (n_blds,n_days)
        print(f"existed n_blds == {n_blds}")
    else:
        print("n_blds == 0")
    
    def get_score(fct,tgt):
        metric_target = 'ic'
        ret = batch_ret(fct,tgt)
        ic = batch_pearsonr(fct,tgt)

        ic_mean = ic.mean().abs().item()
        icir = (ic_mean/ic.std()).item()
        ret_mean = ret.mean().abs().item()
        ret_ir = (ret_mean/ret.std()).item()
        sharpe = ((ret_mean- 0.03/252)/ret.std() * np.sqrt(252)).item()

        def invalid_to_zero(x):
            if not np.isfinite(x):
                return 0.
            else:
                return max(x,0.)
        multi_score = {'ic':ic_mean,'icir':icir,'ret':ret_mean,'sharpe':sharpe,'retir':ret_ir}
        multi_score = {k:invalid_to_zero(v) for k,v in multi_score.items()}
        score = multi_score[metric_target]
        
        #  too many nan
        if torch.isfinite(fct[0]).sum()/torch.isfinite(tgt[0]).sum() <0.8:
            score = 0.
        
        # unique ratio too small
        elif len(torch.unique(fct[0])) / len(torch.unique(tgt[0])) <0.01:
            score = 0.
        
        if n_blds > 0 and score > 0.:
            assert len(ret.shape) == 1 , f"{ret.shape},{n_days}"
            assert len(ret) == n_days , f"{ret.shape},{n_days}"

            all_matrix = torch.concatenate([existed,ret[None]],dim=0) # (n_blds+1,n_days)
            assert all_matrix.shape == (n_blds+1,n_days) , f"{all_matrix.shape}"

            corr_score = torch.corrcoef(all_matrix)[-1,:-1].abs().max().item()

            if corr_score > corr_thresh:
                score = 0.

        return {'score':score,'ret':ret.detach().cpu().numpy(),'multi_score':multi_score}
    return get_score


def main(
        instruments: str = "csi500",
        train_end_year:int = 2020,
        freq:str = 'day',
        seeds:str = '[0]',
        cuda:int = 0,
        save_name:str = 'test',
        zoo_size:int = 100,
        corr_thresh:float = 0.7,
        ic_thresh:float = 0.03,
        icir_thresh:float = 0.1,
):
    if isinstance(seeds,str):
        seeds = eval(seeds)
    assert isinstance(seeds,list)   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(cuda)
    train_end = train_end_year
    returned = get_data_by_year(
        train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
        instruments=instruments, target=target,freq=freq,
    )
    data_all,data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned

    for seed in seeds:
        reseed_everything(seed)
        class cfg:
            name = f'{save_name}_{instruments}_{train_end}_{seed}'
            # 
            max_len = 20

            batch_size = 256
            potential_size = 100
            n_layers = 2
            d_model = 128
            dropout = 0.2
            num_factors = zoo_size

            # generator configuaration
            num_epochs_g = 200
            g_es_score = 'max' # max mean std combined
            g_es = 10
            g_hidden = 128
            g_lr = 1e-3


            # predictor configuration
            p_hidden = 128
            p_lr = 1e-3
            es_p = 10
            batch_size_p = 64
            num_epochs_p = 100
            data_keep_p = 20000

            f_corr_thresh = corr_thresh # threshold to penalize the correlation
            f_add_thresh = corr_thresh # threshold to add new exprs to the zoo
            f_score_thresh = ic_thresh # threshold to filter exprs in the zoo
            f_multi_score_thresh = {'icir':icir_thresh}


            # loss configuaration
            l_pred = 1.
            l_simi = 10.
            l_simi_thresh = 0.4


            l_potential = 10.
            l_potential_thresh = 0.4
            l_potential_epsilon = 1e-7

            l_entropy = 0

            device = 'cuda:0'

        print(f"seed:{seed},name:{cfg.name}")

        from gan.network.predictor import NetP
        from gan.network.generater import NetG_DCGAN
        NetG_CLS = NetG_DCGAN
        NetP_CLS = NetP
        # 1106 normal alla

        def random_call(z):
            return z.normal_()


        netG = NetG_CLS(
            n_chars=SIZE_ACTION,
            latent_size=cfg.potential_size,
            seq_len=cfg.max_len,
            hidden = cfg.g_hidden,
            ).to(cfg.device)

        netM = NetM(max_len=cfg.max_len,size_action=SIZE_ACTION).to(cfg.device)

        netP = NetP_CLS(
            n_chars=SIZE_ACTION, seq_len=cfg.max_len,hidden = cfg.p_hidden,
            ).to(cfg.device)

        z = torch.zeros([cfg.batch_size,cfg.potential_size])
        z = z.to(cfg.device)
        random_call(z)

        # initialize the zoo
        zoo_blds = Builders(0,max_len=cfg.max_len,n_actions=SIZE_ACTION)
        metric = get_metric(zoo_blds,device=cfg.device,corr_thresh=cfg.f_corr_thresh)
        empty_metric = get_metric(
            Builders(0,max_len=cfg.max_len,n_actions=SIZE_ACTION),
            device=cfg.device,corr_thresh=cfg.f_corr_thresh
        )

        coll = Collector(seq_len=cfg.max_len,n_actions=SIZE_ACTION)
        coll.reset(data,target,metric)
        coll.collect_target_num(netG,netM,z,data,target,metric,
                                target_num=10000,reset_net=True,drop_invalid=False,
                                randomly = False,
                                random_method = random_call,max_iter = 200)


        # train and mine untill the zoo is full
        t = 0
        while len(zoo_blds)<cfg.num_factors:
            if not zoo_blds.examined:
                print(' zoo_blds not examined')
                zoo_blds.evaluate(data,target,empty_metric,verbose=True)

            ### update the metric for the current zoo
            metric = get_metric(zoo_blds,device = cfg.device,corr_thresh=cfg.f_corr_thresh)

            ### Prepare data to train predictor
            coll.blds.evaluate(data,target,metric,verbose=True)
            if coll.blds_bak.batch_size>cfg.data_keep_p:
                # sample the training data of predictor to keep the size
                print(f'sample datas to keep {coll.blds_bak.batch_size}to{cfg.data_keep_p}')
                indices = np.random.choice(np.arange(coll.blds_bak.batch_size),cfg.data_keep_p,replace=False)
                coll.blds_bak = coll.blds_bak.filter_by_index(indices)
            coll.blds_bak.evaluate(data,target,metric,verbose=True)



            if coll.blds_bak.batch_size > 0:
                # give the current builders more weights in training p
                blds_list = [coll.blds_bak,coll.blds]
                weight_list = [1.,2.]
            else:
                blds_list = [coll.blds]
                weight_list = [1.]

            x, y, weights = blds_list_to_tensor(blds_list,weight_list)
            y = pre_process_y(y)


            ### train predictor
            netP.initialize_parameters() 
            train_net_p_with_weight(cfg,netP,x,y,weights,lr=cfg.p_lr)

            ### train generator
            netG.initialize_parameters()
            blds_in_train = train_network_generator(netG, netM, netP, cfg, data, target,t,random_method = random_call,
                                    metric=metric,lr=cfg.g_lr,n_actions=SIZE_ACTION)
            
            ### Generate new alpha factors from current Generator
            coll.reset(data,target,metric)
            coll.collect_target_num(netG,netM,z,data,target,metric,
                                    target_num=1000,reset_net=False,drop_invalid=False,
                                    randomly = False,
                                    random_method = random_call,max_iter = 100)

            lengh_s = {"train":len(blds_in_train)}
            lengh_s['new']=len(coll.blds)
            coll.blds = coll.blds + blds_in_train
            coll.blds.drop_duplicated()
            lengh_s['all_new']=len(coll.blds)

            print(f"{lengh_s['train']} (train) + {lengh_s['new']} (new)  =  {lengh_s['all_new']} (all_new)")
            
            ### get the valid alpha factors during the training process and the generating process
            new_zoo = filter_valid_blds(
                coll.blds,
                corr_thresh=cfg.f_add_thresh,
                score_thresh=cfg.f_score_thresh,
                multi_score_thresh = cfg.f_multi_score_thresh,
                device = cfg.device,
                verbose= True,
                )
            lengh_s['zoo_prev'] = len(zoo_blds)
            zoo_blds = zoo_blds + new_zoo

            print(f" zoo_prev:{lengh_s['zoo_prev']},all_new:{len(new_zoo)},current:{len(zoo_blds)}")
            zoo_blds.evaluate(data,target,empty_metric,verbose=True)
            if t % 5 == 2:
                print('#'*20,"zoo_rebalance")
                zoo_blds = filter_valid_blds(
                    zoo_blds,
                    corr_thresh=cfg.f_add_thresh,
                    score_thresh=cfg.f_score_thresh,
                    multi_score_thresh = cfg.f_multi_score_thresh,
                    device = cfg.device,
                    verbose = False,
                    )
            # save the zoo
            save_blds(zoo_blds,f"out/{cfg.name}",'zoo_final')

            # Randomly generate some alpha factors in order to promote exploration and to avoid local minimum
            coll.collect_target_num(netG,netM,z,data,target,metric,
                                    target_num=1000,reset_net=False,drop_invalid=False,
                                    randomly = True,
                                    random_method = random_call,max_iter = 100)

            del x,y,weights
            gc.collect()
            torch.cuda.empty_cache()
            t+=1

        empty_blds = Builders(0,max_len=cfg.max_len,n_actions=SIZE_ACTION)
        metric = get_metric(empty_blds,device = cfg.device,corr_thresh=cfg.f_corr_thresh)
        zoo_blds.evaluate(data,target,metric,verbose=True)
        save_blds(zoo_blds,f"out/{cfg.name}",'zoo_final')

if __name__ == '__main__':
    import fire
    fire.Fire(main)