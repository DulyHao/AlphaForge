from typing import List, Tuple, Union, Dict, Any
import numpy as np,pandas as pd
import os,gc
from alphagen.rl.env.wrapper import *

from alphagen.config import MAX_EXPR_LENGTH
from alphagen.data.tokens import *
from alphagen.data.expression import *
from alphagen.data.tree import ExpressionBuilder
from alphagen.models.alpha_pool import AlphaPoolBase, AlphaPool
from alphagen.utils import reseed_everything

from alphagen.utils.pytorch_utils import normalize_by_day

from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr#,batch_pearsonr_full_y
import copy
from typing import Callable
import inspect

class Builders:
    def __init__(self,batch_size,max_len,n_actions):
        self.batch_size = batch_size
        self.builders = [ExpressionBuilder() for i in range(batch_size)]
        self.builders_done = [False ]*batch_size
        self.builders_tokens = [[] for i in range(batch_size)] 
        self.exprs = [None] * batch_size
        self.exprs_str = [None] * batch_size
        self.max_len = max_len
        self.n_actions = n_actions 
        self.examined = False
        self.scores = []
        self.multi_scores = []
        self.ret_list = []
    
    def filter_by_index(self,indices):
        new = copy.copy(self)
        new.batch_size = len(indices)
        new.builders = [new.builders[i] for i in indices]
        new.builders_done = [new.builders_done[i] for i in indices]
        new.builders_tokens = [new.builders_tokens[i] for i in indices]
        new.exprs = [new.exprs[i] for i in indices]
        new.exprs_str = [new.exprs_str[i] for i in indices]

        if self.examined:
            new.scores = [new.scores[i] for i in indices]
            new.multi_scores = [new.multi_scores[i] for i in indices]
            new.ret_list = [new.ret_list[i] for i in indices]
        return new
    def __add__(self,other):
        if len(self)==0:
            return other
        if len(other)==0:
            return self
        new = Builders(self.batch_size + other.batch_size,max_len=self.max_len,n_actions=self.n_actions)
        new.builders = self.builders + other.builders
        new.builders_done = self.builders_done + other.builders_done
        new.builders_tokens = self.builders_tokens + other.builders_tokens
        new.exprs = self.exprs + other.exprs
        new.exprs_str = self.exprs_str + other.exprs_str
        if self.examined and other.examined:
            new.scores = self.scores + other.scores
            new.multi_scores = self.multi_scores + other.multi_scores
            new.ret_list = self.ret_list + other.ret_list
            new.examined = True

        return new
    def __sub__(self,other):
        
        exist_ones = [str(expr) for expr in other.builders_tokens]
        new = copy.deepcopy(self)
        valid_index = [i for i in range(new.batch_size) if str(new.builders_tokens[i]) not in exist_ones]
        return self.filter_by_index(valid_index)
    
    def __getitem__(self,index):
        if isinstance(index,slice):
            return self.filter_by_index(range(index.start,index.stop,index.step))
        else:
            return self.filter_by_index([index])
        
    def filter_by_score(self,func):
        if not self.examined:
            self.evaluate()
        valid_index = [i for i in range(self.batch_size) if func(self.scores[i])]
        return self.filter_by_index(valid_index)
    
    def sort_by_score(self,ascending=False):
        if not self.examined:
            self.evaluate()
        valid_index = np.argsort(self.scores)
        if not ascending:
            valid_index = valid_index[::-1]
        return self.filter_by_index(valid_index)
        
    def __len__(self):
        assert len(self.builders_tokens) == self.batch_size
        return self.batch_size
    
    def drop_invalid(self):
        valid_idx = [ i for i in range(self.batch_size) if self.builders[i].is_valid() ]
        self.batch_size = len(valid_idx)
        self.builders = [self.builders[i] for i in valid_idx]
        self.builders_done = [self.builders_done[i] for i in valid_idx]
        self.builders_tokens = [self.builders_tokens[i] for i in valid_idx]
        self.exprs = [self.exprs[i] for i in valid_idx]
        self.exprs_str = [self.exprs_str[i] for i in valid_idx]
        if self.examined:
            self.scores = [self.scores[i] for i in valid_idx]
            self.multi_scores = [self.multi_scores[i] for i in valid_idx]
            self.ret_list = [self.ret_list[i] for i in valid_idx]

            
    def drop_duplicated(self):
        def first_occurrences(lst):
            arr = np.array(lst)
            _, indices = np.unique(arr, return_index=True)
            return sorted(indices)
        valid_idx = first_occurrences([str(expr) for expr in self.builders_tokens])
        self.batch_size = len(valid_idx)
        self.builders = [self.builders[i] for i in valid_idx]
        self.builders_done = [self.builders_done[i] for i in valid_idx]
        self.builders_tokens = [self.builders_tokens[i] for i in valid_idx]
        self.exprs = [self.exprs[i] for i in valid_idx]
        self.exprs_str = [self.exprs_str[i] for i in valid_idx]
        if self.examined:
            self.scores = [self.scores[i] for i in valid_idx]
            self.multi_scores = [self.multi_scores[i] for i in valid_idx]
            self.ret_list = [self.ret_list[i] for i in valid_idx]
        
        
    def fill_to_max(self,token):
        if len(token)<self.max_len:
            token = token + [self.n_actions-1]*(self.max_len-len(token))
        assert len(token) == self.max_len
        return token
    def add_token(self,tokens):
        assert tokens.shape[0]==self.batch_size
        for i in range(self.batch_size):
            builder = self.builders[i]
            action = tokens[i]
            if not self.builders_done[i]:
                token = action2token(action)

                if (isinstance(token, SequenceIndicatorToken) and
                    token.indicator == SequenceIndicatorType.SEP):
                    self.builders_done[i] = True
                    self.builders_tokens[i] = self.fill_to_max(self.builders_tokens[i])
                else:
                    self.builders_tokens[i].append(action)
                    self.builders[i].add_token(token)
        return 
    
    def calc_metric(self,factor:torch.Tensor,target_factor:torch.Tensor,metric:Callable):
        result:Dict = metric(factor,target_factor)
        return result
    
    
    def evaluate(self,data,target,metric=None,verbose=False):
        
        # del self.scores, self.ret_list,self.multi_scores
        # gc.collect()   
        # torch.cuda.empty_cache()

        exprs = self.build_exprs()
        from tqdm import tqdm
        target_factor = target.evaluate(data)
        target_factor = normalize_by_day(target_factor)
        metric_params = inspect.signature(metric).parameters.keys()
        scores = []
        ret_list = []
        multi_scores = []
        to_iter = exprs if not verbose else tqdm(exprs,ncols=50)
        # if verbose:
        #     print(f"begin evaluate")
        for cur in to_iter:
            ret_tensor = None
            if cur is not None:
                try:
                    factor = cur.evaluate(data)

                    params = {}
                    if 'fct' in metric_params:
                        factor = normalize_by_day(factor)
                        params['fct']=factor
                    if 'tgt' in metric_params:
                        params['tgt']=target_factor
                    if 'expr' in metric_params:
                        params['expr']=cur

                    result = metric(**params)
                    ic = result['score']
                    multi_score = result['multi_score']
                    ret_tensor = result['ret']
                except OutOfDataRangeError:
                    ic = 0.
                    multi_score = {}
                    ret_tensor = np.array([0.] * data.n_days)
                if np.isnan(ic):
                    ic = 0.
                    multi_score = {}
                    ret_tensor = np.array([0.] * data.n_days)
            else:
                ic = 0.
                multi_score = {}
                ret_tensor = np.array([0.] * data.n_days)
            scores.append(ic)
            ret_list.append(ret_tensor)
            multi_scores.append(multi_score)
        self.scores = scores
        self.ret_list = ret_list
        self.multi_scores = multi_scores
        self.examined = True
        
    def build_exprs(self):
        res = [
            self.builders[i].get_tree()
            if self.builders[i].is_valid() else None
            for i in range(self.batch_size)
              ]
        self.exprs = res
        self.exprs_str = [str(i) for i in res]
        return res
    
    def _valid_action_types(self,builder) -> dict:
        self._builder = builder
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_pair_rolling = self._builder.validate_op(PairRollingOperator)

        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_feature()
        valid_stop = self._builder.is_valid()

        ret = {
            'select': [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            'op': {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                PairRollingOperator: valid_op_pair_rolling
            }
        }
        return ret
    
    def action_masks(self,i) -> np.ndarray:
        builder = self.builders[i]
        
        res = np.zeros(SIZE_ACTION, dtype=bool)
        if self.builders_done[i]:
            res[-1] = True
            return res
        valid = self._valid_action_types(builder)
        for i in range(OFFSET_OP, OFFSET_OP + SIZE_OP):
            if valid['op'][OPERATORS[i - OFFSET_OP].category_type()]:
                res[i - 1] = True
        if valid['select'][1]:  # FEATURE
            for i in range(OFFSET_FEATURE, OFFSET_FEATURE + SIZE_FEATURE):
                res[i - 1] = True
        if valid['select'][2]:  # CONSTANT
            for i in range(OFFSET_CONSTANT, OFFSET_CONSTANT + SIZE_CONSTANT):
                res[i - 1] = True
        if valid['select'][3]:  # DELTA_TIME
            for i in range(OFFSET_DELTA_TIME, OFFSET_DELTA_TIME + SIZE_DELTA_TIME):
                res[i - 1] = True
        if valid['select'][4]:  # SEP
            res[OFFSET_SEP - 1] = True
        return res
    def get_valid_op(self):
        res = np.zeros([self.batch_size,SIZE_ACTION],dtype = bool)
        for i in range(self.batch_size):
            res[i] = self.action_masks(i)
        return res

def get_all_blds(path):
    blds_list = []
    for root,dirs,files in os.walk(path):
        for file in files:
            if file.endswith('.pkl'):
                file_path = os.path.join(root,file)
                blds_list.append(pd.read_pickle(file_path))
    blds = blds_list[0]
    for i in range(1,len(blds_list)):
        blds = blds + blds_list[i]
    blds.drop_invalid()
    blds.drop_duplicated()
    return blds

import pickle
def save_pickle(data,path):
    with open(path,'wb') as f:
        pickle.dump(data,f)
def load_pickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)
    
def save_blds_csv(blds:Builders,path):
    df = pd.DataFrame()
    df['exprs'] = blds.exprs_str
    df['scores'] = blds.scores
    df = df.sort_values('scores',ascending=False)
    df.to_csv(path)

def save_blds(blds:Builders,path,epoch):
    if len(blds)==0:
        print(f"no new blds, len(blds)==0")
        return 
    os.makedirs(path,exist_ok=True)
    df = pd.DataFrame()
    df['exprs'] = blds.exprs_str
    df['scores'] = blds.scores
    df = df.sort_values('scores',ascending=False)
    # df.to_pickle(f"{path}/bld_{epoch}.pkl")    
    save_pickle(blds,f"{path}/z_bld_{epoch}.pkl")
    df.to_csv(f"{path}/csv_{epoch}.csv")

from tqdm import tqdm

def get_blds_df(blds,top=None):
    df=pd.DataFrame({'score':blds.scores,'exprs':blds.exprs,'exprs_str':blds.exprs_str})
    df =df[df['score']>0]
    df = df.sort_values('score',ascending=False)
    if top is not None and top != 0:
        df = df.head(top)
    df = df.reset_index(drop=True)
    df['name']=[i for i in df.index]
    df.index =df['name']
    return df

def get_evaled_df(df,data,target,raw=False):
    def get_evaled(expr,data):
        evaled = expr.evaluate(data)
        dd = pd.DataFrame(
            evaled.detach().cpu().numpy(),

            index = pd.Series(np.arange(evaled.shape[0]),name='datetime'),

            columns = pd.Series(np.arange(evaled.shape[1]),name='instrument'),
        )
        return dd.T.unstack()
    def get_evaled_raw(expr,data):
        evaled = expr.evaluate(data)
        evaled = data.make_dataframe(evaled)
        if isinstance(evaled,pd.DataFrame):
            evaled = evaled.iloc[:,0]
        return evaled
    result = {}
    
    result['target'] = get_evaled_raw(target,data) if raw else get_evaled(target,data)
    for name in tqdm(df.index):
        expr = df.loc[name,'exprs']
        if raw:
            tmp = get_evaled_raw(expr,data)
        else:
            tmp = get_evaled(expr,data)
        result[name]=tmp
    if raw:
        return pd.DataFrame(result)
        return pd.concat(result,axis=1)
    dd = pd.DataFrame(result)
    dd.columns = dd.columns.to_series().rename('name')
    return dd

def filter_valid_blds(new_blds:Builders,corr_thresh,score_thresh,multi_score_thresh,device,verbose):
    blds = new_blds.filter_by_score(lambda x:x>0)

    blds.drop_invalid()
    blds.drop_duplicated()
    blds = blds.sort_by_score(ascending=False)
    if blds.batch_size == 0:
        print(f"no new blds, blds.batch_size == 0")
        return blds
    ret_array = np.vstack(blds.ret_list)
    assert len(ret_array) == len(blds),f"len(ret_array) != len(blds),{len(ret_array)},{len(blds)}"
    ret_tensor = torch.from_numpy(ret_array).to(device)
    if len(ret_tensor)==1:
        ret_corr = [[0.]]
        print('only 1 valid factor')
    else:
        ret_corr = torch.corrcoef(ret_tensor).abs()
        print(len(ret_corr))
        print(f"ret_corr.shape:{ret_corr.shape}")
    
        assert ret_corr.shape[0] == blds.batch_size,f"ret_corr.shape[0] != blds.batch_size,{ret_corr.shape},{blds.batch_size}"

    def multi_score_valid(score,target):
        for k,v in target.items():
            if k not in score:
                return False
            if score[k] < v:
                return False
        return True

    # print(f" begin filter blds, blds.batch_size:{blds.batch_size}")
    indices = []
    for i in tqdm(range(blds.batch_size)):
        if blds.scores[i] > score_thresh and multi_score_valid(blds.multi_scores[i],multi_score_thresh):
            cur_ok = True
            for j in indices:
                if ret_corr[i,j]> corr_thresh:
                    cur_ok = False
                    break
            if cur_ok:
                indices.append(i)
    if verbose:
        for i in indices:
            print(f"i:{i},score:{blds.scores[i]:.4f},expr:{blds.exprs_str[i]}")
        print(f" end filter blds, blds.batch_size:{blds.batch_size},indices:{len(indices)}")
    else:
        print(f"NOVERBOSE: blds.batch_size:{blds.batch_size},indices:{len(indices)}")

    return blds.filter_by_index(indices)
    
def read_model_blds(path):
    blds = []
    for file in os.listdir(path):
        if file.endswith('.pkl'):
            blds.append(load_pickle(os.path.join(path,file)))
    return blds

def exprs2tensor(exprs,data,verbose = False,normalize = True):
    result = []
    to_iter =  tqdm(exprs) if verbose else exprs
    for expr in to_iter:
        cur_tensor = expr.evaluate(data) # torch Tensor:(n_days,n_stocks)
        if normalize:
            cur_tensor = normalize_by_day(cur_tensor)
        # result.append(cur_tensor.detach().cpu())
        result.append(cur_tensor)

    result = torch.stack(result,dim=-1) # (n_days,n_stocks,n_exprs)
    return result

from alphagen.utils.correlation import batch_ret,batch_pearsonr,batch_spearmanr

from alphagen.utils.pytorch_utils import masked_mean_std

def get_df_metrics(df, data, target, norm_by_day=True,prefix=''):
    exprs = df['exprs'].to_list()
    all_vals = []
    tgt = target.evaluate(data)
    for expr in tqdm(exprs):
        fct = expr.evaluate(data)
        if norm_by_day:
            fct = normalize_by_day(fct)
        ic_vals =  batch_pearsonr(fct,tgt)
        ric_vals = batch_spearmanr(fct,tgt)
        ret_vals = batch_ret(fct,tgt)
        
        ic_vals = torch.nan_to_num(ic_vals)
        ric_vals = torch.nan_to_num(ric_vals)
        ret_vals = torch.nan_to_num(ret_vals)
        result = {
            'ic':ic_vals.mean().item(),
            'ic_std':ic_vals.std().item(),
            'icir':(ic_vals.mean()/ic_vals.std()).item(),
            'ic_ttest':(ic_vals.mean()/ic_vals.std()*np.sqrt(len(ic_vals))).item(),
            'ric':ric_vals.mean().item(),
            'ric_std':ric_vals.std().item(),
            'ricir':(ric_vals.mean()/ric_vals.std()).item(),
            'ret':ret_vals.mean().item(),
            'ret_std':ret_vals.std().item(),
            'retir':(ret_vals.mean()/ret_vals.std()).item(),
        }
        result = {prefix+k:v for k,v in result.items()}
        # del fct,ic_vals,ric_vals,ret_vals
        # gc.collect()
        # torch.cuda.empty_cache()
        all_vals.append(result)
    tmp = pd.DataFrame(all_vals,index=df.index)
    df = pd.concat([df,tmp],axis=1)
    return df

def get_blds_list_df(blds_list):
    
    df_s = [get_blds_df(t) for t in tqdm(blds_list)]
    df = pd.concat(df_s,axis=0).sort_values('score',ascending=False)
    # filter the exprs with the same score
    # 筛选掉重复的
    df = df.groupby('score').first().sort_values('score',ascending=False).reset_index()
    return df


# 配置代码
import numpy as np
def numpy2onehot(integer_matrix,max_num_categories=None,min_num_categories=None):
    if max_num_categories is None:
        max_num_categories = np.max(integer_matrix) + 1
    if min_num_categories is None:
        min_num_categories = np.min(integer_matrix)
    # print(f"max_num_categories:{max_num_categories},min_num_categories:{min_num_categories}")
    # print(f"integer_matrix:{integer_matrix}")
    integer_matrix = integer_matrix - min_num_categories
    num_categories = max_num_categories - min_num_categories
    # print(integer_matrix,num_categories,max_num_categories,min_num_categories)
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