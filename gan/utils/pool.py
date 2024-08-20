import torch
from torch import Tensor
import numpy as np
from typing import List, Optional, Tuple, Set
from alphagen.data.expression import Expression
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen_qlib.stock_data import StockData

from alphagen.utils.pytorch_utils import normalize_by_day
from tqdm import tqdm
class MyPool:
    def __init__(self,data:StockData,target:Expression ):
        super().__init__()
        self.data = data
        self.target = target
        self.tgt = target.evaluate(data) #(n_days,n_stocks)
        self.nan_mask = torch.isfinite(self.tgt).flatten() #(n_days,n_stocks)
        self.tgt = normalize_by_day(self.tgt)
        self.flatten_tgt = self.tgt.flatten()[self.nan_mask] #(n_days*n_stocks,)
        n_days,n_stocks = self.tgt.shape

        self.best_score = 0.

        self.exprs = []
        
        self.expr_tensor = torch.ones((n_days,n_stocks,1)).to(self.device)
        self.size = 0
    @property
    def device(self):
        return self.data.data.device
    
    def add_expr(self,expr:Expression):
        print(f"[Pool +] {expr}")
        fct = expr.evaluate(self.data)
        fct = normalize_by_day(fct)

        self.expr_tensor = torch.cat([self.expr_tensor,fct[...,None]],dim=2)
        self.exprs.append(expr)
        self.size += 1
        assert self.size == len(self.exprs) == self.expr_tensor.shape[2] - 1


        cur_tensor = self.expr_tensor #(n_days,n_stocks,n_exprs+1)
        X = cur_tensor.reshape(-1,self.size+1)[self.nan_mask]
        y = self.flatten_tgt #(ndays*nstocks,)

        # Perform multiple linear regression
        coefficients, residuals, _, _ = torch.linalg.lstsq(X, y)

        coefficients#(n_exprs+1,1)
        # Get the predicted values
        predicted = cur_tensor @ coefficients
        predicted = predicted.reshape(*fct.shape)


        cur_score = batch_pearsonr(predicted,self.tgt).mean()
        increment = cur_score - self.best_score
        if increment > 0:
            prev_score = self.best_score
            self.best_score = cur_score
            print(f"[Best Score] {prev_score:.6f} + {increment:.6f} = {self.best_score:.6f}")
        # print(f"coefficients:{coefficients.shape} ,{coefficients.flatten()}")
        self.weights = coefficients
        
    def test_new_expr(self, expr: Expression) -> float:
        fct = expr.evaluate(self.data)
        fct = normalize_by_day(fct)#(n_days,n_stocks)


        y = self.flatten_tgt #(ndays*nstocks,)
        cur_tensor = torch.cat([self.expr_tensor,fct[...,None]],dim=2) #(n_days,n_stocks,n_exprs+1)
        X = cur_tensor.reshape(-1,self.size+2)[self.nan_mask]

        # Perform multiple linear regression
        coefficients, residuals, _, _ = torch.linalg.lstsq(X, y)

        coefficients#(n_exprs+1,1)
        # Get the predicted values
        predicted = cur_tensor @ coefficients
        predicted = predicted.reshape(*fct.shape)

        cur_score = batch_pearsonr(predicted,self.tgt).mean()
        
        increment = cur_score - self.best_score
        increment = max(increment.item(),0)
        if np.isnan(increment):
            increment = 0
        return {'score':increment,'ret':np.array([0.,0.,0.])}

    def test_expr_list(self,exprs:List[Expression],verbose:bool = False):
        to_iter = exprs if not verbose else tqdm(exprs)
        return [self.test_new_expr(expr)['score'] for expr in to_iter]
    
    def __call__(self,data:StockData):
        with torch.no_grad():
            combined_factor: Tensor  = 0
            for i in range(self.size):
                factor = normalize_by_day(self.exprs[i].evaluate(data))   # type: ignore
                weighted_factor = factor * self.weights[i+1]
                combined_factor += weighted_factor

            return combined_factor

    def evaluate_data(self,data:StockData,target:Expression=None):
        if target is None:
            target = self.target
        tgt = target.evaluate(data)
        tgt = normalize_by_day(tgt)

        fct = self(data)
        return batch_pearsonr(fct,tgt).mean().item()*100,batch_spearmanr(fct,tgt).mean().item()*100
