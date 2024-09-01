import torch
import sklearn
import tensorflow as tf
import numpy as np
import os,json

from alphagen.data.expression import *
# from alphagen_qlib.calculator import QLibStockDataCalculator
from dso import DeepSymbolicRegressor
from dso.library import Token, HardCodedConstant
from dso import functions
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from gan.utils.data import get_data_by_year

funcs = {func.name: Token(complexity=1, **func._asdict()) for func in generic_funcs}
for i, feature in enumerate(['open', 'close', 'high', 'low', 'volume', 'vwap']):
    funcs[f'x{i+1}'] = Token(name=feature, arity=0, complexity=1, function=None, input_var=i)
for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]:
    funcs[f'Constant({v})'] = HardCodedConstant(name=f'Constant({v})', value=v)

def main(
        instruments:str='csi300',
        train_end:int=2018,
        seeds:list=[0],
        capacity:int=100,
        cuda:int=0,
        name:str='test',
):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda)
    if isinstance(seeds,str):
        seeds = eval(seeds)
    for seed in seeds:
        tf.set_random_seed(seed)
        reseed_everything(seed)
        returned = get_data_by_year(
            train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
            instruments=instruments, target=target,freq='day',)
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned

        cache = {}
        device = torch.device('cuda:0')

        X = np.array([['open_', 'close', 'high', 'low', 'volume', 'vwap']])
        y = np.array([[1]])
        functions.function_map = funcs

        pool = AlphaPool(capacity=capacity,
                        stock_data=data,
                        target=target,
                        ic_lower_bound=None)
        save_path = f'out_dso/{name}_{instruments}_{capacity}_{train_end}_{seed}/'
        os.makedirs(save_path,exist_ok=True)

        class Ev:
            def __init__(self, pool):
                self.cnt = 0
                self.pool = pool
                self.results = {}

            def alpha_ev_fn(self, key):
                expr = eval(key)
                try:
                    ret = self.pool.try_new_expr(expr)
                except OutOfDataRangeError:
                    ret = -1.
                else:
                    ret = -1.
                finally:
                    self.cnt += 1
                    if self.cnt % 100 == 0:
                        test_ic = pool.test_ensemble(data_test,target)[0]
                        self.results[self.cnt] = test_ic
                        print(self.cnt, test_ic)
                    return ret

        ev = Ev(pool)

        config = dict(
            task=dict(
                task_type='regression',
                function_set=list(funcs.keys()),
                metric='alphagen',
                metric_params=[lambda key: ev.alpha_ev_fn(key)],
            ),
            training={'n_samples': 5000, 'batch_size': 128, 'epsilon': 0.05},
            prior={'length': {'min_': 2, 'max_': 20, 'on': True}},
            experiment={'seed':seed},
        )

        # Create the model
        model = DeepSymbolicRegressor(config=config)
        model.fit(X, y)
        with open(f'{save_path}/pool.json', 'w') as f:
            json.dump(pool.to_dict(), f)
        print(ev.results)

if __name__ == '__main__':
    import fire
    fire.Fire(main)