
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--instrument',type=str,default='csi300')
parser.add_argument('--seed',type=str,default='[0,1,2,3,4]')
parser.add_argument('--years',type=str,default='[2016]')
parser.add_argument('--freq',type=str,default='day')
parser.add_argument('--cuda',type=str,default='0')


args = parser.parse_args()
instruments = args.instrument
args.seed = eval(args.seed)
args.years = eval(args.years)

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
print('instruments',instruments)
print('seed',args.seed)
print('years',args.years)
print('cuda',args.cuda)


import json
from collections import Counter

import numpy as np

from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr
from alphagen.utils.pytorch_utils import normalize_by_day
from alphagen.utils.random import reseed_everything
from alphagen_generic.operators import funcs as generic_funcs
from alphagen_generic.features import *
from gplearn.fitness import make_fitness
from gplearn.functions import make_function
from gplearn.genetic import SymbolicRegressor
from gan.utils.data import get_data_by_year


def _metric(x, y, w):
    key = y[0]

    if key in cache:
        return cache[key]
    token_len = key.count('(') + key.count(')')
    if token_len > 20:
        return -1.

    expr = eval(key)
    try:
        factor = expr.evaluate(data)
        factor = normalize_by_day(factor)
        ic = batch_pearsonr(factor, target_factor)
        ic = torch.nan_to_num(ic).mean().item()
    except OutOfDataRangeError:
        ic = -1.
    if np.isnan(ic):
        ic = -1.
    cache[key] = ic
    return ic




def try_single():
    top_key = Counter(cache).most_common(1)[0][0]
    try:
        v_valid = eval(top_key).evaluate(data_valid)
        v_test = eval(top_key).evaluate(data_test)
        ic_test = batch_pearsonr(v_test, target_factor_test)
        ic_test = torch.nan_to_num(ic_test,nan=0,posinf=0,neginf=0).mean().item()
        ic_valid = batch_pearsonr(v_valid, target_factor_valid)
        ic_valid = torch.nan_to_num(ic_valid,nan=0,posinf=0,neginf=0).mean().item()
        ric_test = batch_spearmanr(v_test, target_factor_test)
        ric_test = torch.nan_to_num(ric_test,nan=0,posinf=0,neginf=0).mean().item()
        ric_valid = batch_spearmanr(v_valid, target_factor_valid)
        ric_valid = torch.nan_to_num(ric_valid,nan=0,posinf=0,neginf=0).mean().item()
        return {'ic_test': ic_test, 'ic_valid': ic_valid, 'ric_test': ric_test, 'ric_valid': ric_valid}
    except OutOfDataRangeError:
        print ('Out of data range')
        print(top_key)
        exit()
        return {'ic_test': -1., 'ic_valid': -1., 'ric_test': -1., 'ric_valid': -1.}


def try_pool(capacity):
    pool = AlphaPool(capacity=capacity,
                    stock_data=data,
                    target=target,
                    ic_lower_bound=None)

    exprs = []
    for key in dict(Counter(cache).most_common(capacity)):
        exprs.append(eval(key))
    pool.force_load_exprs(exprs)
    pool._optimize(alpha=5e-3, lr=5e-4, n_iter=2000)

    ic_test, ric_test = pool.test_ensemble(data_test, target)
    ic_valid, ric_valid = pool.test_ensemble(data_valid, target)
    return {'ic_test': ic_test, 'ic_valid': ic_valid, 'ric_test': ric_test, 'ric_valid': ric_valid}




def ev():
    global generation
    generation += 1
    res = (
        [{'pool': 0, 'res': try_single()}] +
        [{'pool': cap, 'res': try_pool(cap)} for cap in (10, 20, 50, 100)]
    )
    print(res)
    global save_dir
    dir_ = save_dir
    #'/path/to/save/results'
    os.makedirs(dir_, exist_ok=True)
    if generation % 2 == 0:
        with open(f'{dir_}/{generation}.json', 'w') as f:
            json.dump({'cache': cache, 'res': res}, f)





for seed in args.seed:
    for train_end in args.years:
        #'/path/to/save/results'
        save_dir = f'out_gp/{instruments}_{train_end}_{args.freq}_{seed}' 

        Metric = make_fitness(function=_metric, greater_is_better=True)
        funcs = [make_function(**func._asdict()) for func in generic_funcs]

        generation = 0
        cache = {}

        reseed_everything(seed)


        returned = get_data_by_year(
            train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
            instruments=instruments, target=target,freq=args.freq,
        )
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned

        pool = AlphaPool(capacity=10,
                        stock_data=data,
                        target=target,
                        ic_lower_bound=None)

        target_factor = target.evaluate(data)
        target_factor_valid = target.evaluate(data_valid)
        target_factor_test = target.evaluate(data_test)

        
        features = ['open_', 'close', 'high', 'low', 'volume', 'vwap']
        constants = [f'Constant({v})' for v in [-30., -10., -5., -2., -1., -0.5, -0.01, 0.01, 0.5, 1., 2., 5., 10., 30.]]
        terminals = features + constants

        X_train = np.array([terminals])
        y_train = np.array([[1]])

        est_gp = SymbolicRegressor(population_size=1000,
                                generations=40,
                                init_depth=(2, 6),
                                tournament_size=600,
                                stopping_criteria=1.,
                                p_crossover=0.3,
                                p_subtree_mutation=0.1,
                                p_hoist_mutation=0.01,
                                p_point_mutation=0.1,
                                p_point_replace=0.6,
                                max_samples=0.9,
                                verbose=1,
                                parsimony_coefficient=0.,
                                random_state=seed,
                                function_set=funcs,
                                metric=Metric,
                                const_range=None,
                                n_jobs=1)
        est_gp.fit(X_train, y_train, callback=ev)
        print(est_gp._program.execute(X_train))
