import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import json
from typing import Optional
from datetime import datetime

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen_generic.features import *
from alphagen.data.expression import *
from alphagen.models.alpha_pool import AlphaPool, SingleAlphaPool, AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore

import pickle
def save_pickle(data,path):
    with open(path,'wb') as f:
        pickle.dump(data,f)
def load_pickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)

class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 train_data: StockData,
                 train_target: Expression,
                 valid_data: StockData,
                 valid_target: Expression,
                 test_data: StockData,
                 test_target: Expression,
                 name_prefix: str = 'rl_model',
                 timestamp: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

        self.train_data = train_data
        self.train_target = train_target
        self.valid_data = valid_data
        self.valid_target = valid_target
        self.test_data = test_data
        self.test_target = test_target

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.logger is not None
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)

        ic_train, rank_ic_train = self.pool.test_ensemble(self.train_data, self.train_target)
        self.logger.record('_train/ic', ic_train)
        self.logger.record('_train/rank_ic', rank_ic_train)

        ic_test, rank_ic_test = self.pool.test_ensemble(self.test_data, self.test_target)
        self.logger.record('_test/ic', ic_test)
        self.logger.record('_test/rank_ic', rank_ic_test)

        ic_valid, rank_ic_valid = self.pool.test_ensemble(self.valid_data, self.valid_target)
        self.logger.record('_valid/ic', ic_valid)
        self.logger.record('_valid/rank_ic', rank_ic_valid)
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        save_pickle(self.pool,path+'_pool.pkl')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_dict(), f)

    def show_pool_state(self):
        state = self.pool.state
        n = len(state['exprs'])
        print('---------------------------------------------')
        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    @property
    def pool(self) -> AlphaPoolBase:
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def main(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000,
    raw: bool = False,
    train_end: int = 2019,
    freq: str = 'day',
):
    reseed_everything(seed)

    device = torch.device('cuda:0')
    close = Feature(FeatureType.CLOSE)

    from alphagen_generic.features import open_
    from gan.utils import Builders
    from gan.utils.data import get_data_by_year
    import os

    
    reseed_everything(seed)
    returned = get_data_by_year(
        train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
        instruments=instruments, target=target,freq=freq,
    )
    data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,name = returned
    

    pool = AlphaPool(
        capacity=pool_capacity,
        stock_data=data,
        target=target,
        ic_lower_bound=None
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    name_prefix = f"n1227day_{instruments}_{train_end}_{pool_capacity}_{seed}" ## new_time
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path='out_ppo/checkpoints',
        train_data=data,
        train_target=target,
        valid_data=data_valid,
        valid_target=target,
        test_data=data_test,
        test_target=target,
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=1,
    )

    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log='out_ppo/log2',
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f'{name_prefix}_{timestamp}',
    )

    
from gan.utils.qlib import get_data_my
if __name__ == '__main__':
    steps = {
        10: 250_000,
        20: 300_000,
        50: 350_000,
        100: 400_000
    }
    train_end = 2020
    for capacity in [1,10,20,50,100]:
        for seed in range(5):
            for instruments in ["csi300"]:
                main(
                    seed=seed, instruments=instruments, pool_capacity=capacity, 
                    steps=steps[capacity], raw = True,
                    train_end=train_end,
                    )


