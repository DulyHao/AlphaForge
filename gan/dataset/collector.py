from gan.utils import Builders
import torch

class Collector:
    def __init__(self,seq_len ,n_actions):
        super().__init__()
        self.blds = Builders(0,max_len=seq_len,n_actions=n_actions)
        self.blds_bak = Builders(0,max_len=seq_len,n_actions=n_actions)
        self.seq_len = seq_len
        self.n_actions = n_actions

    def reset(self,data,target,metric):
        self.blds_bak += self.blds
        print('Reset bak_len:',self.blds_bak.batch_size)
        self.blds_bak.evaluate(data,target,metric)
        self.blds = Builders(0,max_len=self.seq_len,n_actions=self.n_actions)

    def collect(self,netG,netM,z,reset_net = False,random_method=None):
        netG.eval()
        with torch.no_grad():
            if reset_net:
                netG.initialize_parameters()
            netG.eval()
            z = random_method(z)
            logit_raw = netG(z)
            masked_x,masks,blds= netM(logit_raw)
            return blds
        
    def collect_randomly(self,z,netM):
        logit_raw = torch.randn([z.shape[0],self.seq_len,self.n_actions])
        print('logit_raw',logit_raw.shape)
        masked_x,masks,blds= netM(logit_raw)
        return blds


    def collect_target_num(self,netG,netM,z,
                           data,target,metric = None,
                           target_num=1000,reset_net =False,
                           drop_invalid=False,
                           randomly = False,
                           random_method = lambda x:x.normal_(),
                           max_iter = 1000,
                           ):
        
        cnt = 0
        iter_num = 0
        while self.blds.batch_size<=target_num:
            if randomly:
                builders = self.collect_randomly(z,netM)
            else:
                builders = self.collect(netG,netM,z,reset_net=reset_net,random_method=random_method)
            if drop_invalid:
                builders.drop_invalid()
            self.blds += builders
            self.blds.drop_duplicated()
            cnt += 1
            iter_num += 1
            if iter_num%10==0:
                print(f"cnt:{cnt} builders_len:{builders.batch_size},all_len:{self.blds.batch_size}")
            if iter_num>max_iter and max_iter>0:
                print(f'iter_num>max_iter:{max_iter}')
                break
          
        self.blds.drop_duplicated()
        print(self.blds.batch_size)
        self.blds.evaluate(data,target,metric)
        return
    @property
    def blds_list(self):
        return [self.blds_bak,self.blds]
    



# class Collector_2:
#     def __init__(self):
#         super().__init__()
#         self.blds = Builders(0)
#         self.blds_bak = Builders(0)

#     def reset(self,data,target,metric):
#         self.blds_bak += self.blds
#         print('Reset bak_len:',self.blds_bak.batch_size)
#         self.blds_bak.evaluate(data,target,metric)
#         self.blds = Builders(0)

#     def collect(self,netG,netM,z,reset_net = False,random_method=None):
#         with torch.no_grad():
#             if reset_net:
#                 netG.initialize_parameters()
#             netG.eval()
#             random_method(z)
#             logit_raw = netG(z)
#             masked_x,masks,blds= netM(logit_raw)
#             return blds
        
#     def collect_randomly(self,z,netM):
#         logit_raw = torch.randn([z.shape[0],self.seq_len,self.n_actions])
#         masked_x,masks,blds= netM(logit_raw)
#         return blds
#     def collect_target_num(self,netG,netM,z,data,target,target_num=1000,reset_net =False,drop_invalid=False,
#                            random_method = lambda x:x.normal_(),metric = None,randomly=False):
        
#         cnt = 0

#         iter_num = 0
#         while self.blds.batch_size<=target_num:
#             if randomly:
#                 builders = self.collect_randomly(z,netM)
#             else:
#                 builders = self.collect(netG,netM,z,reset_net=reset_net,random_method=random_method)
#             if drop_invalid:
#                 builders.drop_invalid()
#             self.blds += builders
#             self.blds.drop_duplicated()
#             cnt += 1
#             iter_num += 1
#             if iter_num%10==0:
#                 print(f"cnt:{cnt} builders_len:{builders.batch_size},all_len:{self.blds.batch_size}")
#             if iter_num>100:
#                 print('iter_num>100')
#                 break
          
#         self.blds.drop_duplicated()
#         print(self.blds.batch_size)
#         self.blds.evaluate(data,target,metric)
#         return
#     @property
#     def blds_list(self):
#         return [self.blds_bak,self.blds]
    