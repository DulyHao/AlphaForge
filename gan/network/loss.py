
import torch

def loss_simi(loss_inputs,cfg):
    onehot_tensor_1 = loss_inputs['onehot_tensor_1'] #（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
    onehot_tensor_2 = loss_inputs['onehot_tensor_2']

    simi = torch.sum(onehot_tensor_1*onehot_tensor_2,dim=-1).sum(dim = -1) # (batch_size,)
    simi = simi / onehot_tensor_1.shape[1]

    simi = simi - cfg.l_simi_thresh
    simi = torch.relu(simi)
    # simi = simi**2
    simi = simi.mean()
    return simi

def loss_pred(loss_inputs,cfg):
    pred_1 = loss_inputs['pred_1'][:,0] #（batch_size）

    return - pred_1.mean()

def loss_potential(loss_inputs,cfg):

    
    epsilong=cfg.l_potential_epsilon
    u1,u2=loss_inputs['latent_1'],loss_inputs['latent_2']#(batch_size*n_sample,latent_size_netP)
    u1=u1.clip(epsilong,1-epsilong)
    u2=u2.clip(epsilong,1-epsilong)
    similarity=(u1*u2).sum(axis=1)/ ( 
        ((u1**2).sum(axis=1))**0.5 * ((u2**2).sum(axis=1))**0.5
                            ) -cfg.l_potential_thresh

    similarity=similarity*(similarity>0)# 针对每个元素选择大于0的
    return similarity.mean()

def loss_entropy(loss_inputs,cfg):
    onehot_tensor_1 = loss_inputs['onehot_tensor_1'] #（batch_size,MAX_EXPR_LENGTH,SIZE_ACTION）
    onehot_tensor_2 = loss_inputs['onehot_tensor_2']

    entropy_1 = -torch.sum(onehot_tensor_1*torch.log(onehot_tensor_1),dim=-1).sum(dim = -1) # (batch_size,)
    entropy_1 = entropy_1 / onehot_tensor_1.shape[1]

    entropy_2 = -torch.sum(onehot_tensor_2*torch.log(onehot_tensor_2),dim=-1).sum(dim = -1) # (batch_size,)
    entropy_2 = entropy_2 / onehot_tensor_2.shape[1]

    entropy = entropy_1 + entropy_2
    entropy = entropy.mean()
    return entropy
def get_losses(loss_inputs,cfg):
    
    loss = 0
    if cfg.l_simi != 0 :
        loss += cfg.l_simi *  loss_simi(loss_inputs,cfg)
    if cfg.l_pred != 0 :
        loss += cfg.l_pred * loss_pred(loss_inputs,cfg)
    if cfg.l_potential !=0 :
        loss += cfg.l_potential * loss_potential(loss_inputs,cfg)
    if cfg.l_entropy !=0 :
        loss += cfg.l_entropy * loss_entropy(loss_inputs,cfg)
    
    return loss