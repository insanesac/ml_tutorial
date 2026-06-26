import torch

def ppo(new_log_prob, old_log_prob, advantages, eps=0.2):
    
    ratio = torch.exp(new_log_prob-old_log_prob)
    loss1 = ratio * advantages
    loss2 = torch.clamp(ratio, 1-eps,1+eps) * advantages
    
    return -torch.min(loss1, loss2).mean()
    
    
    
def dpo(chosen_logp, rejected_logp, beta=0.1):
    probs = torch.sigmoid(beta*(chosen_logp - rejected_logp))
    
    return -torch.log(probs).mean()

def grpo_loss(old_logp, new_logp, rewards, eps=0.2):
    group_mean = rewards.mean()
    group_std = rewards.std() + 1e-8
    
    advantages = (rewards - group_mean) / group_std
    
    ratio = torch.exp(new_logp - old_logp)
    
    loss1 = ratio * advantages
    loss2 = torch.clamp(ratio,1-eps,1+eps) * advantages
    
    return -torch.min(loss1, loss2).mean()