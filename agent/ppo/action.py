import torch    

def get_action(obs, policy_network):

        if len(obs.shape) == 2:
              obs = obs.unsqueeze(0)

        mean, std, _ = policy_network(obs)
        dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.cpu().detach().numpy(), log_prob.detach()

def evaluate(obs, actions, policy_network, critic_network):
    V, _ = critic_network(obs)

    mean, std, _ = policy_network(obs)
    dist = torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))

    log_prob = dist.log_prob(actions)

    return V, log_prob
