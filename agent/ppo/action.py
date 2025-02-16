import torch


def get_action(obs, policy_network, cov_mat):

    if len(obs.shape) == 2:
        obs = obs.unsqueeze(0)

    mean, std, env_class, _ = policy_network(obs)
    dist = torch.distributions.MultivariateNormal(mean, cov_mat)

    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.cpu().detach().numpy(), log_prob.detach(), env_class


def evaluate(obs, actions, policy_network, critic_network, cov_mat):
    V, _ = critic_network(obs)

    mean, std, _, _ = policy_network(obs)
    dist = torch.distributions.MultivariateNormal(mean, cov_mat)

    log_prob = dist.log_prob(actions)

    return V, log_prob
