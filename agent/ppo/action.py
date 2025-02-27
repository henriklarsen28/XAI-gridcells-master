import torch


def get_action(obs, policy_network, cov_mat):

    if len(obs.shape) == 2:
        obs = obs.unsqueeze(0)

    mean, std, _, _ = policy_network(obs)
    dist = torch.distributions.MultivariateNormal(mean, cov_mat)

    action = dist.sample()
    log_prob = dist.log_prob(action)

    return action.cpu().detach().numpy(), log_prob.detach()


def evaluate(obs, actions, policy_network, critic_network, cov_mat):
    V, _ = critic_network(obs)
    mean, std, env_class, _ = policy_network(obs)
    dist = torch.distributions.MultivariateNormal(mean, cov_mat)
    log_prob = dist.log_prob(actions)
    entropy = dist.entropy()

    return V, log_prob, entropy.mean(), env_class


def kl_divergence(obs, actions, policy_network, cov_mat):
    mean, std, _, _ = policy_network(obs)
    dist = torch.distributions.MultivariateNormal(mean, cov_mat)
    old_dist = torch.distributions.MultivariateNormal(actions, cov_mat)

    return torch.distributions.kl_divergence(old_dist, dist).mean()
