# Master thesis 2025
By Gro Oleivsgard and Henrik Larsen


## Explainable AI Approaches to Detect Independent Spatial Representations of Environments in Deep Reinforcement Learning Agents

The purpose of our master thesis is to explore how methods from explainable artificial intelligence (XAI) can be applied to a transformed-based reinforcement learning agent to evaluate parallels in how spatial environments are encoded between an artificial neural network and the biological brain. This is an interesting area of research because it bridges the gap between artificial intelligence and neuroscience using novel methods from the field of XAI.

In our thesis, we will determine if spatial environments are independently encoded in a transformer-based reinforcement learning architecture. This is inspired by studies on the biological brain which suggests that spatial representations are stored independently with minimal overlap between different environments. To develop an artificial agent that can act in an environment similar to physical space, we seek to implement the architecture in a continuous, partially observable environment. The thesis will be guided by the following research questions.

*RQ1: What methods improve the training of transformer-based reinforcement learning agents using Proximal Policy Optimization (PPO) in partially observable continuous environments?*

*RQ2: How do we identify spatial mapping in an artificial neural network using methods from explainable artificial intelligence (XAI)?*

*RQ3: How different must an environment be for a cognitive remapping of the environment in an artificial neural network to occur?*

*RQ4: Can static concepts be used to detect cognitive remapping in transformer-based deep reinforcement learning agents?*

*RQ5: Given the identification of cognitive remapping, what are the scalability limits for transformer architectures in handling multiple environments?*

### Explainable AI methods

We are researching multiple XAI methods on their applicability to explain the encodings of navigation in deep neural networks. Below is a summary of the methods we have tested so far.

1. Visualization of Q-values in a Deep Q-learning Network
   '''
   The Deep Q-Learning algorithm is a reinforcement learning algorithm where the agent learns optimal actions by experiencing consequences in the form of immediate positive or negative rewards and uses this to estimate the value of the next state. Each state has an action with a corresponding q-value, which we visualized in the environment in order to depict how the agent has learned to move through the environment.
   ![image](xai/q-values/q-values-visualized.png)
   '''
