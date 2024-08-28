from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Categorical
from torch.optim import Adam

class MARLAgent(ABC):
    @abstractmethod
    def getType(self):
        ...
    @abstractmethod
    def genAgentObs(self):
        ...
    @abstractmethod
    def get_action_predictions(self, state):
        ...
    @abstractmethod
    def save_model(self):
        ...
    @abstractmethod
    def update():
        ...

#Class representing a PPO agent. Adapted from ericyangyu/PPO-for-Beginners
class PPOAgent(MARLAgent):
    def __init__(self, env, config, network, agent_id):
        assert config.mode == "PPO", "Called PPOAgent, but mode is not PPO"
        self.type = "PPO"
        self.env = env
        self.config = config
        self.architecture = network
        self.id = agent_id
        self.actor = network(self.genAgentObs(), self.config, len(self.env.actions), self.env.n_agents, self.id)
        self.critic = network(self.genAgentObs(), self.config, 1, self.env.n_agents, self.id)
        self.actor_optim = Adam(self.actor.parameters(), lr = self.config.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr = self.config.lr)
        #self.cov_var = torch.full(size=(len(self.env.actions),), fill_value=0.5)
        #self.cov_mat = torch.diag(self.cov_var)

        self.memory = {}
        self.memory["observations"] = []
        self.memory["actions"] = []
        self.memory["log_probs"] = []

    def getType(self):
        return self.type

    def genAgentObs(self):
        birdsEye = self.env.gen_obs()
        agentObs = {'image': birdsEye['image'][self.id], 'direction': birdsEye['direction'][self.id]}
        if(self.env.fully_observed):
            agentObs['position'] = self.env.agent_pos[self.id]
        return agentObs

    def get_action_predictions(self, state = None):
        if not state:
            state = self.genAgentObs()

        self.memory["observations"].append(state)

        mean = self.actor(state)
        #dist = MultivariateNormal(mean, self.cov_mat)
        dist = Categorical(logits = mean)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = action.detach().numpy()
        log_prob = log_prob.detach()
        self.memory["actions"].append(action)
        self.memory["log_probs"].append(log_prob)
        return action, log_prob

    def save_model(self):
        print("Saving PPO agent with id " + str(self.id) + "...")
        torch.save(self.actor, "PPO_agent_actor_" + str(self.id))
        torch.save(self.critic, "PPO_agent_critic_" + str(self.id))

    def update(self):
        print("Updating agent #" + str(self.id))
        assert "rewards" in self.memory, "Please inject reward into agent memory from the metacontroller. Thrown from agent #" + str(self.id)
        assert "eps_length" in self.memory, "Please inject lengths of the episodes from the current batch into agent memory from the metacontroller. Thrown from agent #" + str(self.id)

        self.memory["actions"] = torch.tensor(self.memory["actions"], dtype=torch.float)
        self.memory["log_probs"] = torch.tensor(self.memory["log_probs"], dtype=torch.float)


        self.compute_rtgs()
        assert "rtgs" in self.memory, "compute_rtgs failed."

        V, _ = self.evaluate(self.memory["observations"], self.memory["actions"])
        A_k = self.memory["rtgs"] - V.detach()
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for _ in range(self.config.num_iterations):
            V, curr_log_probs = self.evaluate(self.memory["observations"], self.memory["actions"])
            ratios = torch.exp(curr_log_probs - self.memory["log_probs"])
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.config.clip, 1 + self.config.clip) * A_k
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, self.memory["rtgs"])
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()


        self.resetMemory()

    def evaluate(self, observations, actions):
        V = self.critic(observations).squeeze()
        mean = self.actor(observations)
        #dist = MultivariateNormal(mean, self.cov_mat)
        dist = Categorical(logits = mean)
        log_probs = dist.log_prob(actions)
        return V, log_probs


    def compute_rtgs(self):
        batch_rews = self.memory["rewards"]
        self.memory["rtgs"] = []
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            for rew in ep_rews:
                discounted_reward = rew + discounted_reward * self.config.gamma
                self.memory["rtgs"].insert(0, discounted_reward)
        self.memory["rtgs"] = torch.tensor(self.memory["rtgs"], dtype = torch.float)

    def resetMemory(self):
        self.memory = {}
        self.memory["observations"] = []
        self.memory["actions"] = []
        self.memory["log_probs"] = []
