import random
import numpy as np
from toy_grid_dag import FlowNetAgent, TBFlowNetAgent, DBFlowNetAgent
import torch, fastrand
from time import time

_dev = [torch.device("cpu")]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])

methods_to_agent = {
    "fm_egfn": FlowNetAgent,
    "tb_egfn": TBFlowNetAgent,
    "db_egfn": DBFlowNetAgent,
}


class EvolutionGFNAgent:
    def __init__(self, args, env):
        self.args = args
        self.env = env
        self.agent_star = methods_to_agent[args.method](args, env, is_star=True)

        self.population = [
            methods_to_agent[args.method](args, env)
            for _ in range(args.population_size)
        ]
        # turn off gradient for population
        self.turn_off_population_gradients()

        # population stores its experience in the star agent's replay buffer
        self.replace_population_replay_buffer()

        # might add OUNOISE later

    def turn_off_population_gradients(self):
        for agent in self.population:
            for param in agent.parameters():
                param.requires_grad = False

    def replace_population_replay_buffer(self):
        for agent in self.population:
            agent.replay = self.agent_star.replay

    def evaluate(self, agent, add_noise=False):
        data = agent.sample_many(self.args.num_eval_episodes, [])
        rewards = (
            tf([reward for (_, _, reward, _, _) in data])
            if self.args.method == "fm_egfn"
            else tf(data[2])
        )
        return rewards.sum().item() / self.args.num_eval_episodes

    def evolve(self):
        fitness = tf([self.evaluate(agent) for agent in self.population])
        if self.args.random_policy:
            return
        argsort_fitness = fitness.argsort(descending=True)
        elite_index = argsort_fitness[: self.args.num_elites]
        # offspring index does not necessarily have pop_size - num_elites elements (but should be even)
        offspring_index = self.selection_tournament(
            argsort_fitness,
            self.args.population_size - self.args.num_elites,
            self.args.tournament_size,
        )
        unselected_index = self.get_unselected_index(elite_index, offspring_index)

        # all elite agents are copied to the next generation. First replacing the unselected agents, then (if more left) replacing the offspring agents
        elite_index = self.transfer_elite_weights(
            elite_index, unselected_index, offspring_index
        )

        # elite done, now we fill up S - E. First unselected, next offspring
        self.fill_unselected_with_crossovers(
            elite_index, unselected_index, offspring_index
        )
        self.fill_offspring_with_crossovers(offspring_index)

        # mutate all agents except the elite
        if self.args.mutation:
            self.mutate_all_but_elite(elite_index)
        
        # copy the star agent weights to the worst agent
        if self.args.feedback:
            self.clone(self.agent_star, self.population[argsort_fitness[-1]])

    def mutate_all_but_elite(self, elite_index):
        for i in range(self.args.population_size):
            if i not in elite_index:
                self.mutate2(self.population[i])

    def fill_offspring_with_crossovers(self, offspring_index):
        for i, j in zip(offspring_index[::2], offspring_index[1::2]):
            if random.random() < self.args.crossover_prob:
                self.crossover(self.population[i], self.population[j])

    def fill_unselected_with_crossovers(
        self, elite_index, unselected_index, offspring_index
    ):
        if len(unselected_index) & 1:
            unselected_index.append(random.choice(unselected_index))
        for i, j in zip(unselected_index[::2], unselected_index[1::2]):
            elite_ind, offspring_ind = random.choice(elite_index), random.choice(
                offspring_index
            )
            self.clone(self.population[elite_ind], self.population[i])
            self.clone(self.population[offspring_ind], self.population[j])
            self.crossover(self.population[i], self.population[j])

    def crossover(self, agent1: FlowNetAgent, agent2: FlowNetAgent):
        for param1, param2 in zip(agent1.model.parameters(), agent2.model.parameters()):
            W1, W2 = param1.data, param2.data
            num_crossovers = fastrand.pcg32bounded(W1.shape[0])
            for _ in range(num_crossovers):
                crossover_index = fastrand.pcg32bounded(W1.shape[0])
                if random.random() > 0.5:
                    if len(W1.shape) == 1:
                        W1[crossover_index] = W2[crossover_index]
                    else:
                        W1[crossover_index, :] = W2[crossover_index, :]
                else:
                    if len(W1.shape) == 1:
                        W2[crossover_index] = W1[crossover_index]
                    else:
                        W2[crossover_index, :] = W1[crossover_index, :]

    def mutate(self, agent: FlowNetAgent):
        weights = [
            p for p in agent.model.parameters() if len(p.shape) > 1
        ]  # exclude biases
        weight_mutation_probs = np.random.uniform(0, 1, len(weights))
        for weight, weight_mutation_prob in zip(weights, weight_mutation_probs):
            if weight_mutation_prob > random.random():
                continue
            num_mutations = fastrand.pcg32bounded(
                int(self.args.mutation_frac * weight.numel())
            )
            for _ in range(num_mutations):
                i, j = fastrand.pcg32bounded(weight.shape[0]), fastrand.pcg32bounded(
                    weight.shape[1]
                )
                rand_val = random.random()
                if rand_val < self.args.super_mutation_prob:
                    weight[i, j] += random.gauss(
                        0, self.args.super_mutation_strength * weight[i, j]
                    )
                elif rand_val < self.args.reset_prob:
                    weight[i, j] = random.gauss(0, 1)
                else:
                    weight[i, j] += random.gauss(
                        0, self.args.mutation_strength * weight[i, j]
                    )

            weight = torch.clip(weight, -self.args.weight_limit, self.args.weight_limit)

    def mutate2(self, agent: FlowNetAgent):
        # agent's each param has a probability of being mutated (independently)
        # if ¥es, then mutate the param with torch.normal(mean=0, std=1)
        # if no, then do nothing

        for param in agent.model.parameters():
            if random.random() < self.args.mutation_prob:
                param.data += torch.normal(mean=0, std=self.args.gamma, size=param.size())

    def transfer_elite_weights(self, elite_index, unselected_index, offspring_index):
        new_elite_index = []
        for elite_ind in elite_index:
            replace_ind = (
                unselected_index.pop()
                if len(unselected_index) > 0
                else offspring_index.pop()
            )
            self.clone(self.population[elite_ind], self.population[replace_ind])
            new_elite_index.append(replace_ind)
        return new_elite_index

    def get_unselected_index(self, elite_index, offspring_index):
        unselected_index = [
            i
            for i in range(self.args.population_size)
            if i not in elite_index and i not in offspring_index
        ]
        random.shuffle(unselected_index)
        return unselected_index

    def clone(
        self, master: FlowNetAgent, replacee: FlowNetAgent
    ):  # Replace the replacee individual with master
        for target_param, source_param in zip(
            replacee.model.parameters(), master.model.parameters()
        ):
            target_param.data.copy_(source_param.data)

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        # https://github.com/ShawK91/Evolutionary-Reinforcement-Learning/tree/neurips_paper_2018/core/mod_neuro_evo.py
        total_choices = len(
            index_rank
        )  # index rank is the argsort of fitness in descending order
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings


class OUNoise:
    # https://github.com/ShawK91/Evolutionary-Reinforcement-Learning/tree/neurips_paper_2018

    def __init__(self, action_dimension, scale=0.3, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
