import numpy as np
import random

import numpy.random
from tqdm import tqdm
import pandas as pd



class Route:
    def __init__(self, sequence_list):
        self.route_seq = sequence_list

    def __getSeq__(self):
        return self.route_seq

    def __getValue__(self):
        distance = 0

        for point in range(len(self.route_seq)-1):
            distance = distance + ((map.iloc[self.route_seq[point+1], 0] - map.iloc[self.route_seq[point], 1])**2 + (map.iloc[self.route_seq[point+1], 0]- map.iloc[self.route_seq[point], 1])**2)**0.5
        distance = distance + ((map.iloc[0, 0] - map.iloc[self.route_seq[-1], 0])**2 + (map.iloc[0, 1] - map.iloc[self.route_seq[-1], 1])**2)**0.5
        return distance


def init_map(point_num, min_border, max_board):
    state_list = np.random.randint(min_border, max_board, size=(point_num, 2))
    state_df = pd.DataFrame(data=state_list)
    state_df.to_csv('state_list.csv')
    return state_df


def init_pop(population_size, point_num):
    pop_list = []
    while len(pop_list) < population_size:
        sequence_list = list(range(point_num))
        random.shuffle(sequence_list)
        new_indi = Route(sequence_list)
        pop_list.append(new_indi)
    return pop_list

def get_random(key_list):
    L = len(key_list)
    i = np.random.randint(0, L)
    return key_list[i]


def crossover_mute(list, crossover_rate):
    individual_A = get_random(list)
    individual_B = get_random(list)
    individual_C = get_random(list)
    individual_D = get_random(list)
    parent_A = individual_A if individual_A.__getValue__() < individual_B.__getValue__() else individual_B
    parent_B = individual_C if individual_C.__getValue__() < individual_D.__getValue__() else individual_D

    if random.uniform(0, 1) < crossover_rate:
        for i in range(len(parent_A.route_seq)):
            seq_len = len(parent_A.route_seq)
            crossover_point = random.randint(0, seq_len)
            child_seq = parent_A.route_seq[0:crossover_point]
            for state in parent_B.route_seq:
                if state not in child_seq:
                    child_seq.append(state)
            child = Route(child_seq)
        return child
    else:
        child = parent_A if parent_A.__getValue__()< parent_B.__getValue__() else parent_B
        return child

def scaling_window(fitness_list, pop_list):
    if sum(fitness_list) != 0:
        proportion = fitness_list / sum(fitness_list) * len(fitness_list)
        proportion_list = np.ndarray.round(proportion)
    else:
        proportion_list = np.ones(len(fitness_list))
        # pure strategy select
    select_1 = np.where(proportion_list == 1)[0]
    select_2 = np.where(proportion_list > 1)[0]
    if len(select_2) != 0:
        pop_list = 2 * [pop_list[i] for i in select_2] + [pop_list[i] for i in select_1]
    else:
        pop_list = [pop_list[i] for i in select_1]
    return pop_list[:100]


def ga_model(pop_list, epoch):
    eval_list = [state.__getValue__() for state in pop_list]
    fitness_max = [np.min(eval_list)]
    fitness_min = [np.max(eval_list)]
    for i in tqdm(range(epoch)):
        # applying scaling window
        gen_list = [pop_list[np.nanargmin(eval_list)]]
        fmin = fitness_min[0] if i < 6 else fitness_min[i - 5]
        fitness_list = fmin - eval_list
        pop_list = scaling_window(fitness_list, pop_list)
        while len(gen_list) < len(init_list):
            # one-point crossover
            child = crossover_mute(pop_list, 0.6)
            gen_list.append(child)
        # re-evaluate
        eval_list = [state.__getValue__() for state in pop_list]
        fitness_max.append(np.min(eval_list))
        fitness_min.append(np.max(eval_list))
        pop_list = gen_list
    return pop_list, fitness_max


if __name__ == '__main__':
    max_value = 50
    min_value = 0
    point_num = 10
    map = init_map(point_num, min_border=min_value, max_board=max_value)
    init_list = init_pop(100, point_num)
    pop_list, fitness_max = ga_model(init_list, 1000)
    print(fitness_max)
