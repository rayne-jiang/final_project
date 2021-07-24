import os
import random

import pandas as pd
import numpy as np






# def GA_model():
#
#
#
# def crossover():
#
#
# def mutate():
#
#
# define the latest finish time as energy function


def engery_function(df, weight_matrix):
    time_matrix = df / weight_matrix
    return max(time_matrix.sum(axis=1))


def check_constraint(df, weight_matrix):
    task_complete = df * weight_matrix
    unique_matrix = np.unique(task_complete.sum(axis=0))
    return unique_matrix[0] == 1 and len(unique_matrix) == 1


def check_solution(df, assign_list):
    for i in range(len(assign_list)):
        efficient_value = df.iloc[i, assign_list[i]]
        if efficient_value == 0:
            random.shuffle(assign_list)
            check_solution(df, assign_list)

    assign_list = [[task] for task in assign_list]
    return assign_list



if __name__ == '__main__':
    current_path = os.getcwd()
    file_path = current_path + "/worker_task_list.csv"
    data = pd.read_csv(file_path)
    print(data)

    first_step_init = list(range(0, len(data)))
    random.shuffle(first_step_init)
    assign_list = check_solution(data, first_step_init)

    sequence_list = dict(zip(list(range(0, len(assign_list))), assign_list))

    assign_list
