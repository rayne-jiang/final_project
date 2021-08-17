import numpy
import numpy as np
import random


def initial_route(N):
    m = np.random.randint(0, 100, size=(N, N))
    m_symm = (m + m.T) / 2
    return m_symm


def init_sequence(N):
    sequence_list = list(range(N))
    random.shuffle(sequence_list)
    return sequence_list


def mask(list):
    m = np.zeros((len(list), len(list)))
    list.append(list[0])
    for i in range(0, len(list) - 2):
        # set mask of Link
        m[i][i + 1] = 1
        m[i + 1][i] = 1
        m[i][i - 1] = 1
        m[i - 1][i] = 1
    list.pop()  # delete the last element
    return m


def energy_func(route_mat, m_mat):
    # set constraints
    constraint = (1000 * ((m_mat.sum(axis=0) - 2) ** 2) + 1000 * ((m_mat.sum(axis=1) - 2) ** 2)).sum()
    mul_mat = m_mat.__mul__(route_mat)
    sum = mul_mat.sum()
    return 1 / 2 * sum + constraint


def get_u_mat(delta_mat, route_mat, v_mat, rate_A, rate_D, N):
    delta_mat = -1 * (rate_A * v_mat + rate_D * route_mat.__mul__(delta_mat))
    for row in range(N):
        for col in range(N):
            delta_mat[row][col] = -1 * (
                    rate_A * sum(v_mat[row]-2) + rate_A * sum(v_mat[col]-2) + rate_D * route_mat[row][col] * delta_mat[row][
                col])
    return delta_mat


def get_v_mat(delta_mat):
    return (1 + np.tan(delta_mat)) / 2


def threshold(x, N):
    for row in range(N):
        for col in range(N):
            if x[row][col] > 1 and row!=col:
                x[row][col] = 1
            else:
                x[row][col] = 0
    return x


def update(route_mat, time=100):
    rate_A = 1.5
    rate_B = 0.5
    u_mat = numpy.random.randint(0, 100, size=(N, N)) / 10
    u_mat = (u_mat + u_mat.T) / 2
    energy_list = []
    map_list = []
    for s in range(time):
        v_mat = get_v_mat(u_mat)
        route = threshold(v_mat, N)
        energy = energy_func(route_mat, route)
        energy_list.append(energy)
        map_list.append(route)
        print(map_list)
        print(str(s) + 'round:')
        print(min(energy_list))
        u_mat = get_u_mat(u_mat, route_mat, v_mat, rate_A, rate_B, N)
    return energy_list, map_list

if __name__ == '__main__':
    N = 7
    route_mat = initial_route(N)
    sequence = init_sequence(N)
    energy_list, map_list = update(route_mat)
    index_min = energy_list.index(min(energy_list))

    print(map_list[index_min])