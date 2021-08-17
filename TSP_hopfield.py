import numpy as np
import pandas as pd
from py2opt.routefinder import RouteFinder


# Calculate distance matrix
def distance_cal(df):
    cityx = df['0'].values
    cityy = df['1'].values
    n = len(df)
    d_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            d_mat[i, j] = np.sqrt((cityx[i] - cityx[j]) ** 2 + (cityy[i] - cityy[j]) ** 2)
    return d_mat


def check_valid(v_mat):
    # testing whether solution is legal
    t1, t2, t3 = 0, 0, 0
    N = len(v_mat)

    for vx in range(N):
        for vi in range(N):
            t1 += v_mat[vx, vi]

    for x in range(N):
        for i in range(N - 1):
            for j in range(i + 1, N):
                t2 += np.multiply(v_mat[x, i], v_mat[x, j])

    for i in range(N):
        for x in range(N - 1):
            for y in range(x + 1, N):
                t3 += np.multiply(v_mat[x, i], v_mat[y, i])

    if t1 == N and t2 == 0 and t3 == 0:
        return True
    else:
        return False


# x-v-value of each node, y-u-input potential, u0-gamma
def hopfield():
    u0 = 0.02
    toend = 0
    Dist_List = []
    udao = np.zeros([N, N])
    # while toend == 0:
    # print("Step # ", ctr)
    # U initialization
    v = np.random.rand(N, N)
    u = np.ones([N, N]) * (-u0 * np.log(N - 1) / 2)
    u += u * 0.91

    for i in range(1000):
        for round in range(100):
            for vx in range(N):
                for vi in range(N):
                    j1, j2, j3, j4 = 0, 0, 0, 0
                    # derivative 1 (sum over columns j!=vi)
                    for j in range(N):
                        if j != vi:
                            j1 += v[vx, j]
                            # print(j, vi, j1)
                    j1 *= -A
                    # derivative 2 (sum over rows y!=x)
                    for y in range(N):
                        if y != vx:
                            j2 += v[y, vi]
                    j2 *= -B
                    # derivative 3 (overall sum)
                    j3 = np.sum(v)
                    j3 = -C * (j3 - N)
                    # print(j3)
                    # derivative 4
                    for y in range(N):
                        if y != vx:
                            if vi == 0:
                                j4 += dist_mat[vx, y] * (v[y, vi + 1] + v[y, N - 1])
                            elif vi == N - 1:
                                j4 += dist_mat[vx, y] * (v[y, vi - 1] + v[y, 0])
                            else:
                                j4 += dist_mat[vx, y] * (v[y, vi + 1] + v[y, vi - 1])
                    j4 *= -D
                    udao[vx, vi] = -u[vx, vi] + j1 + j2 + j3 + j4
            # update status and derivatives
            u = u + alpha * udao
            # calculate node value from input potential u
            v = (1 + np.tanh(u / u0)) / 2
            # threshold
        for vx in range(N):
            for vi in range(N):
                if v[vx, vi] < 0.7:
                    v[vx, vi] = 0
                if v[vx, vi] >= 0.7:
                    v[vx, vi] = 1
        if check_valid(v):
            toend = 1
            td, _, _ = total_distance(v)
            print(td)
            Dist_List.append([td, round])
        else:
            toend = 0
    # print(v)
    distance_df = pd.DataFrame(data=Dist_List)
    distance_df.to_csv('hoped_distance.csv')
    return v


def total_distance(v):
    cityx_final = np.zeros([N + 1])
    cityy_final = np.zeros([N + 1])
    for j in range(N):
        for i in range(N):
            if v[i, j] == 1:
                cityx_final[j] = cityx[i]
                cityy_final[j] = cityy[i]

    cityx_final[N] = cityx_final[0]
    cityy_final[N] = cityy_final[0]
    # calculate the total distance
    td = 0
    for i in range(N - 1):
        td += np.sqrt((cityx_final[i] - cityx_final[i + 1]) ** 2
                      + (cityy_final[i] - cityy_final[i + 1]) ** 2)
    td += np.sqrt((cityx_final[N - 1] - cityx_final[0]) ** 2
                  + (cityy_final[N - 1] - cityy_final[0]) ** 2)
    return td, cityx_final, cityy_final


def get_route(v):
    route = ""
    for j in range(v.shape[1]):
        route += str(np.argmax(v[:, j])) + '=>'
    return route + str(np.argmax(v[:, 0]))


if __name__ == '__main__':
    raw_data = pd.read_csv("state_list.csv")

    # get Optimization result from HillClimber
    city_names = list(raw_data.index.array)
    dist_mat = distance_cal(raw_data)

    route_finder = RouteFinder(dist_mat, city_names, iterations=5)
    best_distance, best_route = route_finder.solve()
    print(best_distance)
    print(best_route)
    # Number of cities
    N = len(raw_data)
    # City distances
    cityx = raw_data['0'].values
    cityy = raw_data['1'].values

    A = 500;
    B = 500;
    C = 1000;
    D = 500;
    alpha = 0.0001

    hopfield()

    # v_ideal = np.zeros([N, N])
    # seq = best_route
    # j = 0
    # for el in seq:
    #     v_ideal[el, j] = 1
    #     j += 1
    # print(v_ideal)
    #
    # v = np.zeros([N, N])
    # # desired total distance
    # ct = 0
    # optimal_list = []
    # while True:
    #     ct += 1
    #     v, steps = hopfield()
    #     td, _, _ = total_distance(v)
    #     if np.array_equiv(v, v_ideal):
    #         print("Desired city sequence and distance achieved for {} runs".format(ct))
    #         print(v)
    #         print("Distance: ", td)
    #         optimal_list.append(v)
    #         break
    #     elif td <= best_distance:
    #         print("Achieved desired distance for {} runs".format(ct))
    #         print(v)
    #         print(td)
    #         optimal_list.append(v)
    #     else:
    #         print("No desired solution, executed for {} steps, total distance {}".format(steps, td))
    #
    # print(get_route(v_ideal))
    #
    # v = optimal_list[0]
    # td, X, Y = total_distance(v)
    # print("Total distance: ", td)
    # print("Desired city sequence: {} \n Final permutation matrix \n{}".format(get_route(v_ideal), v_ideal))
    # print("Obtained city sequence: {} \n Final permutation matrix \n{}".format(get_route(v), v))
