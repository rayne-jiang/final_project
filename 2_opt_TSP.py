import pandas as pd
from py2opt.routefinder import RouteFinder
import numpy as np


def distance_cal(df):
    cityx = df['0'].values
    cityy = df['1'].values
    n = len(df)
    d_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            d_mat[i, j] = np.sqrt((cityx[i] - cityx[j]) ** 2 + (cityy[i] - cityy[j]) ** 2)
    return d_mat



if __name__ == '__main__':
    raw_data = pd.read_csv("state_list.csv")
    city_names = list(raw_data.index.array)
    dist_mat = distance_cal(raw_data)

    route_finder = RouteFinder(dist_mat, city_names, iterations=5)
    best_distance, best_route = route_finder.solve()

    print(best_distance)
    print(best_route)
