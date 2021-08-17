# -*- coding: utf-8 -*-
import copy
import math
import os
import random
import sys
import threading
import tkinter
from functools import reduce

import pandas as pd

# ----------- TSP问题 -----------


city_number = 5

length_list = []


class TSP(object):
    '''
    Initial the canvas of TSP
    '''

    def __init__(self, root, width=800, height=800, n=city_number):

        self.root = root
        self.width = width
        self.height = height
        self.n = n
        self.city_distance, self.city_x, self.city_y = self.get_city_distance()
        # Tkinter.Canvas
        self.canvas = tkinter.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#EBEBEB",
            xscrollincrement=1,
            yscrollincrement=1
        )
        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)
        self.title("Hopfield network solving tsp problem")
        self.__r = 1
        self.__lock = threading.RLock()

        self.bindEvents()
        self.initial()

    def bindEvents(self):

        self.root.bind("q", self.quit)
        self.root.bind("n", self.initial)
        self.root.bind("e", self.search_path)
        self.root.bind("s", self.stop)

    def title(self, s):

        self.root.title(s)

    def initial(self, evt=None):

        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

        self.clear()
        self.nodes = []
        self.nodes2 = []

        for i in range(self.n):
            self.nodes.append((self.city_x[i], self.city_y[i]))
            node = self.canvas.create_oval(self.city_x[i] - self.__r,
                                           self.city_y[i] - self.__r, self.city_x[i] + self.__r,
                                           self.city_y[i] + self.__r,
                                           fill="#ee0000",
                                           outline="#000000",
                                           tags="node",
                                           )
            self.nodes2.append(node)
        # set path for the city sequence
        self.path = range(city_number)
        self.line(self.path)

        self.A = 1.5
        self.D = 3
        self.u0 = 0.02
        self.step = 0.01
        self.iter = 1

        self.DistanceCity = self.__cal_total_distance()
        print(self.DistanceCity)
        # initial the U matrix
        self.U = [[0.5 * self.u0 * math.log(city_number - 1) for col in range(city_number)] for raw in
                  range(city_number)]

        for raw in range(city_number):
            for col in range(city_number):
                self.U[raw][col] += 2 * random.random() - 1

        # initial the V matrix
        self.V = [[0.0 for col in range(city_number)] for raw in range(city_number)]
        for raw in range(city_number):
            for col in range(city_number):
                self.V[raw][col] = (1 + math.tanh(self.U[raw][col] / self.u0)) / 2

    def line(self, order):
        self.canvas.delete("line")

        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill="#000000", tags="line")
            return i2

        reduce(line2, order, order[-1])

    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

    def quit(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()

        sys.exit()

    def stop(self, evt):
        self.__lock.acquire()
        self.__running = False
        length_sequence = pd.DataFrame(data=length_list)
        length_sequence.to_csv("hop_length.csv")
        self.__lock.release()

    def DeltaU(self):

        rawsum = []
        for raw in range(city_number):
            raw_sum = 0
            for col in range(city_number):
                raw_sum += self.V[raw][col]
            rawsum.append(raw_sum - 1)

        colsum = []
        for col in range(city_number):
            col_sum = 0
            for raw in range(city_number):
                col_sum += self.V[raw][col]
            colsum.append(col_sum - 1)

        deltau = copy.deepcopy(self.V)
        for raw in deltau:
            temp = raw[0]
            del raw[0]
            raw.append(temp)

        for raw in range(city_number):
            for col in range(city_number):
                deltau[raw][col] = -1 * (
                        self.A * rawsum[raw] + self.A * colsum[col] + self.D * self.DistanceCity * deltau[raw][col])

        return deltau

    '''
    Get Energy Function
    '''

    def Energy(self):
        rawsum = []
        for raw in range(city_number):
            raw_sum = 0
            for col in range(city_number):
                raw_sum += self.V[raw][col]
            rawsum.append(raw_sum - 1)
        rawsumsqr = 0
        for raw in rawsum:
            rawsumsqr += raw * raw

        colsum = []
        for col in range(city_number):
            col_sum = 0
            for raw in range(city_number):
                col_sum += self.V[raw][col]
            colsum.append(col_sum - 1)
        colsumsqr = 0
        for col in colsum:
            colsumsqr += col * col

        PermitV = copy.deepcopy(self.V)
        for raw in PermitV:
            temp = raw[0]
            del raw[0]
            raw.append(temp)
            for item in raw:
                item *= self.DistanceCity
        sumV = 0
        for raw in range(city_number):
            for col in range(city_number):
                sumV += PermitV[raw][col] * self.V[raw][col]

        E = 0.5 * (self.A * rawsumsqr + self.A * colsumsqr + self.D * sumV)
        return E

    '''
    Pathcheck for get the valid path
    which maps the maximum of each rows to 1
    '''

    def Pathcheck(self):
        V1 = [[0 for col in range(city_number)] for raw in range(city_number)]
        for col in range(city_number):
            MAX = -1.0
            MAX_raw = -1
            for raw in range(city_number):
                if self.V[raw][col] > MAX:
                    MAX = self.V[raw][col]
                    MAX_raw = raw
            V1[MAX_raw][col] = 1

        rawsum = []
        for raw in range(city_number):
            raw_sum = 0
            for col in range(city_number):
                raw_sum += V1[raw][col]
            rawsum.append(raw_sum)

        colsum = []
        for col in range(city_number):
            col_sum = 0
            for raw in range(city_number):
                col_sum += V1[raw][col]
            colsum.append(col_sum)

        sumV1 = 0
        for item in range(city_number):
            sumV1 += (rawsum[item] - colsum[item]) ** 2

        path = []
        if sumV1 != 0:
            path.append(-1)
        else:
            for col in range(city_number):
                for raw in range(city_number):
                    if V1[raw][col] == 1:
                        path.append(raw)
        return path

    # calculate the total distance
    def __cal_total_distance(self):
        temp = 0.0

        for i in range(1, city_number):
            start, end = self.path[i], self.path[i - 1]
            temp += self.city_distance[start][end]

        # get the distance path
        end = self.path[0]
        temp += self.city_distance[start][end]
        return temp

    def search_path(self, evt=None):

        self.__lock.acquire()
        self.__running = True
        self.__lock.release()

        while self.__running:

            delta_u = self.DeltaU()
            # calculate U
            for raw in range(city_number):
                for col in range(city_number):
                    self.U[raw][col] += delta_u[raw][col] * self.step

            # calculate V
            for raw in range(city_number):
                for col in range(city_number):
                    self.V[raw][col] = (1 + math.tanh(self.U[raw][col] / self.u0)) / 2

            # get Energy
            E = self.Energy()

            # get Path
            path = self.Pathcheck()
            if path[0] != -1:
                self.path = path
                self.line(self.path)

            self.title("Round: %d, City_Distance: %d" % (self.iter, self.__cal_total_distance()))
            length_list.append(self.__cal_total_distance())
            self.canvas.update()
            self.iter += 1

    def mainloop(self):
        self.root.mainloop()

    def get_city_distance(self):
        current_path = os.getcwd()
        file_path = current_path + "/state_list.csv"
        df = pd.read_csv(file_path)
        x = df['0'].values
        y = df['1'].values

        distance_graph = [[0.0 for col in range(self.n)] for raw in range(self.n)]
        for i in range(self.n):
            for j in range(self.n):
                temp = pow((x[i] - x[j]), 2) + pow((y[i] - y[j]), 2)
                temp = pow(temp, 0.5)
                distance_graph[i][j] = float(int(temp))

        return distance_graph, x, y


if __name__ == '__main__':
    TSP(tkinter.Tk()).mainloop()
