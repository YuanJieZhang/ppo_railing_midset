import random
import sqlite3
import os
import sys

import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import copy
from heapq import heappush, heappop, nsmallest
from tqdm import tqdm

from matplotlib import animation
from data.utils import create_graph
from data.utils import import_requests_from_csv
from data.utils import Driver
from data.utils import choose_random_node

class TopEnvironment1(object):
    """
    Implementation of the travelling officer environment.
    """

    FREE = 0
    OCCUPIED = 1

    def __init__(self, gamma, drivers_num=0, speed=5000., observation=None, start_time=None, timestep=1, final_time=50, fairness_discount=0.9):

        self.train_days = [39]
        self.drivers = []
        for i in range(2):
            self.drivers.append(Driver(0))
        # 初始化司机
        for idx, driver in enumerate(self.drivers):
            driver.on_road = 0
            driver.idx = idx
            driver.money = 0
            driver.speed = 5000

        self.observation = observation
        self.events = None
        self.event_idx = None
        self.time = None
        self.done = False
        self.start_time = 0
        self.timestep = 1
        self.final_time = 50
        self.fairness_discount = 0.9
        # 创建地图
        self.graph = create_graph()
        # 导入所有请求
        self.all_requests = import_requests_from_csv()
        # 当前时间下的所有请求
        self.requests = []
        # 所有的点
        self.actions = tuple(self.graph.nodes)


    def close(self):
        pass

    def reset(self, state=None):
        # 初始化司机在任意位置
        for driver in self.drivers:
            driver.on_road = 0
            driver.money = 0
            driver.Request = None
            driver.pos = choose_random_node(self.graph)

        for requests in self.all_requests:
            for r in requests:
                r.state = 0
        self.time = 0
        self.requests = []
        self.requests.extend(self.all_requests[0])
        self.done = False
        return self._state()

    def _state(self):
        return self.observation(self)

    # action是request[] action是一一对应的
    def step(self, action):
        # action把他变成司机->request的形式传入step
        action_map = {}
        select_actions = []
        reward = 0
        if self.drivers[action[1]].on_road == 0:
            node_idx = action[0]
            for r in self.requests:
                if (r.destination == node_idx) & (r.state == 0):
                    select_actions.append(r)
            if len(select_actions) != 0:
                random_action = random.choice(select_actions)
                random_action.state = 1
                reward = (self.graph.get_edge_data(random_action.origin, random_action.destination)["distance"] -
                         self.graph.get_edge_data(self.drivers[action[1]].pos,
                                                  random_action.origin)["distance"])
                self.drivers[action[1]].on_road = 1
                self.drivers[action[1]].Request = random_action


        if self.time >= self.final_time:
            self.done = True
        return self._state(), reward, self.done, {}

class TopEnvironment(object):
    """
    Implementation of the travelling officer environment.
    """

    FREE = 0
    OCCUPIED = 1

    def __init__(self, gamma, drivers_num=0, speed=5000., observation=None, start_time=None, timestep=1, final_time=50, fairness_discount=0.9):

        self.train_days = [39]
        self.drivers = []
        for i in range(drivers_num):
            self.drivers.append(Driver(0))
        # 初始化司机
        for idx, driver in enumerate(self.drivers):
            driver.on_road = 0
            driver.idx = idx
            driver.money = 0
            driver.speed = speed

        self.observation = observation
        self.events = None
        self.event_idx = None
        self.time = None
        self.done = False
        self.start_time = start_time
        self.timestep = timestep
        self.final_time = final_time
        self.fairness_discount = fairness_discount
        # 创建地图
        self.graph = create_graph()
        # 导入所有请求
        self.all_requests = import_requests_from_csv()
        # 当前时间下的所有请求
        self.requests = []
        # 所有的点
        self.actions = tuple(self.graph.nodes)
        super().__init__()

    def close(self):
        pass

    def reset(self, state=None):
        # 初始化司机在任意位置
        for driver in self.drivers:
            driver.on_road = 0
            driver.money = 0
            driver.Request = None
            driver.pos = choose_random_node(self.graph)

        for requests in self.all_requests:
            for r in requests:
                r.state = 0
        self.time = 0
        self.requests = []
        self.requests.extend(self.all_requests[0])
        self.done = False
        return self._state()

    def _state(self):
        return self.observation(self)

    # action是request[] action是一一对应的
    def step(self, action):
        # action把他变成司机->request的形式传入step
        action_map = {}
        select_actions = []
        reward = 0
        if self.drivers[action[1]].on_road == 0:
            node_idx = action[0]
            for r in self.requests:
                if (r.destination == node_idx) & (r.state == 0):
                    select_actions.append(r)
            if len(select_actions) != 0:
                random_action = random.choice(select_actions)
                random_action.state = 1
                reward = (self.graph.get_edge_data(random_action.origin, random_action.destination)["distance"] -
                         self.graph.get_edge_data(self.drivers[action[1]].pos,
                                                  random_action.origin)["distance"])
                self.drivers[action[1]].on_road = 1
                self.drivers[action[1]].Request = random_action


        if self.time >= self.final_time:
            self.done = True
        return self._state(), reward, self.done, {}


'''
class StrictResourceTargetTopEnvironment(TopEnvironment):

    def __init__(self, *args, allow_wait=False, **kvargs):
        super().__init__(*args, allow_wait=allow_wait, **kvargs)
        action_space = len(self.edge_resources)
        if allow_wait:
            action_space += 1
        self._mdp_info.action_space = Discrete(action_space)
        self.edge_ordering = list(self.edge_resources.keys())
        self.shortest_paths = {}
        for edge in self.resource_edges.values():
            self.shortest_paths[edge] = dict(nx.shortest_path(self.graph, target=edge[0], weight='length'))

    def step(self, action):

        if action[0] == len(self.edge_resources):
            return super().step([len(self.actions) - 1])
        edge = self.edge_ordering[action[0]]
        s = None
        rewards = None
        arrived = False
        done = False
        while not arrived and not done:
            path = self.shortest_paths[edge][self.position]
            if len(path) <= 1:
                action = self.edge_indices[edge + (0,)]
                arrived = True
            else:
                next_node = self.shortest_paths[edge][self.position][1]
                action = self.edge_indices[(self.position, next_node, 0)]
            s, r, done, _ = super().step([action])
            self.render(False)
            if rewards is None:
                rewards = r
            else:
                if np.isscalar(r):
                    # rewards += r#每走一步进行累加
                    rewards = r
                else:
                    # rewards[1] += r[1]
                    # rewards[0] += r[0] * self._mdp_info.gamma ** r[1]
                    rewards[1] += r[1]  # 代表travel_time
                    rewards[0][3] += r[0][3] * self._mdp_info.gamma ** r[1]
                    # rewards[0][0]+=r[0][0]
                    for i in range(len(r[0][0])):
                        rewards[0][0][i] += r[0][0][i]
                    rewards[0][1] = r[0][1]
                    rewards[0][2] = r[0][2]
                    if (sum(self.areaInfo.areaCapList) > sum(self.areaInfo.areaVioList)):
                        raise Exception("record error!", rewards)
                    if (sum(r[0][0]) > 0):
                        print(rewards)
            if done:
                break
        return s, rewards, done, {}

'''
