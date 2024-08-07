import random
import sqlite3
import os
import statistics
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
from data.utils import import_requests_from_csv_poission
from data.utils import Driver
from data.utils import choose_random_node


class TopEnvironmentW:
    FREE = 0
    OCCUPIED = 1

    def __init__(self, gamma, drivers_num, speed=5000., start_time=None, timestep=1, final_time=100,
                 fairness_discount=0.9):
        # 新增变量和初始化
        self.agent_num = drivers_num
        self.obs_dim = 4
        self.action_dim = 11
        self.train_days = [39]
        self.drivers = []
        for i in range(self.agent_num):
            self.drivers.append(Driver(0))
        for idx, driver in enumerate(self.drivers):
            driver.on_road = 0
            driver.idx = idx
            driver.money = 0
            driver.speed = 5000
        self.start_time = start_time
        self.timestep = timestep
        self.final_time = final_time
        self.time = 0
        self.done = False
        self.graph = create_graph()
        self.order_count = 0
        self.all_requests,self.random10_list = import_requests_from_csv_poission(1,2,5000)
        self.init_pos=random.sample(self.random10_list,self.agent_num)
        self.max_count = 500
        self.requests = []
        self.reward = [[]]
        self.fairness = []
        self.utility = [[]]
        self.epoch = 0
        self.beta = 1400000

    def _generate_observation(self):
        state = np.zeros((self.agent_num, self.obs_dim))
        for i, driver in enumerate(self.drivers):
            state[i, 0] = 1 if driver.on_road == self.FREE else 0
            state[i, 1] = driver.pos if driver.on_road == 0 else -1

            state[i, 2] = self.time  # 填充更多司机相关信息
            state[i, 3] = driver.money  # 假设只填充4个维度

        return state

    def reset(self):
        i=0
        for driver in self.drivers:
            driver.on_road = self.FREE
            driver.money = 0
            driver.pos = self.init_pos[i]
            i+=1# 随机选择一个位置

        self.time = 0
        self.requests = []
        self.requests.extend(self.all_requests[0])
        self.done = False
        self.utility = np.zeros((self.agent_num,1))
        self.reward = np.zeros((self.agent_num,1))
        self.fairness = []
        self.order_count = 0
        return self._generate_observation()

    def step(self, action):
        for driver in self.drivers:
            if driver.on_road == 1:
                driver.start_time += self.timestep
                if (self.graph.get_edge_data(driver.Request.origin, driver.Request.destination)["distance"] -
                    self.graph.get_edge_data(driver.pos,
                                             driver.Request.origin)[
                        "distance"]) / driver.speed <= driver.start_time:
                    driver.on_road = 0
                    self.order_count += 1
                    driver.Request.state=1
                    driver.pos = driver.Request.destination
                    driver.money += self.graph.get_edge_data(driver.Request.origin,
                                                             driver.Request.destination)["distance"]
        sorted_drivers = sorted(self.drivers, key=lambda d: d.money)
        # sort 目的地
        reward_list = []
        end_list = []
        for idx, driver in enumerate(sorted_drivers):
            actions = []
            # 选出来的点
            actions.append(action[idx])
            actions.append(driver.idx)
            _, single_reward, done, _ = self.single_step(actions)
            reward_list.append(single_reward)
            end_list.append(done)
        self.time += self.timestep

        vec = np.array(reward_list).reshape((1,self.agent_num))
        self.utility=np.hstack((self.utility,vec.T))


        return self._state(), reward_list, end_list, {}

    def single_step(self, action):
        # action把他变成司机->request的形式传入step
        select_actions = []
        reward = 0
        action_onehot = action[0]
        if action_onehot.tolist().index(1) >= len(self.random10_list):
            return self._state(), reward, self.done, {}
        node_idx = self.random10_list[action_onehot.tolist().index(1)]

        if self.drivers[action[1]].on_road == 0:
            for r in self.requests:
                if (r.destination == node_idx) :
                    if (r.state == 0) & r.timestamp<=self.time &r.origin != r.destination:
                        select_actions.append(r)
            if len(select_actions) != 0:
                for aim_action in select_actions:
                    #random_action = random.choice(select_actions)
                    aim_action.state = 1
                    reward = (self.graph.get_edge_data(aim_action.origin, aim_action.destination)["distance"] -
                          self.graph.get_edge_data(self.drivers[action[1]].pos,
                                                   aim_action.origin)["distance"])
                    self.drivers[action[1]].money += reward
                    if self._filter_beta() > self.beta:
                        self.drivers[action[1]].money -= reward
                        continue
                    self.drivers[action[1]].on_road = 1
                    self.drivers[action[1]].Request = aim_action

        if self.order_count >= self.max_count:
            for r in self.requests:
                r.state = 0
            self.epoch += 1
            print("epoch:", self.epoch)
            if self.epoch <= 350:
                print(self.beta,self._filter_beta())
                self.beta -= 500
            print("utility:", np.sum(self.utility))
            print("fairness:", self._filter_beta())
            self.reset()


        return self._state(), reward, self.done, {}

    def _state(self):
        return self._generate_observation()

    def _filter_beta(self):
        reward_list = []
        for driver in self.drivers:
            reward_list.append(driver.money)
        return statistics.stdev(reward_list)

    # def test(self):
    #     ("Testing environment with {} agents".format(self.agent_num))
    #     obs = self.reset()
    #     ("Initial observations:", obs)
    #
    #     # 随机生成并执行动作
    #     actions = [np.random.randint(self.action_dim) for _ in range(self.agent_num)]
    #     print("Actions:", actions)
    #
    #     next_obs, rewards, done, _ = self.step(actions)
    #     print("Next observations:", next_obs)
    #     print("Rewards:", rewards)
    #     print("Done:", done)

# 测试环境
# gamma = 0.99  # 举例用的 gamma 值
# env = TopEnvironment(gamma)
# env.test()
import numpy as np
import matplotlib.pyplot as plt

def plot_agent_lines(agent_lst, marker_list=None):
    # 将 5*100 的记录转换为包含 5 个 agent 的列表
    agent_lst = [agent_lst[i,:len(agent_lst[i]) - len(agent_lst[i]) % 100] for i in range(agent_lst.shape[0])]

    assert isinstance(agent_lst, list), "agent_lst should be a list."

    # 找到 agent 数据中最大的步骤数
    max_steps = max([len(agent) for agent in agent_lst])

    # 获取横坐标
    steps = [(i+1)*100 for i in range(int(max_steps/100))]

    # 遍历每个 agent，提取需要的步骤对应的值，形成折线图
    for i, agent in enumerate(agent_lst):

        # 根据颜色列表绘制折线图
        if marker_list:
            assert len(marker_list) == len(agent_lst), "marker and agent should have same length."
            plt.plot(steps[:len(agent)//100], np.sum(agent.reshape(-1, 100), axis=1), marker=marker_list[i]['marker'], color=marker_list[i]['color'], label=f"Agent {i}")

        else:
            plt.plot(steps[:len(agent)//100], np.sum(agent.reshape(-1, 100), axis=1), label=f"Agent {i}" )

    plt.legend()
    plt.show()