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
import wandb
from data.utils import create_graph
from data.utils import import_requests_from_csv
from data.utils import Driver
from data.utils import choose_random_node
from data.utils import load_budget
from data.utils import load_location
from data.utils import load_minuium_budget


class TopEnvironmentW_1:
    FREE = 0
    OCCUPIED = 1

    def __init__(self, gamma, drivers_num, speed=5000., start_time=None, timestep=1, final_time=100,
                 fairness_discount=0.9):
        # 新增变量和初始化
        self.agent_num = drivers_num
        self.obs_dim = 4
        self.action_dim = 10001
        self.drivers = []
        for i in range(self.agent_num):
            self.drivers.append(Driver(0))
        for idx, driver in enumerate(self.drivers):
            driver.on_road = 0
            driver.idx = idx
            driver.money = 0
            driver.speed = 5000
            driver.start_time=0

        self.start_time = start_time
        self.timestep = timestep
        self.final_time = final_time
        self.time = 0
        self.done = False
        self.graph = create_graph()
        self.order_count = 0
        self.all_requests = import_requests_from_csv()
        self.init_pos = load_location()
        self.max_count = 9900
        self.requests = []
        self.reward = [[]]
        self.fairness = []
        self.utility = [[]]
        self.epoch = 0
        self.factor = 1
        self.beta = load_minuium_budget()
        project_dir = os.path.dirname(os.getcwd())
        data_dir = project_dir + '/output11.txt'
        self.file = open(data_dir, 'w')
        self.wandb = wandb.init(project='ppo_experiment_1')


    def _generate_observation(self):
        state = np.zeros((self.agent_num, self.obs_dim))
        for i, driver in enumerate(self.drivers):
            state[i, 0] = 1 if driver.on_road == self.FREE else 0
            state[i, 1] = driver.pos if driver.on_road == 0 else -1

            state[i, 2] = self.time  # 填充更多司机相关信息
            state[i, 3] = driver.money  # 假设只填充4个维度

        return state

    def reset(self):
        i = 0
        for driver in self.drivers:
            driver.on_road = self.FREE
            driver.money = 0
            driver.pos = self.init_pos[i]
            driver.start_time=0

            i += 1  # 随机选择一个位置

        self.time = 0
        self.requests = []
        self.requests.extend(self.all_requests[0])
        self.done = False
        self.utility = np.zeros((self.agent_num, 1))
        self.reward = np.zeros((self.agent_num, 1))
        self.fairness = []
        self.order_count = 0
        self.step_count = 0
        self.epoch += 1
        self.factor = 1
        msg = 'epoch:{0}, utility:{1}, fairness:{2}'.format(self.epoch, self._filter_sum(), self._filter_beta())
        print(msg)
        self.file.write(msg)
        return self._generate_observation()

    def step(self, action):
        if self.order_count >= self.max_count:
            for r in self.requests:
                r.state = 0
            self.epoch += 1
            msg = 'epoch:{0}, utility:{1}, fairness:{2}'.format(self.epoch, self._filter_sum(), self._filter_beta())
            print(msg)
            self.file.write(msg)
            self.reset()
        if self.epoch > 1000:
            self.file.close()
            sys.exit(0)
        for driver in self.drivers:
            if driver.on_road == 1:
                if (self.graph.get_edge_data(driver.Request.origin, driver.Request.destination)["distance"] +
                    self.graph.get_edge_data(driver.pos,
                                             driver.Request.origin)[
                        "distance"]) / driver.speed <=self.time-driver.start_time:
                    driver.on_road = 0
                    self.order_count += 1
                    driver.Request.state = 1
                    driver.pos = driver.Request.destination
                    driver.start_time=self.time
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

        after_reward_list = [x + (min(reward_list) / self.agent_num) for x in reward_list]
        self.step_count += 1

        msg = 'epoch:{0},step:{1}, utility:{2}, fairness:{3},beta:{4}'.format(self.epoch,self.step_count, self._filter_sum(), self._filter_beta(),self._beta())
        wandb.log({'epoch': self.epoch, 'step':self.step_count,'utility': self._filter_sum(), 'fairness': self._filter_beta()})
        print(msg)
        self.file.write(msg)
        return self._state(), after_reward_list, end_list, {}

    def single_step(self, action):
        # action把他变成司机->request的形式传入step
        select_actions = []
        reward = 0
        action_onehot = action[0]
        select_action_to = action_onehot.tolist().index(1) + 9999
        if select_action_to >= 20000 :
            return self._state(), reward, self.done, {}
        if self.driver_E_fairness(
            select_action_to, action[1]) < self._beta() * self.factor:
            if self.step_count > 100:
                self.factor *= 0.9
            return self._state(), reward, self.done, {}

        node_idx = select_action_to

        if self.drivers[action[1]].on_road == 0:
            for r in self.requests:
                if (r.destination == node_idx):
                    if (r.state == 0) & r.origin != r.destination:
                        select_actions.append(r)
            if len(select_actions) != 0:
                for aim_action in select_actions:
                    # random_action = random.choice(select_actions)
                    aim_action.state = 1
                    reward = (self.graph.get_edge_data(aim_action.origin, aim_action.destination)["distance"] -
                              self.graph.get_edge_data(self.drivers[action[1]].pos,
                                                       aim_action.origin)["distance"])
                    self.drivers[action[1]].money += reward
                    self.drivers[action[1]].on_road = 1
                    self.drivers[action[1]].Request = aim_action
                    break

        if self.order_count >= self.max_count or self.step_count > 300:
            self.done = True
        return self._state(), reward, self.done, {}

    def _state(self):
        return self._generate_observation()

    def _filter_beta(self):
        reward_list = []
        for driver in self.drivers:
            reward_list.append(driver.money)
        return min(reward_list)

    def _filter_sum(self):
        reward_list = []
        for driver in self.drivers:
            reward_list.append(driver.money)
        return sum(reward_list)

    def driver_E_fairness(self, action, driver_idx):
        select_actions = []
        request_money =[]
        if self.drivers[driver_idx].on_road == 0:
            for r in self.requests:
                if r.destination == action:
                    if (r.state == 0) & r.origin != r.destination:
                        select_actions.append(r)
            if len(select_actions) != 0:
                for aim_action in select_actions:
                    # random_action = random.choice(select_actions)
                    aim_action.state = 1
                    reward = (self.graph.get_edge_data(aim_action.origin, aim_action.destination)["distance"] -
                              self.graph.get_edge_data(self.drivers[driver_idx].pos,
                                                       aim_action.origin)["distance"])
                    request_money.append(reward+self.drivers[driver_idx].money)
                return min(request_money)
            else:
                return 0
        return 0

    def _beta(self):
        if self.epoch < 100:
            return -1
        if self.step_count >= len(self.beta)-1:
            return max(self.beta)
        return self.beta[self.step_count]

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
    agent_lst = [agent_lst[i, :len(agent_lst[i]) - len(agent_lst[i]) % 100] for i in range(agent_lst.shape[0])]

    assert isinstance(agent_lst, list), "agent_lst should be a list."

    # 找到 agent 数据中最大的步骤数
    max_steps = max([len(agent) for agent in agent_lst])

    # 获取横坐标
    steps = [(i + 1) * 100 for i in range(int(max_steps / 100))]

    # 遍历每个 agent，提取需要的步骤对应的值，形成折线图
    for i, agent in enumerate(agent_lst):

        # 根据颜色列表绘制折线图
        if marker_list:
            assert len(marker_list) == len(agent_lst), "marker and agent should have same length."
            plt.plot(steps[:len(agent) // 100], np.sum(agent.reshape(-1, 100), axis=1), marker=marker_list[i]['marker'],
                     color=marker_list[i]['color'], label=f"Agent {i}")

        else:
            plt.plot(steps[:len(agent) // 100], np.sum(agent.reshape(-1, 100), axis=1), label=f"Agent {i}")

    plt.legend()
    plt.show()
