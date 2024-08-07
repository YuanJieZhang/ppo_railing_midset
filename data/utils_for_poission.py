import os
import random

import networkx as nx
import numpy as np
import pandas as pd
import csv
import tempfile
import shutil


def change_node_to_int(node):
    try:
        return int(node.replace("A", ""))
    except ValueError:
        return 0

def request_to_row(request,graph):
    return [f"A{request.origin}", request.timestamp, graph.get_edge_data(request.origin,
                                                  request.destination)["distance"], f"A{request.destination}"]
class Request:
    def __init__(self, timestamp, destination, origin):
        self.timestamp = timestamp
        self.destination = destination
        self.origin = origin
        self.state = 0



def create_graph(list):
    project_dir = os.path.dirname(os.getcwd())
    data_dir = project_dir + '/data.txt'
    # 读取CSV文件
    data = pd.read_csv(data_dir + '/dis_CBD_twoPs_03_19.csv')
    # 创建一个无向图
    graph = nx.Graph()

    # 添加节点和边\
    with open("10_sample_node_vio_data.csv", "w") as f:
        for row in data.itertuples(index=False):
            dis = row.distance
            nodes = row.twoPs.split('_')
            node1 = nodes[0]
            node2 = nodes[1]
            destination = change_node_to_int(node1)
            origin = change_node_to_int(node2)
            if not (destination in list and origin in list and destination != origin):
                continue
            try:
                int(node1.replace("A", ""))
                int(node2.replace("A", ""))
            except ValueError:
                continue
            # 检查节点是否已经存在于图中
            line = ",".join(str(col) for col in row)
            f.write(line + "\n")
            if int(node1.replace("A", "")) not in graph.nodes:
                graph.add_node(int(node1.replace("A", "")))
            if int(node2.replace("A", "")) not in graph.nodes:
                graph.add_node(int(node2.replace("A", "")))
            # 检查边是否已经存在于图中
            if not graph.has_edge(int(node1.replace("A", "")), int(node2.replace("A", ""))):
                graph.add_edge(int(node1.replace("A", "")), int(node2.replace("A", "")), distance=dis)

    import sys
    return graph


def import_requests_from_csv_poission(list, lambda_value, request_count, end_time,graph):
    project_dir = os.path.dirname(os.getcwd())
    data_dir = project_dir + '/data.txt'
    # 读取CSV文件
    data = pd.read_csv(data_dir + '/dis_CBD_twoPs_03_19.csv')
    # 创建一个无向图
    requests = [[]]
    # 添加节点和边
    first_request = []
    for row in data.itertuples(index=False):
        nodes = row.twoPs.split('_')
        node1 = nodes[0]
        node2 = nodes[1]
        destination = change_node_to_int(node1)
        origin = change_node_to_int(node2)
        if not (destination in list and origin in list and destination != origin):
            continue
        request = Request(0, destination, origin)
        first_request.append(request)

    # 初始化时间戳偏移量计数和请求计数，将请求均匀分布在给定的时间段内
    poisson_steps = np.random.poisson(lambda_value, request_count)
    time_offsets = np.cumsum(poisson_steps)
    while time_offsets[-1] < end_time:
        poisson_steps = np.concatenate((poisson_steps, np.random.poisson(lambda_value, request_count)))
        time_offsets = np.cumsum(poisson_steps)
    # 将按时间偏移量分配的请求填充回新的 request 列表请求数量符合泊松分布
    requests = [[] for _ in range(end_time)]
    requests[0] = first_request
    for i, offset in enumerate(time_offsets):
        if offset >= end_time:
            break
        request_index = i % len(requests[0])
        dest = requests[0][request_index].destination
        orig = requests[0][request_index].origin
        request = Request(offset, dest, orig)
        requests[offset].append(request)

    with open("10_sample_node_request_data.csv", "w") as f:
        for requestList in requests:
            for request in requestList:
                line = ",".join(str(col) for col in request_to_row(request, graph))
                f.write(line + "\n")
    return requests


random_10_list = random.sample(range(1, 101), 10)
print(random_10_list)
graph=create_graph(random_10_list)
import_requests_from_csv_poission(random_10_list,1,2,5000,graph)