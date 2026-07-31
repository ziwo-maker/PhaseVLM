from .config import DIC_AGENTS
from .my_utils import merge, get_state, get_state_detail, eight_phase_list, dump_json
from copy import deepcopy
from .cityflow_env import CityFlowEnv
from .pipeline import path_check, copy_cityflow_file, copy_conf_file
import os
import time
import numpy as np
import wandb
from tqdm import tqdm
import threading

# OneLine类：用于单轮交通信号控制仿真流程的封装
class OneLine:

    # 初始化方法，设置仿真所需的配置、路径、路网、流量等，并完成环境和智能体的初始化
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.initialize()

    # 初始化仿真环境和智能体，包括拷贝配置文件、创建环境对象、初始化所有路口的智能体
    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()

        agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
        for i in range(self.dic_traffic_env_conf['NUM_INTERSECTIONS']):
            if "ChatGPT" in agent_name:
                agent = DIC_AGENTS[agent_name.split("-")[0]](
                    GPT_version=self.dic_agent_conf["GPT_VERSION"],
                    intersection=self.env.intersection_dict[self.env.list_intersection[i].inter_name],
                    inter_name=self.env.list_intersection[i].inter_name,
                    phase_num=len(self.env.list_intersection[i].list_phases),
                    log_dir=self.dic_agent_conf["LOG_DIR"],
                    dataset=f"{self.roadnet}-{self.trafficflow}"
                )
            elif "open_llm" in agent_name:
                agent = DIC_AGENTS[agent_name.split("-")[0]](
                    ex_api=self.dic_agent_conf["WITH_EXTERNAL_API"],
                    model=agent_name.split("-")[1],
                    intersection=self.env.intersection_dict[self.env.list_intersection[i].inter_name],
                    inter_name=self.env.list_intersection[i].inter_name,
                    phase_num=len(self.env.list_intersection[i].list_phases),
                    log_dir=self.dic_agent_conf["LOG_DIR"],
                    dataset=f"{self.roadnet}-{self.trafficflow}"
                )
            else:
                agent = DIC_AGENTS[agent_name](
                    dic_agent_conf=self.dic_agent_conf,
                    dic_traffic_env_conf=self.dic_traffic_env_conf,
                    dic_path=self.dic_path,
                    cnt_round=0,
                    intersection_id=str(i)
                )
            self.agents.append(agent)

    # 训练方法，执行一轮仿真，记录奖励、排队长度、等待时间、通行时间等，并用wandb记录实验结果
    def train(self, round):
        # 打印训练开始标志
        print("================ start train ================")
        # 获取本轮仿真的最大步数
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # 初始化输出文件路径
        file_name_memory = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "memories.txt")
        # 标记仿真是否结束
        done = False
        # 重置环境，获取初始状态
        state = self.env.reset()
        # 累计奖励
        total_reward = 0.0
        # 每步排队长度记录
        queue_length_episode = []
        # 每步等待时间记录
        waiting_time_episode = []
        # 步数计数器
        step_num = 0
        # 打印环境重置结束
        print("end reset")
        # 获取当前仿真时间（秒）
        current_time = self.env.get_current_time()  # in seconds

        # 合并所有配置，用于wandb记录
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        # 初始化wandb日志记录器
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"round_{round}",
            config=all_config,
        )

        # 记录训练开始时间
        start_time = time.time()
        # 初始化每个路口的状态-动作日志
        state_action_log = [[] for _ in range(len(state))]
        # 主循环：只要仿真未结束且未达到最大步数
        while not done and current_time < total_run_cnt:
            # 当前步所有路口的动作列表
            action_list = []
            # 多线程动作选择线程列表
            threads = []

            # 遍历每个路口，选择动作
            for i in range(len(state)):
                # 获取当前路口对象
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                # 深拷贝路口道路信息
                roads = deepcopy(intersection["roads"])
                # 获取统计状态、入口状态、平均速度
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                # 记录状态信息到日志
                state_action_log[i].append({"state": statistic_state, "state_incoming": statistic_state_incoming, "approaching_speed": mean_speed})

                # 当前路口状态
                one_state = state[i]
                # 当前步数
                count = step_num
                # 如果是ChatGPT/open_llm模型，使用多线程选择动作
                if "ChatGPT" in self.dic_traffic_env_conf["MODEL_NAME"] or "open_llm" in self.dic_traffic_env_conf["MODEL_NAME"]:
                    thread = threading.Thread(target=self.agents[i].choose_action, args=(self.env,))
                    threads.append(thread)
                else:
                    # 其它模型直接选择动作
                    if self.dic_traffic_env_conf["MODEL_NAME"] in ["EfficientPressLight", "EfficientColight",
                                                                   "EfficientMPLight", "Attend",
                                                                   "AdvancedMPLight", "AdvancedColight", "AdvancedDQN"]:
                        if i == 0:
                            action_list = self.agents[i].choose_action(count, state)
                    else:
                        action = self.agents[i].choose_action(count, one_state)
                        action_list.append(action)

            # ChatGPT模型多线程动作选择
            if "ChatGPT" in self.dic_traffic_env_conf["MODEL_NAME"]:
                for thread in threads:
                    thread.start()

                for thread in tqdm(threads):
                    thread.join()

                for i in range(len(state)):
                    action = self.agents[i].temp_action_logger
                    action_list.append(action)

            # open_llm模型多线程动作选择
            if "open_llm" in self.dic_traffic_env_conf["MODEL_NAME"]:
                started_thread_id = []
                # 线程数根据配置决定
                thread_num = self.dic_traffic_env_conf["LLM_API_THREAD_NUM"] if not self.dic_agent_conf["WITH_EXTERNAL_API"] else 2
                for i, thread in enumerate(tqdm(threads)):
                    thread.start()
                    started_thread_id.append(i)

                    # 控制并发线程数
                    if (i + 1) % thread_num == 0:
                        for t_id in started_thread_id:
                            threads[t_id].join()
                        started_thread_id = []

                for i in range(len(state)):
                    action = self.agents[i].temp_action_logger
                    action_list.append(action)

            # 执行动作，环境推进一步，获得新状态、奖励、done标志
            next_state, reward, done, _ = self.env.step(action_list)

            # 记录本步动作到日志
            for i in range(len(state)):
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            # 打开记忆文件，追加写入
            f_memory = open(file_name_memory, "a")
            # 构造本步信息字符串
            memory_str = 'time = {0}\taction = {1}\tcurrent_phase = {2}\treward = {3}'.\
                format(current_time, str(action_list), str([state[i]["cur_phase"][0] for i in range(len(state))]),
                       str(reward),)
            # 写入文件
            f_memory.write(memory_str + "\n")
            f_memory.close()
            # 更新当前仿真时间
            current_time = self.env.get_current_time()  # in seconds

            # 更新状态为下一步
            state = next_state
            # 步数加一
            step_num += 1

            # 累加奖励
            total_reward += sum(reward)
            # 统计所有路口排队车辆数
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            # 记录本步总排队长度
            queue_length_episode.append(sum(queue_length_inter))

            # 统计所有车辆等待时间
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            # 记录本步平均等待时间
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

        # 统计所有车辆的通行时间
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        # 计算平均通行时间
        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        # 汇总本轮仿真结果
        results = {
            "test_reward_over": total_reward,
            "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_travel_time_over": total_travel_time}
        # 记录到wandb
        logger.log(results)
        # 打印结果
        print(results)
        # 保存状态-动作日志到json文件
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)
        # 结束wandb记录
        wandb.finish()

        # 打印训练耗时
        print("Training time: ", time.time()-start_time)

        # 批量日志处理
        self.env.batch_log_2()

        # 返回仿真结果
        return results
