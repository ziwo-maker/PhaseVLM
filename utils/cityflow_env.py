import pickle
import numpy as np
import json
import sys
import pandas as pd
import os
import csv
import cityflow as engine
import time
from datetime import datetime, timedelta
from multiprocessing import Process
from .my_utils import load_json, calculate_road_length
from functools import reduce
from typing import Dict, List, Optional, Set, Tuple

location_dict = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_dict_reverse = {"N": "North", "S": "South", "E": "East", "W": "West"}
direction_dict = {"go_straight": "T", "turn_left": "L", "turn_right": "R"}

class Intersection:
    def __init__(self, inter_id, dic_traffic_env_conf, eng, light_id_dict, path_to_log, lanes_length_dict):
        self.inter_id = inter_id
        self.inter_name = "intersection_{0}_{1}".format(inter_id[0], inter_id[1])
        self.eng = eng
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.lane_length = lanes_length_dict
        self.obs_length = dic_traffic_env_conf["OBS_LENGTH"]

        self.list_approachs = ["W", "E", "N", "S"]
        # corresponding exiting lane for entering lanes
        self.dic_approach_to_node = {"W": 0, "E": 2, "S": 1, "N": 3}
        self.dic_entering_approach_to_edge = {"W": "road_{0}_{1}_0".format(inter_id[0] - 1, inter_id[1])}
        self.dic_entering_approach_to_edge.update({"E": "road_{0}_{1}_2".format(inter_id[0] + 1, inter_id[1])})
        self.dic_entering_approach_to_edge.update({"N": "road_{0}_{1}_3".format(inter_id[0], inter_id[1] + 1)})
        self.dic_entering_approach_to_edge.update({"S": "road_{0}_{1}_1".format(inter_id[0], inter_id[1] - 1)})
        self.dic_exiting_approach_to_edge = {
            approach: "road_{0}_{1}_{2}".format(inter_id[0], inter_id[1], self.dic_approach_to_node[approach]) for
            approach in self.list_approachs}
        self.list_phases = dic_traffic_env_conf["PHASE"]

        # generate all lanes
        self.list_entering_lanes = []
        for (approach, lane_number) in zip(self.list_approachs, dic_traffic_env_conf["NUM_LANES"]):
            self.list_entering_lanes += [self.dic_entering_approach_to_edge[approach] + "_" + str(i) for i in
                                         range(lane_number)]
        self.list_exiting_lanes = []
        for (approach, lane_number) in zip(self.list_approachs, dic_traffic_env_conf["NUM_LANES"]):
            self.list_exiting_lanes += [self.dic_exiting_approach_to_edge[approach] + "_" + str(i) for i in
                                        range(lane_number)]

        self.list_lanes = self.list_entering_lanes + self.list_exiting_lanes

        self.adjacency_row = light_id_dict["adjacency_row"]
        self.neighbor_ENWS = light_id_dict["neighbor_ENWS"]

        # ========== record previous & current feats ==========
        self.dic_lane_vehicle_previous_step = {}
        self.dic_lane_vehicle_previous_step_in = {}
        self.dic_lane_waiting_vehicle_count_previous_step = {}
        self.dic_vehicle_speed_previous_step = {}
        self.dic_vehicle_distance_previous_step = {}

        # in [entering_lanes] out [exiting_lanes]
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        self.dic_vehicle_speed_current_step = {}
        self.dic_vehicle_distance_current_step = {}

        self.list_lane_vehicle_previous_step_in = []
        self.list_lane_vehicle_current_step_in = []

        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        self.dic_feature = {}  # this second
        self.dic_feature_previous_step = {}  # this second

        # =========== signal info set ================
        # -1: all yellow, -2: all red, -3: none
        self.all_yellow_phase_index = -1
        self.all_red_phase_index = -2

        self.current_phase_index = 1
        self.previous_phase_index = 1
        self.eng.set_tl_phase(self.inter_name, self.current_phase_index)
        path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
        df = [self.get_current_time(), self.current_phase_index]
        df = pd.DataFrame(df)
        df = df.transpose()
        df.to_csv(path_to_log_file, mode="a", header=False, index=False)

        self.next_phase_to_set_index = None
        self.current_phase_duration = -1
        self.all_red_flag = False
        self.all_yellow_flag = False
        self.flicker = 0

    def set_signal(self, action, action_pattern, yellow_time, path_to_log):
        # ensure action is an int when it's passed as a string (some agents may return stringified numbers)
        if isinstance(action, str):
            try:
                action = int(action)
            except ValueError:
                pass
        if self.all_yellow_flag:
            # in yellow phase
            self.flicker = 0
            if self.current_phase_duration >= yellow_time:  # yellow time reached
                self.current_phase_index = self.next_phase_to_set_index
                self.eng.set_tl_phase(self.inter_name, self.current_phase_index)  # if multi_phase, need more adjustment
                path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode="a", header=False, index=False)
                self.all_yellow_flag = False
        else:
            # determine phase
            if action_pattern == "switch":  # switch by order
                if action == 0:  # keep the phase
                    self.next_phase_to_set_index = self.current_phase_index
                elif action == 1:  # change to the next phase
                    self.next_phase_to_set_index = (self.current_phase_index + 1) % len(self.list_phases)
                    # if multi_phase, need more adjustment
                else:
                    sys.exit("action not recognized\n action must be 0 or 1")

            elif action_pattern == "set":  # set to certain phase
                # self.next_phase_to_set_index = self.DIC_PHASE_MAP[action] # if multi_phase, need more adjustment
                self.next_phase_to_set_index = action + 1
            # set phase
            if self.current_phase_index == self.next_phase_to_set_index:
                # the light phase keeps unchanged
                pass
            else:  # the light phase needs to change
                # change to yellow first, and activate the counter and flag
                self.eng.set_tl_phase(self.inter_name, 0)  # !!! yellow, tmp
                path_to_log_file = os.path.join(path_to_log, "signal_inter_{0}.txt".format(self.inter_name))
                df = [self.get_current_time(), self.current_phase_index]
                df = pd.DataFrame(df)
                df = df.transpose()
                df.to_csv(path_to_log_file, mode="a", header=False, index=False)
                self.current_phase_index = self.all_yellow_phase_index
                self.all_yellow_flag = True
                self.flicker = 1

    # update inner measurements
    def update_previous_measurements(self):
        self.previous_phase_index = self.current_phase_index
        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step
        self.dic_lane_vehicle_previous_step_in = self.dic_lane_vehicle_current_step_in
        self.dic_lane_waiting_vehicle_count_previous_step = self.dic_lane_waiting_vehicle_count_current_step
        self.dic_vehicle_speed_previous_step = self.dic_vehicle_speed_current_step
        self.dic_vehicle_distance_previous_step = self.dic_vehicle_distance_current_step

    def update_current_measurements(self, simulator_state):
        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []
            for value in dic_lane_vehicle.values():
                list_lane_vehicle.extend(value)
            return list_lane_vehicle

        if self.current_phase_index == self.previous_phase_index:
            self.current_phase_duration += 1
        else:
            self.current_phase_duration = 1

        self.dic_lane_vehicle_current_step = {}
        self.dic_lane_vehicle_current_step_in = {}
        self.dic_lane_waiting_vehicle_count_current_step = {}
        for lane in self.list_entering_lanes:
            self.dic_lane_vehicle_current_step_in[lane] = simulator_state["get_lane_vehicles"][lane]

        for lane in self.list_lanes:
            self.dic_lane_vehicle_current_step[lane] = simulator_state["get_lane_vehicles"][lane]
            self.dic_lane_waiting_vehicle_count_current_step[lane] = simulator_state["get_lane_waiting_vehicle_count"][lane]

        self.dic_vehicle_speed_current_step = simulator_state["get_vehicle_speed"]
        self.dic_vehicle_distance_current_step = simulator_state["get_vehicle_distance"]

        # get vehicle list
        self.list_lane_vehicle_current_step_in = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step_in)
        self.list_lane_vehicle_previous_step_in = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step_in)

        list_vehicle_new_arrive = list(set(self.list_lane_vehicle_current_step_in) - set(self.list_lane_vehicle_previous_step_in))
        # can't use empty set to - real set
        if not self.list_lane_vehicle_previous_step_in:  # previous step is empty
            list_vehicle_new_left = list(set(self.list_lane_vehicle_current_step_in) -
                                         set(self.list_lane_vehicle_previous_step_in))
        else:
            list_vehicle_new_left = list(set(self.list_lane_vehicle_previous_step_in) -
                                         set(self.list_lane_vehicle_current_step_in))
        # update vehicle arrive and left time
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)
        # update feature
        self._update_feature()

    def _update_leave_entering_approach_vehicle(self):
        list_entering_lane_vehicle_left = []
        # update vehicles leaving entering lane
        if not self.dic_lane_vehicle_previous_step:  # the dict is not empty
            for _ in self.list_entering_lanes:
                list_entering_lane_vehicle_left.append([])
        else:
            last_step_vehicle_id_list = []
            current_step_vehilce_id_list = []
            for lane in self.list_entering_lanes:
                last_step_vehicle_id_list.extend(self.dic_lane_vehicle_previous_step[lane])
                current_step_vehilce_id_list.extend(self.dic_lane_vehicle_current_step[lane])

            list_entering_lane_vehicle_left.append(
                list(set(last_step_vehicle_id_list) - set(current_step_vehilce_id_list))
            )
        return list_entering_lane_vehicle_left

    def _update_arrive_time(self, list_vehicle_arrive):
        ts = self.get_current_time()
        # get dic vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = {"enter_time": ts, "leave_time": np.nan}

    def _update_left_time(self, list_vehicle_left):
        ts = self.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
            except KeyError:
                print("vehicle not recorded when entering")
                sys.exit(-1)

    def _update_feature(self):
        dic_feature = dict()
        dic_feature["cur_phase"] = [self.current_phase_index]
        dic_feature["time_this_phase"] = [self.current_phase_duration]
        dic_feature["lane_num_vehicle"] = self._get_lane_num_vehicle_entring()
        dic_feature["lane_num_vehicle_downstream"] = self._get_lane_num_vehicle_downstream()
        dic_feature["delta_lane_num_vehicle"] = [dic_feature["lane_num_vehicle"][i] -
                                                 dic_feature["lane_num_vehicle_downstream"][i]
                                                 for i in range(12)]
        dic_feature["lane_num_waiting_vehicle_in"] = self._get_lane_queue_length(self.list_entering_lanes)
        dic_feature["lane_num_waiting_vehicle_out"] = self._get_lane_queue_length(self.list_exiting_lanes)

        dic_feature["traffic_movement_pressure_queue"] = self._get_traffic_movement_pressure_general(
            dic_feature["lane_num_waiting_vehicle_in"], dic_feature["lane_num_waiting_vehicle_out"])

        dic_feature["traffic_movement_pressure_queue_efficient"] = self._get_traffic_movement_pressure_efficient(
            dic_feature["lane_num_waiting_vehicle_in"], dic_feature["lane_num_waiting_vehicle_out"])

        dic_feature["traffic_movement_pressure_num"] = self._get_traffic_movement_pressure_general(
            dic_feature["lane_num_vehicle"], dic_feature["lane_num_vehicle_downstream"])

        tmp_part_n, tmp_part_q, tmp_efficient_part, enter_running_part, lepq = self._get_part_traffic_movement_features()

        dic_feature["lane_enter_running_part"] = list(enter_running_part)

        dic_feature["pressure"] = self._get_pressure()
        dic_feature["adjacency_matrix"] = self._get_adjacency_row()

        dic_feature["num_in_seg_attend"] = self._orgnize_several_segments_attend(dic_feature["lane_num_waiting_vehicle_in"],
                                                                                 dic_feature["lane_num_waiting_vehicle_out"])
        self.dic_feature = dic_feature

    def _orgnize_several_segments_attend(self, queue_in, queue_out):
        part1, part2, part3 = self._get_several_segments_attend(lane_vehicles=self.dic_lane_vehicle_current_step,
                                                                vehicle_distance=self.dic_vehicle_distance_current_step,
                                                                vehicle_speed=self.dic_vehicle_speed_current_step,
                                                                lane_length=self.lane_length,
                                                                list_lanes=self.list_lanes)
        run_in_part1 = [float(len(part1[lane])) for lane in self.list_entering_lanes]
        run_in_part2 = [float(len(part2[lane])) for lane in self.list_entering_lanes]
        run_in_part3 = [float(len(part3[lane])) for lane in self.list_entering_lanes]

        run_out_part1 = [float(len(part1[lane])) for lane in self.list_exiting_lanes]
        run_out_part2 = [float(len(part2[lane]))for lane in self.list_exiting_lanes]
        run_out_part3 = [float(len(part3[lane])) for lane in self.list_exiting_lanes]

        total_in, total_out = [], []
        for i in range(12):
            total_in.extend([run_in_part1[i], run_in_part2[i], run_in_part3[i], queue_in[i]])
            total_out.extend([run_out_part1[i], run_out_part2[i], run_out_part3[i], queue_out[i]])
        return total_in + total_out

    def _get_several_segments_attend(self, lane_vehicles, vehicle_distance, vehicle_speed,
                                           lane_length, list_lanes):
        obs_length = 100
        part1, part2, part3 = {}, {}, {}
        for lane in list_lanes:
            part1[lane], part2[lane], part3[lane] = [], [], []
            for vehicle in lane_vehicles[lane]:
                # set as num_vehicle
                if "shadow" in vehicle:  # remove the shadow
                    vehicle = vehicle[:-7]
                    continue
                if vehicle_speed[vehicle] > 0.1:
                    temp_v_distance = vehicle_distance[vehicle]
                    if temp_v_distance > lane_length[lane] - obs_length:
                        part1[lane].append(vehicle)
                    elif lane_length[lane] - 2 * obs_length < temp_v_distance <= lane_length[lane] - obs_length:
                        part2[lane].append(vehicle)
                    elif lane_length[lane] - 3 * obs_length < temp_v_distance <= lane_length[lane] - 2 * obs_length:
                        part3[lane].append(vehicle)
        return part1, part2, part3

    @staticmethod
    def _get_traffic_movement_pressure_general(enterings, exitings):
        """
            Created by LiangZhang
            Calculate pressure with entering and exiting vehicles
            only for 3 x 3 lanes intersection
        """
        list_approachs = ["W", "E", "N", "S"]
        index_maps = {
            "W": [0, 1, 2],
            "E": [3, 4, 5],
            "N": [6, 7, 8],
            "S": [9, 10, 11]
        }
        # vehicles in exiting road
        outs_maps = {}
        for approach in list_approachs:
            outs_maps[approach] = sum([exitings[i] for i in index_maps[approach]])
        turn_maps = ["S", "W", "N", "N", "E", "S", "W", "N", "E", "E", "S", "W"]
        t_m_p = [enterings[j] - outs_maps[turn_maps[j]] for j in range(12)]
        return t_m_p

    @staticmethod
    def _get_traffic_movement_pressure_efficient(enterings, exitings):
        """
            Created by LiangZhang
            Calculate pressure with entering and exiting vehicles
            only for 3 x 3 lanes intersection
        """
        list_approachs = ["W", "E", "N", "S"]
        index_maps = {
            "W": [0, 1, 2],
            "E": [3, 4, 5],
            "N": [6, 7, 8],
            "S": [9, 10, 11]
        }
        # vehicles in exiting road
        outs_maps = {}
        for approach in list_approachs:
            outs_maps[approach] = sum([exitings[i] for i in index_maps[approach]])
        turn_maps = ["S", "W", "N", "N", "E", "S", "W", "N", "E", "E", "S", "W"]
        t_m_p = [enterings[j] - outs_maps[turn_maps[j]]/3 for j in range(12)]
        return t_m_p

    def _get_part_traffic_movement_features(self):
        """
        return: part_traffic_movement_pressure_num:     both the end and the beginning of the lane
                part_patrric_movement_pressure_queue:   all at the end of the road
                part_entering_running_vehicles:         part obs of the running vehicles
        """
        f_p_num, l_p_num, l_p_q = self._get_part_observations(lane_vehicles=self.dic_lane_vehicle_current_step,
                                                              vehicle_distance=self.dic_vehicle_distance_current_step,
                                                              vehicle_speed=self.dic_vehicle_speed_current_step,
                                                              lane_length=self.lane_length,
                                                              obs_length=self.obs_length,
                                                              list_lanes=self.list_lanes)
        """calculate traffic_movement_pressure with part queue"""
        list_entering_part_queue = [len(l_p_q[lane]) for lane in self.list_entering_lanes]
        list_exiting_part_queue = [len(l_p_q[lane]) for lane in self.list_exiting_lanes]
        tmp_queue_efficient_part = self._get_traffic_movement_pressure_efficient(list_entering_part_queue,
                                                                                 list_exiting_part_queue)
        tmp_queue_part = self._get_traffic_movement_pressure_general(list_entering_part_queue,
                                                                     list_exiting_part_queue)

        """calculate traffic_movement_pressure with part num vehicle"""
        # entering
        list_entering_num_f = [len(f_p_num[lane]) for lane in self.list_entering_lanes]
        list_entering_num_l = [len(l_p_num[lane]) for lane in self.list_entering_lanes]
        entering_num = np.array(list_entering_num_f) + np.array(list_entering_num_l)
        # exiting
        list_exiting_num_f = [len(f_p_num[lane]) for lane in self.list_exiting_lanes]
        list_exiting_num_l = [len(l_p_num[lane]) for lane in self.list_exiting_lanes]
        exiting_num = np.array(list_exiting_num_f) + np.array(list_exiting_num_l)
        traffic_movement_pressure_nums = self._get_traffic_movement_pressure_general(entering_num, exiting_num)
        # part of entering running vehicles, all at the end of the road
        part_entering_running = np.array(list_entering_num_l) - np.array(list_entering_part_queue)

        return traffic_movement_pressure_nums, tmp_queue_part, tmp_queue_efficient_part, part_entering_running, list_entering_part_queue

    @staticmethod
    def _get_part_observations(lane_vehicles, vehicle_distance, vehicle_speed,
                               lane_length, obs_length, list_lanes):
        """
            Input: lane_vehicles :      Dict{lane_id    :   [vehicle_ids]}
                   vehicle_distance:    Dict{vehicle_id :   float(dist)}
                   vehicle_speed:       Dict{vehicle_id :   float(speed)}
                   lane_length  :       Dict{lane_id    :   float(length)}
                   obs_length   :       The part observation length
                   list_lanes   :       List[lane_ids at the intersection]
        :return:
                    part_vehicles:      Dict{ lane_id, [vehicle_ids]}
        """
        # get vehicle_ids and speeds
        first_part_num_vehicle = {}
        first_part_queue_vehicle = {}  # useless, at the begin of lane, there is no waiting vechiles
        last_part_num_vehicle = {}
        last_part_queue_vehicle = {}

        for lane in list_lanes:
            first_part_num_vehicle[lane] = []
            first_part_queue_vehicle[lane] = []
            last_part_num_vehicle[lane] = []
            last_part_queue_vehicle[lane] = []
            last_part_obs_length = lane_length[lane] - obs_length
            for vehicle in lane_vehicles[lane]:
                """ get the first part of obs
                    That is vehicle_distance <= obs_length 
                """
                # set as num_vehicle
                if "shadow" in vehicle:  # remove the shadow
                    vehicle = vehicle[:-7]
                temp_v_distance = vehicle_distance[vehicle]
                if temp_v_distance <= obs_length:
                    first_part_num_vehicle[lane].append(vehicle)
                    # analyse if waiting
                    if vehicle_speed[vehicle] <= 0.1:
                        first_part_queue_vehicle[lane].append(vehicle)

                """ get the last part of obs
                    That is  lane_length-obs_length <= vehicle_distance <= lane_length 
                """
                if temp_v_distance >= last_part_obs_length:
                    last_part_num_vehicle[lane].append(vehicle)
                    # analyse if waiting
                    if vehicle_speed[vehicle] <= 0.1:
                        last_part_queue_vehicle[lane].append(vehicle)

        return first_part_num_vehicle, last_part_num_vehicle, last_part_queue_vehicle

    def _get_pressure(self):
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_entering_lanes] + \
               [-self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in self.list_exiting_lanes]

    def _get_lane_queue_length(self, list_lanes):
        """
        queue length for each lane
        """
        return [self.dic_lane_waiting_vehicle_count_current_step[lane] for lane in list_lanes]

    def _get_lane_num_vehicle_entring(self):
        """
        vehicle number for each lane
        """
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in self.list_entering_lanes]

    def _get_lane_num_vehicle_downstream(self):
        """
        vehicle number for each lane, exiting
        """
        return [len(self.dic_lane_vehicle_current_step[lane]) for lane in self.list_exiting_lanes]

    # ================= get functions from outside ======================
    def get_current_time(self):
        return self.eng.get_current_time()

    def get_dic_vehicle_arrive_leave_time(self):
        return self.dic_vehicle_arrive_leave_time

    def get_feature(self):
        return self.dic_feature

    def get_state(self, list_state_features):
        dic_state = {state_feature_name: self.dic_feature[state_feature_name] for
                     state_feature_name in list_state_features}
        return dic_state

    def _get_adjacency_row(self):
        return self.adjacency_row

    def get_reward(self, dic_reward_info):
        dic_reward = dict()
        # dic_reward["sum_lane_queue_length"] = None
        dic_reward["pressure"] = np.absolute(np.sum(self.dic_feature["pressure"]))
        dic_reward["queue_length"] = np.absolute(np.sum(self.dic_feature["lane_num_waiting_vehicle_in"]))
        reward = 0
        for r in dic_reward_info:
            if dic_reward_info[r] != 0:
                reward += dic_reward_info[r] * dic_reward[r]
        return reward


class CityFlowEnv:

    def __init__(self, path_to_log, path_to_work_directory, dic_traffic_env_conf, dic_path):
        self.path_to_log = path_to_log
        self.path_to_work_directory = path_to_work_directory
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self.current_time = None
        self.id_to_index = None
        self.traffic_light_node_dict = None
        self.intersection_dict = None
        self.eng = None
        self.list_intersection = None
        self.list_inter_log = None
        self.list_lanes = None
        self.system_states = None
        self.lane_length = None
        self.waiting_vehicle_list = {}
        self.traffic_count_enabled = False
        self.traffic_count_mode = "road"  # "road" | "intersection_movement"
        self.traffic_count_output_format = "csv"  # "csv" | "jsonl"
        self.traffic_road_ids = []
        self.traffic_road_id_to_index = {}
        self.traffic_count_intervals = []
        self.traffic_counts_by_interval = {}
        self._traffic_prev_road_vehicle_sets = None
        self._traffic_curr_road_vehicle_sets = None
        self.traffic_base_dt = None
        self.traffic_intersection_ids = []
        self.traffic_movement_keys = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
        self._traffic_lane_to_feature_indices = None  # Dict[lane_id, List[col_idx]]

        # check min action time
        if self.dic_traffic_env_conf["MIN_ACTION_TIME"] <= self.dic_traffic_env_conf["YELLOW_TIME"]:
            """ include the yellow time in action time """
            print("MIN_ACTION_TIME should include YELLOW_TIME")
            sys.exit()

        # touch new inter_{}.pkl (if exists, remove)
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            f.close()

    def reset(self):
        print(" ============= self.eng.reset() to be implemented ==========")
        cityflow_config = {
            "interval": self.dic_traffic_env_conf["INTERVAL"],
            "seed": int(np.random.randint(0, 100)),
            "laneChange": True,
            "dir": self.path_to_work_directory+"/",
            "roadnetFile": self.dic_traffic_env_conf["ROADNET_FILE"],
            "flowFile": self.dic_traffic_env_conf["TRAFFIC_FILE"],
            "rlTrafficLight": True,
            "saveReplay": True,  # if "GPT" in self.dic_traffic_env_conf["MODEL_NAME"] or "llm" in self.dic_traffic_env_conf["MODEL_NAME"] else False,
            "roadnetLogFile": f"./{self.dic_traffic_env_conf['ROADNET_FILE']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['MODEL_NAME']}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases-roadnetLogFile.json",
            "replayLogFile": f"./{self.dic_traffic_env_conf['ROADNET_FILE']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['MODEL_NAME']}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases-replayLogFile.txt"
        }
        # print(cityflow_config)
        with open(os.path.join(self.path_to_work_directory, "cityflow.config"), "w") as json_file:
            json.dump(cityflow_config, json_file)

        self.eng = engine.Engine(os.path.join(self.path_to_work_directory, "cityflow.config"), thread_num=1)

        # get adjacency
        self.traffic_light_node_dict = self._adjacency_extraction()

        # get lane length
        _, self.lane_length = self.get_lane_length()

        # initialize intersections (grid)
        self.list_intersection = [Intersection((i+1, j+1), self.dic_traffic_env_conf, self.eng,
                                               self.traffic_light_node_dict["intersection_{0}_{1}".format(i+1, j+1)],
                                               self.path_to_log,
                                               self.lane_length)
                                  for i in range(self.dic_traffic_env_conf["NUM_COL"])
                                  for j in range(self.dic_traffic_env_conf["NUM_ROW"])]
        self.list_inter_log = [[] for _ in range(self.dic_traffic_env_conf["NUM_COL"] *
                                                 self.dic_traffic_env_conf["NUM_ROW"])]

        self.id_to_index = {}
        count = 0
        for i in range(self.dic_traffic_env_conf["NUM_COL"]):
            for j in range(self.dic_traffic_env_conf["NUM_ROW"]):
                self.id_to_index["intersection_{0}_{1}".format(i+1, j+1)] = count
                count += 1

        self.list_lanes = []
        for inter in self.list_intersection:
            self.list_lanes += inter.list_lanes
        self.list_lanes = np.unique(self.list_lanes).tolist()

        # get new measurements
        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance(),
                              }

        self._init_traffic_counter()

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)
        state, done = self.get_state()

        # create roadnet dict
        if self.intersection_dict is None:
            self.create_intersection_dict()

        return state


    def create_intersection_dict(self):
        """
        基于 roadnet 构建 self.intersection_dict（主要用于 LLM/日志侧的路口结构信息）。

        新增字段：
        - intersection["roads"][road_id]["num_lanes"]：该 road 在 roadnet 中的总车道数（len(road["lanes"])）。
        - intersection["movement_lane_counts"]：8 个 movement（每方向直行/左转：NT/NL/.../WL）的车道数量统计。
        """
        roadnet_file = self.dic_traffic_env_conf["ROADNET_FILE"]
        if os.path.isabs(roadnet_file):
            roadnet_path = roadnet_file
        else:
            data_root = self.dic_path.get("PATH_TO_DATA", "")
            roadnet_path = os.path.join(data_root, roadnet_file) if data_root else roadnet_file

        if not os.path.exists(roadnet_path):
            fallback_path = os.path.join(self.path_to_work_directory, os.path.basename(roadnet_file))
            if os.path.exists(fallback_path):
                roadnet_path = fallback_path

        roadnet = load_json(os.path.abspath(roadnet_path))

        intersections_raw = roadnet["intersections"]
        roads_raw = roadnet["roads"]

        agent_intersections = {}

        # init agent intersections
        for i, inter in enumerate(intersections_raw):
            inter_id = inter["id"]
            intersection = None
            for env_inter in self.list_intersection:
                if env_inter.inter_name == inter_id:
                    intersection = env_inter
                    break

            if len(inter['roadLinks']) > 0:
                # collect yellow allowed road links
                yellow_time = None
                phases = inter['trafficLight']['lightphases']
                all_sets = []
                yellow_phase_idx = None
                for p_i, p in enumerate(phases):
                    all_sets.append(set(p['availableRoadLinks']))
                    if p["time"] < 30:
                        yellow_phase_idx = p_i
                        yellow_time = p["time"]
                yellow_allowed_links = reduce(lambda x, y: x & y, all_sets)

                # init intersection
                agent_intersections[inter_id] = {"phases": {"Y": {"time": yellow_time, "idx": yellow_phase_idx}},
                                                 "roads": {}}

                # init roads
                roads = {}
                for r in inter["roads"]:
                    roads[r] = {"location": None, "type": "incoming", "go_straight": None, "turn_left": None,
                                "turn_right": None, "length": None, "max_speed": None,
                                # 该 road 在 roadnet 中的总车道数（len(road["lanes"])）
                                "num_lanes": None,
                                "lanes": {"go_straight": [], "turn_left": [], "turn_right": []}}

                # collect road length speed info & init road location
                road_links = inter["roadLinks"]
                for r in roads_raw:
                    r_id = r["id"]
                    if r_id in roads:
                        roads[r_id]["length"] = calculate_road_length(r["points"])
                        roads[r_id]["max_speed"] = r["lanes"][0]["maxSpeed"]
                        roads[r_id]["num_lanes"] = len(r.get("lanes", []) or [])
                        for env_road_location in intersection.dic_entering_approach_to_edge:
                            if intersection.dic_entering_approach_to_edge[env_road_location] == r_id:
                                roads[r_id]["location"] = location_dict_reverse[env_road_location]
                                break
                        for env_road_location in intersection.dic_exiting_approach_to_edge:
                            if intersection.dic_exiting_approach_to_edge[env_road_location] == r_id:
                                roads[r_id]["location"] = location_dict_reverse[env_road_location]
                                break

                # collect signal phase info
                for p_idx, p in enumerate(phases):
                    other_allowed_links = set(p['availableRoadLinks']) - yellow_allowed_links
                    if len(other_allowed_links) > 0:
                        allowed_directions = []
                        for l_idx in other_allowed_links:
                            link = road_links[l_idx]
                            location = roads[link["startRoad"]]["location"]
                            direction = link["type"]
                            allowed_directions.append(f"{location_dict[location]}{direction_dict[direction]}")
                        allowed_directions = sorted(allowed_directions)
                        allowed_directions = f"{allowed_directions[0]}{allowed_directions[1]}"
                        agent_intersections[inter_id]["phases"][allowed_directions] = {"time": p["time"], "idx": p_idx}

                # collect location type direction info
                for r_link in road_links:
                    start = r_link['startRoad']
                    end = r_link['endRoad']
                    lane_links = r_link['laneLinks']

                    for r in roads:
                        if r != start:
                            continue
                        # collect type
                        roads[r]["type"] = "outgoing"

                        # collect directions
                        if r_link["type"] == "go_straight":
                            roads[r]["go_straight"] = end

                            # collect lane info
                            for l_link in lane_links:
                                lane_id = l_link['startLaneIndex']
                                if lane_id not in roads[r]["lanes"]["go_straight"]:
                                    roads[r]["lanes"]["go_straight"].append(lane_id)

                        elif r_link["type"] == "turn_left":
                            roads[r]["turn_left"] = end

                            # collect lane info
                            for l_link in lane_links:
                                lane_id = l_link['startLaneIndex']
                                if lane_id not in roads[r]["lanes"]["turn_left"]:
                                    roads[r]["lanes"]["turn_left"].append(lane_id)

                        elif r_link["type"] == "turn_right":
                            roads[r]["turn_right"] = end

                            # collect lane info
                            for l_link in lane_links:
                                lane_id = l_link['startLaneIndex']
                                if lane_id not in roads[r]["lanes"]["turn_right"]:
                                    roads[r]["lanes"]["turn_right"].append(lane_id)

                # 额外记录：按“8个 movement(每个方向的直行/左转)”统计车道数量，便于 LLM/日志侧使用。
                # 说明：一个路口在 roadnet 中通常有 8 条 road（4 进 4 出）。
                # 这里的 movement 统计基于 roadLinks 的 startRoad（即进入路口的道路），
                # 因此会得到 NT/NL/ST/SL/ET/EL/WT/WL 共 8 个数。
                for r_id in roads:
                    roads[r_id]["lanes"]["go_straight"] = sorted(roads[r_id]["lanes"]["go_straight"])
                    roads[r_id]["lanes"]["turn_left"] = sorted(roads[r_id]["lanes"]["turn_left"])
                    roads[r_id]["lanes"]["turn_right"] = sorted(roads[r_id]["lanes"]["turn_right"])

                movement_lane_counts = {k: 0 for k in ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]}
                for r_id, r_info in roads.items():
                    location = r_info.get("location")
                    if not location:
                        continue
                    # 只对“有 movement lane 记录的道路”做统计（通常是 startRoad/进入路口的道路）
                    if not (r_info["lanes"]["go_straight"] or r_info["lanes"]["turn_left"] or r_info["lanes"]["turn_right"]):
                        continue
                    movement_lane_counts[f"{location_dict[location]}T"] = len(r_info["lanes"]["go_straight"])
                    movement_lane_counts[f"{location_dict[location]}L"] = len(r_info["lanes"]["turn_left"])

                agent_intersections[inter_id]["roads"] = roads
                agent_intersections[inter_id]["movement_lane_counts"] = movement_lane_counts

        self.intersection_dict = agent_intersections

    def _init_traffic_counter(self):
        # 读取配置并初始化“按道路统计过车数”的计数器。
        # 逻辑：加载路网 roadnet 获取 road_id 列表；解析统计时间间隔；
        # 为每个 road_id 建立车辆集合缓存与时间桶计数矩阵。
        self.traffic_count_enabled = bool(self.dic_traffic_env_conf.get("ENABLE_TRAFFIC_COUNT", False))
        if not self.traffic_count_enabled:
            self.traffic_count_intervals = []
            self.traffic_counts_by_interval = {}
            self._traffic_prev_road_vehicle_sets = None
            self._traffic_curr_road_vehicle_sets = None
            return

        self.traffic_count_mode = str(self.dic_traffic_env_conf.get("TRAFFIC_COUNT_MODE", "road")).strip()
        self.traffic_count_output_format = str(
            self.dic_traffic_env_conf.get(
                "TRAFFIC_COUNT_OUTPUT_FORMAT",
                "jsonl" if self.traffic_count_mode == "intersection_movement" else "csv",
            )
        ).strip()

        # 加载 roadnet，拿到所有 road_id 作为统计维度
        roadnet_path = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        try:
            with open(roadnet_path) as json_data:
                roadnet = json.load(json_data)
        except Exception as exc:
            print(f"Traffic count disabled: failed to load roadnet ({exc})")
            self.traffic_count_enabled = False
            return

        # 解析统计周期（秒），支持单值或列表
        intervals = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_INTERVALS", None)
        if intervals is None:
            intervals = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_INTERVAL_SECONDS", [60, 3600])
        if isinstance(intervals, int):
            intervals = [intervals]
        intervals = [int(x) for x in intervals if int(x) > 0]
        if not intervals:
            print("Traffic count disabled: no valid intervals configured.")
            self.traffic_count_enabled = False
            return

        total_run = int(self.dic_traffic_env_conf.get("RUN_COUNTS", 0))
        self.traffic_count_intervals = sorted(set(intervals))
        self.traffic_counts_by_interval = {}

        if self.traffic_count_mode == "intersection_movement":
            self._init_traffic_counter_intersection_movement(roadnet, total_run)
        else:
            self.traffic_count_mode = "road"
            self._init_traffic_counter_road(roadnet, total_run)

        # 计数 CSV 的时间戳起点
        base_date = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_BASE_DATE", "1970-01-01 00:00:00")
        try:
            self.traffic_base_dt = datetime.strptime(base_date, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print(f"Invalid TRAFFIC_COUNT_BASE_DATE: {base_date}, defaulting to 1970-01-01 00:00:00")
            self.traffic_base_dt = datetime(1970, 1, 1)

    def _init_traffic_counter_road(self, roadnet, total_run: int):
        self.traffic_road_ids = [road["id"] for road in roadnet.get("roads", [])]
        if not self.traffic_road_ids:
            print("Traffic count disabled: no roads found in roadnet.")
            self.traffic_count_enabled = False
            return

        self.traffic_intersection_ids = []
        self._traffic_lane_to_feature_indices = None

        self.traffic_road_id_to_index = {road_id: idx for idx, road_id in enumerate(self.traffic_road_ids)}
        num_roads = len(self.traffic_road_ids)
        for interval in self.traffic_count_intervals:
            num_bins = max(1, (total_run + interval - 1) // interval) if total_run else 1
            self.traffic_counts_by_interval[interval] = np.zeros((num_bins, num_roads), dtype=int)
        self._traffic_prev_road_vehicle_sets = [set() for _ in range(num_roads)]
        self._traffic_curr_road_vehicle_sets = [set() for _ in range(num_roads)]

    def _init_traffic_counter_intersection_movement(self, roadnet, total_run: int):
        self.traffic_road_ids = []
        self.traffic_road_id_to_index = {}
        self._traffic_prev_road_vehicle_sets = None
        self._traffic_curr_road_vehicle_sets = None

        movement_keys = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_MOVEMENT_KEYS", None)
        if movement_keys is not None:
            self.traffic_movement_keys = [str(k).strip() for k in movement_keys if str(k).strip()]

        env_intersections = {inter.inter_name: inter for inter in (self.list_intersection or [])}
        if not env_intersections:
            print("Traffic count disabled: intersections not initialized.")
            self.traffic_count_enabled = False
            return

        # 交叉口 -> movement -> lane_id 列表
        inter_to_movement_lanes: Dict[str, Dict[str, List[str]]] = {}
        for inter in roadnet.get("intersections", []) or []:
            inter_id = inter.get("id")
            if not inter_id or inter.get("virtual"):
                continue
            env_inter = env_intersections.get(inter_id)
            if env_inter is None:
                continue

            start_road_to_prefix: Dict[str, str] = {}
            for prefix, road_id in (env_inter.dic_entering_approach_to_edge or {}).items():
                if road_id:
                    start_road_to_prefix[road_id] = prefix  # prefix in {"N","S","E","W"}

            movement_lanes = {k: [] for k in self.traffic_movement_keys}
            for road_link in inter.get("roadLinks", []) or []:
                start_road = road_link.get("startRoad")
                if not start_road or start_road not in start_road_to_prefix:
                    continue
                link_type = road_link.get("type")
                if link_type not in ("go_straight", "turn_left"):
                    continue
                movement_suffix = "T" if link_type == "go_straight" else "L"
                movement_key = f"{start_road_to_prefix[start_road]}{movement_suffix}"
                if movement_key not in movement_lanes:
                    continue
                for lane_link in road_link.get("laneLinks", []) or []:
                    lane_idx = lane_link.get("startLaneIndex")
                    if lane_idx is None:
                        continue
                    movement_lanes[movement_key].append(f"{start_road}_{int(lane_idx)}")

            for k in movement_lanes:
                movement_lanes[k] = sorted(set(movement_lanes[k]))
            inter_to_movement_lanes[inter_id] = movement_lanes

        def _intersection_sort_key(inter_id: str):
            try:
                _, x, y = inter_id.split("_", 2)
                return (int(x), int(y))
            except Exception:
                return (inter_id,)

        self.traffic_intersection_ids = sorted(inter_to_movement_lanes.keys(), key=_intersection_sort_key)
        if not self.traffic_intersection_ids:
            print("Traffic count disabled: no valid intersections found for movement logging.")
            self.traffic_count_enabled = False
            return

        num_features = len(self.traffic_intersection_ids) * len(self.traffic_movement_keys)
        for interval in self.traffic_count_intervals:
            num_bins = max(1, (total_run + interval - 1) // interval) if total_run else 1
            self.traffic_counts_by_interval[interval] = np.zeros((num_bins, num_features), dtype=int)

        lane_to_feature_indices: Dict[str, List[int]] = {}
        for inter_idx, inter_id in enumerate(self.traffic_intersection_ids):
            movement_lanes = inter_to_movement_lanes.get(inter_id) or {}
            base_col = inter_idx * len(self.traffic_movement_keys)
            for mv_i, mv_key in enumerate(self.traffic_movement_keys):
                col_idx = base_col + mv_i
                for lane_id in movement_lanes.get(mv_key, []) or []:
                    lane_to_feature_indices.setdefault(lane_id, []).append(col_idx)
        self._traffic_lane_to_feature_indices = lane_to_feature_indices

    def _update_traffic_counts(self):
        # 核心计数逻辑（两种模式）：
        # - road：lane->road 聚合，写“道路上车辆数”
        # - intersection_movement：lane->(intersection,movement) 聚合，写“路口 8 个 movement 的车辆数”
        if not self.traffic_count_enabled or self.system_states is None:
            return

        total_run = int(self.dic_traffic_env_conf.get("RUN_COUNTS", 0))
        current_time = int(self.get_current_time())
        if total_run and current_time >= total_run:
            return

        if not self.traffic_counts_by_interval:
            return

        interval_bins = {interval: current_time // interval for interval in self.traffic_count_intervals}
        lane_vehicles = self.system_states.get("get_lane_vehicles", {})

        if self.traffic_count_mode == "intersection_movement":
            if not self.traffic_intersection_ids or not self._traffic_lane_to_feature_indices:
                return
            num_features = len(self.traffic_intersection_ids) * len(self.traffic_movement_keys)
            feature_counts = np.zeros((num_features,), dtype=int)
            for lane_id, vehicle_ids in lane_vehicles.items():
                col_idxs = self._traffic_lane_to_feature_indices.get(lane_id)
                if not col_idxs:
                    continue
                c = len(vehicle_ids or [])
                if c <= 0:
                    continue
                for col_idx in col_idxs:
                    feature_counts[col_idx] += c
            for interval, bin_idx in interval_bins.items():
                counts = self.traffic_counts_by_interval.get(interval)
                if counts is None or bin_idx >= counts.shape[0]:
                    continue
                counts[bin_idx, :] = feature_counts
            return

        # road 模式：清空本步集合缓存
        for s in self._traffic_curr_road_vehicle_sets:
            s.clear()
        # 将 lane_id 映射到 road_id（去掉 lane 后缀），按道路聚合车辆
        for lane_id, vehicle_ids in lane_vehicles.items():
            road_id = lane_id.rsplit("_", 1)[0]
            idx = self.traffic_road_id_to_index.get(road_id)
            if idx is None or not vehicle_ids:
                continue
            self._traffic_curr_road_vehicle_sets[idx].update(vehicle_ids)

        for idx, current_set in enumerate(self._traffic_curr_road_vehicle_sets):
            if not current_set:
                continue
            current_count = len(current_set)
            for interval, bin_idx in interval_bins.items():
                counts = self.traffic_counts_by_interval.get(interval)
                if counts is None or bin_idx >= counts.shape[0]:
                    continue
                counts[bin_idx, idx] = current_count

        self._traffic_prev_road_vehicle_sets, self._traffic_curr_road_vehicle_sets = (
            self._traffic_curr_road_vehicle_sets,
            self._traffic_prev_road_vehicle_sets,
        )

    @staticmethod
    def _interval_suffix(interval_seconds):
        if interval_seconds == 60:
            return "minute"
        if interval_seconds == 3600:
            return "hour"
        return f"{interval_seconds}s"

    def _write_traffic_count_csv(self, counts, output_path, interval_seconds):
        if counts is None or not self.traffic_road_ids:
            return
        base_dt = self.traffic_base_dt or datetime(1970, 1, 1)
        dates = [
            (base_dt + timedelta(seconds=i * interval_seconds)).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(counts.shape[0])
        ]
        df = pd.DataFrame(counts, columns=self.traffic_road_ids)
        df.insert(0, "date", dates)
        if self.dic_traffic_env_conf.get("TRAFFIC_COUNT_ADD_TOTAL", True):
            df["OT"] = counts.sum(axis=1)
        df.to_csv(output_path, index=False)

    def _write_traffic_count_intersection_movement_jsonl(self, counts, output_path, interval_seconds):
        if counts is None or not self.traffic_intersection_ids:
            return
        base_dt = self.traffic_base_dt or datetime(1970, 1, 1)
        movement_keys = self.traffic_movement_keys
        num_mv = len(movement_keys)
        with open(output_path, "w", encoding="utf-8") as f:
            for bin_idx in range(counts.shape[0]):
                date_str = (base_dt + timedelta(seconds=bin_idx * interval_seconds)).strftime("%Y-%m-%d %H:%M:%S")
                row = counts[bin_idx, :]
                for inter_idx, inter_id in enumerate(self.traffic_intersection_ids):
                    offset = inter_idx * num_mv
                    movement_counts = {k: int(row[offset + mv_i]) for mv_i, k in enumerate(movement_keys)}
                    record = {
                        "date": date_str,
                        "interval_s": int(interval_seconds),
                        "intersection_id": inter_id,
                        "movement_counts": movement_counts,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def dump_traffic_counts(self):
        # 按 interval 导出交通统计（按道路或按路口-8movement）。
        if not self.traffic_count_enabled:
            return
        output_dir = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_OUTPUT_DIR", self.path_to_work_directory)
        os.makedirs(output_dir, exist_ok=True)
        base_name = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_BASENAME", "traffic_counts")
        for interval in self.traffic_count_intervals:
            suffix = self._interval_suffix(interval)
            if self.traffic_count_mode == "intersection_movement" and self.traffic_count_output_format == "jsonl":
                out_path = os.path.join(output_dir, f"{base_name}_{suffix}.jsonl")
                self._write_traffic_count_intersection_movement_jsonl(
                    self.traffic_counts_by_interval.get(interval), out_path, interval
                )
            else:
                out_path = os.path.join(output_dir, f"{base_name}_{suffix}.csv")
                self._write_traffic_count_csv(self.traffic_counts_by_interval.get(interval), out_path, interval)

    def dump_lane_counts(self):
        """
        导出“车道数量”相关的静态表格（与仿真步无关，方便你直接用 CSV 查看）。

        输出文件（默认写到 PATH_TO_WORK_DIRECTORY）：
        - intersection_movement_lane_counts.csv：每个路口 8 个 movement（NT/NL/.../WL）的车道数量
        - road_num_lanes.csv：每条 road 的总车道数量（roadnet 中 len(road["lanes"])）
        """
        if self.intersection_dict is None:
            self.create_intersection_dict()
        if not self.intersection_dict:
            return

        output_dir = self.dic_traffic_env_conf.get("LANE_COUNT_OUTPUT_DIR", self.path_to_work_directory)
        os.makedirs(output_dir, exist_ok=True)

        movement_keys = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
        inter_out = os.path.join(output_dir, "intersection_movement_lane_counts.csv")
        with open(inter_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["intersection_id", *movement_keys])
            for inter_id in sorted(self.intersection_dict.keys()):
                counts = self.intersection_dict[inter_id].get("movement_lane_counts") or {}
                writer.writerow([inter_id] + [int(counts.get(k, 0) or 0) for k in movement_keys])

        road_out = os.path.join(output_dir, "road_num_lanes.csv")
        with open(road_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["intersection_id", "road_id", "location", "num_lanes"])
            for inter_id in sorted(self.intersection_dict.keys()):
                roads = (self.intersection_dict[inter_id].get("roads") or {}).items()
                for road_id, road_info in sorted(roads):
                    writer.writerow([
                        inter_id,
                        road_id,
                        road_info.get("location"),
                        road_info.get("num_lanes"),
                    ])

    def step(self, action):

        step_start_time = time.time()

        list_action_in_sec = [action]
        list_action_in_sec_display = [action]
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]-1):
            if self.dic_traffic_env_conf["ACTION_PATTERN"] == "switch":
                list_action_in_sec.append(np.zeros_like(action).tolist())
            elif self.dic_traffic_env_conf["ACTION_PATTERN"] == "set":
                list_action_in_sec.append(np.copy(action).tolist())
            list_action_in_sec_display.append(np.full_like(action, fill_value=-1).tolist())

        average_reward_action_list = [0]*len(action)
        for i in range(self.dic_traffic_env_conf["MIN_ACTION_TIME"]):

            action_in_sec = list_action_in_sec[i]
            action_in_sec_display = list_action_in_sec_display[i]

            instant_time = self.get_current_time()
            self.current_time = self.get_current_time()

            before_action_feature = self.get_feature()
            # state = self.get_state()

            if i == 0:
                print("time: {0}".format(instant_time))
                    
            self._inner_step(action_in_sec)

            # get reward
            reward = self.get_reward()
            for j in range(len(reward)):
                average_reward_action_list[j] = (average_reward_action_list[j] * i + reward[j]) / (i + 1)
            self.log(cur_time=instant_time, before_action_feature=before_action_feature, action=action_in_sec_display)
            next_state, done = self.get_state()

        print("Step time: ", time.time() - step_start_time)
        return next_state, reward, done, average_reward_action_list

    def _inner_step(self, action):
        # copy current measurements to previous measurements
        for inter in self.list_intersection:
            inter.update_previous_measurements()
        # set signals
        # multi_intersection decided by action {inter_id: phase}
        for inter_ind, inter in enumerate(self.list_intersection):
            inter.set_signal(
                action=action[inter_ind],
                action_pattern=self.dic_traffic_env_conf["ACTION_PATTERN"],
                yellow_time=self.dic_traffic_env_conf["YELLOW_TIME"],
                path_to_log=self.path_to_log
            )

        # run one step
        for i in range(int(1/self.dic_traffic_env_conf["INTERVAL"])):
            self.eng.next_step()

            # update queuing vehicle info
            vehicle_ids = self.eng.get_vehicles(include_waiting=False)
            for v_id in vehicle_ids:
                v_info = self.eng.get_vehicle_info(v_id)
                speed = float(v_info["speed"])
                if speed < 0.1:
                    if v_id not in self.waiting_vehicle_list:
                        self.waiting_vehicle_list[v_id] = {"time": None, "link": None}
                        self.waiting_vehicle_list[v_id]["time"] = self.dic_traffic_env_conf["INTERVAL"]
                        self.waiting_vehicle_list[v_id]["link"] = v_info['drivable']
                    else:
                        if self.waiting_vehicle_list[v_id]["link"] != v_info['drivable']:
                            self.waiting_vehicle_list[v_id] = {"time": None, "link": None}
                            self.waiting_vehicle_list[v_id]["time"] = self.dic_traffic_env_conf["INTERVAL"]
                            self.waiting_vehicle_list[v_id]["link"] = v_info['drivable']
                        else:
                            self.waiting_vehicle_list[v_id]["time"] += self.dic_traffic_env_conf["INTERVAL"]
                else:
                    if v_id in self.waiting_vehicle_list:
                        self.waiting_vehicle_list.pop(v_id)

                if v_id in self.waiting_vehicle_list and self.waiting_vehicle_list[v_id]["link"] != v_info['drivable']:
                    self.waiting_vehicle_list.pop(v_id)

        self.system_states = {"get_lane_vehicles": self.eng.get_lane_vehicles(),
                              "get_lane_waiting_vehicle_count": self.eng.get_lane_waiting_vehicle_count(),
                              "get_vehicle_speed": self.eng.get_vehicle_speed(),
                              "get_vehicle_distance": self.eng.get_vehicle_distance()
                              }

        for inter in self.list_intersection:
            inter.update_current_measurements(self.system_states)
        # 每个仿真步更新“按道路过车数”统计
        self._update_traffic_counts()

    def get_feature(self):
        list_feature = [inter.get_feature() for inter in self.list_intersection]
        return list_feature

    def get_state(self, list_state_feature=None):
        if list_state_feature is not None:
            list_state = [inter.get_state(list_state_feature) for inter in self.list_intersection]
            done = False
        else:
            list_state = [inter.get_state(self.dic_traffic_env_conf["LIST_STATE_FEATURE"]) for inter in self.list_intersection]
            done = False
        return list_state, done

    def get_reward(self):
        list_reward = [inter.get_reward(self.dic_traffic_env_conf["DIC_REWARD_INFO"]) for inter in self.list_intersection]
        return list_reward

    def get_current_time(self):
        return self.eng.get_current_time()

    def log(self, cur_time, before_action_feature, action):

        for inter_ind in range(len(self.list_intersection)):
            self.list_inter_log[inter_ind].append({"time": cur_time,
                                                   "state": before_action_feature[inter_ind],
                                                   "action": action[inter_ind]})

    def batch_log_2(self):
        """
        Used for model test, only log the vehicle_inter_.csv
        """
        for inter_ind in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")
        self.dump_traffic_counts()
        self.dump_lane_counts()

    def batch_log(self, start, stop):
        for inter_ind in range(start, stop):
            # changed from origin
            if int(inter_ind) % 100 == 0:
                print("Batch log for inter ", inter_ind)
            path_to_log_file = os.path.join(self.path_to_log, "vehicle_inter_{0}.csv".format(inter_ind))
            dic_vehicle = self.list_intersection[inter_ind].get_dic_vehicle_arrive_leave_time()
            df = pd.DataFrame.from_dict(dic_vehicle, orient="index")
            df.to_csv(path_to_log_file, na_rep="nan")
            
            path_to_log_file = os.path.join(self.path_to_log, "inter_{0}.pkl".format(inter_ind))
            f = open(path_to_log_file, "wb")
            pickle.dump(self.list_inter_log[inter_ind], f)
            f.close()

    def bulk_log_multi_process(self, batch_size=100):
        assert len(self.list_intersection) == len(self.list_inter_log)
        if batch_size > len(self.list_intersection):
            batch_size_run = len(self.list_intersection)
        else:
            batch_size_run = batch_size
        process_list = []
        for batch in range(0, len(self.list_intersection), batch_size_run):
            start = batch
            stop = min(batch + batch_size, len(self.list_intersection))
            p = Process(target=self.batch_log, args=(start, stop))
            print("before")
            p.start()
            print("end")
            process_list.append(p)
        print("before join")

        for t in process_list:
            t.join()
        print("end join")

    def _adjacency_extraction(self):
        traffic_light_node_dict = {}
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open("{0}".format(file)) as json_data:
            net = json.load(json_data)
            for inter in net["intersections"]:
                if not inter["virtual"]:
                    traffic_light_node_dict[inter["id"]] = {"location": {"x": float(inter["point"]["x"]),
                                                                         "y": float(inter["point"]["y"])},
                                                            "total_inter_num": None, "adjacency_row": None,
                                                            "inter_id_to_index": None,
                                                            "neighbor_ENWS": None}

            top_k = self.dic_traffic_env_conf["TOP_K_ADJACENCY"]
            total_inter_num = len(traffic_light_node_dict.keys())
            inter_id_to_index = {}

            edge_id_dict = {}
            for road in net["roads"]:
                if road["id"] not in edge_id_dict.keys():
                    edge_id_dict[road["id"]] = {}
                edge_id_dict[road["id"]]["from"] = road["startIntersection"]
                edge_id_dict[road["id"]]["to"] = road["endIntersection"]

            index = 0
            for i in traffic_light_node_dict.keys():
                inter_id_to_index[i] = index
                index += 1

            for i in traffic_light_node_dict.keys():
                location_1 = traffic_light_node_dict[i]["location"]

                row = np.array([0]*total_inter_num)
                # row = np.zeros((self.dic_traffic_env_conf["NUM_ROW"],self.dic_traffic_env_conf["NUM_col"]))
                for j in traffic_light_node_dict.keys():
                    location_2 = traffic_light_node_dict[j]["location"]
                    dist = self._cal_distance(location_1, location_2)
                    row[inter_id_to_index[j]] = dist
                if len(row) == top_k:
                    adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                elif len(row) > top_k:
                    adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                else:
                    adjacency_row_unsorted = [k for k in range(total_inter_num)]
                adjacency_row_unsorted.remove(inter_id_to_index[i])
                traffic_light_node_dict[i]["adjacency_row"] = [inter_id_to_index[i]]+adjacency_row_unsorted
                traffic_light_node_dict[i]["total_inter_num"] = total_inter_num

            for i in traffic_light_node_dict.keys():
                traffic_light_node_dict[i]["total_inter_num"] = inter_id_to_index
                traffic_light_node_dict[i]["neighbor_ENWS"] = []
                for j in range(4):
                    road_id = i.replace("intersection", "road")+"_"+str(j)
                    if edge_id_dict[road_id]["to"] not in traffic_light_node_dict.keys():
                        traffic_light_node_dict[i]["neighbor_ENWS"].append(None)
                    else:
                        traffic_light_node_dict[i]["neighbor_ENWS"].append(edge_id_dict[road_id]["to"])

        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1["x"], loc_dict1["y"]))
        b = np.array((loc_dict2["x"], loc_dict2["y"]))
        return np.sqrt(np.sum((a-b)**2))

    @staticmethod
    def end_cityflow():
        print("============== cityflow process end ===============")

    def get_lane_length(self):
        """
        newly added part for get lane length
        Read the road net file
        Return: dict{lanes} normalized with the min lane length
        """
        file = os.path.join(self.path_to_work_directory, self.dic_traffic_env_conf["ROADNET_FILE"])
        with open(file) as json_data:
            net = json.load(json_data)
        roads = net['roads']
        lanes_length_dict = {}
        lane_normalize_factor = {}

        for road in roads:
            points = road["points"]
            road_length = abs(points[0]['x'] + points[0]['y'] - points[1]['x'] - points[1]['y'])
            for i in range(3):
                lane_id = road['id'] + "_{0}".format(i)
                lanes_length_dict[lane_id] = road_length
        min_length = min(lanes_length_dict.values())

        for key, value in lanes_length_dict.items():
            lane_normalize_factor[key] = value / min_length
        return lane_normalize_factor, lanes_length_dict
