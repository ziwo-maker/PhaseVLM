import numpy as np
import json
import copy
import torch

location_dict_short = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_direction_dict = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
location_incoming_dict = ["N", "S", "E", "W"]
eight_phase_list = ['ETWT', 'NTST', 'ELWL', 'NLSL', 'WTWL', 'ETEL', 'STSL', 'NTNL']
four_phase_list = {'ETWT': 0, 'NTST': 1, 'ELWL': 2, 'NLSL': 3}
phase2code = {0: 'ETWT', 1: 'NTST', 2: 'ELWL', 3: 'NLSL'}
location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
location_dict_detail = {"N": "Northern", "S": "Southern", "E": "Eastern", "W": "Western"}

phase_explanation_dict_detail = {"NTST": "- NTST: Northern and southern through lanes.",
                                 "NLSL": "- NLSL: Northern and southern left-turn lanes.",
                                 "NTNL": "- NTNL: Northern through and left-turn lanes.",
                                 "STSL": "- STSL: Southern through and left-turn lanes.",
                                 "ETWT": "- ETWT: Eastern and western through lanes.",
                                 "ELWL": "- ELWL: Eastern and western left-turn lanes.",
                                 "ETEL": "- ETEL: Eastern through and left-turn lanes.",
                                 "WTWL": "- WTWL: Western through and left-turn lanes."
                                }

def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def load_json(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise e
    return data

def dump_json(data, file, indent=None):
    try:
        with open(file, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise e

def calculate_road_length(road_points):
    length = 0.0
    i = 1
    while i < len(road_points):
        length += np.sqrt((road_points[i]['x'] - road_points[i-1]['x']) ** 2 + (road_points[i]['y'] - road_points[i-1]['y']) ** 2)
        i += 1

    return length

def get_state(roads, env):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    """
    lane_queues = env.eng.get_lane_waiting_vehicle_count()
    lane_vehicles = env.eng.get_lane_vehicles()

    # init statistic info & get queue info
    statistic_state = {}
    statistic_state_incoming = {}
    for r in roads:
        # road info
        location = roads[r]["location"]
        road_length = float(roads[r]["length"])

        # get queue info
        if roads[r]["type"] == "outgoing":
            if roads[r]["go_straight"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["go_straight"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {"cells": [0 for _ in range(3)],
                                                                                    "queue_len": queue_len}

            if roads[r]["turn_left"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["turn_left"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {"cells": [0 for _ in range(3)],
                                                                                    "queue_len": queue_len}

            # get vehicle position info
            straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
            left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
            lanes = straight_lanes + left_lanes

            for lane in lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    if lane in straight_lanes:
                        lane_group = 0
                    elif lane in left_lanes:
                        lane_group = 1
                elif location == "South":
                    if lane in straight_lanes:
                        lane_group = 2
                    elif lane in left_lanes:
                        lane_group = 3
                elif location == "East":
                    if lane in straight_lanes:
                        lane_group = 4
                    elif lane in left_lanes:
                        lane_group = 5
                elif location == "West":
                    if lane in straight_lanes:
                        lane_group = 6
                    elif lane in left_lanes:
                        lane_group = 7
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state[location_direction_dict[lane_group]]["cells"][gpt_lane_cell] += 1

        # incoming lanes
        else:
            queue_len = 0.0
            try:
                num_lanes = int(roads[r].get("num_lanes")) if roads[r].get("num_lanes") is not None else 2
            except (TypeError, ValueError):
                num_lanes = 2
            for lane in range(max(0, num_lanes)):
                queue_len += lane_queues[f"{r}_{lane}"]
            statistic_state_incoming[location_dict_short[roads[r]['location']]] = {"cells": [0 for _ in range(3)], "queue_len": queue_len}
            incoming_lanes = [f"{r}_{idx}" for idx in range(max(0, num_lanes))]

            for lane in incoming_lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    lane_group = 0
                elif location == "South":
                    lane_group = 1
                elif location == "East":
                    lane_group = 2
                elif location == "West":
                    lane_group = 3
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state_incoming[location_incoming_dict[lane_group]]["cells"][gpt_lane_cell] += 1

    return statistic_state, statistic_state_incoming

def get_state_detail(roads, env):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy.

    该函数与 get_state 类似，但细化为 4 段（cells 长度为 4），并额外统计 avg_wait_time 和返回平均速度 mean_speed。
    下面对函数内部每一步做详细中文注释，解释每行代码的作用。
    """
    # 从环境的 SUMO 接口获取每条车道的等待车辆数（早到队列）字典
    lane_queues = env.eng.get_lane_waiting_vehicle_count()
    # 从环境的 SUMO 接口获取每条车道当前的车辆列表（车辆 id 列表）字典
    lane_vehicles = env.eng.get_lane_vehicles()

    # 初始化统计字典：outgoing（出向）和 incoming（入向）
    statistic_state = {}
    statistic_state_incoming = {}
    # 用于收集所有出向车道上正在行驶（接近路口）的车辆速度，用于计算平均速度
    outgoing_lane_speeds = []
    # 遍历所有道路（roads 是一个字典，键为路段 id）
    for r in roads:
        # 读取该路段的方位 (例如 "North", "East" 等)
        location = roads[r]["location"]
        # 将路段长度转换为浮点数，后续用于计算车辆到路口的距离位置
        road_length = float(roads[r]["length"])

        # 如果该路段是出向车道（道路类型为 outgoing）
        if roads[r]["type"] == "outgoing":
            # 如果存在直行车道组（go_straight 不为 None）
            if roads[r]["go_straight"] is not None:
                queue_len = 0.0
                # 累加该路段所有直行车道的等待车辆数（早到队列）
                for lane in roads[r]["lanes"]["go_straight"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                # 为该方向的直行车道在 statistic_state 中创建条目：
                # 键示例 "NT"、"EL" 等（使用 location_dict_short + T/L），
                # cells 初始为长度 4 的零列表（表示 4 段），queue_len 保存早到队列长度，avg_wait_time 初始为 0.0
                statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {"cells": [0 for _ in range(4)],
                                                                                    "queue_len": queue_len,
                                                                                    "avg_wait_time": 0.0}

            # 如果存在左转车道组（turn_left 不为 None）
            if roads[r]["turn_left"] is not None:
                queue_len = 0.0
                # 累加该路段所有左转车道的等待车辆数（早到队列）
                for lane in roads[r]["lanes"]["turn_left"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                # 为该方向的左转车道创建 statistic_state 条目（cells 长度 4）
                statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {"cells": [0 for _ in range(4)],
                                                                                    "queue_len": queue_len,
                                                                                    "avg_wait_time": 0.0}

            # 获取该路段所有直行与左转车道的完整 lane id 列表（格式 "roadId_laneIndex"）
            straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
            left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
            lanes = straight_lanes + left_lanes

            # 遍历每条车道，统计车辆分段信息与等待时间
            for lane in lanes:
                # 用于收集当前车道上静止车辆（速度<=0.1）的等待时间
                waiting_times = []
                # 获取该车道上当前所有车辆 id 列表
                vehicles = lane_vehicles[lane]

                # 根据路段方位与车道类型，确定 lane_group（用于索引 location_direction_dict）
                # lane_group 映射规则：
                # North: straight->0, left->1; South: straight->2, left->3; East: straight->4, left->5; West: straight->6, left->7
                if location == "North":
                    if lane in straight_lanes:
                        lane_group = 0
                    elif lane in left_lanes:
                        lane_group = 1
                elif location == "South":
                    if lane in straight_lanes:
                        lane_group = 2
                    elif lane in left_lanes:
                        lane_group = 3
                elif location == "East":
                    if lane in straight_lanes:
                        lane_group = 4
                    elif lane in left_lanes:
                        lane_group = 5
                elif location == "West":
                    if lane in straight_lanes:
                        lane_group = 6
                    elif lane in left_lanes:
                        lane_group = 7
                else:
                    lane_group = -1  # 未知情况，保底处理

                # 遍历车道上的每辆车，按距离划分 segment，并统计 approaching（接近）或 waiting（等待）信息
                for veh in vehicles:
                    # 从 SUMO 接口获取车辆详细信息（包含 distance, speed 等）
                    veh_info = env.eng.get_vehicle_info(veh)
                    # lane_pos 表示车辆距离路口的距离：用路段总长减去 veh_info["distance"]（veh_info["distance"] 为车辆离路段起点的距离）
                    lane_pos = road_length - float(veh_info["distance"])

                    # 将 lane_pos 映射到 gpt_lane_cell（0..3 四段）
                    # 段划分阈值：
                    # segment 0: 最近路口，lane_pos <= road_length/10
                    # segment 1: (road_length/10, road_length/3]
                    # segment 2: (road_length/3, 2*(road_length/3)]
                    # segment 3: 其余（最远）
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    elif road_length / 3 < lane_pos <= (road_length / 3) * 2:
                        gpt_lane_cell = 2
                    else:
                        gpt_lane_cell = 3

                    # 将车辆速度解析为浮点数
                    speed = float(veh_info["speed"])
                    # 速度 > 0.1 m/s 被视为接近车辆（approaching），否则视为等待车辆（waiting）
                    if speed > 0.1:
                        # 增加对应方向、对应段的车辆计数
                        statistic_state[location_direction_dict[lane_group]]["cells"][gpt_lane_cell] += 1
                        # 将该接近车辆的速度加入集合，用于后面计算平均速度
                        outgoing_lane_speeds.append(speed)
                    else:
                        # 如果车辆速度很小（认为在排队等待），尝试从 env.waiting_vehicle_list 获取该车辆的累计等待时间
                        # 若车辆不在 waiting_vehicle_list 中则默认等待时间为 0.0
                        veh_waiting_time = env.waiting_vehicle_list[veh]['time'] if veh in env.waiting_vehicle_list else 0.0
                        # 将等待时间加入 waiting_times 列表，用于计算该车道的平均等待时间
                        waiting_times.append(veh_waiting_time)
                # 计算该车道上静止车辆的平均等待时间（若无静止车辆则 avg_wait_time 为 0.0）
                avg_wait_time = np.mean(waiting_times) if len(waiting_times) > 0 else 0.0
                # 将计算结果写回对应的 statistic_state 条目中的 avg_wait_time 字段
                statistic_state[location_direction_dict[lane_group]]["avg_wait_time"] = avg_wait_time

        # 如果该路段是入向车道（incoming）
        else:
            # 统计该路段入向所有车道的早到队列长度之和（默认兼容旧逻辑：未提供 num_lanes 时按 2 处理）
            queue_len = 0.0
            try:
                num_lanes = int(roads[r].get("num_lanes")) if roads[r].get("num_lanes") is not None else 2
            except (TypeError, ValueError):
                num_lanes = 2
            for lane in range(max(0, num_lanes)):
                queue_len += lane_queues[f"{r}_{lane}"]
            # 在 statistic_state_incoming 中为该方向（N/S/E/W 的短写）建立条目，cells 长度为 4（与出向保持一致），并记录 queue_len
            statistic_state_incoming[location_dict_short[roads[r]['location']]] = {"cells": [0 for _ in range(4)],
                                                                                   "queue_len": queue_len}
            # 构造入向车道的 lane id 列表（按 num_lanes）
            incoming_lanes = [f"{r}_{idx}" for idx in range(max(0, num_lanes))]

            # 遍历入向车道上的每条车道
            for lane in incoming_lanes:
                vehicles = lane_vehicles[lane]

                # 根据路段方位确定 lane_group（仅 0..3 四类，用于 location_incoming_dict 索引）
                if location == "North":
                    lane_group = 0
                elif location == "South":
                    lane_group = 1
                elif location == "East":
                    lane_group = 2
                elif location == "West":
                    lane_group = 3
                else:
                    lane_group = -1

                # 遍历该车道上的每辆车，按距离划分段并统计接近车辆
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # 将车辆位置映射到 4 个段（与出向逻辑一致）
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    elif road_length / 3 < lane_pos <= (road_length / 3) * 2:
                        gpt_lane_cell = 2
                    else:
                        gpt_lane_cell = 3

                    # 速度 > 0.1 m/s 的车辆被视为接近车辆，统计到对应的 cells 中
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state_incoming[location_incoming_dict[lane_group]]["cells"][gpt_lane_cell] += 1

    # 计算出向车道上接近车辆的平均速度（若没有接近车辆，则 mean_speed 为 0.0）
    mean_speed = np.mean(outgoing_lane_speeds) if len(outgoing_lane_speeds) > 0 else 0.0

    # 返回出向统计、入向统计以及出向接近车辆的平均速度
    return statistic_state, statistic_state_incoming, mean_speed

def get_state_three_segment(roads, env):
    """
    Retrieve the state of the intersection from sumo, in the form of cell occupancy
    """
    lane_queues = env.eng.get_lane_waiting_vehicle_count()
    lane_vehicles = env.eng.get_lane_vehicles()

    # init statistic info & get queue info
    statistic_state = {}
    statistic_state_incoming = {}
    outgoing_lane_speeds = []
    for r in roads:
        # road info
        location = roads[r]["location"]
        road_length = float(roads[r]["length"])

        # get queue info
        if roads[r]["type"] == "outgoing":
            if roads[r]["go_straight"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["go_straight"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}T"] = {"cells": [0 for _ in range(3)],
                                                                                    "queue_len": queue_len,
                                                                                    "avg_wait_time": 0.0}

            if roads[r]["turn_left"] is not None:
                queue_len = 0.0
                for lane in roads[r]["lanes"]["turn_left"]:
                    queue_len += lane_queues[f"{r}_{lane}"]
                statistic_state[f"{location_dict_short[roads[r]['location']]}L"] = {"cells": [0 for _ in range(3)],
                                                                                    "queue_len": queue_len,
                                                                                    "avg_wait_time": 0.0}

            # get vehicle position info
            straight_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["go_straight"]]
            left_lanes = [f"{r}_{idx}" for idx in roads[r]["lanes"]["turn_left"]]
            lanes = straight_lanes + left_lanes

            for lane in lanes:
                waiting_times = []
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    if lane in straight_lanes:
                        lane_group = 0
                    elif lane in left_lanes:
                        lane_group = 1
                elif location == "South":
                    if lane in straight_lanes:
                        lane_group = 2
                    elif lane in left_lanes:
                        lane_group = 3
                elif location == "East":
                    if lane in straight_lanes:
                        lane_group = 4
                    elif lane in left_lanes:
                        lane_group = 5
                elif location == "West":
                    if lane in straight_lanes:
                        lane_group = 6
                    elif lane in left_lanes:
                        lane_group = 7
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    speed = float(veh_info["speed"])
                    if speed > 0.1:
                        statistic_state[location_direction_dict[lane_group]]["cells"][gpt_lane_cell] += 1
                        outgoing_lane_speeds.append(speed)
                    else:
                        veh_waiting_time = env.waiting_vehicle_list[veh]['time'] if veh in env.waiting_vehicle_list else 0.0
                        waiting_times.append(veh_waiting_time)
                avg_wait_time = np.mean(waiting_times) if len(waiting_times) > 0 else 0.0
                statistic_state[location_direction_dict[lane_group]]["avg_wait_time"] = avg_wait_time

        # incoming lanes
        else:
            queue_len = 0.0
            try:
                num_lanes = int(roads[r].get("num_lanes")) if roads[r].get("num_lanes") is not None else 2
            except (TypeError, ValueError):
                num_lanes = 2
            for lane in range(max(0, num_lanes)):
                queue_len += lane_queues[f"{r}_{lane}"]
            statistic_state_incoming[location_dict_short[roads[r]['location']]] = {"cells": [0 for _ in range(3)],
                                                                                   "queue_len": queue_len}
            incoming_lanes = [f"{r}_{idx}" for idx in range(max(0, num_lanes))]

            for lane in incoming_lanes:
                vehicles = lane_vehicles[lane]

                # collect lane group info
                if location == "North":
                    lane_group = 0
                elif location == "South":
                    lane_group = 1
                elif location == "East":
                    lane_group = 2
                elif location == "West":
                    lane_group = 3
                else:
                    lane_group = -1

                # collect lane cell info
                for veh in vehicles:
                    veh_info = env.eng.get_vehicle_info(veh)
                    lane_pos = road_length - float(veh_info["distance"])

                    # update statistic state
                    if lane_pos <= road_length / 10:
                        gpt_lane_cell = 0
                    elif road_length / 10 < lane_pos <= road_length / 3:
                        gpt_lane_cell = 1
                    else:
                        gpt_lane_cell = 2

                    # speed > 0.1 m/s are approaching vehicles
                    if float(veh_info["speed"]) > 0.1:
                        statistic_state_incoming[location_incoming_dict[lane_group]]["cells"][gpt_lane_cell] += 1

    mean_speed = np.mean(outgoing_lane_speeds) if len(outgoing_lane_speeds) > 0 else 0.0

    return statistic_state, statistic_state_incoming, mean_speed

def trans_prompt_llama(message, chat_history, system_prompt):
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)

def state2text(state):
    state_txt = ""
    for p in four_phase_list:
        lane_1 = p[:2]
        lane_2 = p[2:]
        queue_len_1 = int(state[lane_1]['queue_len'])
        queue_len_2 = int(state[lane_2]['queue_len'])

        seg_1_lane_1 = state[lane_1]['cells'][0]
        seg_2_lane_1 = state[lane_1]['cells'][1]
        seg_3_lane_1 = state[lane_1]['cells'][2] + state[lane_1]['cells'][3]

        seg_1_lane_2 = state[lane_2]['cells'][0]
        seg_2_lane_2 = state[lane_2]['cells'][1]
        seg_3_lane_2 = state[lane_2]['cells'][2] + state[lane_2]['cells'][3]

        state_txt += (f"Signal: {p}\n"
                      f"Relieves: {phase_explanation_dict_detail[p][8:-1]}\n"
                      f"- Early queued: {queue_len_1} ({location_dict[lane_1[0]]}), {queue_len_2} ({location_dict[lane_2[0]]}), {queue_len_1 + queue_len_2} (Total)\n"
                      f"- Segment 1: {seg_1_lane_1} ({location_dict[lane_1[0]]}), {seg_1_lane_2} ({location_dict[lane_2[0]]}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                      f"- Segment 2: {seg_2_lane_1} ({location_dict[lane_1[0]]}), {seg_2_lane_2} ({location_dict[lane_2[0]]}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                      f"- Segment 3: {seg_3_lane_1} ({location_dict[lane_1[0]]}), {seg_3_lane_2} ({location_dict[lane_2[0]]}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n")

    return state_txt

def getPrompt(state_txt):
    # fill information
    signals_text = ""
    for i, p in enumerate(four_phase_list):
        signals_text += phase_explanation_dict_detail[p] + "\n"

    prompt = [
        {"role": "system",
         "content": '''You are an expert in traffic management. You can use your knowledge of traffic commonsense to solve this traffic signal control tasks.IMPORTANT:
        - The attached image is the ONLY reliable observation of the intersection state for the last 30 seconds.
        - Do NOT assume the traffic numbers in text are correct. If any text conflicts with the image, trust the image.
        - You MUST look at and use the image to make the decision.

        ### What the image encodes (read carefully)
        The image encodes a 30-second time series sampled every 2 seconds (15 time steps) for 8 lanes at one intersection.
        - X-axis: time from oldest (left) to most recent (right).
        - Y-axis: lanes in this fixed order:
        1) East Through (ET)
        2) West Through (WT)
        3) North Through (NT)
        4) South Through (ST)
        5) East Left (EL)
        6) West Left (WL)
        7) North Left (NL)
        8) South Left (SL)
        - Pixel intensity (or color strength): larger means more vehicles in that lane at that time.
        Interpret the rightmost columns as the latest demand. Consider both the current magnitude and recent trend (growing vs shrinking).
        '''},
        {"role": "user",
         "content": "A traffic light regulates a four-section intersection with northern, southern, eastern, and western "
                    "sections, each containing two lanes: one for through traffic and one for left-turns. Each lane is "
                    "further divided into three segments. Segment 1 is the closest to the intersection. Segment 2 is in the "
                    "middle. Segment 3 is the farthest. In a lane, there may be early queued vehicles and approaching "
                    "vehicles traveling in different segments. Early queued vehicles have arrived at the intersection and "
                    "await passage permission. Approaching vehicles will arrive at the intersection in the future.\n\n"
                    "The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two "
                    "specific lanes. The state of the intersection is listed below. It describes:\n"
                    "- The group of lanes relieving vehicles' flow under each signal phase.\n"
                    "- The number of early queued vehicles of the allowed lanes of each signal.\n"
                    "- The number of approaching vehicles in different segments of the allowed lanes of each signal.\n\n"
                    + state_txt +
                    "Please answer:\n"
                    "Which is the most effective traffic signal that will most significantly improve the traffic "
                    "condition during the next phase?\n\n"
                    "Requirements:\n"
                    "- Let's think step by step.\n"
                    "- You can only choose one of the signals listed above.\n"
                    "- You must follow the following steps to provide your analysis: Step 1: Provide your analysis "
                    "for identifying the optimal traffic signal. Step 2: Answer your chosen signal.\n"
                    "- Your choice can only be given after finishing the analysis.\n"
                    "- Your choice must be identified by the tag: <signal>YOUR_CHOICE</signal>."
         }
    ]

    return prompt

def action2code(action):
    code = four_phase_list[action]

    return code

def code2action(action):
    code = phase2code[action]

    return code

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
