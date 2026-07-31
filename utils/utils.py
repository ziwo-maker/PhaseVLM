from .pipeline import Pipeline
from .oneline import OneLine
from . import config
import wandb
import copy
import numpy as np
import time
import os

location_dict_short = {"North": "N", "South": "S", "East": "E", "West": "W"}
location_direction_dict = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]

# 合并两个字典，返回合并后的新字典（不会修改原字典）
def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

# 封装Pipeline流程，运行一次仿真，记录奖励、排队长度、通行时间等结果，并用wandb记录实验数据
def pipeline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
    results_table = []
    all_rewards = []
    all_queue_len = []
    all_travel_time = []
    for i in range(1):
        dic_path["PATH_TO_MODEL"] = (dic_path["PATH_TO_MODEL"].split(".")[0] + ".json" +
                                     time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        dic_path["PATH_TO_WORK_DIRECTORY"] = (dic_path["PATH_TO_WORK_DIRECTORY"].split(".")[0] + ".json" +
                                              time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        ppl = Pipeline(dic_agent_conf=dic_agent_conf,
                       dic_traffic_env_conf=dic_traffic_env_conf,
                       dic_path=dic_path,
                       roadnet=roadnet,
                       trafficflow=trafficflow)
        round_results = ppl.run(round=i, multi_process=False)
        results_table.append([round_results['test_reward_over'], round_results['test_avg_queue_len_over'], round_results['test_avg_travel_time_over']])
        all_rewards.append(round_results['test_reward_over'])
        all_queue_len.append(round_results['test_avg_queue_len_over'])
        all_travel_time.append(round_results['test_avg_travel_time_over'])

        # delete junk
        cmd_delete_model = 'find <dir> -type f ! -name "round_<round>_inter_*.h5" -exec rm -rf {} \\;'.replace("<dir>", dic_path["PATH_TO_MODEL"]).replace("<round>", str(int(dic_traffic_env_conf["NUM_ROUNDS"] - 1)))
        traffic_base_name = dic_traffic_env_conf.get(
            "TRAFFIC_COUNT_BASENAME",
            config.dic_traffic_env_conf.get("TRAFFIC_COUNT_BASENAME", "traffic_counts"),
        )
        traffic_pattern_csv = f"{traffic_base_name}_*.csv"
        traffic_pattern_jsonl = f"{traffic_base_name}_*.jsonl"
        cmd_delete_work = (
            'find <dir> -type f ! -name "state_action.json" ! -name "<traffic_pattern_csv>" ! -name "<traffic_pattern_jsonl>" -exec rm -rf {} \\;'
            .replace("<dir>", dic_path["PATH_TO_WORK_DIRECTORY"])
            .replace("<traffic_pattern_csv>", traffic_pattern_csv)
            .replace("<traffic_pattern_jsonl>", traffic_pattern_jsonl)
        )
        os.system(cmd_delete_model)
        os.system(cmd_delete_work)

    results_table.append([np.average(all_rewards), np.average(all_queue_len), np.average(all_travel_time)])
    results_table.append([np.std(all_rewards), np.std(all_queue_len), np.std(all_travel_time)])

    table_logger = wandb.init(
        project=dic_traffic_env_conf['PROJECT_NAME'],
        group=f"{dic_traffic_env_conf['MODEL']}-{roadnet}-{trafficflow}-{len(dic_traffic_env_conf['PHASE'])}_Phases",
        name="exp_results",
        config=merge(merge(dic_agent_conf, dic_path), dic_traffic_env_conf),
    )
    columns = ["reward", "avg_queue_len", "avg_travel_time"]
    logger_table = wandb.Table(columns=columns, data=results_table)
    table_logger.log({"results": logger_table})
    wandb.finish()

    print("pipeline_wrapper end")
    return

# 封装OneLine流程，运行一次仿真，记录奖励、排队长度、通行时间等结果，并用wandb记录实验数据
def oneline_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
    results_table = []
    all_rewards = []
    all_queue_len = []
    all_travel_time = []
    for i in range(1):
        dic_path["PATH_TO_MODEL"] = (dic_path["PATH_TO_MODEL"].split(".")[0] + ".json" +
                                     time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        dic_path["PATH_TO_WORK_DIRECTORY"] = (dic_path["PATH_TO_WORK_DIRECTORY"].split(".")[0] + ".json" +
                                              time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
        oneline = OneLine(dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=merge(config.dic_traffic_env_conf, dic_traffic_env_conf),
                          dic_path=merge(config.DIC_PATH, dic_path),
                          roadnet=roadnet,
                          trafficflow=trafficflow
                          )
        round_results = oneline.train(round=i)   #进行训练
        results_table.append([round_results['test_reward_over'], round_results['test_avg_queue_len_over'],
                              round_results['test_avg_travel_time_over']])
        all_rewards.append(round_results['test_reward_over'])
        all_queue_len.append(round_results['test_avg_queue_len_over'])
        all_travel_time.append(round_results['test_avg_travel_time_over'])

        # delete junk
        cmd_delete_model = 'rm -rf <dir>'.replace("<dir>", dic_path["PATH_TO_MODEL"])
        traffic_base_name = dic_traffic_env_conf.get(
            "TRAFFIC_COUNT_BASENAME",
            config.dic_traffic_env_conf.get("TRAFFIC_COUNT_BASENAME", "traffic_counts"),
        )
        traffic_pattern_csv = f"{traffic_base_name}_*.csv"
        traffic_pattern_jsonl = f"{traffic_base_name}_*.jsonl"
        cmd_delete_work = (
            'find <dir> -type f ! -name "state_action.json" ! -name "<traffic_pattern_csv>" ! -name "<traffic_pattern_jsonl>" -exec rm -rf {} \\;'
            .replace("<dir>", dic_path["PATH_TO_WORK_DIRECTORY"])
            .replace("<traffic_pattern_csv>", traffic_pattern_csv)
            .replace("<traffic_pattern_jsonl>", traffic_pattern_jsonl)
        )
        os.system(cmd_delete_model)
        os.system(cmd_delete_work)

    results_table.append([np.average(all_rewards), np.average(all_queue_len), np.average(all_travel_time)])
    results_table.append([np.std(all_rewards), np.std(all_queue_len), np.std(all_travel_time)])

    table_logger = wandb.init(
        project=dic_traffic_env_conf['PROJECT_NAME'],
        group=f"{dic_traffic_env_conf['MODEL_NAME']}-{roadnet}-{trafficflow}-{len(dic_agent_conf['FIXED_TIME'])}_Phases",
        name="exp_results",
        config=merge(merge(dic_agent_conf, dic_path), dic_traffic_env_conf),
    )
    columns = ["reward", "avg_queue_len", "avg_travel_time"]
    logger_table = wandb.Table(columns=columns, data=results_table)
    table_logger.log({"results": logger_table})
    wandb.finish()

    return
