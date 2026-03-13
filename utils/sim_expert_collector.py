import os
import time
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from utils.llm_aft_trainer import LLM_Inference_VLLM, _shared_maybe_take_snapshot
from utils.my_utils import dump_json, get_state_detail, state2text, getPrompt, code2action, eight_phase_list


class SimExpertCollector(LLM_Inference_VLLM):
    def initialize_llm(self):
        mode = str(self.dic_agent_conf.get("CLIP_CAPTION_MODE", "")).strip().lower()
        needs_llm = mode in ("prompt_llm", "prompt", "llm", "prompt_caption")
        llm_path = str(self.dic_agent_conf.get("LLM_PATH", "") or "").strip()
        if needs_llm and llm_path:
            super().initialize_llm()
            return
        if needs_llm and not llm_path:
            print("Clip caption mode requires LLM_PATH; falling back to rule-based captions.")
        # Disable LLM init for simulation-only collector.
        self.llm_model = None
        self.tokenizer = None
        self.generation_kwargs = {}

    def _phase_state_snapshot(self):
        snapshots = []
        for inter in self.env.list_intersection:
            snapshots.append(
                {
                    "current_phase_index": inter.current_phase_index,
                    "previous_phase_index": inter.previous_phase_index,
                    "next_phase_to_set_index": inter.next_phase_to_set_index,
                    "current_phase_duration": inter.current_phase_duration,
                    "all_yellow_flag": inter.all_yellow_flag,
                    "all_red_flag": inter.all_red_flag,
                    "flicker": inter.flicker,
                }
            )
        return snapshots

    def _phase_state_restore(self, snapshots):
        if not snapshots:
            return
        for inter, snap in zip(self.env.list_intersection, snapshots):
            inter.current_phase_index = snap.get("current_phase_index", inter.current_phase_index)
            inter.previous_phase_index = snap.get("previous_phase_index", inter.previous_phase_index)
            inter.next_phase_to_set_index = snap.get("next_phase_to_set_index", inter.next_phase_to_set_index)
            inter.current_phase_duration = snap.get("current_phase_duration", inter.current_phase_duration)
            inter.all_yellow_flag = snap.get("all_yellow_flag", inter.all_yellow_flag)
            inter.all_red_flag = snap.get("all_red_flag", inter.all_red_flag)
            inter.flicker = snap.get("flicker", inter.flicker)

    def _base_action_for_intersection(self, inter):
        candidates = []
        if isinstance(inter.current_phase_index, int) and inter.current_phase_index > 0:
            candidates.append(inter.current_phase_index - 1)
        next_idx = inter.next_phase_to_set_index
        if isinstance(next_idx, int) and next_idx > 0:
            candidates.append(next_idx - 1)
        prev_idx = inter.previous_phase_index
        if isinstance(prev_idx, int) and prev_idx > 0:
            candidates.append(prev_idx - 1)
        for cand in candidates:
            if 0 <= cand < 4:
                return cand
        return 0

    def _apply_action_no_log(self, inter, action):
        try:
            action = int(action)
        except (TypeError, ValueError):
            action = 0
        yellow_time = int(self.dic_traffic_env_conf.get("YELLOW_TIME", 0) or 0)

        if inter.all_yellow_flag:
            inter.flicker = 0
            if inter.current_phase_duration >= yellow_time:
                next_idx = inter.next_phase_to_set_index
                if isinstance(next_idx, int):
                    inter.current_phase_index = next_idx
                    inter.eng.set_tl_phase(inter.inter_name, inter.current_phase_index)
                inter.all_yellow_flag = False
            return

        inter.next_phase_to_set_index = action + 1
        if inter.current_phase_index == inter.next_phase_to_set_index:
            return

        inter.eng.set_tl_phase(inter.inter_name, 0)
        inter.current_phase_index = inter.all_yellow_phase_index
        inter.all_yellow_flag = True
        inter.flicker = 1

    def _sim_wait_speed(self):
        try:
            speed = float(self.dic_agent_conf.get("SIM_WAIT_SPEED", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        return speed

    def _sim_horizon_s(self):
        horizon = self.dic_agent_conf.get("SIM_HORIZON_S", None)
        if horizon is None or horizon == "" or horizon == 0:
            horizon = self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 1)
        try:
            horizon = int(horizon)
        except (TypeError, ValueError):
            horizon = int(self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 1) or 1)
        return max(1, horizon)

    def _simulate_candidate(self, inter_idx, action, base_actions, snapshot_path, phase_snapshot):
        env = self.env
        if env is None or snapshot_path is None:
            return 0.0

        env.eng.load_from_file(snapshot_path)
        self._phase_state_restore(phase_snapshot)

        action_list = list(base_actions)
        action_list[inter_idx] = action

        horizon = self._sim_horizon_s()
        try:
            interval = float(self.dic_traffic_env_conf.get("INTERVAL", 1) or 1)
        except (TypeError, ValueError):
            interval = 1.0
        if interval <= 0:
            interval = 1.0
        substeps = int(1 / interval) if interval > 0 else 1
        wait_speed = self._sim_wait_speed()

        total_wait = 0.0
        unique_waiting = set()
        target_inter = env.list_intersection[inter_idx]
        entering_lanes = list(target_inter.list_entering_lanes)

        for _ in range(horizon):
            for inter in env.list_intersection:
                inter.previous_phase_index = inter.current_phase_index

            for j, inter in enumerate(env.list_intersection):
                self._apply_action_no_log(inter, action_list[j])

            for _ in range(substeps):
                env.eng.next_step()

            for inter in env.list_intersection:
                if inter.current_phase_index == inter.previous_phase_index:
                    inter.current_phase_duration += 1
                else:
                    inter.current_phase_duration = 1

            lane_vehicles = env.eng.get_lane_vehicles()
            vehicle_speed = env.eng.get_vehicle_speed()
            for lane in entering_lanes:
                for veh in lane_vehicles.get(lane, []):
                    if float(vehicle_speed.get(veh, 0.0)) < wait_speed:
                        total_wait += 1.0
                        unique_waiting.add(veh)

        if not unique_waiting:
            return 0.0
        return total_wait / len(unique_waiting)

    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        current_time = self.env.get_current_time()

        if len(self.dic_traffic_env_conf.get("PHASE", {})) != 4:
            raise NotImplementedError(
                "SimExpertCollector supports only 4-phase control. "
                "Update action mappings and prompts before enabling 8-phase."
            )
        if str(self.dic_traffic_env_conf.get("ACTION_PATTERN", "set")).strip().lower() != "set":
            raise NotImplementedError("SimExpertCollector requires ACTION_PATTERN='set'.")

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]

        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf["MIN_ACTION_TIME"]))):
            if done or current_time >= total_run_cnt:
                break

            current_states = []
            for i in range(len(state)):
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                state_action_log[i].append(
                    {
                        "state": statistic_state,
                        "state_incoming": statistic_state_incoming,
                        "approaching_speed": mean_speed,
                    }
                )
                current_states.append(statistic_state)

            prompts = []
            for s in current_states:
                prompt = getPrompt(state2text(s))
                prompt = (
                    prompt[0]["content"]
                    + "\n\n### Instruction:\n"
                    + prompt[1]["content"]
                    + "\n\n### Response:\n"
                )
                prompts.append(prompt)

            snapshot_path = _shared_maybe_take_snapshot(self, current_time)
            snapshot_paths = [snapshot_path or "" for _ in range(len(prompts))]

            base_actions = [self._base_action_for_intersection(inter) for inter in self.env.list_intersection]
            expert_actions = []
            action_list = []
            candidate_waits_all = []
            expert_action_margins = []
            base_action_names = []

            if snapshot_path:
                phase_snapshot = self._phase_state_snapshot()
                for i in range(len(self.env.list_intersection)):
                    base_action = base_actions[i]
                    base_action_names.append(code2action(base_action))
                    best_action = base_action
                    best_wait = None
                    candidate_waits = {}
                    for cand in range(4):
                        avg_wait = self._simulate_candidate(
                            i, cand, base_actions, snapshot_path, phase_snapshot
                        )
                        candidate_waits[code2action(cand)] = float(avg_wait)
                        if best_wait is None or avg_wait < best_wait:
                            best_wait = avg_wait
                            best_action = cand
                        elif abs(avg_wait - best_wait) < 1e-9 and cand == base_action:
                            best_action = cand
                    sorted_waits = sorted(float(value) for value in candidate_waits.values())
                    if len(sorted_waits) >= 2:
                        action_margin = float(sorted_waits[1] - sorted_waits[0])
                    else:
                        action_margin = 0.0
                    candidate_waits_all.append(candidate_waits)
                    expert_action_margins.append(action_margin)
                    expert_actions.append(code2action(best_action))
                    action_list.append(best_action)

                self.env.eng.load_from_file(snapshot_path)
                self._phase_state_restore(phase_snapshot)
            else:
                for base_action in base_actions:
                    base_action_names.append(code2action(base_action))
                    candidate_waits_all.append({})
                    expert_action_margins.append(0.0)
                    expert_actions.append(code2action(base_action))
                    action_list.append(base_action)

            for i in range(len(action_list)):
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            self._record_traffic_prompt_series(
                current_time,
                prompts,
                expert_actions=expert_actions,
                snapshot_paths=snapshot_paths,
                candidate_action_waits=candidate_waits_all,
                expert_action_margins=expert_action_margins,
                base_actions=base_action_names,
            )

            next_state, _, done, _ = self.env.step(action_list)
            rewards = self.get_norm_reward(next_state)
            current_time = self.env.get_current_time()
            state = next_state

            total_reward += sum(rewards)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature["lane_num_waiting_vehicle_in"]))
            queue_length_episode.append(sum(queue_length_inter))

            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]["time"])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

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

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "test_reward_over": total_reward,
            "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_travel_time_over": total_travel_time,
        }
        logger.log(results)
        print("Test Round:", test_round, results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)
        print("Testing time: ", time.time() - start_time)

        self.env.batch_log_2()
        return results
