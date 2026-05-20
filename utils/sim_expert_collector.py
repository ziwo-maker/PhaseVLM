import os
import time
from copy import deepcopy

import numpy as np
from tqdm import tqdm

from utils.llm_aft_trainer import LLM_Inference_VLLM, _shared_maybe_take_snapshot
from utils.my_utils import dump_json, get_state_detail, state2text, getPrompt, code2action, eight_phase_list


class SimExpertCollector(LLM_Inference_VLLM):
    _SPILLBACK_JAM_SPACING_M = 7.5
    _SPILLBACK_BLOCK_THRESHOLD = 0.8
    _SPILLBACK_BLOCK_WEIGHT = 0.5

    def initialize_llm(self):
        if self.dic_agent_conf.get("SKIP_LLM_INIT", False):
            self.llm_model = None
            self.processor = None
            self.tokenizer = None
            print("SimExpert LLM initialization skipped; using simulator expert only.")
            return
        # Load LLM model first, then proceed with simulation
        # Call parent class method to actually load the LLM model
        super().initialize_llm()

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

    def _expert_score_weights(self):
        def _safe_float(key, default):
            try:
                return float(self.dic_agent_conf.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        return {
            "att": _safe_float("EXPERT_ATT_WEIGHT", 1.0),
            "aql": _safe_float("EXPERT_AQL_WEIGHT", 1.0),
            "awt": _safe_float("EXPERT_AWT_WEIGHT", 1.0),
            "spillback": _safe_float("EXPERT_SPILLBACK_WEIGHT", 0.2),
        }

    @staticmethod
    def _metric_mean(values):
        return float(np.mean(values)) if values else 0.0

    @staticmethod
    def _minmax_normalize(values, action_names):
        normalized = {}
        if not values:
            for action in action_names:
                normalized[action] = 0.0
            return normalized

        min_value = min(values)
        max_value = max(values)
        spread = float(max_value - min_value)
        for idx, action in enumerate(action_names):
            if spread <= 1e-8:
                normalized[action] = 0.0
            else:
                normalized[action] = float((values[idx] - min_value) / spread)
        return normalized

    def _score_candidates(self, candidate_metrics, base_action):
        if not candidate_metrics:
            return {}, {}, base_action

        weights = self._expert_score_weights()
        action_names = list(candidate_metrics.keys())
        raw_values_by_metric = {
            "att": [float(candidate_metrics[action].get("att", 0.0)) for action in action_names],
            "aql": [float(candidate_metrics[action].get("aql", 0.0)) for action in action_names],
            "awt": [float(candidate_metrics[action].get("awt", 0.0)) for action in action_names],
            "spillback": [float(candidate_metrics[action].get("spillback", 0.0)) for action in action_names],
        }

        normalized_by_metric = {}
        for metric, values in raw_values_by_metric.items():
            normalized_by_metric[metric] = self._minmax_normalize(values, action_names)

        candidate_scores = {}
        for action in action_names:
            candidate_scores[action] = float(
                weights["att"] * normalized_by_metric["att"][action]
                + weights["aql"] * normalized_by_metric["aql"][action]
                + weights["awt"] * normalized_by_metric["awt"][action]
                + weights["spillback"] * normalized_by_metric["spillback"][action]
            )

        best_action_name = None
        best_score = None
        base_action_name = code2action(base_action)
        for action in action_names:
            score = float(candidate_scores[action])
            if best_score is None or score < best_score:
                best_score = score
                best_action_name = action
            elif best_score is not None and abs(score - best_score) < 1e-9 and action == base_action_name:
                best_action_name = action

        enriched_metrics = {}
        for action in action_names:
            enriched_metrics[action] = {
                "att": float(candidate_metrics[action].get("att", 0.0)),
                "aql": float(candidate_metrics[action].get("aql", 0.0)),
                "awt": float(candidate_metrics[action].get("awt", 0.0)),
                "spillback": float(candidate_metrics[action].get("spillback", 0.0)),
                "spillback_occupancy": float(candidate_metrics[action].get("spillback_occupancy", 0.0)),
                "spillback_blocked_ratio": float(candidate_metrics[action].get("spillback_blocked_ratio", 0.0)),
                "score": float(candidate_scores[action]),
                "score_mode": "composite_spillback",
                "score_norm": "minmax",
                "weight_att": float(weights["att"]),
                "weight_aql": float(weights["aql"]),
                "weight_awt": float(weights["awt"]),
                "weight_spillback": float(weights["spillback"]),
                "norm_att": float(normalized_by_metric["att"][action]),
                "norm_aql": float(normalized_by_metric["aql"][action]),
                "norm_awt": float(normalized_by_metric["awt"][action]),
                "norm_spillback": float(normalized_by_metric["spillback"][action]),
            }
        best_action_code = int(base_action)
        for cand in range(4):
            if code2action(cand) == (best_action_name or base_action_name):
                best_action_code = cand
                break
        return candidate_scores, enriched_metrics, best_action_code

    def _simulate_candidate(self, inter_idx, action, base_actions, snapshot_path, phase_snapshot):
        env = self.env
        if env is None or snapshot_path is None:
            return {
                "att": 0.0,
                "aql": 0.0,
                "awt": 0.0,
                "spillback": 0.0,
                "spillback_occupancy": 0.0,
                "spillback_blocked_ratio": 0.0,
            }

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

        target_inter = env.list_intersection[inter_idx]
        entering_lanes = list(target_inter.list_entering_lanes)
        exiting_lanes = list(target_inter.list_exiting_lanes)
        entering_lane_set = set(entering_lanes)
        lane_lengths = getattr(target_inter, "lane_length", {}) or {}
        previous_lane_vehicles = set()
        lane_travel_times = {}
        waiting_vehicle_times = {}
        queue_trace = []
        wait_trace = []
        spillback_occupancy_trace = []
        spillback_blocked_trace = []

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
            lane_waiting_counts = env.eng.get_lane_waiting_vehicle_count()
            current_time = float(env.eng.get_current_time())
            current_lane_vehicles = set()
            for lane in entering_lanes:
                current_lane_vehicles.update(str(veh) for veh in lane_vehicles.get(lane, []))

            new_arrivals = current_lane_vehicles - previous_lane_vehicles
            new_left = previous_lane_vehicles - current_lane_vehicles
            for veh in new_arrivals:
                if veh not in lane_travel_times:
                    lane_travel_times[veh] = {"enter_time": current_time, "leave_time": np.nan}
            for veh in new_left:
                if veh in lane_travel_times and np.isnan(lane_travel_times[veh]["leave_time"]):
                    lane_travel_times[veh]["leave_time"] = current_time
            previous_lane_vehicles = current_lane_vehicles

            vehicle_ids = env.eng.get_vehicles(include_waiting=False)
            seen_vehicle_ids = set()
            for v_id in vehicle_ids:
                v_info = env.eng.get_vehicle_info(v_id)
                seen_vehicle_ids.add(str(v_id))
                lane_id = str(v_info.get("drivable", ""))
                speed = float(v_info.get("speed", 0.0))
                if lane_id in entering_lane_set and speed < wait_speed:
                    payload = waiting_vehicle_times.get(str(v_id))
                    if payload is None or payload.get("link") != lane_id:
                        waiting_vehicle_times[str(v_id)] = {"time": interval, "link": lane_id}
                    else:
                        payload["time"] = float(payload.get("time", 0.0)) + interval
                elif str(v_id) in waiting_vehicle_times:
                    waiting_vehicle_times.pop(str(v_id), None)

            stale_ids = [veh_id for veh_id in waiting_vehicle_times if veh_id not in seen_vehicle_ids]
            for veh_id in stale_ids:
                waiting_vehicle_times.pop(veh_id, None)

            queue_trace.append(
                float(sum(float(lane_waiting_counts.get(lane, 0.0) or 0.0) for lane in entering_lanes))
            )
            wait_trace.append(
                self._metric_mean([float(payload.get("time", 0.0)) for payload in waiting_vehicle_times.values()])
            )

            exiting_lane_occupancies = []
            exiting_lane_blocked = []
            for lane in exiting_lanes:
                lane_length = float(lane_lengths.get(lane, 0.0) or 0.0)
                jam_capacity = max(1.0, lane_length / self._SPILLBACK_JAM_SPACING_M)
                lane_vehicle_count = float(len(lane_vehicles.get(lane, [])))
                occupancy = min(1.0, lane_vehicle_count / jam_capacity)
                lane_waiting = float(lane_waiting_counts.get(lane, 0.0) or 0.0)
                exiting_lane_occupancies.append(occupancy)
                exiting_lane_blocked.append(
                    1.0 if occupancy >= self._SPILLBACK_BLOCK_THRESHOLD and lane_waiting > 0.0 else 0.0
                )

            spillback_occupancy_trace.append(self._metric_mean(exiting_lane_occupancies))
            spillback_blocked_trace.append(self._metric_mean(exiting_lane_blocked))

        final_time = float(env.eng.get_current_time())
        att_values = []
        for stats in lane_travel_times.values():
            enter_time = stats.get("enter_time")
            if enter_time is None or np.isnan(enter_time):
                continue
            leave_time = stats.get("leave_time")
            resolved_leave = leave_time if leave_time is not None and not np.isnan(leave_time) else final_time
            att_values.append(max(0.0, float(resolved_leave) - float(enter_time)))

        spillback_occupancy = self._metric_mean(spillback_occupancy_trace)
        spillback_blocked_ratio = self._metric_mean(spillback_blocked_trace)
        return {
            "att": self._metric_mean(att_values),
            "aql": self._metric_mean(queue_trace),
            "awt": self._metric_mean(wait_trace),
            "spillback": spillback_occupancy + self._SPILLBACK_BLOCK_WEIGHT * spillback_blocked_ratio,
            "spillback_occupancy": spillback_occupancy,
            "spillback_blocked_ratio": spillback_blocked_ratio,
        }

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

        progress_disabled = os.environ.get("SIM_EXPERT_PROGRESS", "").strip().lower() not in {"1", "true", "yes", "on"} and not os.isatty(2)
        for step_num in tqdm(
            range(int(total_run_cnt / self.dic_traffic_env_conf["MIN_ACTION_TIME"])),
            disable=progress_disabled,
        ):
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
            candidate_metrics_all = []
            expert_action_margins = []
            base_action_names = []

            if snapshot_path:
                phase_snapshot = self._phase_state_snapshot()
                for i in range(len(self.env.list_intersection)):
                    base_action = base_actions[i]
                    base_action_names.append(code2action(base_action))
                    candidate_metrics = {}
                    for cand in range(4):
                        metrics = self._simulate_candidate(
                            i, cand, base_actions, snapshot_path, phase_snapshot
                        )
                        candidate_metrics[code2action(cand)] = metrics
                    candidate_waits, enriched_metrics, best_action = self._score_candidates(
                        candidate_metrics,
                        base_action=base_action,
                    )
                    sorted_waits = sorted(float(value) for value in candidate_waits.values())
                    if len(sorted_waits) >= 2:
                        action_margin = float(sorted_waits[1] - sorted_waits[0])
                    else:
                        action_margin = 0.0
                    candidate_waits_all.append(candidate_waits)
                    candidate_metrics_all.append(enriched_metrics)
                    expert_action_margins.append(action_margin)
                    expert_actions.append(code2action(best_action))
                    action_list.append(best_action)

                self.env.eng.load_from_file(snapshot_path)
                self._phase_state_restore(phase_snapshot)
            else:
                for base_action in base_actions:
                    base_action_names.append(code2action(base_action))
                    candidate_waits_all.append({})
                    candidate_metrics_all.append({})
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
                candidate_action_metrics=candidate_metrics_all,
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
