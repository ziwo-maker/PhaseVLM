#!/usr/bin/env python3
"""Paired demand-intensity evaluation using the existing PhaseVLM evaluator.
No training, downloads, alternative PhaseVLM action rule, or original-data writes.
"""
import argparse
import concurrent.futures
import collections
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
PYTHON = '/home/sysu/anaconda3/envs/hnr5090/bin/python'
MODEL = '/mnt/data/Qwen3-VL-8B-Instruct'
SCENARIOS = {
    'Jinan-3': dict(dataset='jinan', directory='Jinan/3_4', flow='anon_3_4_jinan_real_2500.json', roadnet='roadnet_3_4.json', checkpoint='online_test_best_weights/anon_3_4_jinan_real_2500/paper_main_table_near_jinan3.pt'),
    'Hangzhou-2': dict(dataset='hangzhou', directory='Hangzhou/4_4', flow='anon_4_4_hangzhou_real_5816.json', roadnet='roadnet_4_4.json', checkpoint='online_test_best_weights/anon_4_4_hangzhou_real_5816/epoch_9.pt'),
}
FACTORS = [.6, .8, 1., 1.2, 1.4]
SEEDS = [2024, 2025, 2026]
METRICS = ['ATT', 'AQL', 'AWT', 'Throughput', 'Unfinished_Vehicles']

def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)

def write_csv(path, rows, fields):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    with temp.open('w') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    temp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def scaled_flow(original, factor, seed):
    # All source records are verified to be single vehicle schedules.
    if factor == 1:
        return copy.deepcopy(original)
    rng = random.Random(seed)
    groups = collections.defaultdict(list)
    for i, v in enumerate(original):
        groups[(tuple(v['route']), int(v['startTime']) // 300)].append(i)
    # Largest-remainder allocation gives an exact rounded global target and <=1
    # vehicle error per route/time stratum. Random ties and draws vary by seed.
    targets = []
    for key, ids in groups.items():
        target = len(ids) * factor
        targets.append([key, math.floor(target), target % 1, rng.random()])
    extra = round(len(original)*factor) - sum(x[1] for x in targets)
    for x in sorted(targets, key=lambda x: (-x[2], x[3]))[:extra]:
        x[1] += 1
    selected = []
    for key, target, _, _ in targets:
        ids = groups[key]
        if target <= len(ids):
            selected.extend(copy.deepcopy(original[i]) for i in rng.sample(ids, target))
        else:
            selected.extend(copy.deepcopy(original[i]) for i in ids)
            for i in rng.sample(ids, target-len(ids)):
                vehicle = copy.deepcopy(original[i])
                # Preserve route and 5-minute time profile; <=1s integer-second random jitter
                # avoids inserting cloned vehicles at exactly identical instants.
                lo = key[1]*300
                t = min(lo+299, max(lo, int(vehicle['startTime']) + rng.choice([-1, 0, 1])))
                vehicle['startTime'] = vehicle['endTime'] = int(t)
                selected.append(vehicle)
    return sorted(selected, key=lambda v: v['startTime'])

def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    jobs, stats, sources = [], [], {}
    for scenario, spec in SCENARIOS.items():
        source = REPO/'data'/spec['directory']/spec['flow']
        original = json.loads(source.read_text())
        assert all(v['startTime'] == v['endTime'] and 0 <= v['startTime'] < 3600 for v in original)
        sources[scenario] = dict(spec, original_vehicles=len(original), flow_sha256=digest(source), checkpoint_sha256=digest(REPO/spec['checkpoint']))
        for seed in SEEDS:
            for factor in FACTORS:
                data = root/'flows'/scenario/f'{factor:.1f}'/str(seed)
                data.mkdir(parents=True)
                generated = scaled_flow(original, factor, seed)
                assert len(generated) == round(len(original)*factor)
                routes = {tuple(v['route']) for v in original}
                assert all(tuple(v['route']) in routes and 0 <= v['startTime'] < 3600 and v['startTime']==v['endTime'] for v in generated)
                if factor == 1:
                    shutil.copy2(source, data/spec['flow'])
                else:
                    save(data/spec['flow'], generated)
                shutil.copy2(source.parent/spec['roadnet'], data/spec['roadnet'])
                for start in range(0, 3600, 300):
                    n = sum(start <= v['startTime'] < start+300 for v in generated)
                    base = sum(start <= v['startTime'] < start+300 for v in original)
                    stats.append(dict(scenario=scenario, factor=factor, seed=seed, original_vehicles=len(original), actual_vehicles=len(generated), actual_ratio=len(generated)/len(original), bin_start_s=start, bin_end_s=start+300, original_bin_vehicles=base, actual_bin_vehicles=n, flow_path=str(data/spec['flow']), flow_sha256=digest(data/spec['flow'])))
                for method in ['PhaseVLM', 'MaxPressure']:
                    job = dict(scenario=scenario, factor=factor, seed=seed, method=method, data=str(data), demand=len(generated), **spec)
                    job['id'] = f'{scenario}_x{factor:.1f}_s{seed}_{method}'
                    jobs.append(job)
    save(root/'jobs.json', jobs)
    write_csv(root/'demand_statistics.csv', stats, list(stats[0]))
    save(root/'configuration.json', dict(model_path=MODEL, max_length=512, gr_att=dict(found=False, original=None, new=None, reason='No gr_att in inference code. Training grad_accum_steps does not apply.'), seeds=SEEDS, factors=FACTORS, scenarios=sources, run_counts=3600, decision_interval_s=30, yellow_time_s=3, all_red_time_config_s=2, all_red_time_note='Existing set_signal uses 3-second yellow transition; ALL_RED_TIME is not passed to set_signal.', traffic_series_interval_s=2, seq_len=16, image_size=224, prefix_tokens=8, prompt_mode='queue_wait_short', phase_overrides='disabled', precision='bf16', controllers=['PhaseVLM','MaxPressure'], scope='two representative scenarios, same-flow checkpoint demand scaling', random_sources='Stratified demand subsampling/cloning seeds; CityFlow seed; deterministic greedy inference. At x1.0 demand is identical; seed repetitions are simulator sensitivity checks, not independent demand samples.', demand_scaling='Largest-remainder allocation per route and 300s bin; seeded sampling without replacement for reductions, preserve all originals plus sampled clones for increases with <=1s integer-second within-bin jitter.', python=PYTHON, created=time.strftime('%Y-%m-%dT%H:%M:%S%z'), evaluator_sha256=digest(REPO/'projection_train/run_projection_online.py'), policy_sha256=digest(REPO/'projection_train/prefix_action_policy.py')))
    (root/'status.txt').write_text('prepared\n')
    print(root, 'jobs=', len(jobs), flush=True)


def execute(root, job_id):
    import torch
    import numpy as np
    from projection_train import run_projection_online as online
    from projection_train.action_space import ACTION_LIST
    job = next(x for x in json.loads((root/'jobs.json').read_text()) if x['id']==job_id)
    out = root/'runs'/job_id
    out.mkdir(parents=True, exist_ok=True)
    save(out/'status.json', dict(status='running', pid=os.getpid(), gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), started=time.time()))
    import threading
    stop_monitor = threading.Event()
    def startup_monitor():
        snapshots = []
        started = time.monotonic()
        for tick in (0, 60, 120):
            if stop_monitor.wait(max(0, started + tick - time.monotonic())):
                break
            signals = {}
            for name, command in [('process', ['ps', '-p', str(os.getpid()), '-o', 'pid,ppid,etime,pcpu,pmem,args']), ('tree', ['pstree', '-p', str(os.getpid())]), ('gpu', ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader'])]:
                try:
                    signals[name] = subprocess.run(command, capture_output=True, text=True, timeout=8).stdout[-6000:]
                except Exception as exc:
                    signals[name] = str(exc)
            try:
                with (out/'run.log').open('rb') as handle:
                    handle.seek(0, 2)
                    handle.seek(max(0, handle.tell()-4000))
                    signals['log_tail'] = handle.read(4000).decode(errors='replace')
            except OSError:
                pass
            snapshots.append(dict(elapsed_s=round(time.monotonic()-started, 2), **signals))
            save(out/'startup_monitor.json', snapshots)
    threading.Thread(target=startup_monitor, daemon=True).start()
    torch.set_num_threads(4)
    original_runner = online.ProjectionPolicyRunner
    class IntensityRunner(original_runner):
        def __init__(self, *args, **kwargs):
            kwargs['dic_path'] = dict(PATH_TO_DATA=job['data'], PATH_TO_WORK_DIRECTORY=str(out/'env'), PATH_TO_MODEL=str(out/'model'))
            kwargs['record_snapshots'] = False
            # Input traffic-series recording is needed by the trained TS2I policy.
            super().__init__(*args, **kwargs)
        def initialize_policy(self):
            if job['method']=='MaxPressure':
                from models.maxpressure_agent import MaxPressureAgent
                self.agents = [MaxPressureAgent(self.dic_agent_conf, self.dic_traffic_env_conf, self.dic_path, 0, str(i)) for i in range(len(self.env.list_intersection))]
                self.tokenizer = None
                return
            super().initialize_policy()
            # Refuse silent state truncation, keeping the actual existing encoder
            # and action-tag scoring unchanged when prompts fit the budget.
            encode = self.policy._encode_prompts
            self.prompt_max_tokens = 0
            def checked(prompts):
                formatted = self.policy._format_prompts(prompts)
                ids = self.tokenizer(formatted, truncation=False, add_special_tokens=True)['input_ids']
                maximum = max(map(len, ids), default=0)
                self.prompt_max_tokens = max(self.prompt_max_tokens, maximum)
                if maximum > 512:
                    raise RuntimeError(f'Prompt length {maximum}>512: refusing silent truncation')
                return encode(prompts)
            self.policy._encode_prompts = checked
            for parameter in self.policy.parameters():
                parameter.requires_grad_(False)
        def _predict_actions(self, states, current_time):
            if job['method']=='PhaseVLM':
                return super()._predict_actions(states, current_time)
            state, _ = self.env.get_state()
            indices = [int(a.choose_action(int(current_time/30), s)) for a,s in zip(self.agents,state)]
            names = [ACTION_LIST[i] for i in indices]
            blank = ['']*len(indices)
            return dict(action_indices=indices, predicted_actions=names, policy_prompts=blank, raw_prompts=blank, logits=[{} for _ in indices], probs=[{} for _ in indices])
        def test(self, logger, test_round):
            self.observed = set()
            self.step_audit = []
            inner = self.env._inner_step
            def tracked(actions):
                inner(actions)
                self.observed.update(v for v in self.env.eng.get_vehicles(include_waiting=True) if 'shadow' not in v)
                if len(self.observed) > job['demand']:
                    raise RuntimeError('Actual generated vehicles exceed scheduled demand; invalid flow parsing')
            self.env._inner_step = tracked
            step = self.env.step
            def audited(actions):
                result = step(actions)
                self.step_audit.append(dict(time_s=self.env.get_current_time(), queue_vehicles=sum(sum(i.dic_feature['lane_num_waiting_vehicle_in']) for i in self.env.list_intersection), waiting_time_s=float(np.mean([v['time'] for v in self.env.waiting_vehicle_list.values()])) if self.env.waiting_vehicle_list else 0.0, active_vehicles=len(self.env.eng.get_vehicles(include_waiting=True))))
                return result
            self.env.step = audited
            with torch.inference_mode():
                result = super().test(logger, test_round)
            active = {v for v in self.env.eng.get_vehicles(include_waiting=True) if 'shadow' not in v}
            completed = len(self.observed-active)
            assert self.env.get_current_time()==3600 and len(self.step_audit)==120
            assert 0 <= completed <= len(self.observed) <= job['demand']
            assert abs(result['test_avg_queue_len_over'] - np.mean([x['queue_vehicles'] for x in self.step_audit])) < 1e-7
            assert abs(result['test_avg_waiting_time_over'] - np.mean([x['waiting_time_s'] for x in self.step_audit])) < 1e-7
            result.update(throughput=completed, unfinished_vehicles=job['demand']-completed, observed_vehicle_count=len(self.observed), active_vehicle_count=len(active), not_yet_observed_vehicles=job['demand']-len(self.observed), demand=job['demand'], prompt_max_tokens=getattr(self,'prompt_max_tokens',None), simulation_end_s=self.env.get_current_time(), decisions=len(self.step_audit))
            save(out/'step_metrics.json', self.step_audit)
            # Engine snapshot independently records completion counters, useful
            # for auditing the set-difference throughput without changing control.
            self.env.eng.snapshot().dump(str(out/'final_snapshot.json'))
            native = json.loads((out/'final_snapshot.json').read_text())
            assert int(native['finishedVehicleCnt']) == completed, 'Native completion counter mismatch'
            result['native_finished_vehicle_count'] = int(native['finishedVehicleCnt'])
            return result
    online.ProjectionPolicyRunner = IntensityRunner
    argv = ['--projection_ckpt', str(REPO/job['checkpoint']), '--llm_path', MODEL, '--dataset', job['dataset'], '--traffic_file', job['flow'], '--run_counts', '3600', '--num_rounds', '1', '--seed',str(job['seed']), '--max_length','512', '--min_action_time','30','--traffic_series_interval','2','--traffic_series_output_dir',str(out/'traffic_series'),'--inference_batch_size',str(8 if job['dataset']=='hangzhou' else 16),'--bf16','--deterministic','--disable_snapshots','--result_json',str(out/'result.json')]
    sys.argv = [sys.argv[0]] + argv
    started=time.time()
    try:
        payload=online.main(online.parse_args())
        stop_monitor.set()
        save(out/'status.json', dict(status='completed', elapsed_s=time.time()-started, gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))
    except BaseException:
        stop_monitor.set()
        save(out/'status.json', dict(status='failed', elapsed_s=time.time()-started, error=traceback.format_exc()))
        raise


def report(root):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    jobs=json.loads((root/'jobs.json').read_text())
    rows=[]
    status=[]
    for job in jobs:
        out=root/'runs'/job['id']
        state=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else dict(status='pending')
        status.append(dict(id=job['id'], **state))
        if state['status']!='completed' or not (out/'result.json').exists():
            continue
        r=json.loads((out/'result.json').read_text())['results']
        rows.append(dict(scenario=job['scenario'],factor=job['factor'],seed=job['seed'],method=job['method'],ATT=r['test_avg_travel_time_over'],AQL=r['test_avg_queue_len_over'],AWT=r['test_avg_waiting_time_over'],Throughput=r['throughput'],Unfinished_Vehicles=r['unfinished_vehicles'],demand=r['demand'],active_vehicles=r['active_vehicle_count'],not_yet_observed_vehicles=r['not_yet_observed_vehicles'],prompt_max_tokens=r['prompt_max_tokens'],result_path=str(out/'result.json')))
    write_csv(root/'metrics.csv', rows, ['scenario','factor','seed','method']+METRICS+['demand','active_vehicles','not_yet_observed_vehicles','prompt_max_tokens','result_path'])
    groups=collections.defaultdict(list)
    for r in rows: groups[(r['scenario'],r['factor'],r['method'])].append(r)
    summary=[]
    for (scene,factor,method), rr in sorted(groups.items()):
        item=dict(scenario=scene,factor=factor,method=method,n=len(rr),uncertainty='paired demand-instance variation; x1.0 simulator seed sensitivity only')
        reference=groups.get((scene,1.,method),[])
        for metric in METRICS:
            values=[x[metric] for x in rr]
            item[metric+'_mean']=float(np.mean(values))
            item[metric+'_std']=float(np.std(values,ddof=1)) if len(values)>1 else None
            base=np.mean([x[metric] for x in reference]) if reference else 0
            item[metric+'_change_pct']=(float(np.mean(values))/base-1)*100 if base else None
        summary.append(item)
    fields=['scenario','factor','method','n','uncertainty']+[m+s for m in METRICS for s in ['_mean','_std','_change_pct']]
    write_csv(root/'summary.csv', summary, fields)
    save(root/'status.json', status)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.grid':True,'grid.alpha':.25,'savefig.dpi':300,'pdf.fonttype':42})
    scenes=list(SCENARIOS)
    definitions=[('figure1_att',['ATT'],['Average travel time (s)']),('figure2_queue_wait',['AQL','AWT'],['Average network queue (vehicles)','Average waiting time (s)']),('figure3_throughput_unfinished',['Throughput','Unfinished_Vehicles'],['Completed trips (vehicles)','Unfinished demand (vehicles)'])]
    for name,metrics,labels in definitions:
        fig,axes=plt.subplots(len(metrics),len(scenes),figsize=(10,3.5*len(metrics)),squeeze=False,constrained_layout=True)
        for i,(metric,label) in enumerate(zip(metrics,labels)):
            for j,scene in enumerate(scenes):
                ax=axes[i,j]
                for method,color,marker,style in [('PhaseVLM','#2166ac','o','-'),('MaxPressure','#b2182b','s','--')]:
                    data=sorted([s for s in summary if s['scenario']==scene and s['method']==method],key=lambda s:s['factor'])
                    if data:
                        ax.errorbar([s['factor'] for s in data],[s[metric+'_mean'] for s in data],yerr=[s[metric+'_std'] or 0 for s in data],color=color,marker=marker,linestyle=style,capsize=3,label=method)
                ax.set(title=scene,xlabel='Traffic demand multiplier',ylabel=label,xticks=FACTORS)
                if ax.lines: ax.legend()
        for ext in ['png','pdf']: fig.savefig(root/f'{name}.{ext}')
        plt.close(fig)
    complete=sum(s['status']=='completed' for s in status)
    failed=sum(s['status']=='failed' for s in status)
    conf=json.loads((root/'configuration.json').read_text())
    lines=['# Traffic demand intensity experiment',f'\nUpdated: {time.strftime("%Y-%m-%d %H:%M:%S %z")}',f'\nActual status: {complete}/{len(jobs)} completed, {failed} failed, {len(jobs)-complete-failed} pending/running.', '\n## Protocol and provenance',f'\nModel: `{MODEL}`. Existing trained checkpoints are fixed; no training or downloads. max_length=512, bf16, greedy scoring of four action tags, TS2I plus structured queue/wait prompts. Each checkpoint retains its serialized architecture (Jinan-3 hidden=64; Hangzhou-2 hidden=256; 8 prefix tokens). gr_att is absent in inference; training gradient accumulation is unused. Prompts are token-counted before encoding and execution fails if >512; visual prefix/action tags are added by the original scoring pipeline.', '\nSimulation: 3600 seconds, 120 decisions at 30 seconds; original 3-second phase-change transition included, original environment signal execution. MaxPressure uses existing `models/maxpressure_agent.py` in the same evaluator, on identical demand instances and seeds. Four GPU workers, one CUDA_VISIBLE_DEVICES per process. Each job saves status, raw results, per-step metrics, action traces and a final CityFlow snapshot.', '\n## Metrics', '\nATT follows the evaluator: mean over unique non-shadow observed vehicles of summed intersection enter/leave durations, censored at 3600s for missing leave times. This is not completed-only ATT and does not measure entry-buffer delay before entering an intersection. AQL is the time mean of network total incoming-lane stopped vehicles at 30-second boundaries (vehicles). AWT is the time mean of the current waiting-vehicle mean continuous stop time; stop counters reset when the drivable changes (seconds). Throughput is unique observed non-shadow vehicles minus those still present at 3600s; observed IDs are collected every simulation second, including entry buffers. Unfinished = scheduled demand minus completed trips, including not-yet-observed and entry-buffer vehicles. Counts must be considered jointly with ATT.', '\n## Demand generation and uncertainty', '\nDemand is allocated by original route and 300-second bin using largest remainders with an exact rounded global vehicle target. Decreases sample individual records without replacement. Increases retain every original plus sampled duplicate records with <=1-second integer jitter within the same bin. Original routes/vehicle parameters are preserved; route/bin rounding error is <1 vehicle. Original files are untouched. x1.0 is byte-identical to source. Three seeds generate distinct demand instances for x != 1.0; the x1.0 input is identical across seeds, so its variation measures simulator sensitivity, not independent demand replication. Zero variability must not be presented as independent statistical evidence. `demand_statistics.csv` includes actual totals, ratios and all twelve 5-minute bins.', '\n## Environment', '\n```json',json.dumps(conf.get('environment',{}),indent=2),'```','\n## Actual results','\n|Scenario|Factor|Method|n|ATT (s)|AQL (veh)|AWT (s)|Throughput|Unfinished|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        vals=[]
        for m in METRICS:
            std=s[m+'_std']; vals.append(f"{s[m+'_mean']:.2f}"+(f' ± {std:.2f}' if std is not None else ' (single run)'))
        lines.append(f"|{s['scenario']}|{s['factor']:.1f}|{s['method']}|{s['n']}|"+'|'.join(vals)+'|')
    lines+=['\n## Observed demand response']
    for scene in scenes:
        for method in ['PhaseVLM','MaxPressure']:
            data={s['factor']:s for s in summary if s['scenario']==scene and s['method']==method}
            if .6 in data and 1.4 in data:
                lo,hi=data[.6],data[1.4]
                lines.append(f"\nObserved {scene}, {method}, 0.6→1.4: "+'; '.join(f"{m} {lo[m+'_mean']:.2f}→{hi[m+'_mean']:.2f}" for m in METRICS)+'.')
                tp=[s['Throughput_mean'] for _,s in sorted(data.items())]
                lines.append('Throughput '+('increases monotonically' if all(a<=b for a,b in zip(tp,tp[1:])) else 'does not increase monotonically')+' across the available demand conditions.')
    for scene in scenes:
        for factor in FACTORS:
            pair={s['method']:s for s in summary if s['scenario']==scene and s['factor']==factor}
            if set(pair)=={'PhaseVLM','MaxPressure'}:
                a,b=pair['PhaseVLM'],pair['MaxPressure']
                changes='; '.join(f"{m} {(a[m+'_mean']/b[m+'_mean']-1)*100:+.2f}%" for m in METRICS if b[m+'_mean'])
                lines.append(f"\nObserved {scene}, x{factor:.1f}, PhaseVLM relative to MaxPressure: {changes}. Negative ATT/AQL/AWT means lower measured delay/queue, positive throughput means more completed trips.")
        high=[s for s in summary if s['scenario']==scene and s['factor']==1.4]
        for item in high:
            demand=conf['demand_validation']['actual_counts'][scene][-1]
            lines.append(f"\nObserved high-demand completion fraction, {scene} {item['method']}: {item['Throughput_mean']/demand*100:.2f}%, unfinished fraction {item['Unfinished_Vehicles_mean']/demand*100:.2f}%.")
    lines+=['\n## Interpretation and limitations', '\nPossible mechanism (interpretation, not a measured causal result): increasing arrivals can saturate movement service capacity, growing queues and continuous stop times. Throughput may plateau while unfinished demand rises. Lower completed-trip ATT alone would not demonstrate efficiency in a congested network; this study retains the original censored ATT and reports all unfinished demand. There is no assumed controller advantage. Error bars are sample standard deviations, not confidence intervals. Partial conditions must not be used for definitive stability or superiority claims. Only Jinan-3 and Hangzhou-2 are included in this representative experiment; the other three paper scenarios are not evaluated here. Archive weights are documented as approximate matches to the paper and the new results are not claimed as exact paper reproduction.', '\n## Exceptions / execution audit']
    for anomaly in conf.get('anomalies', []):
        lines.append('\nResolved startup anomaly: ' + json.dumps(anomaly, ensure_ascii=False))
    attempts = list((root/'attempts').glob('*')) if (root/'attempts').exists() else []
    lines.append(f'\nPreserved prior incomplete attempts: {len(attempts)}. These are excluded from accepted metrics.')
    for s in status:
        if s['status']=='failed': lines.append(f"\n- {s['id']}: failed; see `runs/{s['id']}/run.log` and `status.json`. {s.get('error','')[-700:]}")
    if not failed: lines.append('\nNo recorded failures in the current status snapshot.')
    lines+=['\n## Artifacts', '\n`metrics.csv`, `summary.csv`, `demand_statistics.csv`, `configuration.json`, `jobs.json`, `status.json`, `baseline_check.json`, `flows/`, `runs/`, Figure 1–3 PNG/PDF. Resume: `'+PYTHON+' scripts/run_flow_intensity.py launch --root '+str(root)+'`. Completed jobs are skipped; failed job workspaces are preserved before a retry.']
    lines += ['\n## Observed seed sensitivity and inference configuration']
    for (scene, factor, method), samples in groups.items():
        if factor == 1.0 and len(samples) == 3:
            identical = all(max(r[m] for r in samples)-min(r[m] for r in samples) < 1e-9 for m in METRICS)
            lines.append(f'\n{scene}, {method}, x1.0: ' +
                         ('all five metrics are identical across the three seeds. These are deterministic sensitivity repeats, not three independent demand samples.' if identical else
                          'simulator seed sensitivity is observed with identical input demand; see metric standard deviations in summary.csv.'))
    for adjustment in conf.get('inference_optimizations', []):
        lines.append('\nRecorded inference adjustment: ' + json.dumps(adjustment))
    lines.append('\nScope: this run completes the authorized representative subset (2 scenarios × 5 factors × 3 seeds × 2 controllers = 60 conditions). The full five-scenario matrix would contain 150 conditions and is not part of the accepted result count here. No missing scenario results are inferred.')
    lines.append('\nUnfinished demand includes normal trips still in progress at the finite simulation horizon as well as entry-buffer congestion. An increase alone does not prove gridlock; interpret completion fractions jointly with queues and waiting time. Five demand levels cannot precisely locate a critical capacity threshold.')
    (root/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    temporal_report(root)
    print(f'completed={complete}/{len(jobs)} failed={failed}',flush=True)
    return complete,failed



def temporal_report(root):
    import numpy as np
    import matplotlib.pyplot as plt
    jobs=json.loads((root/'jobs.json').read_text())
    grouped=collections.defaultdict(list)
    for job in jobs:
        out=root/'runs'/job['id']
        if not (out/'status.json').exists() or json.loads((out/'status.json').read_text())['status']!='completed':
            continue
        for step in json.loads((out/'step_metrics.json').read_text()):
            grouped[(job['scenario'],job['factor'],job['method'],step['time_s'])].append(step)
    rows=[]
    for (scene,factor,method,t),rr in sorted(grouped.items()):
        row=dict(scenario=scene,factor=factor,method=method,time_s=t,n=len(rr))
        for key in ['queue_vehicles','waiting_time_s','active_vehicles']:
            row[key+'_mean']=float(np.mean([r[key] for r in rr]))
            row[key+'_std']=float(np.std([r[key] for r in rr],ddof=1)) if len(rr)>1 else None
        rows.append(row)
    fields=['scenario','factor','method','time_s','n']+[k+s for k in ['queue_vehicles','waiting_time_s','active_vehicles'] for s in ['_mean','_std']]
    write_csv(root/'temporal_summary.csv',rows,fields)
    demand=list(csv.DictReader((root/'demand_statistics.csv').open()))
    lines=['\n## Temporal demand and response (observations)', '\nFigure 4 upper panel shows seed-2024 scheduled arrivals; lower panels show means over completed seed instances (three in the final results). It displays demand factors 0.6, 1.0 and 1.4.']
    for scene in SCENARIOS:
        bins=[r for r in demand if r['scenario']==scene and float(r['factor'])==1 and int(r['seed'])==2024]
        counts=np.array([int(r['actual_bin_vehicles']) for r in bins])
        peak=int(counts.argmax());start=int(bins[peak]['bin_start_s'])
        lines.append(f"\n{scene} original five-minute arrivals: {counts.tolist()}; peak {counts.max()} vehicles at {start//60}–{(start+300)//60} min; peak/mean={counts.max()/counts.mean():.2f}; bin CV={counts.std()/counts.mean():.3f}. These describe scheduled demand, not measured flow discharge.")
    fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True,constrained_layout=True)
    for factor,color in [(.6,'#4393c3'),(1.,'#2166ac'),(1.4,'#b2182b')]:
        bins=[r for r in demand if r['scenario']=='Hangzhou-2' and float(r['factor'])==factor and int(r['seed'])==2024]
        axes[0].step([int(r['bin_start_s'])/60 for r in bins]+[60],[int(r['actual_bin_vehicles']) for r in bins]+[int(bins[-1]['actual_bin_vehicles'])],where='post',color=color,label=f'{factor:.1f}× demand')
        for method,style in [('PhaseVLM','-'),('MaxPressure','--')]:
            rr=[r for r in rows if r['scenario']=='Hangzhou-2' and r['factor']==factor and r['method']==method]
            if not rr:continue
            axes[1].plot([r['time_s']/60 for r in rr],[r['queue_vehicles_mean'] for r in rr],color=color,ls=style,label=f'{method}, {factor:.1f}×')
            axes[2].plot([r['time_s']/60 for r in rr],[r['waiting_time_s_mean'] for r in rr],color=color,ls=style)
            if factor==1.4:
                q=max(rr,key=lambda r:r['queue_vehicles_mean']);w=max(rr,key=lambda r:r['waiting_time_s_mean'])
                lines.append(f"\nHangzhou-2 x1.4 {method}: maximum mean boundary queue {q['queue_vehicles_mean']:.2f} vehicles at {q['time_s']/60:.1f} min; maximum mean boundary waiting time {w['waiting_time_s_mean']:.2f} s at {w['time_s']/60:.1f} min. Temporal offsets from the demand peak are observations; buildup and limited discharge capacity are possible explanations, not experimentally isolated causes.")
    axes[0].set(title='Hangzhou-2: temporal demand and network response',ylabel='Arrivals (vehicles / 5 min)')
    axes[1].set(ylabel='Network queue (vehicles)')
    axes[2].set(ylabel='Mean waiting time (s)',xlabel='Simulation time (min)')
    axes[0].legend(ncol=3);axes[1].legend(ncol=3,fontsize=8)
    for ext in ['png','pdf']:fig.savefig(root/f'figure4_hangzhou_temporal.{ext}')
    plt.close(fig)
    with (root/'experiment_report.md').open('a') as handle:handle.write('\n'.join(lines)+'\n')


def audit(root, require_complete=True):
    """Independently reconcile accepted results against saved simulator artifacts."""
    import pandas as pd
    import numpy as np
    jobs = json.loads((root/'jobs.json').read_text())
    audits = []
    for job in jobs:
        out = root/'runs'/job['id']
        if not (out/'status.json').exists() and not require_complete:
            continue
        status = json.loads((out/'status.json').read_text())
        if status['status'] != 'completed' and not require_complete:
            continue
        assert status['status'] == 'completed', (job['id'], status)
        results = json.loads((out/'result.json').read_text())['results']
        cached = out/'independent_metrics_audit.json'
        if cached.exists():
            audits.append(json.loads(cached.read_text()))
            continue
        frames = []
        for path in (out/'env').glob('vehicle_inter_*.csv'):
            frame = pd.read_csv(path, index_col=0)
            frame = frame[~frame.index.str.contains('shadow') & frame.enter_time.notna()]
            frame['duration'] = frame.leave_time.fillna(3600) - frame.enter_time
            frames.append(frame[['duration']])
        att = float(pd.concat(frames).groupby(level=0).duration.sum().mean())
        steps = json.loads((out/'step_metrics.json').read_text())
        aql = float(np.mean([s['queue_vehicles'] for s in steps]))
        awt = float(np.mean([s['waiting_time_s'] for s in steps]))
        native = json.loads((out/'final_snapshot.json').read_text())
        flow = json.loads((Path(job['data'])/job['flow']).read_text())
        errors = dict(ATT=abs(att-results['test_avg_travel_time_over']),
                      AQL=abs(aql-results['test_avg_queue_len_over']),
                      AWT=abs(awt-results['test_avg_waiting_time_over']))
        assert all(error < 1e-8 for error in errors.values()), (job['id'], errors)
        assert len(steps) == 120 and steps[-1]['time_s'] == 3600
        assert native['finishedVehicleCnt'] == results['throughput']
        assert len(flow) == results['demand'] == results['throughput'] + results['unfinished_vehicles']
        if job['method'] == 'PhaseVLM':
            assert results['prompt_max_tokens'] <= 512
        actions = json.loads((out/'env'/'state_action.json').read_text())
        assert len(actions) == (12 if job['dataset']=='jinan' else 16)
        assert all(len(intersection)==120 for intersection in actions)
        assert all(set(step.get('action_logits',{})) == {'ETWT','NTST','ELWL','NLSL'}
                   for intersection in actions for step in intersection) if job['method']=='PhaseVLM' else True
        item = dict(id=job['id'], passed=True, metric_absolute_errors=errors,
                    native_finished_vehicles=native['finishedVehicleCnt'], demand=len(flow),
                    decisions=120, intersections=len(actions))
        save(cached, item)
        audits.append(item)
    if require_complete:
        conf = json.loads((root/'configuration.json').read_text())
        for spec in conf['scenarios'].values():
            assert digest(REPO/'data'/spec['directory']/spec['flow']) == spec['flow_sha256']
            assert digest(REPO/spec['checkpoint']) == spec['checkpoint_sha256']
        assert digest(REPO/'projection_train/run_projection_online.py') == conf['evaluator_sha256']
        assert digest(REPO/'projection_train/prefix_action_policy.py') == conf['policy_sha256']
        metrics = list(csv.DictReader((root/'metrics.csv').open()))
        summary = list(csv.DictReader((root/'summary.csv').open()))
        demand_rows = list(csv.DictReader((root/'demand_statistics.csv').open()))
        assert len(metrics) == 60 and len(summary) == 20 and len(demand_rows) == 360
        assert all(row['n'] == '3' for row in summary)
        for name in ['figure1_att','figure2_queue_wait','figure3_throughput_unfinished','figure4_hangzhou_temporal']:
            assert (root/(name+'.png')).stat().st_size > 10000
            assert (root/(name+'.pdf')).read_bytes().startswith(b'%PDF')
    save(root/('completion_audit.json' if require_complete else 'partial_metrics_audit.json'), dict(passed=True, accepted_jobs=len(audits),
         expected_jobs=len(jobs), audits=audits,
         excluded_attempts=len(list((root/'attempts').glob('*')))))
    print('PASS:', len(audits), 'accepted results reconcile with saved simulator artifacts')


def run_job(root, job, gpu):
    out=root/'runs'/job['id']; out.mkdir(parents=True,exist_ok=True)
    state=out/'status.json'
    if state.exists() and json.loads(state.read_text()).get('status')=='completed': return True
    if (out/'env').exists():
        previous=root/'attempts'/f"{job['id']}_{time.time_ns()}"
        previous.parent.mkdir(exist_ok=True); out.rename(previous); out.mkdir()
    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(REPO), MPLCONFIGDIR=str(root/'mplcache'))
    with (out/'run.log').open('w') as log:
        p=subprocess.Popen([PYTHON,'-u',str(Path(__file__).resolve()),'execute','--root',str(root),'--job',job['id']],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(out/'process.json',dict(pid=p.pid,gpu=gpu,started=time.time()))
        try: rc=p.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(p.pid,signal.SIGTERM); rc=124
    if rc:
        save(state,dict(status='failed',exit_code=rc,error='See run.log',gpu=gpu))
    return rc==0


def launch(root):
    jobs=json.loads((root/'jobs.json').read_text())
    conf=json.loads((root/'configuration.json').read_text())
    import platform
    conf.setdefault('environment',{}).update(platform=platform.platform(),python=sys.version, gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10).stdout)
    save(root/'configuration.json',conf)
    # Gate the full matrix on a real x1.0 PhaseVLM end-to-end simulation.
    baseline=next(j for j in jobs if j['scenario']=='Jinan-3' and j['factor']==1 and j['seed']==2024 and j['method']=='PhaseVLM')
    (root/'status.txt').write_text('baseline_check_running\n')
    if not run_job(root,baseline,0):
        save(root/'baseline_check.json',dict(passed=False,job=baseline['id']))
        report(root); (root/'status.txt').write_text('blocked_baseline_failed\n'); return
    result=json.loads((root/'runs'/baseline['id']/'result.json').read_text())
    r=result['results']
    assert all(math.isfinite(r[k]) and r[k]>0 for k in ['test_avg_travel_time_over','test_avg_queue_len_over','test_avg_waiting_time_over'])
    assert r['simulation_end_s']==3600 and r['decisions']==120 and r['prompt_max_tokens']<=512
    save(root/'baseline_check.json',dict(passed=True,job=baseline['id'],results=r,checks=['checkpoint loaded strictly with existing trained projectors','120 original control decisions completed','no prompt truncation','AQL/AWT independently match per-step aggregation','original evaluator ATT retained','vehicle counts conserved']))
    (root/'status.txt').write_text('matrix_running\n')
    report(root)
    # Dynamic queue keeps all four GPUs available to model inference, rather
    # than accidentally grouping all CPU baselines on the same worker IDs.
    import queue
    pending=queue.Queue()
    busy={}
    active_ids=set()
    for job in jobs:
        out=root/'runs'/job['id']
        state=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
        if state.get('status')=='completed':
            continue
        proc=json.loads((out/'process.json').read_text()) if (out/'process.json').exists() else {}
        alive=False
        if proc.get('pid'):
            try:
                os.kill(proc['pid'],0)
                alive=True
            except ProcessLookupError:
                pass
        if alive:
            busy[int(proc['gpu'])]=(job,proc['pid'])
            active_ids.add(job['id'])
        else:
            pending.put(job)
    def worker(gpu):
        if gpu in busy:
            job,pid=busy[gpu]
            # Adopt existing detached workers after a scheduler-only restart.
            started=time.monotonic()
            while True:
                try:
                    os.kill(pid,0)
                except ProcessLookupError:
                    break
                if time.monotonic()-started>3600:
                    import signal
                    os.killpg(pid,signal.SIGTERM)
                    save(root/'runs'/job['id']/'status.json',dict(status='failed',error='Adopted worker timeout',gpu=gpu))
                    break
                time.sleep(1)
        while True:
            try:
                job=pending.get_nowait()
            except queue.Empty:
                break
            run_job(root,job,gpu)
            print('finished',job['id'],flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,gpu) for gpu in range(4)]
        while not all(f.done() for f in futures):
            time.sleep(30)
            report(root)
        for f in futures: f.result()
    complete,failed=report(root)
    (root/'status.txt').write_text('completed\n' if complete==len(jobs) else 'completed_with_failures\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','execute','launch','report','audit']);p.add_argument('--root',required=True);p.add_argument('--job')
    args=p.parse_args();root=Path(args.root).resolve()
    if args.command=='prepare': prepare(root)
    elif args.command=='execute': execute(root,args.job)
    elif args.command=='launch': launch(root)
    elif args.command=='audit': audit(root)
    else: report(root)
