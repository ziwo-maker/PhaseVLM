from collections import defaultdict
from utils import parse_road_id

class IntersectionAnalyzer:
    def __init__(self, vehicle_data, roadnet_data):
        self.vehicles = vehicle_data
        self.roadnet = roadnet_data
        # 结构: { intersection_id: { road_id: [events] } }
        self.results = defaultdict(lambda: defaultdict(list))

    def analyze(self):
        """执行分析：将以车辆为中心的数据转换为以路口为中心"""
        print(f"Analyzing {len(self.vehicles)} vehicles...")
        
        for vehicle in self.vehicles:
            # 获取关键信息
            v_route = vehicle.get('route', [])
            v_start_time = vehicle.get('startTime', 0)
            # v_id = vehicle.get('vehicle', {}).get('id', 'unknown') # 假设车辆ID在详情里，或者列表索引
            
            # 遍历该车辆经过的所有道路
            for road_id in v_route:
                src, dst = parse_road_id(road_id)
                
                if src is None or dst is None:
                    continue
                
                # 构建事件记录
                event = {
                    "time": v_start_time, # 注意：这是进入系统的时间，代表流量需求的产生
                    "vehicle_count": 1
                    # "vehicle_info": vehicle # 如果需要详细车辆参数可取消注释
                }
                
                # 逻辑1: 归属到源路口 (作为出发道路)
                self.results[f"intersection_{src}"][road_id].append(event)
                
                # 逻辑2: 也可以归属到目标路口 (作为进入道路)，视分析需求而定
                # 这里我们主要按“路口辖区”来分，通常关注连接该路口的道路
        
        # 对结果按时间排序
        self._sort_events()
        return self.results

    def _sort_events(self):
        """对每条道路的事件按时间排序"""
        for intersect_id, roads in self.results.items():
            for road_id, events in roads.items():
                # 按时间升序排序
                events.sort(key=lambda x: x['time'])