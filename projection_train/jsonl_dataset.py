import json
from dataclasses import dataclass, field
from typing import List, Dict, Iterable

from utils.my_utils import location_direction_dict


@dataclass
class PromptSample:
    # 单条样本：文本提示、对应的序列特征、日期
    prompt: str
    sequence: List[List[float]]
    date: str
    lane_details: Dict = field(default_factory=dict)
    snapshot_path: str = ""
    intersection_id: str = ""


class TrafficPromptSeriesDataset:
    # 读取 jsonl 文件并构造“提示词 + 交通序列”样本
    def __init__(
        self,
        path: str,
        intersection_id: str,
        seq_len: int = 15,
        movement_keys: Iterable[str] = None,
    ) -> None:
        # 数据文件路径（jsonl，一行一个记录）
        self.path = path
        # 需要筛选的路口 id
        self.intersection_id = intersection_id
        # 序列长度，确保为 int
        self.seq_len = int(seq_len)
        # 交通流向/车道方向的键顺序；若未传入则使用预设字典顺序
        self.movement_keys = list(movement_keys) if movement_keys is not None else list(location_direction_dict)
        # 加载后的样本列表
        self.samples: List[PromptSample] = []
        # 立即读取并构建样本
        self._load()

    def _load(self) -> None:
        # 按 prompt 切分的时间序列缓存（仅包含目标路口）
        group_history: List[Dict[str, int]] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                # 跳过空行
                if not line.strip():
                    continue
                # 解析单行 json
                record = json.loads(line)
            

                # 取出当前时刻的各方向流量计数
                movement_counts = record.get("movement_counts") or {}
                # 将当前计数加入组内历史（时间序列）
                group_history.append(movement_counts)
                # 取出提示词；按 prompt 切分为一组样本
                prompt = record.get("prompt") or ""
                if not prompt:
                    continue
                # 用当前组历史构造固定长度的序列
                seq = self._build_sequence(group_history)
                # 收集样本（日期字段可为空）
                lane_details = record.get("lane_details") or {}
                snapshot_path = record.get("snapshot_path") or ""
                raw_intersection_id = record.get("intersection_id")
                if raw_intersection_id is None:
                    raw_intersection_id = record.get("road_id")
                intersection_id = "" if raw_intersection_id is None else str(raw_intersection_id)
                self.samples.append(
                    PromptSample(
                        prompt=prompt,
                        sequence=seq,
                        date=str(record.get("date", "")),
                        lane_details=lane_details,
                        snapshot_path=str(snapshot_path),
                        intersection_id=intersection_id,
                    )
                )
                # 当前组完成，清空历史等待下一组
                group_history = []

    def _build_sequence(self, history: List[Dict[str, int]]) -> List[List[float]]:
        # 将历史计数转换成固定长度的序列向量
        seq = []
        # 只取最后 seq_len 个时刻
        start_index = max(0, len(history) - self.seq_len)
        window = history[start_index:]
        # 如果不足 seq_len，则用最早一条记录进行前置填充
        if len(window) < self.seq_len:
            pad = window[0] if window else {}
            for _ in range(self.seq_len - len(window)):
                seq.append(self._counts_to_vector(pad))
        # 将窗口内的每个时刻转换为向量
        for counts in window:
            seq.append(self._counts_to_vector(counts))
        return seq

    def _counts_to_vector(self, counts: Dict[str, int]) -> List[float]:
        # 按固定顺序将计数字典映射为向量，缺失键默认 0
        return [float(counts.get(key, 0.0)) for key in self.movement_keys]

    def __len__(self) -> int:
        # 样本数量
        return len(self.samples)

    def __getitem__(self, idx: int) -> PromptSample:
        # Indexable access for shuffled sampling.
        return self.samples[idx]

    def __iter__(self):
        # 允许直接遍历数据集
        return iter(self.samples)
