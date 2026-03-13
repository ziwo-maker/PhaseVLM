import importlib


class _LazyAgentDict:
    def __init__(self, mapping):
        self._mapping = dict(mapping)
        self._cache = {}

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        if key not in self._mapping:
            raise KeyError(key)
        module_name, class_name = self._mapping[key]
        try:
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        except Exception as exc:
            raise ImportError(
                f"Failed to import agent '{key}' from {module_name}.{class_name}: {exc}"
            ) from exc
        self._cache[key] = cls
        return cls

    def __contains__(self, key):
        return key in self._mapping

    def __iter__(self):
        return iter(self._mapping)

    def __len__(self):
        return len(self._mapping)

    def keys(self):
        return self._mapping.keys()

    def items(self):
        return ((key, self[key]) for key in self._mapping)

    def values(self):
        return (self[key] for key in self._mapping)

_AGENT_CLASS_PATHS = {
    "Random": ("models.random_agent", "RandomAgent"),
    "Fixedtime": ("models.fixedtime_agent", "FixedtimeAgent"),
    "MaxPressure": ("models.maxpressure_agent", "MaxPressureAgent"),
    "EfficientMaxPressure": ("models.efficient_maxpressure_agent", "EfficientMaxPressureAgent"),
    "AdvancedMaxPressure": ("models.advanced_maxpressure_agent", "AdvancedMaxPressureAgent"),
    "EfficientPressLight": ("models.presslight_one", "PressLightAgentOne"),
    "EfficientColight": ("models.colight_agent", "CoLightAgent"),
    "EfficientMPLight": ("models.mplight_agent", "MPLightAgent"),
    "MPLight": ("models.mplight_agent", "MPLightAgent"),
    "Colight": ("models.colight_agent", "CoLightAgent"),
    "AdvancedMPLight": ("models.advanced_mplight_agent", "AdvancedMPLightAgent"),
    "AdvancedColight": ("models.colight_agent", "CoLightAgent"),
    "AdvancedDQN": ("models.simple_dqn_one", "SimpleDQNAgentOne"),
    "Attend": ("models.attendlight_agent", "AttendLightAgent"),
    "ChatGPTTLCSWaitTimeForecast": ("models.chatgpt", "ChatGPTTLCS_Wait_Time_Forecast"),
    "ChatGPTTLCSCommonsense": ("models.chatgpt", "ChatGPTTLCS_Commonsense"),
}

DIC_AGENTS = _LazyAgentDict(_AGENT_CLASS_PATHS)

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
}
dic_traffic_env_conf = {

    # 支持的模型/算法名称列表（用于选择训练/对比的控制器）
    "LIST_MODEL": ["Random", "Fixedtime", "MaxPressure", "EfficientMaxPressure", "AdvancedMaxPressure",
                   "EfficientPressLight", "EfficientColight", "EfficientMPLight",
                   "AdvancedMPLight", "AdvancedColight", "AdvancedDQN", "Attend"],

    # 需要在训练过程中进行参数更新（学习/反向传播/更新Q网络等）的模型列表
    # 一般 Random/Fixedtime/MaxPressure 这类规则法不需要更新
    "LIST_MODEL_NEED_TO_UPDATE": ["EfficientPressLight", "EfficientColight", "EfficientMPLight",
                                  "AdvancedMPLight", "AdvancedColight", "AdvancedDQN", "Attend"],

    # 路口总车道（lane）数量（用于状态维度、lane统计等）
    "NUM_LANE": 12,

    # 相位映射表：每个“相位”对应哪些信号灯索引为绿/通行（具体索引含义取决于你的仿真环境定义）
    # 注：上方注释的 'WT_ET'... 是相位语义命名（西向直行/东向直行等）
    "PHASE_MAP": [
        [1, 4, 12, 13, 14, 15, 16, 17],
        [7, 10, 18, 19, 20, 21, 22, 23],
        [0, 3, 18, 19, 20, 21, 22, 23],
        [6, 9, 12, 13, 14, 15, 16, 17]
    ],
    # 备用的相位映射（注释掉的另一套相位定义）
    # "PHASE_MAP": [[0, 1, 15, 16, 17, 18, 19, 20], [3, 4, 12, 13, 14, 21, 22, 23],
    #               [9, 10, 18, 19, 20, 12, 13, 14], [6, 7, 21, 22, 23, 15, 16, 17]],

    # “遗忘/重置”轮数：用于某些训练机制（如经验缓存/历史统计）定期清空或衰减（具体取决于你的实现）
    "FORGET_ROUND": 20,

    # 仿真运行总步数/总时长（常见为秒数或环境 step 数；这里 3600 通常表示 1 小时）
    "RUN_COUNTS": 3600,

    # 当前实验使用的模型名（运行时再填；None表示由外部逻辑指定）
    "MODEL_NAME": None,

    # 邻接矩阵中每个路口保留的 Top-K 相邻路口数量（用于图结构/邻接稀疏化）
    "TOP_K_ADJACENCY": 5,

    # 动作执行模式：set 通常表示“直接设置相位”（而非 step/extend 等其它模式）
    "ACTION_PATTERN": "set",

    # 路口数量（多路口协同训练时会 >1）
    "NUM_INTERSECTIONS": 1,

    # 观测向量长度（state展开后的总维度；与你选择的 LIST_STATE_FEATURE 强相关）
    "OBS_LENGTH": 167,

    # 最小动作持续时间：一次决策（一次相位设置/切换）至少保持多少秒
    "MIN_ACTION_TIME": 30,    # 决策时间/控制步长

    # 统计/评估时间窗口：用于计算reward或统计交通指标的时间粒度（常与 MIN_ACTION_TIME 一致）
    "MEASURE_TIME": 30,

    # 是否对相位进行二进制展开（如 one-hot / binary 编码），用于状态表示或模型输入
    "BINARY_PHASE_EXPANSION": True,

    # 黄灯持续时间（秒）
    "YELLOW_TIME": 3,

    # 全红持续时间（秒），用于相位切换安全过渡（有的仿真为0）
    "ALL_RED_TIME": 2,

    # 相位数量（这里为4相位控制）
    "NUM_PHASES": 4,

    # 每个方向/进口道的车道数（与路口拓扑定义一致；这里四个进口各3条车道）
    "NUM_LANES": [3, 3, 3, 3],

    # 仿真最小时间粒度（每个 step 的秒数）；1 表示每步 1 秒
    "INTERVAL": 1,

    # 状态特征列表：决定智能体能观测到哪些信息（也决定 OBS_LENGTH）
    "LIST_STATE_FEATURE": [
        "cur_phase",                          # 当前相位编号/状态
        "time_this_phase",                    # 当前相位已经持续的时间
        "lane_num_vehicle",                   # 各车道车辆数（队列/占用近似）
        "lane_num_vehicle_downstream",        # 下游车道车辆数（反映溢出/拥堵传播）
        "traffic_movement_pressure_num",      # 以“车辆数差”定义的转向压力（movement级）
        "traffic_movement_pressure_queue",    # 以“排队长度”定义的转向压力（movement级）
        "traffic_movement_pressure_queue_efficient",  # 改进/高效版队列压力（你的实现定义）
        "pressure",                           # 路口压力（通常是上游-下游的差分指标）
        "adjacency_matrix"                    # 邻接矩阵（用于多路口图模型/协同控制）
    ],

    # reward 各项权重/开关（通常为线性加权；0 表示不计入）
    "DIC_REWARD_INFO": {
        "queue_length": 0,   # 队列长度惩罚项权重
        "pressure": 0,       # 压力惩罚/奖励项权重
    },

    # 相位灯色/通行模式定义：每个相位对应的8维二进制向量（与 list_lane_order 对齐）
    # 一般 1 表示该 movement 放行，0 表示不放行
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],  # 相位1：例如 WT/ET 放行（具体看 lane_order 映射）
        2: [0, 0, 0, 0, 0, 1, 0, 1],  # 相位2：例如 NT/ST 放行
        3: [1, 0, 1, 0, 0, 0, 0, 0],  # 相位3：例如 WL/EL 放行
        4: [0, 0, 0, 0, 1, 0, 1, 0]   # 相位4：例如 NL/SL 放行
    },

    # 车道/转向 movement 的固定顺序（用于把相位向量和具体 movement 对齐）
    "list_lane_order": ["WL", "WT", "EL", "ET", "NL", "NT", "SL", "ST"],

    # 相位名称列表（与 NUM_PHASES、PHASE_MAP/PHASE 对齐）
    "PHASE_LIST": ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL'],
}

DIC_BASE_AGENT_CONF = {
    # MLP/DNN隐藏层宽度（dense层神经元数），用于价值网络/策略网络等
    "D_DENSE": 20,

    # 学习率（optimizer 的 step size）
    "LEARNING_RATE": 0.001,

    # Early-stopping耐心值：验证指标多少轮不提升就停止（或用于某些调参逻辑）
    "PATIENCE": 10,

    # batch大小：每次训练更新使用的样本数量
    "BATCH_SIZE": 20,

    # 训练轮数/epoch 数
    "EPOCHS": 100,

    # 每轮训练采样的样本量（从replay buffer或数据集中抽取）
    "SAMPLE_SIZE": 3000,

    # 经验回放缓存最大长度（replay buffer容量）
    "MAX_MEMORY_LEN": 12000,

    # 目标网络/延迟网络更新频率（例如 DQN 里更新 target Q 网络的周期）
    "UPDATE_Q_BAR_FREQ": 5,

    # 是否按“通信轮(C round)”更新目标网络（联邦/分轮逻辑相关；False 表示按 freq 固定更新）
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    # 折扣因子 gamma：未来奖励折扣程度（RL常用）
    "GAMMA": 0.8,

    # 归一化因子：用于对 reward / state 做缩放（避免数值过大）
    "NORMAL_FACTOR": 20,

    # epsilon-greedy 的探索率（初始探索概率）
    "EPSILON": 0.8,

    # 探索率衰减系数：每轮将 epsilon 乘以该值
    "EPSILON_DECAY": 0.95,

    # 最小探索率：epsilon 衰减到该值后不再降低
    "MIN_EPSILON": 0.2,

    # 损失函数类型（用于监督更新Q值/价值网络；这里是MSE）
    "LOSS_FUNCTION": "mean_squared_error",
}


DIC_CHATGPT_AGENT_CONF = {
    "GPT_VERSION": "gpt-4",
    "LOG_DIR": "../GPT_logs"
}

DIC_FIXEDTIME_AGENT_CONF = {
    "FIXED_TIME": [30, 30, 30, 30]
}

DIC_MAXPRESSURE_AGENT_CONF = {
    "FIXED_TIME": [30, 30, 30, 30]
}
