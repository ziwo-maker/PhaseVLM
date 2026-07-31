import os
import torch
import warnings

_OPTIONAL_MODEL_IMPORT_ERROR = None
_OPTIONAL_MODELS = {}
try:
    from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
        Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, FiLM, iTransformer, \
        Koopa, TiDE, FreTS, TimeMixer, TSMixer, SegRNN, MambaSimple, TemporalFusionTransformer, SCINet, PAttn, TimeXer, TimeLLM, VisionTS
    _OPTIONAL_MODELS = {
        'TimeLLM': TimeLLM,
        'VisionTS': VisionTS,
        'TimesNet': TimesNet,
        'Autoformer': Autoformer,
        'Transformer': Transformer,
        'Stationary': Nonstationary_Transformer,
        'DLinear': DLinear,
        'FEDformer': FEDformer,
        'Informer': Informer,
        'LightTS': LightTS,
        'Reformer': Reformer,
        'ETSformer': ETSformer,
        'PatchTST': PatchTST,
        'Pyraformer': Pyraformer,
        'MICN': MICN,
        'Crossformer': Crossformer,
        'FiLM': FiLM,
        'iTransformer': iTransformer,
        'Koopa': Koopa,
        'TiDE': TiDE,
        'FreTS': FreTS,
        'MambaSimple': MambaSimple,
        'TimeMixer': TimeMixer,
        'TSMixer': TSMixer,
        'SegRNN': SegRNN,
        'TemporalFusionTransformer': TemporalFusionTransformer,
        "SCINet": SCINet,
        'PAttn': PAttn,
        'TimeXer': TimeXer,
    }
except Exception as exc:
    _OPTIONAL_MODEL_IMPORT_ERROR = exc

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        # 模型注册表：字符串名称 -> 具体的模型类（可选模型失败时允许仅用 TimeVLM）
        self.model_dict = dict(_OPTIONAL_MODELS)
        # 模型差异概览（便于快速定位特性与适用场景）：
        # 1) Transformer 家族：Autoformer/Informer/FEDformer/ETSformer/PatchTST/iTransformer/TimeXer/Crossformer 等
        #    侧重长序列依赖与注意力机制改造（稀疏注意力、分解、频域/多尺度等）。
        # 2) 线性/混合结构：DLinear/LightTS/TimesNet/TimeMixer/TSMixer 等
        #    假设序列可用线性或混合模块解释，训练更快、参数更少。
        # 3) RNN/状态空间：SegRNN/MambaSimple/SCINet 等
        #    强调时序递归或状态更新，适合流式/长序列建模。
        # 4) 任务/模态扩展：TimeLLM/VisionTS/TemporalFusionTransformer/TiDE/FreTS 等
        #    融合大模型、视觉或多变量外生信息，强调可解释或多模态能力。
        if args.model == 'TimeVLM':
            # TimeVLM 仅在需要时动态导入，避免无关依赖加载
            from src.TimeVLM import model as TimeVLM
            self.model_dict['TimeVLM'] = TimeVLM
        elif _OPTIONAL_MODEL_IMPORT_ERROR:
            warnings.warn(
                "Optional forecasting models failed to import; only TimeVLM is available. "
                f"Original error: {_OPTIONAL_MODEL_IMPORT_ERROR}"
            )

        if args.model not in self.model_dict:
            available = ", ".join(sorted(self.model_dict.keys()))
            raise ValueError(f"Unknown or unavailable model '{args.model}'. Available: {available}")

        # 设备选择与模型构建
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        if args.is_training:
            # 训练阶段输出参数量，方便对比模型规模
            self._log_model_parameters()
        
        
    def _log_model_parameters(self):
        """
        打印模型参数。
        说明：区分可学习参数与总参数，便于评估模型规模与训练成本。
        """
        def count_learnable_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        def count_total_parameters(model):
            return sum(p.numel() for p in model.parameters())

        learable_params = count_learnable_parameters(self.model)
        total_params = count_total_parameters(self.model)
        print(f"Learnable model parameters: {learable_params:,}")
        print(f"Total model parameters: {total_params:,}")
        

    def _build_model(self):
        # 子类需要根据 args.model 构建具体模型
        raise NotImplementedError
        return None

    def _acquire_device(self):
        # 根据配置选择 GPU 或 CPU
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
