import math
from typing import Dict, List, Tuple, Optional, Union
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

# 尝试导入 pypinyin，如果失败则使用备用方案
try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    print("Warning: pypinyin not available, using fallback pinyin mapping")


# ============================================================
# 配置类
# ============================================================

@dataclass
class KeyboardSimilarityConfig:
    """键盘相似度配置"""
    # 权重范围 (论文中的值)
    w_min: float = 3.75  # 最小权重
    w_max: float = 8.91  # 最大权重
    
    # 距离阈值
    tau: float = 3.0  # τ 是"多远算完全不相似"的阈值
    
    # 编辑操作代价
    insert_cost: float = 1.0  # 插入代价
    delete_cost: float = 1.0  # 删除代价
    
    # 温度/缩放参数
    gamma: float = 1.0  # 全局缩放因子
    temperature: float = 1.0  # 温度参数
    
    # 是否使用键盘距离（False 则退化为原始拼音相似度）
    use_keyboard_distance: bool = True


# ============================================================
# QWERTY 键盘布局
# ============================================================

# 标准 QWERTY 键盘布局坐标 (行, 列)，考虑实际物理位置偏移
QWERTY_LAYOUT: Dict[str, Tuple[float, float]] = {
    # 第一行 (Q-P)
    'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4),
    'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
    # 第二行 (A-L)，有 0.25 偏移
    'a': (1, 0.25), 's': (1, 1.25), 'd': (1, 2.25), 'f': (1, 3.25), 'g': (1, 4.25),
    'h': (1, 5.25), 'j': (1, 6.25), 'k': (1, 7.25), 'l': (1, 8.25),
    # 第三行 (Z-M)，有 0.75 偏移
    'z': (2, 0.75), 'x': (2, 1.75), 'c': (2, 2.75), 'v': (2, 3.75), 'b': (2, 4.75),
    'n': (2, 5.75), 'm': (2, 6.75),
}

# 计算归一化用的最大距离
_positions = list(QWERTY_LAYOUT.values())
D_MAX = max(
    math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    for i, p1 in enumerate(_positions)
    for p2 in _positions[i+1:]
)


# ============================================================
# 核心计算函数
# ============================================================

@lru_cache(maxsize=1024)
def keyboard_letter_cost(a: str, b: str) -> float:
    """
    键盘字母替换代价 (公式 2.1)
    
    cost_kbd(a → b) = ||pos(a) - pos(b)||_2 / d_max
    
    Args:
        a: 源字母
        b: 目标字母
        
    Returns:
        归一化的键盘距离 [0, 1]
    """
    if a == b:
        return 0.0
    
    pos_a = QWERTY_LAYOUT.get(a.lower())
    pos_b = QWERTY_LAYOUT.get(b.lower())
    
    if pos_a is None or pos_b is None:
        return 1.0
    
    dist = math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
    return min(dist / D_MAX, 1.0)


def weighted_edit_distance(s1: str, s2: str,
                          insert_cost: float = 1.0,
                          delete_cost: float = 1.0,
                          use_keyboard: bool = True) -> float:
    """
    加权编辑距离 (公式 2.2)
    
    使用键盘距离作为替换代价的 Levenshtein 算法
    
    Args:
        s1: 源字符串
        s2: 目标字符串
        insert_cost: 插入代价
        delete_cost: 删除代价
        use_keyboard: 是否使用键盘距离计算替换代价
        
    Returns:
        加权编辑距离
    """
    m, n = len(s1), len(s2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i * delete_cost
    for j in range(n + 1):
        dp[0][j] = j * insert_cost
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                if use_keyboard:
                    sub_cost = keyboard_letter_cost(s1[i-1], s2[j-1])
                else:
                    sub_cost = 1.0
                dp[i][j] = min(
                    dp[i-1][j] + delete_cost,
                    dp[i][j-1] + insert_cost,
                    dp[i-1][j-1] + sub_cost
                )
    
    return dp[m][n]


# ============================================================
# 拼音处理
# ============================================================

# 备用拼音映射表（常用字）
FALLBACK_PINYIN: Dict[str, List[str]] = {
    '机': ['ji'], '鸡': ['ji'], '几': ['ji'], '积': ['ji'], '级': ['ji'],
    '制': ['zhi'], '智': ['zhi'], '知': ['zhi'], '只': ['zhi'],
    '天': ['tian'], '田': ['tian'], '甜': ['tian'], '添': ['tian'],
    '大': ['da'], '达': ['da'], '打': ['da'],
    '小': ['xiao'], '笑': ['xiao'], '校': ['xiao'],
    '人': ['ren'], '任': ['ren'], '认': ['ren'],
    '的': ['de', 'di'], '地': ['de', 'di'], '得': ['de', 'dei'],
    '是': ['shi'], '事': ['shi'], '时': ['shi'], '十': ['shi'],
    # ... 可以继续添加更多常用字
}


def get_char_pinyin(char: str) -> List[str]:
    """
    获取汉字的拼音列表（无声调）
    
    Args:
        char: 单个汉字
        
    Returns:
        拼音列表（多音字返回多个读音）
    """
    if PYPINYIN_AVAILABLE:
        try:
            readings = pinyin(char, style=Style.NORMAL, heteronym=True)
            if readings and readings[0]:
                return [p.replace('ü', 'v') for p in readings[0]]
        except:
            pass
    
    # 使用备用映射
    return FALLBACK_PINYIN.get(char, [])


@lru_cache(maxsize=65536)
def pinyin_keyboard_distance(x: str, y: str, 
                             use_keyboard: bool = True) -> float:
    """
    计算两个汉字之间的拼音键盘距离
    
    d(x, y) = min_{p∈P(x), q∈P(y)} d_kbd(p, q)
    
    多音字取最小距离
    
    Args:
        x: 源汉字
        y: 目标汉字
        use_keyboard: 是否使用键盘距离
        
    Returns:
        最小拼音键盘距离
    """
    if x == y:
        return 0.0
    
    pinyin_x = get_char_pinyin(x)
    pinyin_y = get_char_pinyin(y)
    
    if not pinyin_x or not pinyin_y:
        return float('inf')
    
    min_dist = float('inf')
    for px in pinyin_x:
        for py in pinyin_y:
            dist = weighted_edit_distance(px, py, use_keyboard=use_keyboard)
            min_dist = min(min_dist, dist)
    
    return min_dist


# ============================================================
# 主要权重计算类
# ============================================================

class KeyboardPinyinWeightCalculator:
    """
    基于键盘拼音相似度的权重计算器
    
    这个类实现了图片中描述的核心思路：
    1. 用键盘距离定义拼音字符间的"软距离"
    2. 用加权 Levenshtein 计算拼音串距离
    3. 将距离映射为 DistL 的替换权重 w_S
    
    使用方法：
        calculator = KeyboardPinyinWeightCalculator()
        weight = calculator.get_substitution_weight('机', '制')
    """
    
    def __init__(self, config: KeyboardSimilarityConfig = None):
        """
        初始化计算器
        
        Args:
            config: 配置对象，None 则使用默认配置
        """
        self.config = config or KeyboardSimilarityConfig()
    
    def get_substitution_weight(self, x_char: str, y_char: str) -> float:
        """
        获取替换权重 (公式 2.3)
        
        w_S(x, y) = w_min + (w_max - w_min) · clip(d(x,y)/τ, 0, 1)
        
        Args:
            x_char: 原字符
            y_char: 候选替换字符
            
        Returns:
            替换权重
        """
        if x_char == y_char:
            return 0.0
        
        # 计算拼音键盘距离
        d = pinyin_keyboard_distance(
            x_char, y_char, 
            use_keyboard=self.config.use_keyboard_distance
        )
        
        if d == float('inf'):
            return self.config.w_max * self.config.gamma * self.config.temperature
        
        # 应用 clip 和线性映射
        tau = self.config.tau * self.config.temperature
        clipped = min(max(d / tau, 0.0), 1.0)
        weight = self.config.w_min + (self.config.w_max - self.config.w_min) * clipped
        
        return weight * self.config.gamma
    
    def get_insertion_weight(self) -> float:
        """获取插入权重"""
        return self.config.w_max * self.config.gamma * self.config.temperature
    
    def get_deletion_weight(self) -> float:
        """获取删除权重"""
        return self.config.w_max * self.config.gamma * self.config.temperature
    
    def wS(self, x: str, y: str) -> float:
        """w_S(x, y) 的简写别名"""
        return self.get_substitution_weight(x, y)


# ============================================================
# Simple-CSC 框架适配器
# ============================================================

class TransformationTypeAdapter:
    """
    用于替换 simple-csc 中 TransformationType 类的适配器
    
    这个类可以直接替换原代码中基于分箱（Same Pinyin / Similar Pinyin / 
    Similar Shape / Other Similar）的权重计算逻辑。
    
    使用方法：
    
    1. 直接替换模式：
       在 transformation_type.py 中:
       ```python
       from keyboard_similarity_integration import TransformationTypeAdapter
       adapter = TransformationTypeAdapter()
       # 替换原有的 get_edit_weight 函数调用
       weight = adapter.get_edit_weight(x_char, y_char, op_type)
       ```
    
    2. 继承模式：
       创建新类继承原 TransformationType 并覆盖权重计算方法
    """
    
    def __init__(self, 
                 w_min: float = 3.75,
                 w_max: float = 8.91,
                 tau: float = 3.0,
                 gamma: float = 1.0,
                 temperature: float = 1.0):
        """
        初始化适配器
        
        Args:
            w_min: 最小权重
            w_max: 最大权重
            tau: 距离阈值
            gamma: 缩放因子
            temperature: 温度参数
        """
        config = KeyboardSimilarityConfig(
            w_min=w_min,
            w_max=w_max,
            tau=tau,
            gamma=gamma,
            temperature=temperature
        )
        self.calculator = KeyboardPinyinWeightCalculator(config)
        self.w_min = w_min
        self.w_max = w_max
    
    def get_edit_weight(self, 
                        x_char: str, 
                        y_char: str, 
                        op_type: str = 'sub') -> float:
        """
        获取编辑权重（兼容原始接口）
        
        这个方法可以直接替换原代码中的 get_edit_weight 或 wS 函数
        
        Args:
            x_char: 原字符
            y_char: 候选字符
            op_type: 操作类型 ('sub', 'insert', 'delete', 'keep')
            
        Returns:
            编辑权重
        """
        if op_type == 'keep' or x_char == y_char:
            return 0.0
        elif op_type == 'sub':
            return self.calculator.get_substitution_weight(x_char, y_char)
        elif op_type == 'insert':
            return self.calculator.get_insertion_weight()
        elif op_type == 'delete':
            return self.calculator.get_deletion_weight()
        else:
            return self.w_max
    
    def wS(self, xm: str, yn: str) -> float:
        """
        论文中的 w_S(x_m, y_n) 函数
        
        可以直接替换原代码中的 wS(xm, yn) 调用
        """
        return self.get_edit_weight(xm, yn, 'sub')
    
    def get_similarity_category(self, x_char: str, y_char: str) -> str:
        """
        获取相似度分类（用于调试和分析）
        
        基于权重值反推相似度类别
        """
        if x_char == y_char:
            return "SAME"
        
        weight = self.calculator.get_substitution_weight(x_char, y_char)
        
        # 根据权重划分类别
        if weight <= self.w_min + 0.1:
            return "SAME_PINYIN"
        elif weight <= self.w_min + (self.w_max - self.w_min) * 0.3:
            return "SIMILAR_PINYIN"
        elif weight <= self.w_min + (self.w_max - self.w_min) * 0.6:
            return "SIMILAR_KEYBOARD"
        else:
            return "OTHER"


# ============================================================
# 工具函数
# ============================================================

def create_weight_function(w_min: float = 3.75,
                          w_max: float = 8.91,
                          tau: float = 3.0) -> callable:
    """
    创建一个权重计算函数
    
    方便在函数式编程风格中使用
    
    Args:
        w_min: 最小权重
        w_max: 最大权重
        tau: 距离阈值
        
    Returns:
        权重计算函数 (x_char, y_char) -> float
    """
    adapter = TransformationTypeAdapter(w_min=w_min, w_max=w_max, tau=tau)
    return adapter.wS


def batch_compute_weights(char_pairs: List[Tuple[str, str]],
                         w_min: float = 3.75,
                         w_max: float = 8.91,
                         tau: float = 3.0) -> List[float]:
    """
    批量计算替换权重
    
    Args:
        char_pairs: 字符对列表 [(x, y), ...]
        w_min: 最小权重
        w_max: 最大权重
        tau: 距离阈值
        
    Returns:
        权重列表
    """
    weight_fn = create_weight_function(w_min, w_max, tau)
    return [weight_fn(x, y) for x, y in char_pairs]


# ============================================================
# 参数搜索工具
# ============================================================

def grid_search_tau(validation_data: List[Tuple[str, str, bool]],
                    tau_values: List[float] = None,
                    w_min: float = 3.75,
                    w_max: float = 8.91,
                    threshold: float = None) -> Dict:
    """
    网格搜索最优 τ 参数
    
    Args:
        validation_data: 验证数据 [(src_char, tgt_char, should_replace), ...]
        tau_values: 要搜索的 τ 值列表
        w_min: 最小权重
        w_max: 最大权重
        threshold: 判断阈值（低于此权重则替换）
        
    Returns:
        包含最优参数和各 τ 值指标的字典
    """
    if tau_values is None:
        tau_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    
    if threshold is None:
        threshold = (w_min + w_max) / 2
    
    results = {}
    
    for tau in tau_values:
        adapter = TransformationTypeAdapter(w_min=w_min, w_max=w_max, tau=tau)
        
        tp = fp = tn = fn = 0
        for src, tgt, should_replace in validation_data:
            weight = adapter.wS(src, tgt)
            predicted_replace = weight < threshold
            
            if should_replace and predicted_replace:
                tp += 1
            elif should_replace and not predicted_replace:
                fn += 1
            elif not should_replace and predicted_replace:
                fp += 1
            else:
                tn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[tau] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        }
    
    # 找到最优 τ
    best_tau = max(results.keys(), key=lambda t: results[t]['f1'])
    
    return {
        'best_tau': best_tau,
        'best_f1': results[best_tau]['f1'],
        'all_results': results
    }


# ============================================================
# 演示和测试
# ============================================================

def demo_integration():
    """演示如何集成到 simple-csc"""
    print("=" * 70)
    print("键盘拼音相似度权重计算 - Simple-CSC 集成演示")
    print("=" * 70)
    
    # 创建适配器
    adapter = TransformationTypeAdapter(
        w_min=3.75,
        w_max=8.91,
        tau=3.0
    )
    
    # 测试案例
    test_cases = [
        ('机', '机', '相同'),
        ('机', '鸡', '同音'),
        ('机', '制', '近音'),
        ('机', '人', '远音'),
        ('天', '添', '同音'),
        ('大', '小', '不同'),
    ]
    
    print("\n替换权重计算示例:")
    print("-" * 50)
    print(f"{'原字':<6}{'候选':<6}{'权重':<12}{'分类':<15}")
    print("-" * 50)
    
    for x, y, desc in test_cases:
        weight = adapter.wS(x, y)
        category = adapter.get_similarity_category(x, y)
        print(f"{x:<6}{y:<6}{weight:<12.4f}{category:<15}({desc})")
    
    # 演示如何在代码中使用
    print("\n" + "=" * 70)
    print("代码集成示例")
    print("=" * 70)
    
    code_example = '''
# 在 simple-csc 代码中的使用方式:

# 方式 1: 直接使用适配器
from keyboard_similarity_integration import TransformationTypeAdapter

adapter = TransformationTypeAdapter(
    w_min=3.75,   # 论文值
    w_max=8.91,   # 论文值
    tau=3.0       # 可调参数，推荐 2.5~3.5
)

# 替换原有的权重计算
weight = adapter.get_edit_weight(x_char, y_char, 'sub')
# 或者
weight = adapter.wS(x_char, y_char)


# 方式 2: 使用函数式接口
from keyboard_similarity_integration import create_weight_function

wS = create_weight_function(w_min=3.75, w_max=8.91, tau=3.0)
weight = wS('机', '制')


# 方式 3: 批量计算
from keyboard_similarity_integration import batch_compute_weights

pairs = [('机', '鸡'), ('天', '添'), ('大', '小')]
weights = batch_compute_weights(pairs, tau=3.0)
'''
    print(code_example)
    
    # 演示参数敏感性
    print("\n" + "=" * 70)
    print("τ 参数敏感性分析 (用于调参)")
    print("=" * 70)
    
    tau_values = [2.0, 2.5, 3.0, 3.5, 4.0]
    pairs = [('机', '制'), ('机', '鸡'), ('大', '小')]
    
    print(f"\n{'字符对':<12}" + "".join([f"τ={t:<8}" for t in tau_values]))
    print("-" * 60)
    
    for x, y in pairs:
        weights = []
        for tau in tau_values:
            adapter = TransformationTypeAdapter(tau=tau)
            weights.append(f"{adapter.wS(x, y):.2f}")
        print(f"'{x}'→'{y}'".ljust(12) + "".join([f"{w:<8}" for w in weights]))


if __name__ == "__main__":
    demo_integration()