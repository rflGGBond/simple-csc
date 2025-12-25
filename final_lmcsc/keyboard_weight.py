"""
keyboard_weight.py - 键盘拼音相似度权重计算模块

功能：
- 基于 QWERTY 键盘布局计算拼音字母替换代价
- 使用加权 Levenshtein 距离计算拼音相似度
- 将连续的拼音距离映射为替换权重

公式：
- cost_kbd(a → b) = ||pos(a) - pos(b)||_2 / d_max
- d(x, y) = min_{p∈pinyin(x), q∈pinyin(y)} weighted_levenshtein(p, q)
- w_S(x, y) = w_min + (w_max - w_min) · clip(d(x,y)/τ, 0, 1)
"""

import math
from functools import lru_cache
from typing import Tuple, Dict

try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    print("[keyboard_weight] Warning: pypinyin not available, using fallback")


# ============================================================
# QWERTY 键盘布局
# ============================================================

QWERTY_POSITIONS: Dict[str, Tuple[float, float]] = {
    # 第一行
    'q': (0, 0), 'w': (0, 1), 'e': (0, 2), 'r': (0, 3), 't': (0, 4),
    'y': (0, 5), 'u': (0, 6), 'i': (0, 7), 'o': (0, 8), 'p': (0, 9),
    # 第二行（向右偏移 0.25）
    'a': (1, 0.25), 's': (1, 1.25), 'd': (1, 2.25), 'f': (1, 3.25), 'g': (1, 4.25),
    'h': (1, 5.25), 'j': (1, 6.25), 'k': (1, 7.25), 'l': (1, 8.25),
    # 第三行（向右偏移 0.75）
    'z': (2, 0.75), 'x': (2, 1.75), 'c': (2, 2.75), 'v': (2, 3.75), 'b': (2, 4.75),
    'n': (2, 5.75), 'm': (2, 6.75),
}

# 计算最大键盘距离（用于归一化）
_positions = list(QWERTY_POSITIONS.values())
D_MAX = max(
    math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    for i, p1 in enumerate(_positions) for p2 in _positions[i+1:]
)


# ============================================================
# 核心算法
# ============================================================

@lru_cache(maxsize=4096)
def keyboard_letter_cost(a: str, b: str) -> float:
    """
    键盘字母替换代价
    
    cost_kbd(a → b) = ||pos(a) - pos(b)||_2 / d_max
    
    返回值范围：[0, 1]
    """
    if a == b:
        return 0.0
    pos_a = QWERTY_POSITIONS.get(a.lower())
    pos_b = QWERTY_POSITIONS.get(b.lower())
    if pos_a is None or pos_b is None:
        return 1.0  # 非键盘字符，最大代价
    dist = math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)
    return min(dist / D_MAX, 1.0)


def weighted_levenshtein(p: str, q: str, insert_cost: float = 1.0, delete_cost: float = 1.0) -> float:
    """
    加权 Levenshtein 距离
    
    替换代价使用键盘距离，插入/删除使用固定代价
    """
    m, n = len(p), len(q)
    
    # 创建 DP 表
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化边界
    for i in range(m + 1):
        dp[i][0] = i * delete_cost
    for j in range(n + 1):
        dp[0][j] = j * insert_cost
    
    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[i-1] == q[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                sub_cost = keyboard_letter_cost(p[i-1], q[j-1])
                dp[i][j] = min(
                    dp[i-1][j] + delete_cost,      # 删除
                    dp[i][j-1] + insert_cost,      # 插入
                    dp[i-1][j-1] + sub_cost        # 替换
                )
    
    return dp[m][n]


@lru_cache(maxsize=65536)
def get_char_pinyin(char: str) -> Tuple[str, ...]:
    """
    获取汉字的拼音列表（支持多音字）
    
    返回：拼音元组，如 ('ji', 'qi') 或空元组
    """
    if not PYPINYIN_AVAILABLE:
        return ()
    try:
        readings = pinyin(char, style=Style.NORMAL, heteronym=True)
        if readings and readings[0]:
            # 将 ü 转换为 v（键盘上没有 ü）
            return tuple(p.replace('ü', 'v') for p in readings[0])
    except:
        pass
    return ()


@lru_cache(maxsize=65536)
def pinyin_keyboard_distance(x: str, y: str) -> float:
    """
    计算两个汉字之间的拼音键盘距离
    
    对于多音字，取所有读音组合中的最小距离
    
    返回：
    - 0.0: 完全相同（同字）
    - 0.0: 同音（如 机/鸡）
    - >0: 不同音，返回最小编辑距离
    - inf: 无法计算（非汉字）
    """
    if x == y:
        return 0.0
    
    pinyin_x = get_char_pinyin(x)
    pinyin_y = get_char_pinyin(y)
    
    if not pinyin_x or not pinyin_y:
        return float('inf')
    
    # 多音字取最小距离
    min_dist = float('inf')
    for px in pinyin_x:
        for py in pinyin_y:
            dist = weighted_levenshtein(px, py)
            min_dist = min(min_dist, dist)
            if min_dist == 0:
                return 0.0  # 找到同音，提前返回
    
    return min_dist


def get_keyboard_weight(
    x_char: str, 
    y_char: str, 
    w_min: float = 3.75, 
    w_max: float = 8.91, 
    tau: float = 3.0
) -> float:
    """
    获取字符替换权重
    
    公式：w_S(x, y) = w_min + (w_max - w_min) · clip(d(x,y)/τ, 0, 1)
    
    Args:
        x_char: 原字符（观测序列中的字符）
        y_char: 候选字符（词表中 token 的字符）
        w_min: 最小权重（同音字），默认 3.75
        w_max: 最大权重（完全不同），默认 8.91
        tau: 距离阈值，控制权重变化的敏感度，默认 3.0
             - τ 小：对距离敏感，远距离替换惩罚大（高 precision）
             - τ 大：对距离不敏感，允许更多替换（高 recall）
    
    Returns:
        替换权重（正值），范围 [0, w_max]
        - 0.0: 相同字符
        - w_min: 同音字
        - w_max: 完全不同或非汉字
    """
    if x_char == y_char:
        return 0.0
    
    d = pinyin_keyboard_distance(x_char, y_char)
    
    if d == float('inf'):
        return w_max
    
    # clip(d/τ, 0, 1)
    normalized = min(max(d / tau, 0.0), 1.0)
    
    return w_min + (w_max - w_min) * normalized


# ============================================================
# 类封装（可选）
# ============================================================

class KeyboardWeightCalculator:
    """
    键盘拼音相似度权重计算器（类封装）
    
    用于需要固定参数的场景
    """
    
    def __init__(self, w_min: float = 3.75, w_max: float = 8.91, tau: float = 3.0):
        self.w_min = w_min
        self.w_max = w_max
        self.tau = tau
    
    def __call__(self, x_char: str, y_char: str) -> float:
        return get_keyboard_weight(x_char, y_char, self.w_min, self.w_max, self.tau)
    
    def get_weight(self, x_char: str, y_char: str) -> float:
        return self(x_char, y_char)


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("键盘拼音相似度权重测试")
    print("=" * 60)
    
    test_pairs = [
        ('机', '机', '相同字符'),
        ('机', '鸡', '同音 jī'),
        ('机', '几', '同音 jī/jǐ'),
        ('机', '制', '近音 jī/zhì'),
        ('机', '智', '近音 jī/zhì'),
        ('机', '人', '远音 jī/rén'),
        ('天', '添', '同音 tiān'),
        ('大', '小', '不同 dà/xiǎo'),
        ('的', '得', '近音 de/dé'),
        ('他', '她', '同音 tā'),
    ]
    
    print(f"\n{'原字':<4}{'候选':<4}{'拼音距离':<10}{'权重':<8}说明")
    print("-" * 60)
    
    for x, y, desc in test_pairs:
        d = pinyin_keyboard_distance(x, y)
        w = get_keyboard_weight(x, y)
        d_str = f"{d:.4f}" if d != float('inf') else "inf"
        print(f"{x:<4}{y:<4}{d_str:<10}{w:.4f}  {desc}")
    
    print("\n" + "=" * 60)
    print("τ 参数敏感性分析")
    print("=" * 60)
    
    tau_values = [2.0, 2.5, 3.0, 3.5, 4.0]
    print(f"\n{'字符对':<12}" + "".join(f"τ={t:<7}" for t in tau_values))
    print("-" * 60)
    
    for x, y, _ in [('机', '制', ''), ('机', '鸡', ''), ('大', '小', ''), ('机', '人', '')]:
        weights = [f"{get_keyboard_weight(x, y, tau=t):.2f}" for t in tau_values]
        print(f"'{x}' → '{y}'    " + "   ".join(f"{w:<7}" for w in weights))
    
    print("\n" + "=" * 60)
    print("与原有离散权重的对比")
    print("=" * 60)
    print("""
原有离散权重（从配置文件）:
  SAP (同音):  -3.75
  SIP (近音):  -4.85
  SIS (形近):  -5.40
  OTP (其他):  -8.91

键盘相似度连续权重（τ=3.0）:
  同音 (d=0):   3.75  ← 与 SAP 相同
  近音 (d≈0.6): 4.79  ← 介于 SAP 和 SIP 之间
  远音 (d≈1.2): 5.87  ← 介于 SIP 和 SIS 之间
  很远 (d≥3):   8.91  ← 与 OTP 相同

优势：连续权重能更精细地区分不同程度的拼音相似度
""")