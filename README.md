# 阿里云拼图滑块

`slider.py` 是单文件实现，依赖 `opencv-python` + `numpy` + `playwright`。

```bash
python slider.py --selftest        # 缺口检测精度自检（5 张标注样本）
```

```python
import slider
slider.solve(page)                 # 传 playwright Page，返回 True/False
```

## 算法

### 1. 缺口定位

缺口不是"原图上的一个洞"，而是原图被一层半透明白色遮罩重新填充过。所以：

- 缺口**内部**的饱和度 S 下降、亮度 V 上升；
- 缺口**周围一圈**的 S/V 基本不变。

拿拼图块的 alpha 掩码做形状模板，内缩成「内部」、外扩成「环」，
在背景图上滑窗比较两者的 S/V 均值差，差最大的位置就是缺口：

```
score(x) = mean_S(ring) − mean_S(inner) + k · [mean_V(inner) − mean_V(ring)]
```

不靠颜色匹配、不靠边缘检测——缺口的内容已经被遮罩改过，但遮罩这个**物理事实**改不掉。

### 2. y 轴约束

拼图块只做水平平移，缺口 y 恒等于拼图块不透明区的 y。所以 y 不搜索，
直接锁死在块 y（±2px）。这一步剔除掉了绝大部分评分更高但位置错误的候选。

### 3. 滑块行程换算

滑块行程与拼图位移不是 1:1（拖 100px 拼图只动 43px），按二次关系反解：

```
12·s² + maxTravel·s − 13·maxTravel·puzzle = 0
```

### 4. 拖动

回放真人录制的手动拖拽轨迹（`traces.json`）：水平分量按目标距离缩放，
其余原样，按浏览器自身的派发节奏走。

## 精度

缺口检测对 5 张人工标注样本的平均误差 **2.2px**，最大 4px。

## 文件

| 文件 | 内容 |
|---|---|
| `slider.py` | 检测 + 换算 + 拖动，单文件 |
| `traces.json` | 真人拖拽轨迹（3 条，`[t, dx, dy]`） |
| `samples/` | 5 组 `bg_N.png` + `puzzle_N.png`，`--selftest` 用 |
