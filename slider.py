"""阿里云拼图滑块：缺口检测 + 拟人拖动。

缺口检测
--------
缺口是**半透明白色遮罩**盖在原图上（实测 α≈0.7），所以缺口内部的
饱和度 S 明显下降、亮度 V 明显上升；而缺口周围一圈的 S/V 基本不变。
拿拼图块的 alpha 掩码做形状模板：内缩成「内部」，外扩成「环」，
在背景上滑动比较两者的 S/V 均值差，差最大的位置就是缺口。

y 不用搜：拼图块只做水平平移，缺口 y 恒等于拼图块不透明区的 y。
把 y 锁死在块 y（±2px）能干掉绝大部分假高分候选。

拖动
----
真人拖拽轨迹是唯一能过行为检测的形状 —— 自己造的 S 曲线会被判失败。
所以这里回放真人录制的 (t, dx, dy) 序列：水平位移按目标距离缩放，
其余原样，按浏览器自身的派发节奏走。

自检
----
    python slider.py --selftest       # 缺口检测对 5 张标注样本求误差
"""
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# 缺口检测
# ---------------------------------------------------------------------------

ALPHA_THRESHOLD = 20        # 拼图块不透明判据
INNER_KERNEL = 7            # 内缩核：只取缺口内部，避开边缘
RING_KERNEL = 9             # 外扩核：取缺口周围一圈
Y_TOLERANCE = 2             # 缺口 y 与块 y 的容差
V_WEIGHT = 60.0 / 255.0     # 亮度差的权重（把 V 的 0~255 折算到 S 的量级）


def piece_mask(puzzle):
    """拼图块的不透明区域。返回 (mask, x0, y0, w, h)，坐标为块图内坐标。"""
    alpha = (puzzle[:, :, 3] if puzzle.shape[2] == 4
             else 255 - cv2.cvtColor(puzzle, cv2.COLOR_BGR2GRAY))
    ys, xs = np.where(alpha > ALPHA_THRESHOLD)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    mask = (alpha[y0:y1, x0:x1] > ALPHA_THRESHOLD).astype(np.uint8)
    return mask, x0, y0, x1 - x0, y1 - y0


def detect_gap(bg, puzzle):
    """返回 {"x","y","score","piece_offset_x",...}，x/y 为缺口左上角（背景原图坐标）。"""
    got = piece_mask(puzzle)
    if got is None:
        return None
    mask, px0, py0, pw, ph = got
    H, W = bg.shape[:2]
    if pw < 8 or ph < 8 or pw >= W or ph >= H:
        return None

    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV).astype(np.float32)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]

    inner = cv2.erode(mask, np.ones((INNER_KERNEL,) * 2, np.uint8)).astype(np.float32)
    ring = (cv2.dilate(mask, np.ones((RING_KERNEL,) * 2, np.uint8))
            - mask).astype(np.float32)
    n_inner, n_ring = inner.sum(), ring.sum()
    if n_inner < 50 or n_ring < 50:
        return None

    best = None
    for dy in range(-Y_TOLERANCE, Y_TOLERANCE + 1):
        y = py0 + dy
        if y < 0 or y + ph > H:
            continue
        for x in range(0, W - pw + 1):
            s_win = sat[y:y + ph, x:x + pw]
            v_win = val[y:y + ph, x:x + pw]
            d_s = (s_win * ring).sum() / n_ring - (s_win * inner).sum() / n_inner
            d_v = (v_win * inner).sum() / n_inner - (v_win * ring).sum() / n_ring
            score = d_s + d_v * V_WEIGHT
            if best is None or score > best[0]:
                best = (score, x, y, d_s, d_v)

    if best is None:
        return None
    score, x, y, d_s, d_v = best
    return {
        "x": int(x), "y": int(y),
        "score": round(float(score), 2),
        "dS": round(float(d_s), 1), "dV": round(float(d_v), 1),
        "w": int(pw), "h": int(ph),
        "piece_offset_x": int(px0),     # 块图里不透明区的左偏移，换算位移时要减掉
    }


# ---------------------------------------------------------------------------
# 滑块行程 ↔ 拼图位移
# ---------------------------------------------------------------------------

def puzzle_to_slider(puzzle_px, max_travel=260.0):
    """拼图需要移动的距离 → 滑块需要拖的距离。

    阿里云不是 1:1 映射（拖 100px 时拼图只动 43px），解这条二次方程：

        12·s² + max_travel·s - 13·max_travel·puzzle = 0
    """
    a, b = 12.0, float(max_travel)
    c = -13.0 * max_travel * float(puzzle_px)
    disc = b * b - 4 * a * c
    if disc < 0:
        return 0.0
    return max(0.0, min(max_travel, (-b + math.sqrt(disc)) / (2 * a)))


# ---------------------------------------------------------------------------
# 拟人拖动（回放真人轨迹）
# ---------------------------------------------------------------------------

TRACES = os.path.join(HERE, "traces.json")
_traces = None


def _load_traces():
    global _traces
    if _traces is None:
        with open(TRACES, encoding="utf-8") as f:
            _traces = json.load(f)
    return _traces


def human_drag(page, start_x, start_y, distance):
    """从 (start_x, start_y) 按下，水平拖 distance 像素。

    轨迹取自真人成功通过验证的录制：先按目标距离缩放水平分量，
    再按原顺序回放。不额外 sleep —— CDP 每次派发的自身开销
    （约 10~16ms/次）正好落在真人 6~17ms 的采样区间里。
    """
    pts = random.choice(_load_traces())
    span = pts[-1][1]
    if abs(span) < 1:
        return
    k = float(distance) / span

    page.mouse.move(start_x, start_y)
    time.sleep(random.uniform(0.05, 0.12))
    page.mouse.down()
    time.sleep(random.uniform(0.02, 0.06))

    for (_t, dx, dy) in pts:
        page.mouse.move(start_x + dx * k, start_y + dy)
    page.mouse.move(start_x + distance, start_y + pts[-1][2])
    time.sleep(random.uniform(0.25, 0.45))
    page.mouse.up()


# ---------------------------------------------------------------------------
# 页面级封装：读几何 → 取图 → 算距离 → 拖
# ---------------------------------------------------------------------------

GEOM_JS = r"""
() => {
  const g = s => { const e = document.querySelector(s); if (!e) return null;
    const r = e.getBoundingClientRect();
    return {x: r.x, y: r.y, w: r.width, h: r.height,
            nw: e.naturalWidth || 0, nh: e.naturalHeight || 0}; };
  return {bg: g('#aliyunCaptcha-img'), puzzle: g('#aliyunCaptcha-puzzle'),
          slider: g('#aliyunCaptcha-sliding-slider'),
          body: g('#aliyunCaptcha-sliding-body'),
          text: (document.getElementById('aliyunCaptcha-sliding-text') || {}).textContent || '',
          present: !!document.querySelector('#aliyunCaptcha-img')};
}
"""

_TO_DATAURL = r"""
async (sel) => {
  const e = document.querySelector(sel);
  if (!e || !e.src) return null;
  const r = await fetch(e.src); const b = await r.blob();
  return await new Promise(res => { const f = new FileReader();
    f.onload = () => res(f.result); f.onerror = () => res(null); f.readAsDataURL(b); });
}
"""


def _decode(data_url):
    import base64
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)


def is_present(page):
    """滑块是否真的展开。

    注意 #aliyunCaptcha-img 在挑战没展开时也存在，只是 display:none / 尺寸为 0，
    所以必须看可见尺寸，不能只判元素存在。
    """
    return bool(page.evaluate("""() => {
        const e = document.querySelector('#aliyunCaptcha-img');
        if (!e) return false;
        const r = e.getBoundingClientRect();
        return r.width > 50 && r.height > 50;
    }"""))


def open_challenge(page, hover_moves=12):
    """点「Click to verify」，等拼图出现。元素刚渲染时还没布局，要等它有宽度。"""
    box = None
    for _ in range(20):
        b = page.locator("#aliyunCaptcha-captcha-body").bounding_box()
        if b and b["width"] > 0 and b["height"] > 0:
            box = b
            break
        time.sleep(0.5)
    if not box:
        return False

    for attempt in range(3):
        cx = box["x"] + box["width"] * 0.75
        cy = box["y"] + box["height"] / 2
        page.mouse.move(cx - 180, cy + 30)
        for i in range(1, hover_moves + 1):
            page.mouse.move(cx - 180 + i * 15, cy + 30 - i * 2.5)
            time.sleep(0.015)
        page.mouse.click(cx, cy)
        for _ in range(16):
            time.sleep(0.5)
            if is_present(page):
                return True
        if not page.locator("#aliyunCaptcha-captcha-body").count():
            return False
    return False


def solve(page, max_tries=4, wait_after=3.0):
    """过滑块。返回 True/False。"""
    for attempt in range(1, max_tries + 1):
        if not is_present(page):
            if not open_challenge(page):
                time.sleep(2.0)
                continue
        time.sleep(1.5)
        geom = page.evaluate(GEOM_JS)
        if not geom.get("bg") or not geom["bg"]["w"]:
            continue
        bg = _decode(page.evaluate(_TO_DATAURL, "#aliyunCaptcha-img"))
        pz = _decode(page.evaluate(_TO_DATAURL, "#aliyunCaptcha-puzzle"))
        if bg is None or pz is None:
            continue
        gap = detect_gap(bg, pz)
        if not gap:
            continue

        scale = geom["bg"]["w"] / geom["bg"]["nw"] if geom["bg"]["nw"] else 1.0
        puzzle_px = (gap["x"] - gap["piece_offset_x"]) * scale
        max_travel = geom["body"]["w"] - geom["slider"]["w"]
        dx = puzzle_to_slider(puzzle_px, max_travel)

        sx = geom["slider"]["x"] + geom["slider"]["w"] / 2
        sy = geom["slider"]["y"] + geom["slider"]["h"] / 2
        human_drag(page, sx, sy, dx)

        t0 = time.time()
        while time.time() - t0 < wait_after + 6:
            time.sleep(0.5)
            if not is_present(page):
                return True
            txt = page.evaluate(GEOM_JS)["text"] or ""
            if "fail" in txt.lower() or "失败" in txt:
                break
        # 失败：等挑战自动刷新后重试
        time.sleep(1.0)
    return False


def _selftest():
    """缺口检测自检：5 张人工标注样本，误差应 <= 5px。"""
    base = os.path.join(HERE, "samples")
    truth = {"1": 226, "2": 226, "3": 241, "4": 227, "5": 200}
    errs = []
    for n, want in sorted(truth.items()):
        bp = os.path.join(base, "bg_%s.png" % n)
        pp = os.path.join(base, "puzzle_%s.png" % n)
        if not os.path.exists(bp):
            continue
        r = detect_gap(cv2.imread(bp), cv2.imread(pp, cv2.IMREAD_UNCHANGED))
        err = r["x"] - want
        errs.append(abs(err))
        print("bg_%s  x=%3d  真值=%3d  误差=%+.0f" % (n, r["x"], want, err))
    if errs:
        print("平均误差 %.2fpx" % (sum(errs) / len(errs)))
    assert errs, "没有样本"
    assert sum(errs) / len(errs) <= 5.0, "缺口检测精度退化"

    for puzzle, actual in ((185.5, 218.0), (192.1, 222.0), (224.7, 241.0)):
        pred = puzzle_to_slider(puzzle)
        assert abs(pred - actual) < 1.0, (puzzle, pred, actual)
    print("映射公式 OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
