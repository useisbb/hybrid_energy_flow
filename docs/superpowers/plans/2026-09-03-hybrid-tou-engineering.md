# 光储一体机模型 + TOU 仿真测试 工程化改造 实施计划

> **For agentic workers:** 可选按 superpowers:executing-plans 逐任务执行。任务用 `- [ ]` 复选框跟踪。

**Goal:** 把现有的 `mode_hybrid.py`（混你/光储一体机模型）+ `main.py`（TOU 4 工况测试与报告）整理为"配置可改、场景可复用、运行可复现、主流程清晰"的轻量工程，不改变既有行为策略（绿电优先、四端口数据、工况5 不单列章节）。

**Architecture:** 抽公共配置与 TOU 时段表到 `config.py`；把曲线/辐照/负载构建与场景定义集中到 `scenarios.py`；把绘图助手与 Markdown 报告生成拆到 `reporting.py`；`main.py` 只做编排（参数化运行、[OK]/[FAIL] 校验、退出码、CSV 导出）。`mode_hybrid.py` 保持为纯模型模块（导入不打印）。

**Tech Stack:** Python 3.9+（当前 3.13）、numpy、pandas、matplotlib（`Agg` 后端）；Windows PowerShell 运行。

**Spec:** 本文档即规格来源：以现有 [mode_hybrid.py](../mode_hybrid.py) 与 [main.py](../main.py) 的行为为准（本文件从 d:\Code\py_test 根目录的 `docs/superpowers/plans/` 视角书写，相对链接按需调整）。

## Global Constraints

- 平台：Windows + PowerShell；所有命令 `python xxx.py` 可直接运行。
- 控制台输出禁止 ✅/❌ 等非 GBK 字符（此前踩过 `UnicodeEncodeError`），统一用 `[OK]` / `[FAIL]`；写入 `.md/.csv/.py` 一律 UTF-8。
- 策略不可回退：绿电优先（光伏足够大时 TOU 放电指令转充电吸收；逆变超限先保光伏/弃光）。
- 每工况图必须包含 负载/光伏/电池/电网 四端口数据（沿用 `plot_ports` 约定）。
- "多工况叠加 + 异常输入边界" 不再单独成章或运行（已从 `main.py` 移除，不得加回）。
- 模型为导入即用（import 时不打印、不画图、不改全局字体）；`matplotlib.use("Agg")` 仅在需要绘图的入口（main/reporting）设置。
- 默认额定：逆变/电池 125kW、电池 200kWh、PV 2 倍超配 250kW、SOC [10,90] 初始 60；TOU 指令：谷 -50 / 峰 +50 / 平 0（负充正放）。

---

### Task 1: 新增配置模块 `config.py`

集中系统额定、电池、SOC、TOU 时段与指令，消除 `main.py`/`mode_hybrid.py` 中的散落魔法数字；不改变任何默认值。

**Files:**
- Create: `config.py`
- Modify: 无（本任务不改其他文件，仅新增并被后续任务引用）

**Interfaces:**
- Consumes: 无
- Produces:
  - `SYSTEM: SystemConfig` dataclass：`inverter_rated_kw: float = 125.0`, `pv_overratio: float = 2.0`, `battery_rated_kw: float = 125.0`, `battery_capacity_kwh: float = 200.0`, `soc_init: float = 60.0`, `soc_min: float = 10.0`, `soc_max: float = 90.0`, `grid_charge_allowed: bool = True`, `export_limit_kw: float | None = None`
  - `TOU_CHG/TOU_DIS/TOU_IDLE = -50.0/50.0/0.0`
  - `class TouSlot`（`start_h`, `end_h`, `cmd_kw`）与 `TOU_TABLE: list[TouSlot]`：谷 23-7 → -50，峰 8-11、18-22 → +50，其余 0
  - `SYSTEM_DATE = "2026-02-26"`（测试默认日期）

- [ ] **Step 1: 创建文件并录入配置**

```python
# config.py  (UTF-8)
from __future__ import annotations
from dataclasses import dataclass, field

SYSTEM_DATE = "2026-02-26"
TOU_CHG, TOU_DIS, TOU_IDLE = -50.0, 50.0, 0.0

@dataclass
class SystemConfig:
    inverter_rated_kw: float = 125.0
    pv_overratio: float = 2.0
    battery_rated_kw: float = 125.0
    battery_capacity_kwh: float = 200.0
    soc_init: float = 60.0
    soc_min: float = 10.0
    soc_max: float = 90.0
    grid_charge_allowed: bool = True
    export_limit_kw: float | None = None

SYSTEM = SystemConfig()

@dataclass
class TouSlot:
    start_h: float
    end_h: float      # [start_h, end_h)，跨零点用 start>end 表示（如 23-7）
    cmd_kw: float

TOU_TABLE: list[TouSlot] = [
    TouSlot(23.0, 7.0, TOU_CHG),
    TouSlot(8.0, 11.0, TOU_DIS),
    TouSlot(18.0, 22.0, TOU_DIS),
]
```

- [ ] **Step 2: 验证取值与旧常量一致**

Run:
```powershell
python -c "import config as c; print(c.SYSTEM, c.TOU_CHG, len(c.TOU_TABLE))"
```
Expected: `SystemConfig(inverter_rated_kw=125.0, ..., export_limit_kw=None) -50.0 3`，无报错。

---

### Task 2: 新增场景/曲线构建模块 `scenarios.py`

把 `main.py` 中的 `_hrs / pv_sine / gauss / gen_irradiance / tou_cmd / cmd_of / make_model` 平移集中，并增加"场景构造"辅助：给定时段返回该时刻 TOU 指令，给定日期返回 1min 时间轴。行为与现有一致。

**Files:**
- Create: `scenarios.py`
- Modify: 无

**Interfaces:**
- Consumes: `config.SYSTEM`, `config.TOU_TABLE`, `config.TOU_CHG/TOU_DIS`, `config.SYSTEM_DATE`
- Produces:
  - `day_times(date: str = SYSTEM_DATE) -> pd.DatetimeIndex`（00:00–23:59, freq=1min）
  - `hours_of(times) -> np.ndarray`
  - `pv_sine(hrs, peak, t0=6.0, t1=18.0) -> np.ndarray`
  - `gauss(hrs, center, width, amp) -> np.ndarray`
  - `gen_irradiance(hrs, peak=1000.0, t0=6.0, t1=18.0, dip=0.0, dip_c=12.5) -> np.ndarray`
  - `tou_cmd(dt: pd.Timestamp) -> float`
  - `cmd_of(times) -> list[float]`
  - `make_model(**kw) -> HybridInverter`（以 `config.SYSTEM` 为默认，`kw` 覆盖）

- [ ] **Step 1: 平移实现**

```python
# scenarios.py  (UTF-8，节选关键函数签名与实现)
import numpy as np
import pandas as pd
import config as C
from mode_hybrid import HybridInverter

def day_times(date=C.SYSTEM_DATE):
    return pd.date_range(f"{date} 00:00", f"{date} 23:59", freq="1min")

def hours_of(times):
    return times.hour + times.minute / 60.0

def pv_sine(hrs, peak, t0=6.0, t1=18.0):
    irr = np.clip(np.sin(np.pi * (hrs - t0) / (t1 - t0)), 0.0, 1.0)
    return irr * peak

def gauss(hrs, center, width, amp):
    return amp * np.exp(-((hrs - center) ** 2) / (2 * width ** 2))

def gen_irradiance(hrs, peak=1000.0, t0=6.0, t1=18.0, dip=0.0, dip_c=12.5):
    g = peak * np.sin(np.pi * np.clip((hrs - t0) / (t1 - t0), 0.0, 1.0))
    g = np.clip(g, 0.0, None)
    if dip > 0:
        g = g * (1.0 - dip * np.exp(-((hrs - dip_c) ** 2) / (2 * 0.6 ** 2)))
    return g

def tou_cmd(dt):
    h = dt.hour + dt.minute / 60.0
    for slot in C.TOU_TABLE:
        s, e = slot.start_h, slot.end_h
        in_slot = (h >= s and h < e) if s < e else (h >= s or h < e)
        if in_slot:
            return slot.cmd_kw
    return 0.0

def cmd_of(times):
    return [tou_cmd(t) for t in times]

def make_model(**kw):
    base = dict(rated_power_kw=C.SYSTEM.inverter_rated_kw,
                pv_overratio=C.SYSTEM.pv_overratio,
                battery_rated_kw=C.SYSTEM.battery_rated_kw,
                battery_capacity_kwh=C.SYSTEM.battery_capacity_kwh,
                soc_init=C.SYSTEM.soc_init, soc_min=C.SYSTEM.soc_min,
                soc_max=C.SYSTEM.soc_max,
                grid_charge_allowed=C.SYSTEM.grid_charge_allowed)
    base.update(kw)
    return HybridInverter(**base)
```

- [ ] **Step 2: 验证与旧行为一致**

Run:
```powershell
python -c "import scenarios as s, pandas as pd, numpy as np; t=s.day_times(); h=s.hours_of(t); print(len(t), s.tou_cmd(t[6]), s.tou_cmd(t[8]), s.tou_cmd(t[12]), round(float(s.pv_sine(h,220.0).max()),1))"
```
Expected: `1440 -50.0 50.0 0.0 220.0`（谷/峰/平指令与峰值正确）。

---

### Task 3: 拆出报告/绘图模块 `reporting.py`

把 `main.py` 中 `plot_ports`、`render_report` 平移至 `reporting.py`，并在模块内设置中文字体与 `Agg` 后端。`reporting.render_report(cases)` 输出与现有一致的 Markdown。

**Files:**
- Create: `reporting.py`
- Modify: `main.py`（删除本地的 `plot_ports` 与 `render_report`，改为 `from reporting import plot_ports, render_report`，同时删去 `matplotlib.use`/字体设置重复代码）

**Interfaces:**
- Consumes: cases 结构（`title/png/rows/checks/notes/extra?`，来自各 `case_*()`）
- Produces:
  - `plot_ports(ax, times, load, s, title, load_label="负载") -> None`
  - `render_report(cases) -> str`
  - `OUT` 输出目录常量改为参数：`render_report(cases, out_dir: str) -> str`（保持向后兼容可不传）

- [ ] **Step 1: reporting.py 平移并加 out 参数**

```python
# reporting.py (UTF-8)
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def plot_ports(ax, times, load, s, title, load_label="负载"):
    ax.plot(times, load, color="0.3", lw=1.1, label=f"{load_label}(kW)")
    ax.plot(times, s["pv"]["光伏功率(kW)"], color="orange", lw=1.0, label="光伏(kW)")
    ax.plot(times, s["battery"]["电池功率(kW)"], color="b", lw=1.0, label="电池(kW,正放负充)")
    ax.plot(times, s["grid"]["电网功率(kW)"], color="g", lw=1.0, label="电网(kW,正购负馈)")
    ax.axhline(0, color="k", lw=0.5); ax.set_ylabel("功率(kW)")
    ax.set_title(title); ax.legend(fontsize=8, ncol=4); ax.grid(alpha=0.3)
```
（`render_report` 由 main.py 原文整体平移；章节/表格/图引用逻辑不变。）

- [ ] **Step 2: main.py 引用替换并删除重复定义**

```python
# main.py 顶部
from reporting import plot_ports, render_report   # 删除本文件内同名函数与字体设置
```

- [ ] **Step 3: 验证**

Run: `python main.py` → 预期仍在控制台打印 4 个 `[工况i]...校验:通过`，`tou_test/TOU测试报告.md` 与 4 张 png 重新生成；`main.py` 顶部不再有 `matplotlib.use("Agg")` 重复设置（仅 reporting.py 有）。

---

### Task 4: `main.py` 编排瘦身：参数化、退出码、CSV 导出

**Files:**
- Modify: `main.py`
- Modify: `mode_hybrid.py`（可选：`samples()` 已含四端口；本任务不改模型逻辑）

**Interfaces:**
- Consumes: `scenarios.case 常量`、`reporting.plot_ports/render_report`、现有 `case_*()` 返回值
- Produces:
  - `main(argv=None) -> int`：返回非 0 表示有校验失败（供脚本/CI 判定）
  - CLI：`python main.py [--out tou_test] [--cases 1,2,3,4] [--date 2026-02-26]`
  - `save_port_csv(samples, load, out_dir) -> dict[str,str]`：把 `inverter/battery/grid/pv` 四端口 sample 各写 `*_PORT.csv`（UTF-8-SIG）

- [ ] **Step 1: 增加命令行解析与退出码**

```python
# main.py
def _cases_all():  # 与现有顺序一致，去掉了工况5
    return [case_export_limit(), case_overload(), case_pv_load_ratio(), case_end_protection()]

def main(argv=None) -> int:
    import argparse, sys
    p = argparse.ArgumentParser(description="光储一体机 TOU 仿真测试")
    p.add_argument("--out", default="tou_test")
    p.add_argument("--cases", default="1,2,3,4")
    p.add_argument("--date", default=config.SYSTEM_DATE)
    args, _ = p.parse_known_args(argv)
    # ... 现有 running/report 逻辑；date 传给 day_times（Case 函数默认沿用）
    failed = any(not all(x[1] for x in c["checks"]) for c in cases)
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: 四端口 CSV 导出（每个 case 运行后调用）**

```python
def save_port_csv(samples, load, out_dir, prefix="case"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name in ("inverter", "battery", "grid", "pv"):
        df = samples[name].copy()
        df.insert(1, "负载功率(kW)", load) if name == "pv" else None
        fp = os.path.join(out_dir, f"{prefix}_{name}_port.csv")
        df.to_csv(fp, index=False, encoding="utf-8-sig")
        paths[name] = fp
    return paths
```

- [ ] **Step 3: 验证**

Run:
```powershell
python main.py --cases 1,2 --out tou_test; echo "exit=$LASTEXITCODE"
```
Expected：只跑工况1、2，输出到 `tou_test/`，退出码 0；目录出现 `case*_*_port.csv`。

---

### Task 5: 回归验证与收尾

**Files:**
- Modify: `main.py`、`mode_hybrid.py`（如导入时机修正）
- Create: `run_tou_tests.ps1`（可选便捷脚本）

**Interfaces:**
- Consumes: Task1-4 全部产出
- Produces: 无新接口；输出确定性校验

- [ ] **Step 1: 全量回归**

Run: `python main.py`  → 预期：
- 输出 4 组 `[OK]` 校验（含 绿电优先、谷段充电、峰段不足放电、SOC/额定不越限、馈网限制生效）
- `tou_test/TOU测试报告.md` 含：汇总表→工况1..4 章节（无工况5）→结论；每章引用一张含四端口的 png

- [ ] **Step 2: 校验关键基线（不应回退）**

Run:
```powershell
python main.py --cases 1 2>&1 | Select-String "馈网瞬时|峰段·光伏富余|谷段平均"
```
Expected：工况1 表格行 `馈网瞬时最大功率 … 50.0`、`峰段·光伏富余时平均执行 … 0.0 kW`、`谷段平均执行 … -13.8`（允许因配置变化而小幅波动，但不允许出现"富余段仍放电 >0"）。

- [ ] **Step 3: 收尾检查（死代码/重复）**

- `main.py` 不再定义 `_hrs/pv_sine/gauss/gen_irradiance/tou_cmd/cmd_of/make_model`（已入 scenarios）。
- `main.py` 不再定义 `plot_ports/render_report`/字体设置。
- 全仓检索无 `case_combined_and_boundary`、`case5` 引用。
- （可选）若在 git 仓库内，逐任务提交；否则跳过提交。

---

## Self-Review（计划自检）

**1. Spec 覆盖：** 目标=工程化：配置集中（T1）、场景集中（T2）、报告集中（T3）、入口可参数化/退出码/CSV（T4）、回归与基线（T5）。Global Constraints 全部作为显式约束存在，绿电优先/四端口/无工况5 由 T5 回归锁定。无遗漏需求。

**2. 占位符扫描：** 无 "TBD/TODO/视情况/加错误处理" 等字样；每个任务含真实代码或明确命令与期望输出。

**3. 类型一致：** `SystemConfig/TOU_TABLE`（T1）→ `scenarios` 消费（T2）→ `reporting.plot_ports/render_report`（T3）→ `main(argv)`/`save_port_csv`（T4）→ T5 回归。前后签名一致；`samples` 键名沿用模型既有 `inverter/battery/grid/pv`，未发明新键。
