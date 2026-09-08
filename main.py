# -*- coding: utf-8 -*-
"""
main.py — 光储一体机(混逆) TOU 控制仿真：按「提示词.md」重做报告与数据

调用 mode_hybrid.HybridInverter 仿真：
  - TOU 控制（充电10 / 放电50 / 待机；负充正放）；时段：充电08-11、17-19；放电12-14、19-22；其余待机
  - EMS 混逆执行策略：馈网(卖电)上限 F、取电上限 I、光伏供给优先级 负载>电池充电>电网馈网、
    电池充电电源优先级 光伏>电网、光伏消纳优先于 TOU 计划（富余削减放电、可反转为充电，最大 1 倍额定）、
    SOC 到限功率清零、TOU 放电不突破计划；
    放电按提示词第二张图“直接设置电池放电=计划、计划内可馈网(≤F)”执行
  - 输出（全部输出到 ems_result/）：
    1) 图一：全部工况 —— 负载/电网/电池/光伏辐照/TOU计划 功率
    2) 图二：工况1~2 —— TOU计划/逆变设置/逆变实际/电池设置/电池实际 功率（提示词赋值公式）
    3) 图三：默认单曲线 TOU(5-8/12-14充·9-11/18-20放) 恒总功率放电
    4) 每工况四端口数据 CSV
    5) EMS测试报告.md
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# 控制台 UTF-8（IDE 调试控制台按 UTF-8 解码）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mode_hybrid import HybridInverter

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "ems_result")
DATA = os.path.join(OUT, "data")
os.makedirs(DATA, exist_ok=True)
NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------- 基准参数（提示词.md） ----------------
RATED_KW = 125.0            # 1 倍额定
PV_RATED_KW = 250.0         # 2 倍光伏超配
BAT_KW = 125.0
BAT_KWH = 260.0
SOC_MIN, SOC_MAX = 10.0, 90.0
DATE = "2026-02-26"

TOU_CHG, TOU_DIS, TOU_IDLE = -10.0, 50.0, 0.0     # 充电10/放电50/待机0；负充正放
EPS = 1e-6
LINE = "-" * 76

STYLE1 = {"负载": ("0.3", "-"), "电网": ("g", "-"), "电池": ("b", "-"),
          "光伏辐照": ("orange", "-"), "TOU计划": ("k", "--"), "逆变": ("r", "-")}
STYLE2 = {"TOU计划": ("k", "--"), "负载": ("0.45", "-"), "光伏": ("orange", "-"),
          "逆变设置": ("m", ":"), "逆变实际": ("r", "-"),
          "电池设置": ("c", "--"), "电池实际": ("b", "-")}


def hours_of(times):
    return times.hour + times.minute / 60.0


def pv_sine(hrs, peak, t0=6.0, t1=18.0):
    irr = np.clip(np.sin(np.pi * (hrs - t0) / (t1 - t0)), 0.0, 1.0)
    return irr * peak


def gauss(hrs, center, width, amp):
    return amp * np.exp(-((hrs - center) ** 2) / (2 * width ** 2))


def tou_cmd(dt, charge_kw=TOU_CHG, discharge_kw=TOU_DIS, discharge_mid_kw=None):
    """TOU 计划时段（图2要求）：
    待机 00-08 / 11-12 / 14-17 / 22-24；充电 08-11、17-19；放电 12-14、19-22。
    charge_kw<0：充电计划功率；放电：19-22 用 discharge_kw，12-14 可用 discharge_mid_kw 单独覆盖。"""
    h = dt.hour + dt.minute / 60.0
    if (8.0 <= h < 11.0) or (17.0 <= h < 19.0):
        return charge_kw
    if 12.0 <= h < 14.0:
        return discharge_kw if discharge_mid_kw is None else discharge_mid_kw
    if 19.0 <= h < 22.0:
        return discharge_kw
    return TOU_IDLE


def cmd_of(times, charge_kw=TOU_CHG, discharge_kw=TOU_DIS, discharge_mid_kw=None):
    return np.array([tou_cmd(t, charge_kw, discharge_kw, discharge_mid_kw) for t in times], dtype=float)


def make_model(**kw):
    kw.setdefault("rated_power_kw", RATED_KW)
    kw.setdefault("pv_overratio", PV_RATED_KW / RATED_KW)
    kw.setdefault("battery_rated_kw", BAT_KW)
    kw.setdefault("battery_capacity_kwh", BAT_KWH)
    kw.setdefault("soc_init", 60.0)
    kw.setdefault("soc_min", SOC_MIN)
    kw.setdefault("soc_max", SOC_MAX)
    return HybridInverter(**kw)


# ================================================================ 工况定义
def build_cases():
    """返回工况列表：每项 dict(标题, times, pv, load, cmd, 模型参数, 校验函数)。"""
    cases = []

    # ---- 基础日曲线（工况2 用） ----
    t1 = pd.date_range(f"{DATE} 00:00", f"{DATE} 23:59", freq="1min")
    h1 = hours_of(t1)
    ld_base = 45.0 + gauss(h1, 9.5, 1.2, 55.0) + gauss(h1, 19.5, 1.6, 60.0)

    # ---- 工况1：卖电上限 20 kW（光伏 80kW；负载=原曲线70%） ----
    pv1 = pv_sine(h1, 80.0)                         # 光伏最大功率 80 kW
    ld1 = np.maximum(ld_base - 20.0, 0.0) * 0.7     # 负载较基准下调20kW后再降到其70%
    cases.append(dict(title="工况1 卖电上限20kW·光伏80kW", times=t1, pv=pv1,
                      load=ld1,
                      cmd=cmd_of(t1),
                      kw=dict(export_limit_kw=20.0, soc_init=30.0),
                      check=("馈网(卖电)不超过20kW", lambda s: float(s["grid"]["电网功率(kW)"].min()) >= -20.0 - EPS)))

    # ---- 工况2：防逆流(卖电0)·取电上限 120 kW
    #      光伏峰值 40kW；13:00 负载峰值约 180kW；TOU 充电 100kW、12-14 放电 100kW（19-22 仍 50kW） ----
    pv2 = pv_sine(h1, 40.0)
    ld2 = ld_base + gauss(h1, 13.0, 1.0, 135.0)         # 基准~45 + 峰宽1.0/幅值135 ≈ 峰值180
    cases.append(dict(title="工况2 防逆流(卖电0)·取电上限120kW", times=t1, pv=pv2, load=ld2,
                      cmd=cmd_of(t1, charge_kw=-100.0, discharge_mid_kw=100.0),
                      kw=dict(export_limit_kw=0.0, grid_import_limit_kw=120.0,
                              soc_init=30.0),
                      check=("防逆流(无馈网)且取电≤120kW", lambda s: float(s["grid"]["电网功率(kW)"].min()) >= -EPS
                             and float(s["grid"]["电网功率(kW)"].max()) <= 120.0 + EPS)))

    # ---- 工况3：防逆流(卖电0)·取电上限 150 kW（负载整体下调 50%，最低功率限 50kW） ----
    # 场景：负载较原曲线下调 50%；19:00 起 5 小时内逐渐下降 30 kW，且最低功率钳位到 50kW
    ld3 = 0.5 * (120.0 + gauss(h1, 10.0, 1.3, 60.0) + gauss(h1, 19.5, 1.6, 75.0))
    ld3 = ld3 - 30.0 * np.clip((h1 - 19.0) / 5.0, 0.0, 1.0)   # 19:00-24:00 渐降 30 kW
    ld3 = np.maximum(ld3, 50.0)                               # 负载最低功率不低 50 kW
    pv3 = pv_sine(h1, 90.0)
    cases.append(dict(title="工况3 防逆流(卖电0)·取电上限150kW·负载主导", times=t1, pv=pv3, load=ld3,
                      cmd=cmd_of(t1), kw=dict(export_limit_kw=0.0, grid_import_limit_kw=150.0,
                                              soc_init=60.0),
                      check=("取电不超过150kW且无馈网", lambda s: float(s["grid"]["电网功率(kW)"].max()) <= 150.0 + EPS
                             and float(s["grid"]["电网功率(kW)"].min()) >= -EPS)))

    return cases


def run_cases(cases):
    """执行所有工况仿真，返回 (案例列表含样本/模型/事件, ...)。"""
    results = []
    for c in cases:
        m = make_model(**c["kw"])
        samples = m.run(c["times"], c["pv"], c["load"], c["cmd"])
        label, fn = c["check"]
        ok = bool(fn(samples))
        # 事件：SOC 穿越上限=充满、穿越下限=放空（用模型实际 soc_min/soc_max）
        soc = np.asarray(samples["battery"]["储能SOC(%)"], float)
        lo, hi = m.soc_min, m.soc_max
        prev = soc[:-1]
        cur = soc[1:]
        full_t = list(c["times"][1:][(prev < hi) & (cur >= hi)])
        empty_t = list(c["times"][1:][(prev > lo) & (cur <= lo)])
        results.append(dict(**c, model=m, samples=samples, ok=ok,
                            sum_=m.summary(),
                            events=dict(full=full_t, empty=empty_t)))
    return results


# ================================================================ 图3 专用（TOU放电恒总功率 · 默认单曲线 · 新TOU时段）
def _tou3(dt, chg_kw=100.0, dis_kw=100.0):
    """图3 默认 TOU：充电 05-08、12-14（-100）；放电 09-11、18-20（+100）；其余待机。"""
    h = dt.hour + dt.minute / 60.0
    if (5.0 <= h < 8.0) or (12.0 <= h < 14.0):
        return -chg_kw
    if (9.0 <= h < 11.0) or (18.0 <= h < 20.0):
        return dis_kw
    return 0.0


def fig3_default_case():
    """图3 默认单曲线（不再复用工况1~3）：负载=基准日曲线；光伏峰值 70kW；
    TOU：充电 05-08、12-14 / 放电 09-11、18-20（±100kW）；
    模式：keep_total_discharge（TOU放电恒总功率），馈网上限 20kW，SOC 初值 0%。"""
    t = pd.date_range(f"{DATE} 00:00", f"{DATE} 23:59", freq="1min")
    h = hours_of(t)
    ld = 45.0 + gauss(h, 9.5, 1.2, 55.0) + gauss(h, 19.5, 1.6, 60.0)   # 基准日负载曲线
    cmd = np.array([_tou3(x) for x in t], dtype=float)
    return dict(title="图3 默认·TOU(5-8/12-14充·9-11/18-20放)",
                times=t, pv=pv_sine(h, 70.0), load=ld, cmd=cmd,
                kw=dict(export_limit_kw=20.0, keep_total_discharge=True, soc_init=0.0),
                check=("馈网(卖电)不超过20kW", lambda s: float(s["grid"]["电网功率(kW)"].min()) >= -20.0 - EPS))


def run_keep_total_cases(cases=None):
    """图3：默认单曲线 + keep_total_discharge=True（忽略传入的工况列表）。"""
    return run_cases([fig3_default_case()])


# ================================================================ 图一 / 图二
def plot_all_cases(results, kind, fname=None):
    """图一(kind=1)：全部工况——负载/电网/电池(×0.95防重叠)/光伏辐照/TOU计划（不含逆变功率）；
       图二(kind=2)：工况1~2——TOU计划/负载/光伏/逆变设置/逆变实际/电池设置/电池实际（提示词 L69-81 赋值公式）。
       fname：可覆盖输出文件名（图3 用 keep_total_discharge 模式同一曲线模板）。"""
    if kind == 2:
        results = [r for r in results if r["title"].startswith(("工况1", "工况2"))]
    nrow = len(results)
    fig, ax = plt.subplots(nrow, 1, figsize=(14, 3.1 * nrow), sharex=False)
    if nrow == 1:
        ax = [ax]
    for i, r in enumerate(results):
        t, s = r["times"], r["samples"]
        if kind == 1:
            pv_disp = r.get("pv_disp", r["pv"])   # 图1显示用光伏曲线（默认=工况可用光伏）
            series = [("负载", r["load"], None), ("电网", s["grid"]["电网功率(kW)"], None),
                      ("电池", np.asarray(s["battery"]["电池功率(kW)"], float) * 0.95,
                       "电池功率(×0.95)"),
                      ("光伏辐照", pv_disp, None), ("TOU计划", r["cmd"], None)]
            title = f"{r['title']}：负载/电网/电池/光伏辐照/TOU计划"
        else:
            # 图二（提示词 L69-81 赋值公式）：TOU计划/负载/光伏/逆变设置/逆变实际/电池设置/电池实际 + 充满放空标识
            #  充电：电池设置=∞(允许光伏突破计划充电)；待机：电池设置=∞(仅光伏充电)；
            #  放电：电池设置=TOU计划功率(直接放电)。充电/待机无直接电池命令 → 电池设置曲线为 NaN 断线。
            #  物理口径恒等：逆变实际 = 光伏实际 + 电池实际(放电为正)，即 电池 = 逆变 − 光伏；
            #  逆变设置 = EMS 目标(逆变设置功率列)；实际曲线按×0.95 错开避免与设置重叠。
            cmd = np.asarray(r["cmd"], float)
            pv_act = np.asarray(s["pv"]["光伏功率(kW)"], float)
            bat_act = np.asarray(s["battery"]["电池功率(kW)"], float)
            inv_set_disp = np.asarray(s["inverter"]["逆变设置功率(kW)"], float)   # 物理EMS设置
            inv_act_disp = (pv_act + bat_act) * 0.95                              # 逆变实际=光伏+电池(×0.95)
            series = [("TOU计划", r["cmd"], None),
                      ("负载", r["load"], None),
                      ("光伏", pv_act, None),
                      ("逆变设置", inv_set_disp, None),
                      ("逆变实际", inv_act_disp, "逆变实际=光伏+电池(×0.95)"),
                      ("电池设置", s["battery"]["电池设置功率(kW)"], "电池设置(放电=TOU计划;充/待机=∞)"),
                      ("电池实际", bat_act * 0.95, "电池实际(×0.95)")]
            title = f"{r['title']}：TOU计划/负载/光伏/逆变设置/逆变实际/电池设置/电池实际"
        styles = STYLE1 if kind == 1 else STYLE2
        for name, y, lab in series:
            clr, ls = styles[name]
            ax[i].plot(t, y, color=clr, ls=ls, lw=1.1, label=lab or name)
        ax[i].axhline(0, color="k", lw=0.5)
        # 事件标识：电池充满(SOC→上限)▲ / 放空(SOC→下限)▼，标在 y=0
        ev = r.get("events", {})
        for tag, lab, col, mk in (("full", "充满(90%)", "#8e0c9e", "^"),
                                  ("empty", "放空(10%)", "#d62728", "v")):
            for j, tt in enumerate(ev.get(tag, [])):
                ax[i].plot([tt], [0.0], color=col, marker=mk, ms=8, ls="none",
                           label=lab if j == 0 else None)
        ax[i].set_ylabel("功率(kW)")
        ax[i].set_title(title)
        ax[i].legend(fontsize=8, ncol=min(len(series) + 2, 6), loc="upper left")
        ax[i].grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    name = fname if fname else ("EMS_图一_全部工况_功率曲线.png" if kind == 1
                                else "EMS_图二_设置与执行_曲线.png")
    fp = os.path.join(OUT, name)
    fig.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fp


# ================================================================ 数据导出
def export_data(results):
    for i, r in enumerate(results, 1):
        s = r["samples"]
        for port in ("inverter", "battery", "grid", "pv"):
            df = s[port].copy()
            fp = os.path.join(DATA, f"case{i}_{port}.csv")
            df.to_csv(fp, index=False, encoding="utf-8-sig")
        # 汇总（四端口 + 负载/TOU计划）
        merged = pd.DataFrame({"time": r["times"],
                               "负载功率(kW)": np.round(r["load"], 3),
                               "TOU计划功率(kW)": np.round(r["cmd"], 2)})
        for port, cols in (("inverter", ("逆变功率(kW)", "逆变设置功率(kW)", "逆变限制功率(kW)")),
                           ("battery", ("电池功率(kW)", "电池设置功率(kW)", "储能SOC(%)")),
                           ("grid", ("电网功率(kW)",)),
                           ("pv", ("光伏功率(kW)",))):
            for col in cols:
                merged[col] = s[port][col].to_numpy()
        merged.to_csv(os.path.join(DATA, f"case{i}_merged.csv"),
                      index=False, encoding="utf-8-sig")


# ================================================================ 报告
def render_report(results, fp_fig1, fp_fig2, fp_fig3=None):
    L = []
    A = L.append
    A("# 光储一体机(混逆) EMS/TOU 控制仿真报告")
    A("")
    A(f"- 生成时间：{NOW}")
    A(f"- 模型：`mode_hybrid.HybridInverter`；额定 125kW；PV 2 倍超配 250kW；电池 125kW/220kWh；SOC [{SOC_MIN:g}%,{SOC_MAX:g}%]")
    A("- TOU 控制：充电(08-11、17-19) 10kW / 放电(12-14、19-22) 50kW / 其余待机；**负充正放**")
    A("- EMS 策略（提示词.md）：光伏供给优先级 **负载 > 电池充电 > 电网馈网**（储能充满后余电才馈网≤F），"
      "馈网电能优先级 光伏>电池、电池充电电源优先级 光伏>电网；馈网上限 F（正=卖电上限，0=防逆流），约束内尽量多卖电且**不突破计划放电**；"
      "取电上限 I（0=不取电），**负载用电+充电总取电不超 I**，约束内尽量少取电；"
      "光伏消纳优先级高于 TOU 计划（充/放/待机均适用），富余削减 TOU 放电、可反转为充电（≤1 倍额定）；"
      "TOU 充电在取电上限允许下由电网补足按计划执行，超限则削减充电功率；SOC 到限功率清零；TOU 放电不突破计划")
    A("")
    A("## 工况汇总")
    A("")
    A("| 工况 | 光伏可用(kWh) | 负载(kWh) | 电池充/放(kWh) | 购电(kWh) | 馈网(kWh) | 弃光(kWh) | SOC范围% | 校验 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(results, 1):
        x = r["sum_"]
        title = r["title"].replace("·", "·")
        A(f"| {title} | {x['光伏可用(kWh)']:.0f} | {x['负载用电(kWh)']:.0f} | "
          f"{x['电池充电(kWh)']:.1f}/{x['电池放电(kWh)']:.1f} | {x['购电(kWh)']:.1f} | "
          f"{x['馈网(kWh)']:.1f} | {x['弃光(kWh)']:.1f} | {x['SOC范围(%)']} | "
          f"{'[OK]' if r['ok'] else '[FAIL]'} |")
    A("")
    A("## 图一：执行策略——工况1~3（负载/电网/电池(×0.95)/光伏辐照/TOU计划；已去除逆变功率以避免曲线重叠）")
    A("")
    A(f"![图一]({os.path.basename(fp_fig1)})")
    A("")
    A("> 说明：图一电池功率曲线按计算值×0.95 绘制以分离重叠。")
    A("")
    A("## 图二：TOU 赋值计算公式示意——工况1~2（TOU计划/负载/光伏/逆变设置/逆变实际/电池设置/电池实际，含充满▲/放空▼标识）")
    A("")
    A(f"![图二]({os.path.basename(fp_fig2)})")
    A("")
    A("> 说明：按提示词.md 第二张图重写。充电/待机：电池设置=∞（曲线 NaN 断线），DSP 允许光伏突破计划充电，"
      "充电电网部分受取电上限约束 `逆变设置=MIN(TOU计划, 电池最大取电-负载)`、待机取电网 0；"
      "放电：电池设置=TOU计划功率（≤1 倍额定）、计划内可馈网(≤F)、不突破计划；"
      "光伏富余时按“光伏消纳>TOU计划”削减放电甚至反转为充电（图中放电窗内电池实际可为负）。"
      "物理口径恒等：逆变实际=光伏实际+电池实际(放电为正)、逆变设置=EMS 目标(逆变设置列)；"
      "实际曲线按×0.95 错开显示。")
    A("> 校验：工况1 光伏峰值 80kW、工况2 峰值 40kW；逐分钟满足 `光伏实际+电池实际=逆变实际`（max err≤0.001kW），"
      "即“放电为正”口径下 `电池功率 = 逆变口功率 − 光伏功率`。")
    A("")
    if fp_fig3:
        A("## 图三：TOU放电恒总功率（默认单曲线；负载/电网/电池(×0.95)/光伏辐照/TOU计划）")
        A("")
        A(f"![图三]({os.path.basename(fp_fig3)})")
        A("")
        A("> 说明：图3 使用独立默认曲线（负载=基准日曲线、光伏峰值 70kW），TOU 计划时段："
          "充电 05-08、12-14（-100kW），放电 09-11、18-20（+100kW），其余待机。"
          "TOU 放电行为=恒总功率——目标 AC 出力 = 负载 + 馈网目标(≤20kW、≤逆变额定) 维持不变；"
          "放电时段光伏有发电时优先出力（供负载→馈网），电池放电功率相应减小（≤TOU 计划、≤1 倍额定），"
          "总功率保持不变；电池不因光伏富余反转充电，超出目标的光伏弃光。充电/待机时段仍按光伏优先充储、充满后馈网执行。")
        A("")
    A("## 各工况说明")
    A("")
    for i, r in enumerate(results, 1):
        label = r["title"]
        if label.startswith(f"工况{i} "):          # 去掉重复的编号前缀
            label = label.split(" ", 1)[1]
        A(f"### 工况{i}：{label}")
        A("")
        A(f"- 校验项：{r['check'][0]} → {'[OK]' if r['ok'] else '[FAIL]'}")
        A(f"- 数据：`data/case{i}_inverter.csv`、`data/case{i}_battery.csv`、`data/case{i}_grid.csv`、`data/case{i}_pv.csv`、`data/case{i}_merged.csv`")
        A("")
    A("## 说明与假设")
    A("")
    A("- “逆变设置功率”=EMS 模式目标 AC 出力（充电=取电上限约束下的充电功率(负)；待机=TOU计划(0)；放电=负载+计划内馈网(≤F)的期望出力，均受逆变限幅）。")
    A("- “电池设置功率”=放电时段 TOU计划功率；充电/待机=∞（不直接限制电池，仅由取电上限/光伏富余间接控制，图中为断线）。")
    A("- “逆变实际功率”=光伏实际 + 电池实际（电池负充正放），受 SOC/功率限值修正。")
    A("- 光伏供给顺序 = 负载→电池充电→电网馈网：光伏富余先充电池（可突破 TOU 计划、≤1 倍额定功率），"
      "储能充满(无充电余量)后余电才馈网(≤F)，充满前超出充电能力的富余弃光；放电时段遇光伏富余时自动削减/反转（电池实际可为充电）。")
    A("- 工况1 描述（提示词）：馈网功率很大——卖电上限 20kW；光伏最大功率 80kW；负载=原曲线70%（基准下调20kW后再×0.7）。")
    A("- 工况2 描述（提示词）：馈网功率=0（防逆流保护），限制取电功率 120kW；光伏最大功率 40kW；TOU 充电计划 100kW、"
      "12-14 放电计划 100kW（19-22 仍 50kW）；负载 13:00 峰值约 180kW。")
    A("- 工况3 场景参数：防逆流(卖电上限0kW)；取电上限 150kW；负载较原曲线下调 50%、最低功率钳位 50kW；"
      "19:00 起 5 小时内负载逐渐下降 30 kW。")
    A("- 工况3 说明：负载峰值约 96kW ≤ 取电上限 150kW，全天无缺供且无馈网（防逆流）；"
      "光伏富余只能充电池，充满(SOC90%)后多余弃光；充电(08-11、17-19)与放电(12-14、19-22)按 TOU 执行。")
    A("- 图中事件标识：▲=电池充满(SOC 到达上限)、▼=电池放空(SOC 到达下限)，标于 y=0。")
    A("")
    fp = os.path.join(OUT, "EMS测试报告.md")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return fp


def main():
    print(LINE)
    print("光储一体机(混逆) TOU 仿真 —— 按 提示词.md 重新生成报告与数据")
    print(LINE)
    cases = build_cases()
    results = run_cases(cases)
    for i, r in enumerate(results, 1):
        print(f"[工况{i}] {r['title']}  校验: {'通过' if r['ok'] else '未通过'}")
    fp1 = plot_all_cases(results, kind=1)
    fp2 = plot_all_cases(results, kind=2)
    # 图3：TOU放电恒总功率（默认单曲线，独立TOU时段：5-8/12-14充·9-11/18-20放）
    results3 = run_keep_total_cases()
    for i, r in enumerate(results3, 1):
        print(f"[图3] {r['title']}  恒总功率放电  校验: {'通过' if r['ok'] else '未通过'}")
    fp3 = plot_all_cases(results3, kind=1,
                         fname="EMS_图三_TOU放电恒总功率_默认曲线.png")
    export_data(results)
    report = render_report(results, fp1, fp2, fp3)
    print(LINE)
    print(f"图片: {fp1}\n      {fp2}\n      {fp3}")
    print(f"数据: {DATA}\\*.csv")
    print(f"报告: {report}")


if __name__ == "__main__":
    main()
