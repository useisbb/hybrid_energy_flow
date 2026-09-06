# -*- coding: utf-8 -*-
"""
main.py — 光储一体机(混你) TOU 控制仿真：按「提示词.md」重做报告与数据

调用 mode_hybrid.HybridInverter 仿真：
  - TOU 控制（谷充10 / 峰放50 / 平待机；负充正放）
  - EMS 混逆执行策略：馈网(卖电)上限 F、取电上限 I、光伏供给优先级 负载>馈网>电池、
    电池充电优先级 光伏>电网、SOC 到限功率清零、TOU 放电不突破计划
  - 输出（全部输出到 ems_result/）：
    1) 图一：全部工况 —— 负载/电网/电池/光伏辐照/TOU计划/逆变 功率
    2) 图二：全部工况 —— TOU计划/逆变设置/逆变实际/电池设置/电池实际 功率
    3) 每工况四端口数据 CSV
    4) EMS测试报告.md
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
BAT_KWH = 200.0
SOC_MIN, SOC_MAX = 10.0, 90.0
DATE = "2026-02-26"

TOU_CHG, TOU_DIS, TOU_IDLE = -10.0, 50.0, 0.0     # 谷充10/峰放50/平待机；负充正放
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


def tou_cmd(dt):
    """谷充(02:00-07:00)-10 / 峰放(8-11、18-22)+50 / 平待机0。"""
    h = dt.hour + dt.minute / 60.0
    if 2.0 <= h < 7.0:
        return TOU_CHG
    if (8.0 <= h < 11.0) or (18.0 <= h < 22.0):
        return TOU_DIS
    return TOU_IDLE


def cmd_of(times):
    return np.array([tou_cmd(t) for t in times], dtype=float)


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

    # ---- 工况1：卖电上限 70 kW（光伏充裕）。输入 PV 峰值300，但 >250 按 250 限幅 ----
    pv1 = pv_sine(h1, 300.0)                        # 模拟输入最高点 300 kW
    pv1_disp = np.minimum(pv1, PV_RATED_KW)         # 图/仿真限幅到 250 kW
    ld1 = np.maximum(ld_base - 20.0, 0.0)           # 负载整体下调 20 kW
    cases.append(dict(title="工况1 卖电上限70kW·光伏充裕", times=t1, pv=pv1,
                      pv_disp=pv1_disp, load=ld1,
                      cmd=cmd_of(t1),
                      kw=dict(export_limit_kw=70.0, soc_init=30.0),   # PV额定按默认2x=250限幅
                      check=("馈网(卖电)不超过70kW", lambda s: float(s["grid"]["电网功率(kW)"].min()) >= -70.0 - EPS)))

    # ---- 工况2：防逆流 0 kW（光伏发电量改为原方案的 25%，峰值 220→55 kW） ----
    pv2 = pv_sine(h1, 220.0 * 0.25)
    cases.append(dict(title="工况2 防逆流(卖电0)", times=t1, pv=pv2, load=ld_base,
                      cmd=cmd_of(t1), kw=dict(export_limit_kw=0.0, soc_init=30.0),
                      check=("防逆流：无馈网(电网功率≥0)", lambda s: float(s["grid"]["电网功率(kW)"].min()) >= -EPS)))

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


# ================================================================ 图一 / 图二
def plot_all_cases(results, kind):
    """图一(kind=1)：负载/电网/电池(×0.95防重叠)/光伏辐照/TOU计划（不含逆变功率）；
       图二(kind=2)：TOU计划/逆变设置/逆变实际/电池设置/电池实际。"""
    nrow = len(results)
    fig, ax = plt.subplots(nrow, 1, figsize=(14, 3.1 * nrow), sharex=False)
    if nrow == 1:
        ax = [ax]
    for i, r in enumerate(results):
        t, s = r["times"], r["samples"]
        if kind == 1:
            pv_disp = r.get("pv_disp", r["pv"])   # 工况1 用限幅250后的光伏曲线
            series = [("负载", r["load"], None), ("电网", s["grid"]["电网功率(kW)"], None),
                      ("电池", np.asarray(s["battery"]["电池功率(kW)"], float) * 0.95,
                       "电池功率(×0.95)"),
                      ("光伏辐照", pv_disp, None), ("TOU计划", r["cmd"], None)]
            title = f"{r['title']}：负载/电网/电池/光伏辐照/TOU计划"
        else:
            # 图二：逆变实际功率 = 实际光伏功率 + 实际电池功率(负充正放)；按×0.95 错开显示避免重叠
            inv_set = np.asarray(s["inverter"]["逆变设置功率(kW)"], float)
            pv_act = np.asarray(s["pv"]["光伏功率(kW)"], float)
            bat_act = np.asarray(s["battery"]["电池功率(kW)"], float)
            inv_act = (pv_act + bat_act) * 0.95      # 逆变口功率 = 光伏+电池（电池充电为负）
            series = [("TOU计划", r["cmd"], None),
                      ("负载", r["load"], None),
                      ("光伏", pv_act, None),
                      ("逆变设置", inv_set, None),
                      ("逆变实际", inv_act, "逆变实际=光伏+电池(×0.95)"),
                      ("电池设置", s["battery"]["电池设置功率(kW)"], None),
                      ("电池实际", bat_act * 0.95, "电池实际(×0.95)")]
            title = f"{r['title']}：TOU计划/负载/光伏/逆变(设置,实际=光伏+电池×0.95)/电池(设置,实际×0.95)"
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
    name = "EMS_图一_全部工况_功率曲线.png" if kind == 1 else "EMS_图二_设置与执行_曲线.png"
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
def render_report(results, fp_fig1, fp_fig2):
    L = []
    A = L.append
    A("# 光储一体机(混你) EMS/TOU 控制仿真报告")
    A("")
    A(f"- 生成时间：{NOW}")
    A(f"- 模型：`mode_hybrid.HybridInverter`；额定 125kW；PV 2 倍超配 250kW；电池 125kW/200kWh；SOC [{SOC_MIN:g}%,{SOC_MAX:g}%]")
    A("- TOU 控制：谷充(02:00-07:00) 10kW / 峰放(08-11、18-22) 50kW / 平待机；**负充正放**")
    A("- EMS 策略（提示词.md）：馈网上限 F（正=卖电上限，0=防逆流），约束内尽量多卖电且**不突破计划放电**；"
      "取电上限 I（0=不取电），**负载用电+充电总取电不超 I**，约束内尽量少取电；"
      "光伏消纳优先级高于 TOU 计划（充/放/待机均适用），富余先削减 TOU 放电、再反转为充电（≤1 倍额定）；"
      "TOU 充电在取电上限允许下可由电网补足按计划执行，超限则削减充电功率；SOC 到限功率清零；TOU 放电不突破计划")
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
    A("## 图二：执行策略——工况1~3（TOU计划/负载/光伏/逆变设置·逆变实际(=光伏+电池,×0.95)/电池设置·电池实际(×0.95)）")
    A("")
    A(f"![图二]({os.path.basename(fp_fig2)})")
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
        A(f"- 数据：`data/case{i}_inverter.csv`、`case{i}_battery.csv`、`case{i}_grid.csv`、`case{i}_pv.csv`、`case{i}_merged.csv`")
        A("")
    A("## 说明与假设")
    A("")
    A("- “逆变设置功率”=EMS 目标 AC 出力（下发值，未计限幅修正）；“逆变实际功率”=执行后实际值（限幅/弃光后）。")
    A("- “电池设置功率”=TOU 计划值（光伏富余时可被削减甚至反转为充电，体现为与实际功率的偏差）。")
    A("- 工况1 场景参数：模拟输入光伏峰值 300kW、>250kW 按 250kW 限幅；负载较基准下调 20kW（仅作场景配置，不写入图/表标题）。")
    A("- 工况2 场景参数：光伏发电量调整为原方案的 25%（峰值 220→55kW）。")
    A("- SOC 到上/下限触发保护时电池功率清零；充电期间优先光伏，光伏不足且取电上限允许时由电网补足。")
    A("- 工况3 场景参数：防逆流(卖电上限0kW)；取电上限 150kW；负载较原曲线下调 50%、最低功率钳位 50kW；"
      "19:00 起 5 小时内负载逐渐下降 30 kW。")
    A("- 工况3 说明：负载峰值约 96kW ≤ 取电上限 150kW，全天无缺供且无馈网（防逆流）；"
      "光伏富余只能充电池，充满(SOC90%)后多余弃光；谷充(02-07)与峰放(08-11、18-22)按 TOU 执行。")
    A("- 图中事件标识：▲=电池充满(SOC 到达上限)、▼=电池放空(SOC 到达下限)，标于 y=0。")
    A("")
    fp = os.path.join(OUT, "EMS测试报告.md")
    with open(fp, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return fp


def main():
    print(LINE)
    print("光储一体机(混你) TOU 仿真 —— 按 提示词.md 重新生成报告与数据")
    print(LINE)
    cases = build_cases()
    results = run_cases(cases)
    for i, r in enumerate(results, 1):
        print(f"[工况{i}] {r['title']}  校验: {'通过' if r['ok'] else '未通过'}")
    fp1 = plot_all_cases(results, kind=1)
    fp2 = plot_all_cases(results, kind=2)
    export_data(results)
    report = render_report(results, fp1, fp2)
    print(LINE)
    print(f"图片: {fp1}\n      {fp2}")
    print(f"数据: {DATA}\\*.csv")
    print(f"报告: {report}")


if __name__ == "__main__":
    main()
