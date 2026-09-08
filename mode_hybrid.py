# -*- coding: utf-8 -*-
"""
光储一体机（Hybrid PV+Storage All-in-One Inverter）仿真模型

拓扑（参考常见光储逆变器产品，如阳光电源 ST125CS、华为 LUNA 系列等）：
    光伏(PV) ──┐
               ├─(DC 直流母线并联)── 双向DC/AC逆变口 ──┬── 负载
    电池(BAT) ─┘                                       └── 电网(GRID)

  - PV 端口:   单向 DC/DC（MPPT），支持 2 倍超配，即阵列容量 = 2×额定
  - 电池端口:  双向 DC/DC，额定 1 倍超配，支持功率指令(充电/放电)与 SOC 限值
  - 逆变端口:  双向 AC/DC，额定 1 倍，交流侧与负载、电网并联，可限制输出功率
  - 电网端口:  交流节点，计量购电/馈电，功率自动平衡（削峰填谷/自发自用）

单位约定：
  功率 kW，能量 kWh，时间以秒为单位传入、内部换算为小时。
  电池口功率指令：正 = 放电(向直流母线供电)，负 = 充电(吸收母线功率)。
  SOC：0~100%，由名义电池容量 energy_kwh 与充放电累计能量积分得到。

提供的采样/仿真接口：
  - step(pv, load, dt_s, cmd)    单步平衡计算，返回四个端口的瞬时采样(联动)
  - run(times, pv, load)         整段时序仿真，返回四端口 DataFrame 采样数据
  - samples()                    返回 {'inverter','battery','grid','pv'} 采样表
"""

from __future__ import annotations

import sys

# 强制标准输出/错误为 UTF-8：IDE 运行/调试控制台按 UTF-8 解码，避免中文乱码
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd

# 便捷采样列名
T = "time"
P = "功率(kW)"
FWD = "正向有功总电能"   # 电能累计，方向与 PCS 电表约定一致（放电/输出为正方向累计）
REV = "反向有功总电能"   # 电能累计（充电/输入方向累计）


class HybridInverter:
    """光储一体机模型：PV + 电池直流母线并联，逆变口带负载并与电网并联。"""

    # ---------------------------------------------------------------- 初始化
    def __init__(
        self,
        rated_power_kw: float = 125.0,       # 逆变额定 1 倍功率 kW
        pv_overratio: float = 2.0,           # PV 超配倍数（2 倍）
        battery_rated_kw: float = 125.0,     # 电池口额定功率 kW（充=放）
        battery_capacity_kwh: float = 220.0, # 电池名义容量 kWh
        soc_init: float = 50.0,
        soc_min: float = 10.0,
        soc_max: float = 90.0,
        grid_charge_allowed: bool = False,   # 是否允许电网经逆变口整流给电池充电
        export_limit_kw: float | None = None,  # 馈网功率限制（None=不限，受逆变额定约束）
        grid_import_limit_kw: float | None = None,  # 购电功率限制（None=不限）
        keep_total_discharge: bool = False,   # 图3：TOU放电时段维持总功率不变（光伏置换电池放电）
    ):
        # ---- 额定/端口容量
        self.inverter_rated_kw = float(rated_power_kw)
        self.pv_rated_kw = float(rated_power_kw) * pv_overratio   # 2 倍光伏超配
        self.battery_rated_kw = float(battery_rated_kw)           # 电池口 1 倍

        # ---- 电池状态参数
        self.battery_capacity_kwh = float(battery_capacity_kwh)
        self.soc_min = float(soc_min)
        self.soc_max = float(soc_max)

        # ---- 控制接口状态
        self.inv_power_limit_kw = self.inverter_rated_kw   # 逆变功率限制（可下调）
        self.export_limit_kw = export_limit_kw
        self.grid_import_limit_kw = grid_import_limit_kw
        self.grid_charge_allowed = grid_charge_allowed
        self.keep_total_discharge = bool(keep_total_discharge)

        # ---- 运行状态
        self.reset(soc_init=soc_init)

        # ---- 采样记录
        self._records: list[dict] = []

    # ================================================================ 控制接口
    def set_inverter_power_limit(self, limit_kw: float) -> None:
        """逆变口功率限制接口：限制逆变交/直流转换功率（0 < limit <= 额定）。"""
        if not (0 < limit_kw <= self.inverter_rated_kw):
            raise ValueError(f"逆变功率限制须在 (0, {self.inverter_rated_kw}] kW 内")
        self.inv_power_limit_kw = float(limit_kw)

    def set_battery_power_limit(self, charge_kw=None, discharge_kw=None) -> None:
        """电池口功率控制接口：设置最大充电/放电功率(kW)，均不超过电池口额定。"""
        if discharge_kw is not None:
            if not (0 <= discharge_kw <= self.battery_rated_kw):
                raise ValueError("放电功率限制须在 [0, battery_rated_kw] 内")
            self.battery_discharge_limit_kw = float(discharge_kw)
        if charge_kw is not None:
            if not (0 <= charge_kw <= self.battery_rated_kw):
                raise ValueError("充电功率限制须在 [0, battery_rated_kw] 内")
            self.battery_charge_limit_kw = float(charge_kw)

    def set_battery_soc_limits(self, soc_min: float = None, soc_max: float = None) -> None:
        """电池 SOC 上下限接口：低于下限停止放电，高于上限停止充电。"""
        lo = self.soc_min if soc_min is None else float(soc_min)
        hi = self.soc_max if soc_max is None else float(soc_max)
        if not (0 <= lo < hi <= 100):
            raise ValueError("需满足 0 <= soc_min < soc_max <= 100")
        self.soc_min, self.soc_max = lo, hi

    # ================================================================ 状态重置
    def reset(self, soc_init: float = None) -> None:
        if soc_init is None:
            soc_init = self.soc_init if hasattr(self, "soc_init") else self.soc_min
        soc_init = float(soc_init)
        if not (0 <= soc_init <= 100):
            raise ValueError("初始 SOC 须在 [0,100]")
        self.soc_init = soc_init
        self.soc = soc_init
        self._stored_kwh = self.battery_capacity_kwh * soc_init / 100.0
        self.battery_charge_limit_kw = self.battery_rated_kw
        self.battery_discharge_limit_kw = self.battery_rated_kw
        # 各端口累计电能（起始置 0，只做相对积分）
        self.grid_fwd = self.grid_rev = 0.0
        self.inv_fwd = self.inv_rev = 0.0
        self.bat_fwd = self.bat_rev = 0.0
        self.pv_fwd = 0.0
        self._curtail_kwh = 0.0
        self._unmet_kwh = 0.0
        self._load_kwh = 0.0
        self._records = []

    # ================================================================ 单步平衡
    def _clamp_battery(self, dt_h: float):
        """根据 SOC 限值/容量，计算本步电池功率允许范围 [b_min(充电为负), b_max]。"""
        b_max = min(                    # 最大放电
            self.battery_discharge_limit_kw,
            max(0.0, self._stored_kwh - self.battery_capacity_kwh * self.soc_min / 100) / dt_h,
        )
        b_min = -min(                   # 最大充电（负值）
            self.battery_charge_limit_kw,
            max(0.0, self.battery_capacity_kwh * self.soc_max / 100 - self._stored_kwh) / dt_h,
        )
        return b_min, b_max

    def step(self, pv_avail_kw: float, load_kw: float, dt_s: float,
             battery_power_cmd: float | None = None) -> dict:
        """
        单步仿真：给定该时刻光伏可用功率与负载功率（+可选电池功率指令），
        完成直流/交流母线功率平衡，更新 SOC 与各端口累计电能。

        返回各端口瞬时采样 dict（支持外部调度/EMS 逐拍联动）：
          {'inverter','battery','grid','pv','load','soc', ...}
        """
        dt_h = dt_s / 3600.0
        if dt_h <= 0:
            raise ValueError("dt_s 必须 > 0")

        # ---- 输入（异常输入防护：NaN/±inf 按缺省处理，超限钳制到端口容量）
        def _num(x, default=0.0):
            x = float(x)
            return default if not np.isfinite(x) else x

        a0 = min(max(_num(pv_avail_kw), 0.0), self.pv_rated_kw)   # 光伏可用
        l = max(_num(load_kw), 0.0)                                # 负载
        cmd = battery_power_cmd
        if cmd is not None and not np.isfinite(float(cmd)):
            cmd = None                                            # 非有限指令 -> 交由自动策略
        inv_max = self.inv_power_limit_kw

        # ================= EMS 混逆执行策略（按 提示词.md 修改后语义） =================
        # F: 馈网(卖电)上限 export_limit_kw；0=防逆流(不馈网)；None=不限(受逆变额定约束)
        # I: 电网最大取电功率 grid_import_limit_kw；0=不从电网取电
        # 控制优先级（提示词.md）：
        #   SOC 到上/下限保护时电池功率清零；
        #   光伏供给优先级 负载 > 电池充电 > 电网馈网；
        #   储能充满(无充电余量)前富余不馈网（弃光），充满后余电才馈网 ≤F；
        #   电池充电电源优先级 光伏 > 电网；馈网电能优先级 光伏 > 电池；
        #   光伏消纳 > TOU 计划（充/放/待机均适用，富余削减计划放电、可反转为充电，
        #   最大充电功率 1 倍额定）；TOU 放电不突破计划；总取电(负载+充电) ≤ I。
        F = self.export_limit_kw
        I = self.grid_import_limit_kw
        plan = 0.0 if cmd is None else float(cmd)
        plan_mag = abs(plan)
        inv_max = self.inv_power_limit_kw

        b_min, b_max = self._clamp_battery(dt_h)     # 本步电池允许范围（负=充，正=放）
        chg_cap = max(-b_min, 0.0)                   # 充电能力（≤1 倍额定 + SOC 余量）
        dis_cap = max(b_max, 0.0)                    # 放电能力

        # ---- 光伏分配（优先级：负载 → 电池充电 → 电网馈网）
        pv_to_load = min(a0, l)                      # 1) 供本地负载
        surplus = max(a0 - l, 0.0)                   #    负载外富余
        deficit = max(l - a0, 0.0)                   #    负载缺口（光伏不足）

        const_disch = bool(self.keep_total_discharge and plan > 0.0)
        if const_disch:
            # 图3 —— TOU放电恒总功率：
            #   目标 AC 出力 = 负载 + 馈网目标(≤F、≤逆变额定) 维持不变；
            #   放电时段光伏有出力时优先供 AC（负载优先、剩余馈网），电池放电随之减小
            #   (≤TOU计划/放电能力)，保证总功率(负载+馈网)不变；电池不反转充电，超目标弃光。
            e_tgt = 0.0 if F is None else max(F, 0.0)
            out_tgt = min(l + e_tgt, inv_max)
            pv_use = min(a0, out_tgt)                  # 光伏优先（供负载→馈网）
            pv_to_load = min(pv_use, l)
            pv_export = pv_use - pv_to_load
            need = max(out_tgt - pv_use, 0.0)          # 光伏之后仍需电池补足
            d = min(need, plan_mag, dis_cap)
            c_pv, c_grid = 0.0, 0.0                    # 恒总功率放电：不充电、不从电网补充
            a_use = pv_use
            out = pv_use + d
            inv_set = float(np.clip(out, -inv_max, inv_max))
        else:
            # 图1/图2 —— 光伏供给优先级：负载 → 电池充电 → 电网馈网
            c_pv = min(surplus, chg_cap)               # 富余充电池（可突破 TOU 计划，≤1 倍额定）
            pv_left = surplus - c_pv
            export_cap = inv_max if F is None else min(inv_max, max(F, 0.0))
            # 储能充满(无充电余量)后才允许馈网；充满前富余不馈网（≤F、≤逆变能力）
            pv_export = 0.0 if chg_cap > 0.0 else min(pv_left, export_cap)

            # 电池放电（仅光伏无富余时按 TOU 计划执行；不突破计划，计划内可馈网 ≤F）
            d = 0.0
            if plan > 0.0 and surplus <= 0.0 and dis_cap > 0.0:
                if F is None:
                    room = plan_mag                    # 无馈网上限：按计划全额放电
                else:
                    room = deficit + max(F, 0.0)       # 补负载缺口 + 计划内馈网空间(≤F)
                d = min(plan_mag, room, dis_cap)

            # TOU 充电计划：光伏已充后不足部分由电网补足；总取电(负载+充电) ≤ I
            c_grid = 0.0
            if plan < 0.0 and chg_cap > 0.0:
                need_grid_chg = max(plan_mag - c_pv, 0.0)   # 光伏已充后仍需电网部分
                if need_grid_chg > 0.0:
                    load_imp = max(deficit - d, 0.0)        # 放电后负载仍缺（电网供电）
                    if I is None:
                        c_grid = need_grid_chg              # 充电电源优先级 光伏>电网
                    else:
                        c_grid = min(need_grid_chg, max(I - load_imp, 0.0))

            # ---- 交流功率平衡（逆变口口径 = 光伏实际 + 电池净，负 = 整流充电）
            a_use = pv_to_load + pv_export + c_pv          # 光伏实际利用（负载+馈网+充电池）
            out = pv_to_load + pv_export + d - c_grid      # 逆变口净送出（含整流充电）
            inv_set = float(np.clip(out, -inv_max, inv_max))   # EMS 目标 AC 出力（限幅前）
            # 逆变限幅：AC 输出过高先削光伏馈网→削放电→削充电消纳(弃光)；整流充电不超逆变能力
            if out > inv_max:
                red = out - inv_max
                red_export = min(pv_export, red)           # 先削减光伏馈网（保负载/保充电）
                pv_export -= red_export
                a_use -= red_export
                red_rest = red - red_export
                red_dis = min(d, red_rest)                 # 再削减电池放电
                d -= red_dis
                red_rest -= red_dis
                c_pv = max(c_pv - red_rest, 0.0)           # 仍超出→削减充电/弃光消纳
                a_use = max(pv_to_load + pv_export + c_pv, 0.0)
                out = pv_to_load + pv_export + d - c_grid
            if out < -inv_max:
                c_grid = min(c_grid, inv_max + pv_to_load + pv_export + d)
                out = pv_to_load + pv_export + d - c_grid
        b = d - (c_pv + c_grid)                            # 电池净功率（正放负充），限幅后重算

        # ---- 电网交换：g>0 取电(购电)，g<0 馈网
        g = l - out
        unmet = 0.0
        if I is not None and g > I:
            unmet = g - I                              # 负载缺供(取电上限内优先保负载)
            g = I
        if F is not None and g < -F:                   # 馈网不超过上限（防逆流时 F=0）
            g = -F
        curtail = max(a0 - a_use, 0.0)

        # 电池“设置功率”（提示词公式）：放电=TOU计划功率(≤1倍额定)；充电/待机=∞(置 NaN 断线)
        battery_set_kw = min(plan, self.battery_rated_kw) if plan > 0 else float("nan")

        # ---- 状态/电能更新
        self._stored_kwh = float(np.clip(
            self._stored_kwh - b * dt_h,
            self.battery_capacity_kwh * self.soc_min / 100,
            self.battery_capacity_kwh * self.soc_max / 100,
        ))
        self.soc = self._stored_kwh / self.battery_capacity_kwh * 100.0
        self.grid_fwd += max(g, 0.0) * dt_h
        self.grid_rev += max(-g, 0.0) * dt_h
        self.inv_fwd += max(out, 0.0) * dt_h
        self.inv_rev += max(-out, 0.0) * dt_h
        self.bat_fwd += max(b, 0.0) * dt_h
        self.bat_rev += max(-b, 0.0) * dt_h
        self.pv_fwd += a_use * dt_h
        self._curtail_kwh += curtail * dt_h
        self._unmet_kwh += unmet * dt_h
        self._load_kwh += l * dt_h

        # ---- 返回四端口瞬时采样（联动接口）
        return {
            "pv_avail_kw": a0, "pv_used_kw": a_use,
            "battery_pw_kw": b, "battery_cmd_kw": float("nan") if cmd is None else plan,
            "battery_set_kw": battery_set_kw,
            "inverter_pw_kw": out, "inverter_set_kw": inv_set,
            "inv_cap_kw": inv_max,           # 逆变口功率限制(额定/可设定限制)
            "grid_pw_kw": g, "load_kw": l,
            "soc": self.soc,
            "curtail_kw": curtail, "unmet_kw": unmet,
        }

    # ================================================================ 整段仿真
    def run(self, times, pv_avail_kw, load_kw, battery_power_cmd=None) -> dict:
        """
        按时间序列仿真，返回 {'inverter','battery','grid','pv'} 四端口采样 DataFrame。
        times: 长度 N 的时间序列；pv_avail_kw / load_kw: 长度 N 的序列或标量。
        """
        times = pd.DatetimeIndex(times)
        pv = np.broadcast_to(np.asarray(pv_avail_kw, dtype=float), times.shape)
        ld = np.broadcast_to(np.asarray(load_kw, dtype=float), times.shape)
        if battery_power_cmd is None:
            cmd = [None] * len(times)
        else:
            cmd = list(np.broadcast_to(np.asarray(battery_power_cmd, dtype=float), times.shape))

        self.reset()
        # 相邻时间差（秒）。注意：此环境 pandas 时间分辨率可能为微秒，
        # 不能用 asi8/1e9，须用 timedelta64 精确换算。
        if len(times) > 1:
            dts = np.asarray((times[1:] - times[:-1]) / np.timedelta64(1, "s"), dtype=float)
        else:
            dts = np.asarray([], dtype=float)
        if len(dts) and dts.min() <= 0:
            raise ValueError("times 必须严格递增")
        if len(times) == 0:
            return self.samples()

        dt0 = float(dts[0]) if len(dts) else 60.0
        for i, t in enumerate(times):
            dt_s = dt0 if i == 0 else float(dts[i - 1])
            row = self.step(pv[i], ld[i], dt_s, cmd[i])
            row["time"] = t
            row["grid_fwd_kwh"] = self.grid_fwd
            row["grid_rev_kwh"] = self.grid_rev
            row["inv_fwd_kwh"] = self.inv_fwd
            row["inv_rev_kwh"] = self.inv_rev
            row["bat_fwd_kwh"] = self.bat_fwd
            row["bat_rev_kwh"] = self.bat_rev
            row["pv_fwd_kwh"] = self.pv_fwd
            self._records.append(row)
        return self.samples()

    # ================================================================ 采样数据
    def samples(self) -> dict:
        """返回四端口仿真采样表（DataFrame），含累计电能，支持继续绘图/落盘。"""
        if not self._records:
            return {}
        r = pd.DataFrame(self._records)

        def build(pw, fwd, rev, extra=None):
            df = pd.DataFrame({T: r[T], P: np.round(r[pw], 3),
                               FWD: np.round(r[fwd], 4), REV: np.round(r[rev], 4)})
            if extra:
                df[extra] = np.round(r[extra], 2)
            return df

        inv = build("inverter_pw_kw", "inv_fwd_kwh", "inv_rev_kwh")
        inv = inv.rename(columns={P: "逆变功率(kW)"})
        inv["逆变限制功率(kW)"] = np.round(r["inv_cap_kw"], 3)
        inv["逆变设置功率(kW)"] = np.round(r["inverter_set_kw"], 3)
        bat = build("battery_pw_kw", "bat_fwd_kwh", "bat_rev_kwh", "soc")
        bat = bat.rename(columns={P: "电池功率(kW)", "soc": "储能SOC(%)"})
        bat["电池设置功率(kW)"] = np.round(r["battery_set_kw"], 2)
        grid = build("grid_pw_kw", "grid_fwd_kwh", "grid_rev_kwh")
        grid = grid.rename(columns={P: "电网功率(kW)"})
        pv = pd.DataFrame({
            T: r[T],
            "光伏功率(kW)": np.round(r["pv_used_kw"], 3),
            "光伏可用功率(kW)": np.round(r["pv_avail_kw"], 3),
            "弃光功率(kW)": np.round(r["pv_avail_kw"] - r["pv_used_kw"], 3),
            FWD: np.round(r["pv_fwd_kwh"], 4),
        })
        return {"inverter": inv, "battery": bat, "grid": grid, "pv": pv}

    # ================================================================ 统计汇总
    def summary(self) -> dict:
        if not self._records:
            return {}
        socs = [x["soc"] for x in self._records]
        return {
            "光伏可用(kWh)": round(self.pv_fwd + self._curtail_kwh, 1),
            "光伏利用(kWh)": round(self.pv_fwd, 1),
            "弃光(kWh)": round(self._curtail_kwh, 1),
            "电池放电(kWh)": round(self.bat_fwd, 1),
            "电池充电(kWh)": round(self.bat_rev, 1),
            "购电(kWh)": round(self.grid_fwd, 1),
            "馈网(kWh)": round(self.grid_rev, 1),
            "负载用电(kWh)": round(self._load_kwh - self._unmet_kwh, 1),
            "负载缺供(kWh)": round(self._unmet_kwh, 1),
            "SOC范围(%)": f"{min(socs):.1f} ~ {max(socs):.1f}",
        }


if __name__ == "__main__":
    # ---- 演示：24 小时整段仿真，验证四端口联动 ----
    times = pd.date_range("2026-02-26 00:00", "2026-02-26 23:59", freq="1min")
    hrs = times.hour + times.minute / 60.0

    # 光伏可用：250 kW(2 倍超配) 白天正弦形状
    irr = np.clip(np.sin(np.pi * (hrs - 6.0) / 12.0), 0, 1)
    pv_avail = irr * 250.0

    # 负载：基础 30kW + 上午/傍晚双峰（傍晚高于逆变额定，验证电网补充）
    base = 30.0
    mor = 120.0 * np.exp(-((hrs - 10.0) ** 2) / (2 * 1.5 ** 2))
    eve = 160.0 * np.exp(-((hrs - 19.5) ** 2) / (2 * 2.0 ** 2))
    load = base + mor + eve

    model = HybridInverter(
        rated_power_kw=125.0, pv_overratio=2.0, battery_rated_kw=125.0,
        battery_capacity_kwh=220.0, soc_init=60.0, soc_min=10.0, soc_max=90.0,
    )
    samples = model.run(times, pv_avail, load)

    print("光储一体机模型配置：逆变 125kW / 电池 125kW·220kWh / PV 2x=250kW")
    print("仿真时长: 24h, 采样: 1min")
    s = model.summary()
    for k, v in s.items():
        print(f"  {k}: {v if isinstance(v, str) else round(v, 1)}")
    for name, df in samples.items():
        print(f"\n[{name}] 列: {list(df.columns)}  行数: {len(df)}")
