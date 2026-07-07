"""Оптимизатор режима поверх цифрового двойника (Optuna).

Идея: при ТЕКУЩЕМ состоянии печи (состав шихты + рабочая точка) найти такие значения
рычагов, которые снижают потери меди в шлак и расход ресурсов, не выходя за
технологические ограничения.

Валидность (критично): рычаги ищем в ЛОКАЛЬНОЙ окрестности текущей точки
(±LOCAL_FRAC, пересечённой с перцентилями данных). Двойник моделирует локальный отклик
и не экстраполирует в нефизичную зону — защита от «красивого, но невозможного» режима.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from config import LEVERS, CHARGE_COMPOSITION, OBJECTIVE_MINIMIZE, CONSTRAINTS
from data import load
from twin import DigitalTwin, build

optuna.logging.set_verbosity(optuna.logging.WARNING)

LOCAL_FRAC = 0.15        # рычаги крутим в пределах ±15% от текущего значения
PENALTY = 50.0           # штраф за нарушение технологического ограничения
REPORTS = Path("reports")


def lever_bounds(df: pd.DataFrame, point: dict) -> dict:
    """Границы поиска по каждому рычагу: локальная окрестность ∩ перцентили данных."""
    bounds = {}
    for lv in LEVERS:
        cur = point[lv]
        p1, p99 = np.percentile(df[lv], [1, 99])
        lo = max(cur * (1 - LOCAL_FRAC), p1)
        hi = min(cur * (1 + LOCAL_FRAC), p99)
        lo, hi = min(lo, cur), max(hi, cur)   # границы всегда включают текущую точку
        if lo >= hi:                          # вырожденный случай — берём перцентили
            lo, hi = float(min(p1, cur)), float(max(p99, cur))
        bounds[lv] = (float(lo), float(hi))
    return bounds


def objective_value(levers: dict, outputs: dict, refs: dict) -> float:
    """Целевая: взвешенная сумма НОРМИРОВАННЫХ минимизируемых величин.
    Каждый член делим на его типичный масштаб (refs) — иначе крупные по величине ресурсы
    (natural_gas≈800) подавляют главный таргет Cu_slag≈0.9. После нормировки веса —
    чистые приоритеты, и Cu_slag доминирует.
    """
    src = {**levers, **outputs}
    return sum(w * src[k] / (refs[k] + 1e-9) for k, w in OBJECTIVE_MINIMIZE.items())


def constraint_penalty(outputs: dict) -> tuple[float, list[str]]:
    """Штраф за выход за технологические ограa-ничения + список нарушений."""
    pen, viol = 0.0, []
    for var, lim in CONSTRAINTS.items():
        val = outputs[var]
        if "min" in lim and val < lim["min"]:
            pen += (lim["min"] - val) / abs(lim["min"]); viol.append(f"{var}<{lim['min']}")
        if "max" in lim and val > lim["max"]:
            pen += (val - lim["max"]) / abs(lim["max"]); viol.append(f"{var}>{lim['max']}")
    return pen, viol


def optimize(twin: DigitalTwin, df: pd.DataFrame, point: dict,
             n_trials: int = 400) -> dict:
    bounds = lever_bounds(df, point)
    fixed_charge = {c: point[c] for c in CHARGE_COMPOSITION}
    refs = {k: float(df[k].mean()) for k in OBJECTIVE_MINIMIZE}   # масштабы для нормировки

    def objective(trial: optuna.Trial) -> float:
        levers = {lv: trial.suggest_float(lv, *bounds[lv]) for lv in LEVERS}
        outputs = twin.predict_one({**levers, **fixed_charge})
        pen, _ = constraint_penalty(outputs)
        return objective_value(levers, outputs, refs) + PENALTY * pen

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial({lv: point[lv] for lv in LEVERS})   # стартуем с текущего режима
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_levers = {lv: study.best_params[lv] for lv in LEVERS}
    out_before = twin.predict_one({**{lv: point[lv] for lv in LEVERS}, **fixed_charge})
    out_after = twin.predict_one({**best_levers, **fixed_charge})
    return {
        "study": study,
        "bounds": bounds,
        "best_levers": best_levers,
        "current_levers": {lv: point[lv] for lv in LEVERS},
        "out_before": out_before,
        "out_after": out_after,
    }


def save_plots(study: optuna.Study) -> None:
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history, plot_param_importances)
        import matplotlib.pyplot as plt
        (REPORTS / "figures").mkdir(parents=True, exist_ok=True)
        for fn, name in [(plot_optimization_history, "opt_history"),
                         (plot_param_importances, "opt_param_importance")]:
            ax = fn(study); ax.figure.tight_layout()
            ax.figure.savefig(REPORTS / "figures" / f"{name}.png", dpi=120)
            plt.close(ax.figure)
    except Exception as e:      # графики — не критичный шаг
        print(f"(графики пропущены: {e})")


def batch_gain(twin, df, points, n_trials=200):
    """Средний выигрыш по выборке рабочих точек — устойчивая демо-метрика."""
    cu_before, cu_after, res_before, res_after, viol_free = [], [], [], [], 0
    resource_keys = [k for k in OBJECTIVE_MINIMIZE if k in LEVERS]
    for p in points:
        r = optimize(twin, df, p, n_trials=n_trials)
        cu_before.append(r["out_before"]["Cu_slag"])
        cu_after.append(r["out_after"]["Cu_slag"])
        res_before.append(sum(r["current_levers"][k] for k in resource_keys))
        res_after.append(sum(r["best_levers"][k] for k in resource_keys))
        if not constraint_penalty(r["out_after"])[1]:
            viol_free += 1
    cu_b, cu_a = np.mean(cu_before), np.mean(cu_after)
    rs_b, rs_a = np.mean(res_before), np.mean(res_after)
    return {
        "n_points": len(points),
        "cu_slag_before": round(float(cu_b), 4),
        "cu_slag_after": round(float(cu_a), 4),
        "cu_slag_gain_pct": round(float((cu_a - cu_b) / cu_b * 100), 1),
        "resource_gain_pct": round(float((rs_a - rs_b) / rs_b * 100), 1),
        "feasible_pct": round(viol_free / len(points) * 100, 0),
    }


def main():
    df = load()
    twin = build(recent_window=2016, save=True)   # двойник на актуальном окне

    # Выборка недавних точек с наибольшим потенциалом (высокий Cu_slag = есть что снижать)
    recent = df.iloc[-2016:]
    sample = recent.nlargest(15, "Cu_slag")
    points = [row.to_dict() for _, row in sample.iterrows()]

    print("Средний выигрыш по 15 недавним точкам с высокими потерями меди:")
    gain = batch_gain(twin, df, points, n_trials=200)
    print(f"  Cu_slag:  {gain['cu_slag_before']} -> {gain['cu_slag_after']}  "
          f"({gain['cu_slag_gain_pct']}%)")
    print(f"  Ресурсы (O2+газ+уголь): {gain['resource_gain_pct']}%")
    print(f"  Режимов в допуске: {gain['feasible_pct']}%\n")

    # Детальный разбор одной репрезентативной точки
    point = points[0]
    res = optimize(twin, df, point, n_trials=400)

    def fmt(d, keys): return {k: round(d[k], 3) for k in keys}
    obj_keys = list(OBJECTIVE_MINIMIZE)
    con_keys = list(CONSTRAINTS)
    # объединённые источники: рычаги (ресурсы) + выходы двойника (Cu_slag и пр.)
    src_before = {**res["current_levers"], **res["out_before"]}
    src_after = {**res["best_levers"], **res["out_after"]}

    print("=" * 64)
    print("ОПТИМИЗАЦИЯ РЕЖИМА — текущая точка vs рекомендация")
    print("=" * 64)
    print("\nРычаги (было -> стало):")
    for lv in LEVERS:
        a, b = res["current_levers"][lv], res["best_levers"][lv]
        print(f"  {lv:22} {a:9.2f} -> {b:9.2f}  ({(b-a)/a*100:+.1f}%)")

    print("\nЦелевые показатели (Cu_slag — прогноз двойника, ресурсы — рычаги):")
    for k in obj_keys:
        a, b = src_before[k], src_after[k]
        arrow = "↓" if b < a else "↑"
        print(f"  {k:18} {a:9.3f} -> {b:9.3f}  {arrow}")

    print("\nОграничения (после оптимизации):")
    pen, viol = constraint_penalty(res["out_after"])
    for k in con_keys:
        print(f"  {k:18} = {res['out_after'][k]:9.3f}  лимит {CONSTRAINTS[k]}")
    print(f"  Нарушения: {viol if viol else 'нет — режим допустим'}")

    cu_a, cu_b = res["out_before"]["Cu_slag"], res["out_after"]["Cu_slag"]
    print(f"\nПотери меди в шлак: {cu_a:.3f} -> {cu_b:.3f}  "
          f"({(cu_b-cu_a)/cu_a*100:+.1f}%)")

    save_plots(res["study"])
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "optimization_result.json").write_text(json.dumps({
        "batch_summary": gain,
        "current_levers": fmt(res["current_levers"], LEVERS),
        "best_levers": fmt(res["best_levers"], LEVERS),
        "targets_before": fmt(src_before, obj_keys + con_keys),
        "targets_after": fmt(src_after, obj_keys + con_keys),
        "violations": viol,
    }, ensure_ascii=False, indent=2))
    print("\nСохранено: reports/optimization_result.json, reports/figures/*.png")


if __name__ == "__main__":
    main()
