"""Сквозной поток системы (end-to-end): от онлайн-датчиков до рекомендации режима.

Замыкает всё в одну цепочку, работающую БЕЗ лабораторных данных на входе:

    онлайн-датчики (рычаги + состояние печи)
        -> [1] оценка состава шихты      (charge soft-sensor)     — убираем лаг шихты
        -> [2] оценка штейна Cu_matte     (matte soft-sensor)      — убираем лаг штейна
        -> [3] прогноз продуктов          (двойник на ОЦЕНЁННОМ составе)
        -> [4] рекомендация режима         (Optuna: минимум Cu_slag + ресурсы)
        + [5] флаги: смена штабеля / отклонение режима

Ключевое отличие от отдельных скриптов: двойник и оптимизатор питаются ОЦЕНЁННЫМ
составом шихты (из слоя 1), а не истинным лабораторным. Это и есть работа «как на
реальной печи», где состав онлайн неизвестен.

Запуск:  PYTHONPATH=src .venv/bin/python src/pipeline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import (LEVERS, ONLINE, ONLINE_STATE, CHARGE_COMPOSITION,
                    OBJECTIVE_MINIMIZE, CONSTRAINTS)
from data import load, time_split
from soft_sensor import SoftSensor
from twin import DigitalTwin
from optimize import optimize, constraint_penalty

WINDOW = 2016          # актуальное окно обучения (~2 недели)
REPORTS = Path("reports")
MODELS_DIR = Path("models")


class FurnacePipeline:
    """Единый объект: обучается на истории, принимает онлайн-срез, выдаёт всё сразу."""

    def __init__(self):
        self.charge_sensor = SoftSensor(targets=CHARGE_COMPOSITION, features=ONLINE)
        self.matte_sensor = SoftSensor(targets=["Cu_matte"], features=ONLINE)
        self.twin = DigitalTwin()
        self.ref: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, window: int = WINDOW) -> "FurnacePipeline":
        recent = df.iloc[-window:]
        self.charge_sensor.fit(recent)
        self.matte_sensor.fit(recent)
        self.twin.fit(recent)
        self.ref = df
        return self

    def estimate_charge(self, online: pd.DataFrame) -> dict:
        """[1] Оценка состава шихты по онлайн-сигналам (без лаборатории)."""
        return self.charge_sensor.predict(online).iloc[0].to_dict()

    def run(self, snapshot: dict, last_lab_charge: dict | None = None,
            n_trials: int = 200) -> dict:
        """Полный проход для одного онлайн-среза датчиков.

        last_lab_charge — последний известный лабораторный состав (персистентность).
        Состав шихты кусочно-постоянен, поэтому это точная оценка между сменами партий;
        если не задан, откатываемся к онлайн soft-sensor'у.
        """
        online = pd.DataFrame([snapshot])[ONLINE]

        # [1] оценка шихты: персистентность последнего лаб-замера, иначе soft-sensor.
        soft_charge = self.estimate_charge(online)
        est_charge = dict(last_lab_charge) if last_lab_charge else soft_charge
        # [2] оценка штейна — из онлайн-сигналов
        est_matte = float(self.matte_sensor.predict(online).iloc[0]["Cu_matte"])

        # [3] текущие продукты по двойнику на ОЦЕНЁННОМ составе
        levers = {k: snapshot[k] for k in LEVERS}
        cur_out = self.twin.predict_one({**levers, **est_charge})

        # [4] рекомендация режима — оптимизатор на оценённом составе
        point = {**snapshot, **est_charge}
        res = optimize(self.twin, self.ref, point, n_trials=n_trials)
        _, viol = constraint_penalty(res["out_after"])

        cu_b, cu_a = res["out_before"]["Cu_slag"], res["out_after"]["Cu_slag"]
        return {
            "charge_source": "lab_persistence" if last_lab_charge else "soft_sensor",
            "estimated_charge": {k: round(v, 3) for k, v in est_charge.items()},
            "soft_S_charge": round(soft_charge["S_charge"], 3),   # онлайн-кросс-чек по сере
            "estimated_matte": round(est_matte, 3),
            "current_products": {k: round(v, 3) for k, v in cur_out.items()},
            "recommended_levers": {k: round(res["best_levers"][k], 3) for k in LEVERS},
            "cu_slag_before": round(cu_b, 4),
            "cu_slag_after": round(cu_a, 4),
            "cu_slag_gain_pct": round((cu_a - cu_b) / cu_b * 100, 1),
            "feasible": not viol,
            "violations": viol,
        }

    def save(self) -> None:
        import joblib
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(self, MODELS_DIR / "pipeline.pkl")


LAB_INTERVAL = 48   # лаборатория даёт состав раз в смену (8ч)


def validate_charge(df: pd.DataFrame) -> dict:
    """Честная точность оценки шихты (обучение на train, тест на val — БЕЗ перекрытия).

    Сравниваем два источника оценки состава между лабораторными замерами:
      - online soft-sensor (по датчикам);
      - persistence (последний лаб-замер, LAB_INTERVAL назад) — реалистичный базовый.
    """
    tr, va = time_split(df, 0.2)
    sensor = SoftSensor(targets=CHARGE_COMPOSITION, features=ONLINE).fit(tr)
    est = sensor.predict(va).reset_index(drop=True)
    va = va.reset_index(drop=True)
    out = {}
    for c in CHARGE_COMPOSITION:
        soft_r2 = round(float(r2_score(va[c], est[c])), 3)
        pers = va[c].shift(LAB_INTERVAL).bfill()
        pers_mae = float((va[c] - pers).abs().mean())
        soft_mae = float((va[c] - est[c]).abs().mean())
        out[c] = {"soft_R2": soft_r2, "soft_MAE": round(soft_mae, 3),
                  "persist_MAE": round(pers_mae, 3)}
    return out


def main():
    df = load()
    pipe = FurnacePipeline().fit(df)
    pipe.save()

    print("=" * 66)
    print("СКВОЗНОЙ ПОТОК (end-to-end): онлайн-датчики -> рекомендация")
    print("=" * 66)

    # --- [1] Оценка состава шихты: онлайн soft-sensor vs персистентность лаб-замера ---
    ch = validate_charge(df)
    print("\n[1] Оценка состава шихты между лабораторными замерами (held-out):")
    print(f"    {'компонент':14} {'soft R²':>8} {'soft MAE':>9} {'persist MAE':>12}")
    for c, m in ch.items():
        print(f"    {c:14} {m['soft_R2']:>8} {m['soft_MAE']:>9} {m['persist_MAE']:>12}")
    print("    Сера наблюдаема онлайн (soft-sensor); остальное точнее вести")
    print("    персистентно (состав кусочно-постоянен) + детектор смены партии.")

    # --- Демо: «текущий» онлайн-срез, состав известен из прошлой смены (персистентность) ---
    _, va = time_split(df, 0.2)
    va = va.reset_index(drop=True)
    # берём проблемный режим (высокие потери меди) — когда оператору и нужен совет
    cand = va.iloc[LAB_INTERVAL:]
    i = int(cand["Cu_slag"].idxmax())
    row = va.iloc[i].to_dict()
    snapshot = {k: row[k] for k in ONLINE}                       # оператор видит только это
    last_lab = {c: va.iloc[i - LAB_INTERVAL][c] for c in CHARGE_COMPOSITION}  # прошлая смена
    result = pipe.run(snapshot, last_lab_charge=last_lab, n_trials=250)

    print("\n[2/3] По онлайн-срезу система оценила (без ожидания лаборатории):")
    print(f"    Cu в штейне:  {result['estimated_matte']:.2f}  (истина {row['Cu_matte']:.2f})")
    print(f"    S в шихте (онлайн-кросс-чек): {result['soft_S_charge']:.2f}  "
          f"(истина {row['S_charge']:.2f})")
    print(f"    Cu_slag сейчас: {result['cu_slag_before']:.3f}  (истина {row['Cu_slag']:.3f})")

    print("\n[4] Рекомендация режима (оптимизатор на составе из прошлой смены):")
    print(f"    Cu_slag: {result['cu_slag_before']:.3f} -> {result['cu_slag_after']:.3f}"
          f"  ({result['cu_slag_gain_pct']}%)")
    print(f"    Режим в допуске: {'да' if result['feasible'] else 'НЕТ: ' + str(result['violations'])}")
    key_levers = ["oxygen_flow", "oxygen_concentration", "natural_gas_flow", "addition_flow"]
    print("    Ключевые рычаги (сейчас -> рекомендация):")
    for lv in key_levers:
        print(f"      {lv:20} {snapshot[lv]:8.2f} -> {result['recommended_levers'][lv]:8.2f}")

    # --- [5] Флаг смены штабеля по онлайн-потоку (детектор) ---
    from anomaly import detect_charge_shifts
    _, true_sh, sh_recall, lead_h = detect_charge_shifts(df)
    print(f"\n[5] Детектор смены штабеля онлайн: recall {sh_recall*100:.0f}%, "
          f"опережение лаборатории {lead_h:.1f} ч")

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "pipeline_result.json").write_text(json.dumps({
        "charge_estimation": ch,
        "demo": result,
        "shift_detector": {"recall_pct": round(sh_recall * 100, 1),
                           "lead_hours": round(float(lead_h), 1)},
    }, ensure_ascii=False, indent=2))
    print("\nСохранено: models/pipeline.pkl, reports/pipeline_result.json")


if __name__ == "__main__":
    main()
