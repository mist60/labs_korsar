import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # чтобы графики сохранялись без открытия окна
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import os
import json
from datetime import datetime


# ============================================================
# НАСТРОЙКА: ПУТЬ К ФАЙЛУ result.dat
# ============================================================
filepath = r"C:\Users\user\Desktop\labs_korsar\010_accumulator_drain\Final\KORSAR\result.dat"


# ============================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================

def load_data(filepath):
    """
    Загрузка данных из файла result.dat.
    Столбцы:
    0 - время, с
    1 - давление газовой подушки, Па
    2 - уровень воды, м
    3 - начальная температура, К
    4 - температура газовой подушки, К
    5 - расход, кг/с
    6 - время начала закрытия задвижки, с
    7 - время полного закрытия, с
    """
    data = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    row = [float(x) for x in parts[:8]]
                    data.append(row)
                except ValueError:
                    continue

    df = pd.DataFrame(data, columns=[
        'time', 'pressure', 'level', 'T_initial',
        'T_gas', 'flow_rate', 't_valve_start', 't_valve_end'
    ])
    return df


# ============================================================
# 2. РАСЧЁТ ПОКАЗАТЕЛЕЙ
# ============================================================

def compute_metrics(df, rho_water=990.5, tank_area=0.5):
    """
    Расчёт показателей Корсар:
    - время начала закрытия задвижки (7-й столбец)
    - время полного закрытия задвижки (8-й столбец)
    - длительность закрытия
    - время слива = время полного закрытия задвижки
    - масса слитой воды
    - давление перед закрытием задвижки
    - остаточный уровень
    - невязка баланса
    """
    t = df['time'].values
    P = df['pressure'].values
    h = df['level'].values
    G = df['flow_rate'].values
    t_start = df['t_valve_start'].values
    t_end = df['t_valve_end'].values

    # --- Время начала закрытия задвижки (7-й столбец) ---
    mask_start = t_start > 0
    if np.any(mask_start):
        t_valve_start = float(t_start[mask_start][0])
    else:
        t_valve_start = None

    # --- Время полного закрытия задвижки (8-й столбец) ---
    mask_end = t_end > 0
    if np.any(mask_end):
        t_valve_end = float(t_end[mask_end][0])
    else:
        t_valve_end = None

    # --- ВРЕМЯ СЛИВА = ВРЕМЯ ПОЛНОГО ЗАКРЫТИЯ ЗАДВИЖКИ ---
    if t_valve_end is not None:
        t_drain = t_valve_end
    else:
        # если задвижка не закрывалась — берём последний момент записи
        t_drain = float(t[-1])

    # --- Длительность закрытия задвижки ---
    if t_valve_start is not None and t_valve_end is not None:
        t_valve_duration = t_valve_end - t_valve_start
    else:
        t_valve_duration = None

    # --- Масса слитой воды (интегрируем расход до момента t_drain) ---
    mask_before = t <= t_drain
    if np.sum(mask_before) > 1:
        mass_drained = trapezoid(G[mask_before], t[mask_before])
    else:
        mass_drained = 0.0

    # --- Давление перед закрытием (в момент t_drain) ---
    if np.any(mask_before):
        P_before_close = float(P[mask_before][-1])
    else:
        P_before_close = float(P[0])

    # --- Остаточный уровень (в момент t_drain) ---
    if np.any(mask_before):
        h_residual = float(h[mask_before][-1])
    else:
        h_residual = float(h[0])

    # --- Невязка баланса ---
    V_change = (h[0] - h_residual) * tank_area
    V_drained = mass_drained / rho_water
    balance_residual = float(V_change - V_drained)

    return {
        't_valve_start': float(t_valve_start),
        't_valve_end': float(t_valve_end),
        't_valve_duration': float(t_valve_duration),
        't_drain': float(t_drain),
        'mass_drained': float(mass_drained),
        'P_before_close': float(P_before_close),
        'h_residual': float(h_residual),
        'balance_residual': float(balance_residual)
    }


# ============================================================
# 3. ПОСТРОЕНИЕ И СОХРАНЕНИЕ ГРАФИКОВ
# ============================================================

def plot_results(df, metrics, title_prefix="КОРСАР", save_dir=None):
    """
    Строит 4 отдельных графика и сохраняет их в save_dir.
    На каждом графике — вертикальные линии:
      жёлтая  — начало закрытия задвижки
      красная — полное закрытие задвижки (= время слива)
    """
    t = df['time'].values
    P = df['pressure'].values
    h = df['level'].values
    T_gas = df['T_gas'].values
    G = df['flow_rate'].values

    t_close_start = metrics.get('t_valve_start')
    t_close_end = metrics.get('t_valve_end')   # = t_drain

    plots = [
        ('flow', G, 'b-', 'Расход, кг/с',
         f'{title_prefix}: Расход от времени', None),
        ('pressure', P / 1e5, 'g-', 'Давление, бар',
         f'{title_prefix}: Давление газовой подушки',
         (metrics['P_before_close'] / 1e5,
          f"P перед закрытием = {metrics['P_before_close']/1e5:.4f} бар")),
        ('level', h, 'm-', 'Уровень, м',
         f'{title_prefix}: Уровень воды',
         (metrics['h_residual'],
          f"Остаточный уровень = {metrics['h_residual']:.4f} м")),
        ('temp', T_gas, 'r-', 'Температура, К',
         f'{title_prefix}: Температура газовой подушки', None),
    ]

    for name, y, style, ylabel, title, extra in plots:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t, y, style, linewidth=1.5, label=ylabel)

        # --- Жёлтая линия: начало закрытия задвижки ---
        if t_close_start is not None:
            ax.axvline(t_close_start, color='gold', linestyle='--',
                       linewidth=1.5,
                       label=f'Начало закрытия ({t_close_start:.2f} с)')

        # --- Красная линия: полное закрытие задвижки = время слива ---
        if t_close_end is not None:
            ax.axvline(t_close_end, color='r', linestyle='--',
                       linewidth=1.5,
                       label=f'Полное закрытие = время слива ({t_close_end:.2f} с)')

        # --- Горизонтальная линия показателя ---
        if extra is not None:
            ax.axhline(extra[0], color='orange', linestyle=':', label=extra[1])

        ax.set_xlabel('Время, с')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        if save_dir:
            path = os.path.join(save_dir, f'{name}.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"  Сохранён график: {path}")
            plt.close(fig)
        else:
            plt.show()


# ============================================================
# 4. ОСНОВНАЯ ПРОГРАММА
# ============================================================

def fmt(value, unit="", digits=4, none_str="не обнаружено"):
    """Красиво форматирует число или пишет 'не обнаружено'."""
    if value is None:
        return none_str
    return f"{value:.{digits}f} {unit}".strip()


def main():
    # --- Проверка файла ---
    if not os.path.exists(filepath):
        print(f"ОШИБКА: файл не найден по пути:\n  {filepath}")
        print("\nПроверь переменную filepath в начале скрипта.")
        return

    abs_path = os.path.abspath(filepath)
    print(f"Используется файл: {abs_path}")

    # --- Папка для результатов ---
    results_dir = os.path.join(os.path.dirname(abs_path), 'results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"Папка для результатов: {results_dir}")

    # --- Загрузка ---
    df = load_data(filepath)
    print(f"Загружено строк: {len(df)}")
    print(df.head())

    # --- Расчёт показателей ---
    metrics = compute_metrics(df, rho_water=990.5, tank_area=0.5)

    # --- Печать в консоль ---
    print("\n" + "=" * 60)
    print("ПОКАЗАТЕЛИ КОРСАР")
    print("=" * 60)
    print(f"  Начало закрытия задвижки:  {fmt(metrics['t_valve_start'], 'с')}")
    print(f"  Полное закрытие задвижки:  {fmt(metrics['t_valve_end'], 'с')}")
    print(f"  Длительность закрытия:     {fmt(metrics['t_valve_duration'], 'с')}")
    print(f"  Время слива:               {metrics['t_drain']:.4f} с "
          f"(= время полного закрытия задвижки)")
    print(f"  Масса слитой воды:         {metrics['mass_drained']:.4f} кг")
    print(f"  Давление перед закрытием:  {metrics['P_before_close']:.4f} Па "
          f"({metrics['P_before_close'] / 1e5:.6f} бар)")
    print(f"  Остаточный уровень:        {metrics['h_residual']:.6f} м")
    print(f"  Невязка баланса:           {metrics['balance_residual']:.6g} м³")

    # --- Сохранение показателей в TXT ---
    txt_path = os.path.join(results_dir, 'metrics.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("ПОКАЗАТЕЛИ КОРСАР\n")
        f.write("=" * 60 + "\n")
        f.write(f"Файл данных:               {abs_path}\n")
        f.write(f"Дата расчёта:              {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Начало закрытия задвижки:  {fmt(metrics['t_valve_start'], 'с')}\n")
        f.write(f"Полное закрытие задвижки:  {fmt(metrics['t_valve_end'], 'с')}\n")
        f.write(f"Длительность закрытия:     {fmt(metrics['t_valve_duration'], 'с')}\n")
        f.write(f"Время слива:               {metrics['t_drain']:.4f} с "
                f"(= время полного закрытия задвижки)\n")
        f.write(f"Масса слитой воды:         {metrics['mass_drained']:.4f} кг\n")
        f.write(f"Давление перед закрытием:  {metrics['P_before_close']:.4f} Па "
                f"({metrics['P_before_close'] / 1e5:.6f} бар)\n")
        f.write(f"Остаточный уровень:        {metrics['h_residual']:.6f} м\n")
        f.write(f"Невязка баланса:           {metrics['balance_residual']:.6g} м³\n")
    print(f"  Сохранены показатели: {txt_path}")

    # --- Сохранение показателей в JSON ---
    json_path = os.path.join(results_dir, 'metrics.json')
    metrics_serializable = dict(metrics)
    metrics_serializable['source_file'] = abs_path
    metrics_serializable['timestamp'] = datetime.now().isoformat()
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_serializable, f, indent=2, ensure_ascii=False,
                  default=lambda x: None if x is None else float(x))
    print(f"  Сохранены показатели: {json_path}")

    # --- Сохранение графиков ---
    plot_results(df, metrics, title_prefix="КОРСАР", save_dir=results_dir)

    print("\nГотово. Все результаты в папке:")
    print(f"  {results_dir}")


if __name__ == "__main__":
    main()