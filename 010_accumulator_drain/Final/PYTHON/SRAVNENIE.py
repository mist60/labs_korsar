"""
Сравнение экспериментальных данных с решением системы ДУ
для изотермического (n=1) и адиабатического (n=γ) процессов.

Флаг модели HCS относится к ЭКСПЕРИМЕНТАЛЬНОЙ кривой:
  - в легенде подписываются экспериментальные точки;
  - в заголовке графика указывается режим эксперимента;
  - в имени файла — суффикс _HCS / _noHCS.

Столбцы .dat:
    0: τ, время, с
    1: P, давление газовой подушки, Па
    2: z, уровень воды, м
    3: T0, начальная температура, К (постоянна)
    4: Tг, температура газовой подушки, К
    5: G, расход воды, кг/с
    6: t7, время начала закрытия задвижки, с
    7: t8, время полного закрытия задвижки, с
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


# ---------------------------------------------------------------------------
# 0. Пути и defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(SCRIPT_DIR, "params.json")
LAST_DIR_FILE = os.path.join(SCRIPT_DIR, "last_dir.json")
LAST_HCS_FILE = os.path.join(SCRIPT_DIR, "last_hcs.json")

DEFAULTS = {
    "Fпрох":    1.0e-2,   # площадь проходного сечения, м^2
    "Fсосуда":  1.0,      # площадь поперечного сечения сосуда, м^2
    "ρ":        1000.0,   # плотность воды, кг/м^3
    "g":        9.81,     # ускорение свободного падения, м/с^2
    "Pb":       1.0e5,    # выходное (противодавление), Па
    "λ":        0.03,     # коэффициент трения
    "l":        10.0,     # длина трубопровода, м
    "d":        0.1,      # диаметр трубопровода, м
    "ξ":        1.0,      # коэффициент местного сопротивления
    "H":        5.0,      # высота сосуда, м
    "R":        287.0,    # газовая постоянная, Дж/(кг·К)
    "m":        1.0,      # масса газа, кг
    "γ":        1.4,      # показатель адиабаты
}


def load_saved_params():
    if not os.path.isfile(PARAMS_FILE):
        return dict(DEFAULTS)
    try:
        with open(PARAMS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        result = dict(DEFAULTS)
        for k, v in saved.items():
            if k in result:
                result[k] = float(v)
        print(f"Загружены сохранённые параметры из {PARAMS_FILE}")
        return result
    except Exception as ex:
        print(f"Не удалось прочитать {PARAMS_FILE}: {ex}. "
              f"Используются значения по умолчанию.")
        return dict(DEFAULTS)


def save_params(params):
    try:
        with open(PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        print(f"Параметры сохранены в {PARAMS_FILE}")
    except Exception as ex:
        print(f"Не удалось сохранить параметры: {ex}")


def load_last_dir():
    if not os.path.isfile(LAST_DIR_FILE):
        return None
    try:
        with open(LAST_DIR_FILE, "r", encoding="utf-8") as f:
            d = json.load(f).get("outdir")
        return d if d and os.path.isdir(d) else None
    except Exception:
        return None


def save_last_dir(path):
    try:
        with open(LAST_DIR_FILE, "w", encoding="utf-8") as f:
            json.dump({"outdir": path}, f, ensure_ascii=False, indent=2)
    except Exception as ex:
        print(f"Не удалось сохранить путь: {ex}")


def load_last_hcs():
    """Возвращает True/False (последний выбор пользователя) или None."""
    if not os.path.isfile(LAST_HCS_FILE):
        return None
    try:
        with open(LAST_HCS_FILE, "r", encoding="utf-8") as f:
            v = json.load(f).get("hcs")
        return bool(v) if v is not None else None
    except Exception:
        return None


def save_last_hcs(flag: bool):
    try:
        with open(LAST_HCS_FILE, "w", encoding="utf-8") as f:
            json.dump({"hcs": bool(flag)}, f, ensure_ascii=False, indent=2)
    except Exception as ex:
        print(f"Не удалось сохранить флаг HCS: {ex}")


# ---------------------------------------------------------------------------
# 1. Диалоговое окно ввода параметров
# ---------------------------------------------------------------------------
class ParamDialog(tk.Tk):
    def __init__(self, defaults):
        super().__init__()
        self.title("Параметры системы ДУ")
        self.resizable(False, False)
        self.result = None

        frm = ttk.Frame(self, padding=12)
        frm.grid()

        ttk.Label(frm, text="Параметры системы",
                  font=("", 11, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(0, 8))

        self.entries = {}
        for i, (name, val) in enumerate(defaults.items(), start=1):
            ttk.Label(frm, text=name, width=12, anchor="e").grid(
                row=i, column=0, padx=4, pady=3, sticky="e")
            e = ttk.Entry(frm, width=20)
            e.insert(0, repr(val))
            e.grid(row=i, column=1, padx=4, pady=3, sticky="w")
            self.entries[name] = e

        btns = ttk.Frame(frm)
        btns.grid(row=len(defaults) + 1, column=0, columnspan=2, pady=(12, 0))
        ttk.Button(btns, text="OK", command=self.on_ok).pack(side="left", padx=6)
        ttk.Button(btns, text="Отмена", command=self.on_cancel).pack(side="left", padx=6)
        ttk.Button(btns, text="Сброс к умолчанию",
                   command=self.on_reset).pack(side="left", padx=6)

        self.bind("<Return>", lambda e: self.on_ok())
        self.bind("<Escape>", lambda e: self.on_cancel())

    def on_ok(self):
        try:
            self.result = {k: float(e.get().replace(",", "."))
                           for k, e in self.entries.items()}
        except ValueError as ex:
            messagebox.showerror("Ошибка", f"Некорректное число: {ex}")
            return
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()

    def on_reset(self):
        for k, e in self.entries.items():
            e.delete(0, tk.END)
            e.insert(0, repr(DEFAULTS[k]))


# ---------------------------------------------------------------------------
# 1b. Диалог выбора папки сохранения
# ---------------------------------------------------------------------------
class SaveDirDialog(tk.Tk):
    def __init__(self, initial_dir):
        super().__init__()
        self.title("Папка для сохранения графиков")
        self.resizable(False, False)
        self.result = None

        frm = ttk.Frame(self, padding=12)
        frm.grid()

        ttk.Label(frm, text="Куда сохранять PNG-графики:",
                  font=("", 11, "bold")).grid(
            row=0, column=0, columnspan=3, pady=(0, 8), sticky="w")

        self.var = tk.StringVar(value=initial_dir or "")
        e = ttk.Entry(frm, textvariable=self.var, width=55)
        e.grid(row=1, column=0, columnspan=2, padx=(0, 6), pady=4, sticky="we")

        ttk.Button(frm, text="Обзор…",
                   command=self.on_browse).grid(row=1, column=2, pady=4)

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=3, pady=(12, 0))
        ttk.Button(btns, text="OK", command=self.on_ok).pack(side="left", padx=6)
        ttk.Button(btns, text="Отмена", command=self.on_cancel).pack(side="left", padx=6)

        self.bind("<Return>", lambda e: self.on_ok())
        self.bind("<Escape>", lambda e: self.on_cancel())

    def on_browse(self):
        init = self.var.get() if os.path.isdir(self.var.get()) else SCRIPT_DIR
        d = filedialog.askdirectory(
            title="Выберите папку для сохранения графиков",
            initialdir=init)
        if d:
            self.var.set(d)

    def on_ok(self):
        path = self.var.get().strip()
        if not path:
            messagebox.showerror("Ошибка", "Укажите папку.")
            return
        if not os.path.isdir(path):
            if messagebox.askyesno(
                    "Папки нет",
                    f"Папки не существует:\n{path}\n\nСоздать её?"):
                try:
                    os.makedirs(path, exist_ok=True)
                except Exception as ex:
                    messagebox.showerror("Ошибка", f"Не удалось создать: {ex}")
                    return
            else:
                return
        self.result = path
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()


# ---------------------------------------------------------------------------
# 1c. Диалог выбора модели HCS (относится к эксперименту)
# ---------------------------------------------------------------------------
class HcsDialog(tk.Tk):
    """
    Спрашивает: в эксперименте использовалась модель HCS (учёт теплообмена)?
    Результат: True / False / None (отмена).
    """
    def __init__(self, default_flag=None):
        super().__init__()
        self.title("Модель HCS (эксперимент)")
        self.resizable(False, False)
        self.result = None

        frm = ttk.Frame(self, padding=16)
        frm.grid()

        ttk.Label(
            frm,
            text="В эксперименте учитывалась модель HCS?",
            font=("", 11, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

        ttk.Label(
            frm,
            text="Пометка будет отнесена к экспериментальной кривой:\n"
                 "в легенде, заголовке графика и имени файла\n"
                 "(суффикс _HCS / _noHCS).",
            foreground="#555"
        ).grid(row=1, column=0, columnspan=2, pady=(0, 12), sticky="w")

        self.var = tk.BooleanVar(
            value=bool(default_flag) if default_flag is not None else True
        )
        ttk.Radiobutton(frm, text="Да — эксперимент с HCS",
                        variable=self.var, value=True).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Radiobutton(frm, text="Нет — эксперимент без HCS",
                        variable=self.var, value=False).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=2)

        btns = ttk.Frame(frm)
        btns.grid(row=4, column=0, columnspan=2, pady=(14, 0))
        ttk.Button(btns, text="OK", command=self.on_ok).pack(side="left", padx=6)
        ttk.Button(btns, text="Отмена", command=self.on_cancel).pack(side="left", padx=6)

        self.bind("<Return>", lambda e: self.on_ok())
        self.bind("<Escape>", lambda e: self.on_cancel())

    def on_ok(self):
        self.result = bool(self.var.get())
        self.destroy()

    def on_cancel(self):
        self.result = None
        self.destroy()


# ---------------------------------------------------------------------------
# 2. Чтение .dat
# ---------------------------------------------------------------------------
def load_dat(path):
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError("Ожидается 8 столбцов в .dat файле")
    return data


def find_t7(data):
    col = data[:, 6]
    nz = np.where(col > 0)[0]
    if len(nz) == 0:
        raise ValueError("В файле нет ненулевого значения t7 (столбец 7).")
    t7 = col[nz[0]]
    if not np.allclose(col[nz], t7):
        print("Предупреждение: значения t7 не одинаковы — берётся первое.")
    return float(t7)


# ---------------------------------------------------------------------------
# 3. Правые части ОДУ
# ---------------------------------------------------------------------------
def make_rhs_iso(p):
    Fпрох, Fсосуда, rho, g = p["Fпрох"], p["Fсосуда"], p["ρ"], p["g"]
    Pb, lam, l, d, xi = p["Pb"], p["λ"], p["l"], p["d"], p["ξ"]
    H = p["H"]

    def G_of(P, z):
        num = 2.0 * rho * (P + rho * g * z - Pb)
        den = 1.0 + lam * l / d + xi
        return Fпрох * np.sqrt(np.maximum(num / den, 0.0))

    def rhs(tau, state, P0, z0):
        z = state[0]
        Hmz = max(H - z, 1e-9)
        P = P0 * (H - z0) / Hmz
        G = G_of(P, z)
        dzdt = -G / (Fсосуда * rho)
        return [dzdt]

    return rhs, G_of


def make_rhs_adiab(p):
    Fпрох, Fсосуда, rho, g = p["Fпрох"], p["Fсосуда"], p["ρ"], p["g"]
    Pb, lam, l, d, xi = p["Pb"], p["λ"], p["l"], p["d"], p["ξ"]
    H, R, m, gamma = p["H"], p["R"], p["m"], p["γ"]

    def G_of(P, z):
        num = 2.0 * rho * (P + rho * g * z - Pb)
        den = 1.0 + lam * l / d + xi
        return Fпрох * np.sqrt(np.maximum(num / den, 0.0))

    def rhs(tau, state, P0, z0, T0):
        z, P = state
        Hmz = max(H - z, 1e-9)

        k = (gamma - 1.0) / gamma
        dTg_dP = T0 * k * (P ** (k - 1.0)) / (P0 ** k)

        G = G_of(P, z)
        dzdt = -G / (Fсосуда * rho)

        coef = 1.0 - (m * R) / (Fсосуда * Hmz) * dTg_dP
        if abs(coef) < 1e-12:
            coef = 1e-12 if coef >= 0 else -1e-12
        dPdt = (P / Hmz) * dzdt / coef

        return [dzdt, dPdt]

    return rhs, G_of


# ---------------------------------------------------------------------------
# 4. Решение
# ---------------------------------------------------------------------------
def solve_iso(p, data, t7):
    z0 = data[0, 2]
    P0 = data[0, 1]
    T0 = data[0, 3]
    Fсосуда = p["Fсосуда"]
    H = p["H"]

    rhs, G_of = make_rhs_iso(p)

    tau_eval = np.linspace(0.0, t7, 1500)
    sol = solve_ivp(lambda t, y: rhs(t, y, P0, z0),
                    (0.0, t7), [z0],
                    t_eval=tau_eval, method="LSODA",
                    rtol=1e-8, atol=1e-10)

    z = sol.y[0]
    Hmz = np.maximum(H - z, 1e-9)
    P = P0 * (H - z0) / Hmz
    G = G_of(P, z)
    Tg = np.full_like(z, T0)
    return sol.t, P, z, Tg, G


def solve_adiab(p, data, t7):
    z0 = data[0, 2]
    P0 = data[0, 1]
    T0 = data[0, 3]
    gamma = p["γ"]

    rhs, G_of = make_rhs_adiab(p)

    tau_eval = np.linspace(0.0, t7, 1500)
    sol = solve_ivp(lambda t, y: rhs(t, y, P0, z0, T0),
                    (0.0, t7), [z0, P0],
                    t_eval=tau_eval, method="LSODA",
                    rtol=1e-8, atol=1e-10)

    z = sol.y[0]
    P = sol.y[1]
    G = G_of(P, z)
    Tg = T0 * (P / P0) ** ((gamma - 1.0) / gamma)
    return sol.t, P, z, Tg, G


# ---------------------------------------------------------------------------
# 5. Построение графиков
# ---------------------------------------------------------------------------
def plot_pair(tau_exp, y_exp, tau_num, y_num, ylabel, title, fname,
              hcs_flag=None):
    """
    Строит один график. Метка HCS относится к ЭКСПЕРИМЕНТАЛЬНОЙ кривой:
      - подпись в легенде для экспериментальных точек;
      - пометка в заголовке;
      - маркеры эксперимента при включённом HCS выделяются цветом.
    Расчётная кривая подписывается нейтрально.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # --- подпись эксперимента в зависимости от HCS ---
    if hcs_flag is True:
        exp_label = "эксперимент (HCS)"
        exp_color = "tab:red"
        exp_marker = "o"
    elif hcs_flag is False:
        exp_label = "эксперимент (без HCS)"
        exp_color = "tab:blue"
        exp_marker = "o"
    else:
        exp_label = "эксперимент"
        exp_color = "C0"
        exp_marker = "o"

    ax.plot(tau_exp, y_exp, exp_marker, ms=3,
            color=exp_color, label=exp_label, alpha=0.75)
    ax.plot(tau_num, y_num, "-", lw=1.8, color="k",
            label="расчёт (ДУ)")

    ax.set_xlabel("τ, с")
    ax.set_ylabel(ylabel)

    # --- заголовок с пометкой, относящейся к эксперименту ---
    if hcs_flag is True:
        full_title = f"{title}  [эксперимент: HCS]"
    elif hcs_flag is False:
        full_title = f"{title}  [эксперимент: без HCS]"
    else:
        full_title = title
    ax.set_title(full_title)

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  сохранено: {fname}")


def hcs_suffix(hcs_flag):
    """Возвращает 'HCS' или 'noHCS' для имени файла."""
    if hcs_flag is True:
        return "HCS"
    if hcs_flag is False:
        return "noHCS"
    return ""


def build_plots(outdir, data, t7, iso, adiab, hcs_flag=None):
    tau_exp = data[:, 0]
    mask = tau_exp <= t7 + 1e-12
    te = tau_exp[mask]

    P_exp = data[mask, 1]
    z_exp = data[mask, 2]
    Tg_exp = data[mask, 4]
    G_exp = data[mask, 5]

    suf = hcs_suffix(hcs_flag)
    sfx = f"_{suf}" if suf else ""

    # Изотермический
    t_i, P_i, z_i, Tg_i, G_i = iso
    plot_pair(te, G_exp,  t_i, G_i,  "G, кг/с",
              "Расход воды (изотермический процесс)",
              os.path.join(outdir, f"расход_изотермический{sfx}.png"),
              hcs_flag=hcs_flag)
    plot_pair(te, z_exp,  t_i, z_i,  "z, м",
              "Уровень воды (изотермический процесс)",
              os.path.join(outdir, f"уровень_изотермический{sfx}.png"),
              hcs_flag=hcs_flag)
    plot_pair(te, P_exp,  t_i, P_i,  "P, Па",
              "Давление газовой подушки (изотермический процесс)",
              os.path.join(outdir, f"давление_изотермический{sfx}.png"),
              hcs_flag=hcs_flag)
    plot_pair(te, Tg_exp, t_i, Tg_i, "Tг, К",
              "Температура газа (изотермический процесс)",
              os.path.join(outdir, f"температура_изотермический{sfx}.png"),
              hcs_flag=hcs_flag)

    # Адиабатический
    t_a, P_a, z_a, Tg_a, G_a = adiab
    plot_pair(te, G_exp,  t_a, G_a,  "G, кг/с",
              "Расход воды (адиабатический процесс)",
              os.path.join(outdir, f"расход_адиабатический{sfx}.png"),
              hcs_flag=hcs_flag)
    plot_pair(te, z_exp,  t_a, z_a,  "z, м",
              "Уровень воды (адиабатический процесс)",
              os.path.join(outdir, f"уровень_адиабатический{sfx}.png"),
              hcs_flag=hcs_flag)
    plot_pair(te, P_exp,  t_a, P_a,  "P, Па",
              "Давление газовой подушки (адиабатический процесс)",
              os.path.join(outdir, f"давление_адиабатический{sfx}.png"),
              hcs_flag=hcs_flag)
    plot_pair(te, Tg_exp, t_a, Tg_a, "Tг, К",
              "Температура газа (адиабатический процесс)",
              os.path.join(outdir, f"температура_адиабатический{sfx}.png"),
              hcs_flag=hcs_flag)


# ---------------------------------------------------------------------------
# 6. main
# ---------------------------------------------------------------------------
def main():
    # --- файл данных ---
    root = tk.Tk()
    root.withdraw()
    dat_path = filedialog.askopenfilename(
        title="Выберите файл данных (.dat)",
        filetypes=[("Data files", "*.dat"), ("All files", "*.*")])
    root.destroy()
    if not dat_path:
        print("Файл не выбран. Выход.")
        return

    data = load_dat(dat_path)
    print(f"Загружено строк: {len(data)}")

    # --- параметры ---
    initial = load_saved_params()
    dlg = ParamDialog(initial)
    dlg.mainloop()
    if dlg.result is None:
        print("Ввод отменён. Выход.")
        return
    p = dlg.result
    save_params(p)
    print("Параметры:", p)

    # --- выбор модели HCS (относится к эксперименту) ---
    last_hcs = load_last_hcs()
    hcs_dlg = HcsDialog(default_flag=last_hcs)
    hcs_dlg.mainloop()
    if hcs_dlg.result is None:
        print("Выбор HCS отменён. Выход.")
        return
    hcs_flag = hcs_dlg.result
    save_last_hcs(hcs_flag)
    print(f"Эксперимент: {'с HCS' if hcs_flag else 'без HCS'}")

    # --- папка для сохранения графиков ---
    last_dir = load_last_dir()
    initial_outdir = last_dir or os.path.dirname(os.path.abspath(dat_path))
    dir_dlg = SaveDirDialog(initial_outdir)
    dir_dlg.mainloop()
    if dir_dlg.result is None:
        print("Выбор папки отменён. Выход.")
        return
    outdir = dir_dlg.result
    save_last_dir(outdir)
    print(f"Папка для сохранения: {outdir}")

    # --- t7 ---
    try:
        t7 = find_t7(data)
    except ValueError as ex:
        messagebox.showerror("Ошибка", str(ex))
        return
    print(f"t7 = {t7} с")

    # --- решение ---
    print("Решение изотермического процесса...")
    iso = solve_iso(p, data, t7)
    print("Решение адиабатического процесса...")
    adiab = solve_adiab(p, data, t7)

    # --- графики ---
    print(f"Сохранение графиков в: {outdir}")
    build_plots(outdir, data, t7, iso, adiab, hcs_flag=hcs_flag)

    exp_txt = "эксперимент с HCS" if hcs_flag else "эксперимент без HCS"
    messagebox.showinfo(
        "Готово",
        f"Графики ({exp_txt}) сохранены в:\n{outdir}")
    print(f"Готово. HCS (эксперимент) = {hcs_flag}")


if __name__ == "__main__":
    main()