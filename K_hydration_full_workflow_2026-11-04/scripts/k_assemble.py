#!/usr/bin/env python3
"""Build the K-(H2O)n summary from real run files. Missing numbers stay blank."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from k_pipeline import read_xyz, write_xyz
from validate_k_shell import analyze_one


def start_index(source):
    parts = source.split("_")
    return int(parts[1])


def mean_sd(values):
    arr = np.asarray(list(values), float)
    if len(arr) == 0:
        return "", ""
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def geometry_row(frame, n):
    res = analyze_one([s.upper() for s in frame[1]], np.asarray(frame[2], float))
    if res["n"] != n:
        raise RuntimeError("geometry stoichiometry changed")
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="full_100")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runs = root / "runs"
    folder = root / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    summary, catalog, verified_rows, saturation_rows = [], [], [], []
    lowest_rows, boltz_rows = [], []
    absent = []
    all_unique, all_cn = [], []

    for n in range(4, 15):
        base = runs / f"K_{n:02d}" / args.tag
        status_path = base / "final" / "RESULTS_STATUS.json"
        unique_csv = base / "unique_minima" / "unique_minima.csv"
        if not status_path.is_file() or not unique_csv.is_file():
            absent.append(n)
            summary.append({
                "n": n, "status": "NOT_CALCULATED", "n_starts": "", "n_verified_structures": "",
                "n_unresolved_starts": "", "n_cocos_groups": "", "n_unique_minima": "",
                "n_groups_cn_eq_n": "", "n_structures_cn_eq_n": "", "cn_mean_verified": "",
                "cn_histogram": "", "fraction_cn_eq_n": "", "mean_KO_unique_A": "",
                "sd_KO_unique_A": "", "mean_KO_cn_eq_n_A": "", "sd_KO_cn_eq_n_A": "",
                "mean_orientation_unique": "", "mean_orientation_cn_eq_n": "",
                "mean_surface_index_unique": "", "mean_n_KO_gt_3.4_unique": "",
                "mean_ww_hbonds_unique": "", "Emin_hartree_within_n": "",
                "deltaE_span_mEh_within_n": "", "new_groups_in_starts_81_100": "",
            })
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        incomplete = (base / "verified" / "INCOMPLETE.txt").read_text(encoding="utf-8")
        unresolved = int(incomplete.split("EXCLUDED_OR_UNRESOLVED_STARTS=")[1].splitlines()[0])
        with unique_csv.open(newline="", encoding="utf-8") as handle:
            unique = list(csv.DictReader(handle))
        unique_frames = read_xyz(base / "unique_minima" / "unique_minima.xyz")
        if len(unique) != len(unique_frames):
            raise RuntimeError(f"n={n}: unique catalog length mismatch")
        with (base / "final" / "selection_audit.csv").open(newline="", encoding="utf-8") as handle:
            audit = list(csv.DictReader(handle))
        with (base / "verified" / "free_sorted_metadata.csv").open(newline="", encoding="utf-8") as handle:
            meta = list(csv.DictReader(handle))
        with (base / "cocos" / "group_members.csv").open(newline="", encoding="utf-8") as handle:
            groups = list(csv.DictReader(handle))
        meta_by_frame = {int(row["sorted_frame"]): row for row in meta}
        group_by_frame = {int(row["sorted_frame"]): int(row["group_id"]) for row in groups}
        cn_counts = Counter(float(row["coordination_mdal"]) for row in audit)
        orient = {int(row["sorted_frame"]): float(row["orientation_cos_mdatom"]) for row in audit}
        passed = [row for row in audit if str(row["pass_full_first_shell"]) in {"True", "true", "1"}]
        geos = []
        for row, frame in zip(unique, unique_frames):
            geo = geometry_row(frame, n)
            geos.append(geo)
            catalog.append({
                "set": "unique_minimum", "n": n, "group_id": row["group_id"],
                "energy_hartree": row["energy_hartree"], "delta_energy_hartree": row["delta_energy_hartree"],
                "sorted_frame": row["sorted_frame"], **{k: geo[k] for k in (
                    "mean_KO_A", "min_KO_A", "max_KO_A", "mean_cos_unweighted",
                    "surface_index", "n_KO_gt_3.4", "n_ww_hbonds", "cn_ko_2.8",
                    "cn_ko_3.0", "cn_ko_3.2", "cn_kh_3.2_130")},
            })
            all_unique.append(frame)
        cn_geos = []
        models = base / "final" / "models.xyz"
        if models.is_file() and status["final_representatives"]:
            cn_frames = read_xyz(models)
            with (base / "final" / "models.csv").open(newline="", encoding="utf-8") as handle:
                model_rows = list(csv.DictReader(handle))
            if len(cn_frames) != len(model_rows):
                raise RuntimeError(f"n={n}: CN=n model count mismatch")
            if model_rows:
                sorted_models=sorted(model_rows,key=lambda a:float(a['energy_hartree']))
                best=sorted_models[0]
                lowest_rows.append({'n':n,'energy_hartree':best['energy_hartree'],
                    'radius_A':best['radius_mdatom_A'],'cos_x':best['orientation_cos_mdatom'],
                    'water_cn':best['coordination_mdal'],'group_id':best['group_id'],
                    'sorted_frame':best['sorted_frame']})
                # Exploratory E-only weights: NOT rigorous solution-state conformer populations.
                en=np.array([float(a['energy_hartree']) for a in sorted_models])
                weights=np.exp(-(en-en.min())/(3.166811563e-6*300.0))
                weights/=weights.sum()
                boltz_rows.append({'n':n,'T_K':300.0,'n_unique_shell_groups':len(sorted_models),
                    'radius_A_weighted_E':float(np.dot(weights,[float(a['radius_mdatom_A']) for a in sorted_models])),
                    'cos_x_weighted_E':float(np.dot(weights,[float(a['orientation_cos_mdatom']) for a in sorted_models])),
                    'water_cn_weighted_E':float(np.dot(weights,[float(a['coordination_mdal']) for a in sorted_models])),
                    'Emin_hartree':float(en.min()),'method':'E-only, one representative per CoCoS group; exploratory, not G(T)'})
            for row, frame in zip(model_rows, cn_frames):
                geo = geometry_row(frame, n)
                cn_geos.append(geo)
                all_cn.append(frame)
        verified_frames = read_xyz(base / "verified" / "free_sorted_for_cocos.xyz")
        for frame_i, frame in enumerate(verified_frames, 1):
            geo = geometry_row(frame, n)
            verified_rows.append({
                "n": n, "sorted_frame": frame_i, "source": meta_by_frame[frame_i]["source"],
                "start": start_index(meta_by_frame[frame_i]["source"]),
                "group_id": group_by_frame[frame_i],
                "energy_hartree": meta_by_frame[frame_i]["energy_hartree"],
                "coordination_mdal": next(float(r["coordination_mdal"]) for r in audit if int(r["sorted_frame"]) == frame_i),
                "orientation_cos_mdatom": orient[frame_i],
                "mean_KO_A": geo["mean_KO_A"], "min_KO_A": geo["min_KO_A"], "max_KO_A": geo["max_KO_A"],
                "surface_index": geo["surface_index"], "n_KO_gt_3.4": geo["n_KO_gt_3.4"],
                "n_ww_hbonds": geo["n_ww_hbonds"],
            })
        # Saturation: groups whose earliest parent start is in each block of 20.
        earliest = {}
        for row in verified_rows:
            if row["n"] != n:
                continue
            earliest[row["group_id"]] = min(earliest.get(row["group_id"], 999), row["start"])
        previous = 0
        for limit in (20, 40, 60, 80, 100):
            now = sum(v <= limit for v in earliest.values())
            saturation_rows.append({"n": n, "starts": limit, "groups_reached": now, "new_groups": now - previous})
            previous = now
        ko_u, _ = mean_sd(g["mean_KO_A"] for g in geos)
        sd_u = mean_sd(g["mean_KO_A"] for g in geos)[1]
        ori_u, _ = mean_sd(g["mean_cos_unweighted"] for g in geos)
        ko_c, sd_c = mean_sd(g["mean_KO_A"] for g in cn_geos)
        ori_c, _ = mean_sd(float(r["orientation_cos_mdatom"]) for r in (model_rows if cn_geos else []))
        energies = [float(r["energy_hartree"]) for r in unique]
        summary.append({
            "n": n,
            "status": "ANALYZED",
            "n_starts": 100,
            "n_verified_structures": status["total_verified"],
            "n_unresolved_starts": unresolved,
            "n_cocos_groups": status["cocos_groups_total"],
            "n_unique_minima": len(unique),
            "n_groups_cn_eq_n": status["final_representatives"],
            "n_structures_cn_eq_n": status["passed_full_shell"],
            "cn_mean_verified": float(np.mean([float(r["coordination_mdal"]) for r in audit])),
            "cn_histogram": json.dumps(status["cn_histogram"], ensure_ascii=False),
            "fraction_cn_eq_n": status["passed_full_shell"] / status["total_verified"],
            "mean_KO_unique_A": ko_u,
            "sd_KO_unique_A": sd_u,
            "mean_KO_cn_eq_n_A": ko_c,
            "sd_KO_cn_eq_n_A": sd_c,
            "mean_orientation_unique": ori_u,
            "mean_orientation_cn_eq_n": ori_c,
            "mean_surface_index_unique": mean_sd(g["surface_index"] for g in geos)[0],
            "mean_n_KO_gt_3.4_unique": mean_sd(g["n_KO_gt_3.4"] for g in geos)[0],
            "mean_ww_hbonds_unique": mean_sd(g["n_ww_hbonds"] for g in geos)[0],
            "Emin_hartree_within_n": min(energies),
            "deltaE_span_mEh_within_n": (max(energies) - min(energies)) * 1000.0,
            "new_groups_in_starts_81_100": sum(1 for v in earliest.values() if v > 80),
        })

    if absent and not args.allow_incomplete:
        raise SystemExit(f"Series is incomplete. No final/unique result for n={absent}. Completed n={[r['n'] for r in summary if r['status']=='ANALYZED']}.")

    write_csv(root / "K_SUMMARY.csv", summary)
    write_csv(folder / "lowest_energy_shell_model_per_n.csv", lowest_rows)
    write_csv(folder / "boltzmann_E_only_300K.csv", boltz_rows)
    write_csv(folder / "unique_minima_geometry.csv", catalog)
    write_csv(folder / "all_verified_structures.csv", verified_rows)
    write_csv(folder / "search_saturation.csv", saturation_rows)
    if all_unique:
        write_xyz(folder / "all_unique_minima.xyz", all_unique)
    if all_cn:
        write_xyz(folder / "all_cn_eq_n_representatives.xyz", all_cn)
    plot(summary, saturation_rows, folder)
    write_readme(root / "README_RESULTS.md", summary, absent)
    archive(root, folder, args.tag)
    print("Wrote K_SUMMARY.csv, README_RESULTS.md, analysis plots, and one archive")


def plot(summary, saturation, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    done = [r for r in summary if r["status"] == "ANALYZED"]
    ns = [r["n"] for r in done]

    def series(key):
        return [r[key] for r in done if r[key] != ""]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    if done:
        ax.errorbar(ns, [r["mean_KO_unique_A"] for r in done], yerr=[r["sd_KO_unique_A"] for r in done],
                    marker="o", label="All unique minima")
        cn = [(r["n"], r["mean_KO_cn_eq_n_A"], r["sd_KO_cn_eq_n_A"]) for r in done if r["mean_KO_cn_eq_n_A"] != ""]
        if cn:
            ax.errorbar([a for a, _b, _c in cn], [b for _a, b, _c in cn], yerr=[c for _a, _b, c in cn],
                        marker="s", label="CN = n representatives")
    ax.set_xlabel("Water molecules n")
    ax.set_ylabel("Mean K–O distance (Å)")
    ax.set_xticks(range(4, 15))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "radius_vs_n.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    if done:
        ax.plot(ns, [r["mean_orientation_unique"] for r in done], marker="o", label="Geometric bisector, unique minima")
        cn = [(r["n"], r["mean_orientation_cn_eq_n"]) for r in done if r["mean_orientation_cn_eq_n"] != ""]
        if cn:
            ax.plot([a for a, _b in cn], [b for _a, b in cn], marker="s", label="MDAtom cosine, CN = n")
    ax.set_xlabel("Water molecules n")
    ax.set_ylabel("Orientation parameter")
    ax.set_xticks(range(4, 15))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "orientation_vs_n.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    if done:
        ax.plot(ns, [r["cn_mean_verified"] for r in done], marker="o", label="Mean MDAtom coordination")
        ax.plot(ns, ns, linestyle="--", alpha=0.55, label="CN = n reference")
    ax.set_xlabel("Water molecules n")
    ax.set_ylabel("Weighted coordination (#water)")
    ax.set_xticks(range(4, 15))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "coordination_vs_n.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    if done:
        ax.plot(ns, [r["fraction_cn_eq_n"] for r in done], marker="s", label="Fraction with CN=n")
    ax.set_xlabel("Water molecules n")
    ax.set_ylabel("Fraction of verified structures")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(range(4, 15))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "first_shell_fraction_vs_n.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    by_n = defaultdict(list)
    for row in saturation:
        by_n[row["n"]].append(row)
    for n, rows in sorted(by_n.items()):
        ax.plot([r["starts"] for r in rows], [r["groups_reached"] for r in rows], marker="o", label=f"n={n}")
    ax.set_xlabel("First k random starts")
    ax.set_ylabel("CoCoS groups reached")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(folder / "search_saturation.png", dpi=180)
    plt.close(fig)


def write_readme(path, summary, absent):
    done = [r for r in summary if r["status"] == "ANALYZED"]
    lines = [
        "# K⁺(H₂O)ₙ, n = 4–14: результаты расчёта",
        "",
        "Метод: GFN2-xTB/ALPB(water), заряд +1, мультиплетность 1.",
        "Программа: xTB 6.7.1; фактическая сборка и SHA256 GWH фиксируются в `gwh_provenance.json`.",
        "Стартовое расстояние K–O = 3,33 Å задаёт только опорные точки GWH. Это не равновесный радиус.",
        "Опорная точка — центр зарядов SPC/E молекулы воды, а не ядро кислорода.",
        "Вращения GWH: `Rotation.from_euler('zyz')`. Центрирование: WaterCharges, без `--DoNotCenter`.",
        "Свободная оптимизация: `--opt normal`, затем `--opt tight`; структуры с мнимыми внутренними частотами исключаются.",
        "Критерий минимума: внутренние ν > 0 см⁻¹, шесть внешних мод исключены.",
        "CoCoS сравнивает структуры после свободной оптимизации, с зеркальным совмещением по умолчанию автора.",
        "КЧ — дробный результат MDAtomLocalEnvironment (--scale 1.0, без округления). В отборе используется заранее заданное условие 0 ≤ n−КЧ ≤ 0,15 на точности 0,1 в файле .res; чувствительность к порогу приводится отдельно.",
        "Абсолютные электронные энергии разных n не являются энергиями гидратации и между собой не сравниваются. ΔE считается только внутри одного n.",
        "Основной отбор Дениса — минимум энергии СРЕДИ РУЧНО ПРОВЕРЕННЫХ геометрически координированных структур; `review/output/K_coordination_minima_review.md`. `analysis/lowest_energy_shell_model_per_n.csv` — только диагностический отбор по MDAtom и порогу 0.15.",
        "Вторичная проверка — условные больцмановские веса exp[-(E_i-Emin)/(kBT)] при T=300 K для одного представителя каждой группы CoCoS; `analysis/boltzmann_E_only_300K.csv`. Это не распределение по свободным энергиям; ZPVE, энтальпии и G(T) без подтверждённых термохимических данных не вычислялись.",
        "Минимумы с водой во внешней области сохранены в наборе уникальных минимумов.",
        "",
        f"Незавершённые составы: {absent if absent else 'нет'}.",
        "",
        "## Сводка",
        "",
    ]
    for row in done:
        lines.append(
            f"- n={row['n']}: стартов {row['n_starts']}, подтверждённых структур {row['n_verified_structures']}, "
            f"неразрешённых стартов {row['n_unresolved_starts']}, групп CoCoS {row['n_cocos_groups']}, "
            f"уникальных минимумов {row['n_unique_minima']}, представителей с КЧ=n {row['n_groups_cn_eq_n']} "
            f"(структур с КЧ=n: {row['n_structures_cn_eq_n']}). "
            f"Средний K–O уникальных минимумов {row['mean_KO_unique_A']} Å. "
            f"Гистограмма КЧ {row['cn_histogram']}. "
            f"Новых групп среди стартов 81–100: {row['new_groups_in_starts_81_100']}."
        )
    lines += [
        "",
        "Поверхностное положение K⁺ оценивается индексом `surface_index`: норма среднего единичного вектора K→O. Значение около 0 отвечает оболочке вокруг иона, значение около 1 — скоплению воды по одну сторону от K⁺.",
        "Число контактов K–O > 4,0 Å и число водородных связей вода–вода приведены как геометрическая диагностика, а не как определение первой оболочки.",
        "Весовое координационное число вычислено MDAtom с --DoNotUseRounding --scale 1.0; критерий n−КЧ ≤0,15 отдельно проверяется по чувствительности к порогу.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive(root, folder, tag):
    dest = root / "K_hydration_results.zip"
    if dest.exists():
        dest.unlink()
    runs = root / "runs"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in [root / "K_SUMMARY.csv", root / "README_RESULTS.md", root / "EXPERIMENT_LOG.md", root / "runs/MDAtom_water.xlsx", root / "README_НАЧАТЬ_ЗДЕСЬ.md", root / "inputs_manifest.csv"]:
            if path.is_file():
                zipped.write(path, arcname=path.name)
        for path in sorted((root / "scripts").glob("*.py")):
            zipped.write(path, arcname=str(path.relative_to(root)))
        for path in sorted((root / "inputs").glob("K_*/packing.vecfle")):
            zipped.write(path, arcname=str(path.relative_to(root)))
        for path in folder.rglob("*"):
            if path.is_file():
                zipped.write(path, arcname=str(path.relative_to(root)))
        for n in range(4, 15):
            base = runs / f"K_{n:02d}" / tag
            for name in (f"constrained_100_multiframe.xyz", "constrained_100.zip", "energies.csv"):
                path = base / name
                if path.is_file():
                    zipped.write(path, arcname=f"gwh/K_{n:02d}/{name}")
            for extra in (
                "unique_minima/unique_minima.xyz",
                "unique_minima/unique_minima.csv",
                "final/models.xyz",
                "final/models.csv",
                "final/selection_audit.csv",
                "final/RESULTS_STATUS.json",
                "gwh_provenance.json",
                "collect_audit.json",
            ):
                path = base / extra
                if path.is_file():
                    zipped.write(path, arcname=f"catalog/K_{n:02d}/{Path(extra).name}")


if __name__ == "__main__":
    main()
