#!/usr/bin/env python3
"""Reproducible K-(H2O)n driver. Quantum-chemical results are never invented.

Stages: check, gwh, collect, relax, export, cocos, mdatom, select, status. No mode-follow.
Successful jobs are skipped on resume. Constrained GWH files are not replaced
by freely optimized geometries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT_RESOLVED = Path(__file__).resolve().parents[1]


def prefer_junction(path: Path) -> Path:
    """Stay on the ASCII junction so xTB/GWH do not see a non-ASCII cwd."""
    junction = Path(r"C:\khyd")
    try:
        target = junction.resolve()
        resolved = path.resolve()
        if junction.exists() and resolved.is_relative_to(target):
            return junction / resolved.relative_to(target)
    except OSError:
        pass
    return path


ROOT = prefer_junction(ROOT_RESOLVED)
RUN = ROOT / "runs"
EH_TO_KCAL = 627.509474
WATER_Q_ABS = np.array([0.8476, 0.4238, 0.4238], dtype=float)  # abs(SPC/E), as GWH ReCenterMolecule
EMATCH = re.compile(r"\bE\s*=\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s*Eh", re.I)
XEN = re.compile(r"\benergy\s*:\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)", re.I)
LOG_EN = re.compile(r"TOTAL ENERGY\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)\s+Eh", re.I)
FREQ_IMAG_LIMIT = 0.0


def parse_xyz(text, label="xyz"):
    ls = text.splitlines()
    i = 0
    out = []
    while i < len(ls):
        if not ls[i].strip():
            i += 1
            continue
        try:
            na = int(ls[i].strip())
        except Exception as exc:
            raise ValueError(f"{label}:{i+1}: invalid XYZ atom count") from exc
        if na < 1 or i + na + 2 > len(ls):
            raise ValueError(f"{label}: incomplete XYZ")
        sy, xyz = [], []
        for s in ls[i + 2 : i + 2 + na]:
            v = s.split()
            if len(v) < 4:
                raise ValueError(f"{label}: invalid coordinate line {s}")
            sy.append(v[0])
            xyz.append(tuple(float(a.replace("D", "E")) for a in v[1:4]))
        out.append((ls[i + 1], sy, xyz))
        i += na + 2
    return out


def read_xyz(path):
    p = Path(path)
    return parse_xyz(p.read_text(encoding="utf-8", errors="replace"), str(p))


def write_xyz(path, frames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for co, sy, xyz in frames:
            fh.write(f"{len(sy)}\n{co}\n")
            for el, (x, y, z) in zip(sy, xyz):
                fh.write(f"{el:2s} {x:.12f} {y:.12f} {z:.12f}\n")


def check_formula(frame, n):
    _co, sy, xyz = frame
    c = Counter(s.title() for s in sy)
    if c != Counter({"K": 1, "O": n, "H": 2 * n}):
        raise ValueError(f"wrong composition {dict(c)} for K(H2O){n}")
    if len(xyz) != 3 * n + 1:
        raise ValueError("wrong number of atoms")
    return True


def xyz_energy(comment):
    m = EMATCH.search(comment) or XEN.search(comment)
    return float(m.group(1)) if m else None


def work(n, tag):
    return RUN / f"K_{n:02d}" / tag


def run(cmd, cwd, logfile, env=None):
    cwd = Path(cwd)
    cwd.mkdir(parents=True, exist_ok=True)
    with (cwd / logfile).open("wb") as handle:
        proc = subprocess.run(
            [str(t) for t in cmd],
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    return proc.returncode


def xtb_home():
    matches = sorted((ROOT / "vendors").glob("xtb-*/xtb-*/bin/xtb.exe"))
    if not matches:
        return None
    return matches[-1].parents[1]


def resolve_xtb(name):
    home = xtb_home()
    if home and (name in ("xtb", "xtb.exe")):
        return home / "bin" / "xtb.exe"
    p = Path(name)
    if p.is_file():
        return p
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f"Real xTB executable missing: {name}")
    return Path(found)


def xtb_env(threads):
    env = os.environ.copy()
    home = xtb_home()
    if home:
        env["PATH"] = str(home / "bin") + os.pathsep + env.get("PATH", "")
        env["XTBHOME"] = str(home)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    return env


def xtb_version_text(name="xtb"):
    try:
        proc = subprocess.run(
            [str(resolve_xtb(name)), "--version"],
            capture_output=True,
            env=xtb_env(1),
            timeout=30,
            check=False,
        )
        return (proc.stdout + proc.stderr).decode("utf-8", "replace").strip()
    except Exception as exc:
        return f"NOT VERIFIED: {exc}"


def read_vecfle(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        a = line.split()
        vec = np.array([float(a[1]), float(a[2]), float(a[3])], float)
        rows.append((a[0], vec, float(a[4])))
    return rows


def load_pack_vectors(path):
    nums = [float(x) for x in Path(path).read_text(encoding="utf-8").split()]
    if len(nums) % 3:
        raise ValueError(f"{path}: pack component count is not a multiple of 3")
    out = []
    for i in range(0, len(nums), 3):
        v = np.array(nums[i : i + 3], float)
        nrm = float(np.linalg.norm(v))
        if nrm < 1e-8:
            raise ValueError(f"{path}: zero packing vector")
        out.append(v / nrm)
    return out


def assert_gwh_source(path):
    text = Path(path).read_text(encoding="utf-8")
    required = ["zyz", "WaterCharges", "-0.8476", "Powell", "--charge"]
    missing = [k for k in required if k not in text]
    if missing:
        raise RuntimeError(f"GWH source {path} is missing original markers {missing}")
    if "from_euler('zyz'" not in text and 'from_euler("zyz"' not in text:
        raise RuntimeError("GWH rotation convention is not lowercase zyz")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(args):
    with (ROOT / "inputs_manifest.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 11:
        raise RuntimeError(f"expected 11 input rows, found {len(rows)}")
    pack_dir = Path(args.pack_dir) if args.pack_dir else (ROOT / "inputs" / "packings")
    for row in rows:
        n = int(row["n_water"])
        if row.get("ion") != "K" or int(row["charge"]) != 1 or int(row["multiplicity"]) != 1 or int(row["n_start_target"]) != 100:
            raise RuntimeError(f"K manifest mismatch: {row}")
        p = ROOT / "inputs" / row["case"] / "packing.vecfle"
        raw = p.read_text(encoding="utf-8")
        digest = hashlib.sha256(raw.encode()).hexdigest()
        if digest != row["vecfle_sha256"]:
            raise RuntimeError(f"{row['case']}: SHA256 mismatch")
        vecs = read_vecfle(p)
        if len(vecs) != n:
            raise RuntimeError(f"{row['case']}: expected {n} directions")
        dots = []
        for i, (name, vec, radius) in enumerate(vecs):
            if name != "K" or abs(radius - 3.04) > 1e-9:
                raise RuntimeError(f"{row['case']}: bad label/radius")
            if abs(float(np.linalg.norm(vec)) - 1) > 1e-7:
                raise RuntimeError(f"{row['case']}: direction {i} is not a unit vector")
            for j in range(i):
                dots.append(float(np.dot(vec, vecs[j][1])))
        msg = f"K_{n:02d}: {n} unit directions, r=3.04 A SPC/E reference, SHA256 OK"
        if dots:
            ang = [math.degrees(math.acos(max(-1.0, min(1.0, d)))) for d in dots]
            msg += f", neighbor angles {min(ang):.2f}..{max(ang):.2f} deg"
        if pack_dir and pack_dir.is_dir():
            raw_pack = pack_dir / f"pack.3.{n}.txt"
            if hashlib.sha256(raw_pack.read_bytes()).hexdigest() != row["packing_sha256"]:
                raise RuntimeError(f"pack.3.{n}.txt: SHA256 mismatch")
            pack = load_pack_vectors(raw_pack)
            if len(pack) != n:
                raise RuntimeError(f"pack.3.{n}.txt has {len(pack)} vectors")
            worst = max(float(np.linalg.norm(pack[i] - vecs[i][1])) for i in range(n))
            if worst > 2e-9:
                raise RuntimeError(f"{row['case']}: disagrees with pack.3.{n}.txt by {worst}")
            msg += f", pack.3.{n}.txt match {worst:.2e}"
        print(msg)
    v4 = [v for _name, v, _r in read_vecfle(ROOT / "inputs" / "K_04" / "packing.vecfle")]
    for i in range(4):
        for j in range(i):
            if abs(float(np.dot(v4[i], v4[j])) + 1.0 / 3.0) > 2e-6:
                raise RuntimeError("K_04 packing is not a regular tetrahedron")
    print("K_04 reference points are a regular tetrahedron (dot = -1/3).")
    if not getattr(args,'skip_xtb',False):
        version = xtb_version_text(args.xtb)
        if "6.7.1" not in version or version.startswith("NOT VERIFIED"):
            raise RuntimeError(f"xTB 6.7.1 executable required for actual runs, got: {version[:300]}")
        print("External xTB:", version.splitlines()[0][:300])
    else:
        print("xTB executable check skipped (inputs-only mode); actual runs require real xTB.")
    print("Inputs checked. No xTB energies were generated by this command.")


def write_radius_vecfle(n, radius, dest):
    src = read_vecfle(ROOT / "inputs" / f"K_{n:02d}" / "packing.vecfle")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, vec, _r in src:
        u = vec / np.linalg.norm(vec)
        lines.append(f"{name} {u[0]:.10f} {u[1]:.10f} {u[2]:.10f} {radius:.2f}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def gwh_output_ok(folder, index, n):
    p = folder / f"mol_str{index:04d}_fin.xyz"
    err = folder / "qc_tmp.err"
    if not p.is_file() or not err.is_file():
        return False
    if "normal termination of xtb" not in err.read_text(encoding="utf-8", errors="replace").lower():
        return False
    try:
        frames = read_xyz(p)
        if len(frames) != 1:
            return False
        check_formula(frames[0], n)
        energy = xyz_energy(frames[0][0])
    except Exception:
        return False
    return energy is not None and math.isfinite(energy) and energy < -1.0 and energy > -1.0e6


def retry_io(action, attempts=40, delay=0.5):
    """Retry a filesystem action when Windows or OneDrive still holds a file."""
    last = None
    for attempt in range(attempts):
        try:
            return action()
        except PermissionError as exc:
            last = exc
            time.sleep(delay)
    raise last


def publish_gwh_folder(job, src, folder, index, n, rc):
    if folder.exists():
        retry_io(lambda: shutil.rmtree(folder))
    retry_io(lambda: src.rename(folder))
    log = job / "gwh_one.log"
    if log.is_file():
        retry_io(lambda: log.replace(folder / "gwh_one.log"))
    retry_io(lambda: shutil.rmtree(job))
    if not gwh_output_ok(folder, index, n):
        return {"index": index, "status": "fail", "exit": rc, "reason": "output failed validation"}
    return {"index": index, "status": "ok"}


def gwh_one(index, gwh, vecfle, dest_root, n, threads):
    folder = dest_root / f"gwh_opt_str{index:04d}"
    if gwh_output_ok(folder, index, n):
        return {"index": index, "status": "skip"}
    job = dest_root / f"_tmp_{index:04d}"
    src = job / "gwh_opt_str0000"
    renamed = src / f"mol_str{index:04d}_fin.xyz"
    original = src / "mol_str0000_fin.xyz"
    if renamed.is_file() or original.is_file():
        if original.is_file() and not renamed.is_file():
            for path in list(src.iterdir()):
                if "mol_str0000" in path.name:
                    target = src / path.name.replace("mol_str0000", f"mol_str{index:04d}")
                    retry_io(lambda path=path, target=target: path.rename(target))
        return publish_gwh_folder(job, src, folder, index, n, 0)
    if job.exists():
        retry_io(lambda: shutil.rmtree(job))
    job.mkdir(parents=True)
    seed = 20261104 + n * 1000 + index
    # GWH uses Python's random.uniform. Seed it in the same interpreter before
    # running the original, unmodified GWH program.  Each start has its own seed.
    launcher = (
        "import random,runpy,sys\n"
        "random.seed(int(sys.argv[1]))\n"
        "path=sys.argv[2]\n"
        "sys.argv=[path]+sys.argv[3:]\n"
        "runpy.run_path(path,run_name='__main__')\n"
    )
    cmd = [
        sys.executable,
        "-c", launcher, str(seed), str(gwh),
        "-f",
        str(vecfle),
        "--Charge",
        "+1",
        "--Mult",
        "1",
        "--NumStruct",
        "1",
        "--Random",
        "--Opt",
        "--AddMethod",
        " --alpb water --gfn 2 ",
    ]
    rc = run(cmd, job, "gwh_one.log", xtb_env(threads))
    src = job / "gwh_opt_str0000"
    fin = src / "mol_str0000_fin.xyz"
    if rc != 0 or not fin.is_file():
        failed = dest_root / f"_failed_{index:04d}"
        if failed.exists():
            retry_io(lambda: shutil.rmtree(failed))
        retry_io(lambda: job.rename(failed))
        return {"index": index, "status": "fail", "exit": rc, "kept": str(failed)}
    for path in list(src.iterdir()):
        if "mol_str0000" in path.name:
            target = src / path.name.replace("mol_str0000", f"mol_str{index:04d}")
            retry_io(lambda path=path, target=target: path.rename(target))
    return publish_gwh_folder(job, src, folder, index, n, rc)


def gwh(args):
    if args.numstruct < 1 or args.numstruct > 100:
        raise ValueError("numstruct must be 1..100")
    gwh_path = Path(args.gwh).expanduser()
    gwh_path = gwh_path if gwh_path.is_absolute() else (Path.cwd() / gwh_path)
    if not gwh_path.is_file():
        raise RuntimeError(f"GWH source missing: {gwh_path}")
    digest = assert_gwh_source(gwh_path)
    resolve_xtb(args.xtb)
    version = xtb_version_text(args.xtb)
    if "6.7.1" not in version:
        raise RuntimeError(f"xTB 6.7.1 is required, got: {version[:300]}")
    base = work(args.n, args.tag)
    out = base / "gwh"
    out.mkdir(parents=True, exist_ok=True)
    radius = float(args.radius)
    if abs(radius - 3.04) < 1e-12:
        vecfle = ROOT / "inputs" / f"K_{args.n:02d}" / "packing.vecfle"
    else:
        vecfle = write_radius_vecfle(args.n, radius, base / f"packing_r{radius:.2f}.vecfle")
    provenance = {
        "command_template": "gloriouswaterhole.py -f packing.vecfle --Charge +1 --Mult 1 --NumStruct 1 --Random --Opt --AddMethod ' --alpb water --gfn 2 '",
        "parallel_processes": args.workers,
        "random_seed_rule": "python.random.seed(20261104 + n*1000 + start_index)",
        "random_seed_base": 20261104 + args.n*1000,
        "n": args.n,
        "charge": +1,
        "multiplicity": 1,
        "numstruct": args.numstruct,
        "xtb_version": version,
        "gwh_source": str(gwh_path),
        "gwh_sha256": digest,
        "water_centering": "WaterCharges SPC/E abs-weights, GWH default; --DoNotCenter not used",
        "rotation_convention": "scipy Rotation.from_euler('zyz', ..., degrees=True)",
        "optimizer": "scipy Powell on Euler angles; oxygen reference points fixed",
        "initial_K_to_water_reference_A": radius,
        "vecfle": str(vecfle),
        "method": "GFN2-xTB/ALPB(water)",
        "stage": "CONSTRAINED ONLY",
    }
    prov_file=base / "gwh_provenance.json"
    if prov_file.is_file():
        previous=json.loads(prov_file.read_text(encoding="utf-8"))
        for key in ("n", "charge", "multiplicity", "numstruct", "initial_K_to_water_reference_A",
                    "gwh_sha256", "rotation_convention", "method", "random_seed_rule"):
            if previous.get(key)!=provenance.get(key):
                raise RuntimeError(f"GWH provenance changed for {key}: {previous.get(key)!r} vs "
                                   f"{provenance.get(key)!r}. Use a NEW --tag, never mix starts.")
        if previous.get('xtb_version')!=provenance['xtb_version']:
            raise RuntimeError('xTB version changed within this tag. Use new --tag or explicitly validate compatibility.')
    prov_file.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    pending = [i for i in range(args.numstruct) if not gwh_output_ok(out / f"gwh_opt_str{i:04d}", i, args.n)]
    print(f"GWH n={args.n} tag={args.tag}: {args.numstruct - len(pending)} done, {len(pending)} to run", flush=True)
    failures = []
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(gwh_one, i, gwh_path, vecfle, out, args.n, args.threads): i for i in pending
            }
            for fut in as_completed(futures):
                result = fut.result()
                idx = result["index"]
                geo = out / f"gwh_opt_str{idx:04d}" / f"mol_str{idx:04d}_fin.xyz"
                energy = xyz_energy(read_xyz(geo)[0][0]) if result["status"] != "fail" and geo.is_file() else None
                val = f"E={energy:.7f} Ha" if energy is not None else result["status"]
                print(f"  start {idx:03d} {val}", flush=True)
                if result["status"] == "fail":
                    failures.append(result)
    if failures:
        raise RuntimeError(f"GWH failed for {len(failures)} starts; inspect runs/.../gwh/_failed_*")
    done = sum(1 for i in range(args.numstruct) if gwh_output_ok(out / f"gwh_opt_str{i:04d}", i, args.n))
    print(f"GWH complete {done}/{args.numstruct}", flush=True)


def charge_centers(symbols, xyz):
    if symbols[0].title() != "K":
        raise ValueError(f"atom 0 is {symbols[0]}, expected K")
    expected = ["O", "H", "H"] * ((len(symbols) - 1) // 3)
    if [s.title() for s in symbols[1:]] != expected:
        raise ValueError("atom order is not K,(O,H,H)*n")
    coords = np.asarray(xyz, float)
    centers = []
    for k in range((len(symbols) - 1) // 3):
        block = coords[1 + 3 * k : 4 + 3 * k]
        centers.append((WATER_Q_ABS[:, None] * block).sum(axis=0) / WATER_Q_ABS.sum())
    return coords[0], np.asarray(centers, float)


def constrained_paths(base, count):
    return {
        "xyz": base / f"constrained_{count}_multiframe.xyz",
        "zip": base / f"constrained_{count}.zip",
        "csv": base / "energies.csv",
    }


def collect(args):
    base = work(args.n, args.tag)
    prov_path = base / "gwh_provenance.json"
    if not prov_path.is_file():
        raise RuntimeError("No GWH provenance. Run gwh first.")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    count = int(prov["numstruct"])
    gwh_dir = base / "gwh"
    frames, energies, errors = [], [], []
    for i in range(count):
        folder = gwh_dir / f"gwh_opt_str{i:04d}"
        path = folder / f"mol_str{i:04d}_fin.xyz"
        try:
            if not gwh_output_ok(folder, i, args.n):
                raise RuntimeError("missing file, xtb termination, formula, or real energy")
            frame = read_xyz(path)[0]
            energy = xyz_energy(frame[0])
            log = (folder / "qc_tmp.log").read_text(encoding="utf-8", errors="replace") if (folder / "qc_tmp.log").is_file() else ""
            logged = [float(x) for x in LOG_EN.findall(log)]
            if logged and abs(logged[-1] - energy) > 1e-6:
                raise RuntimeError(f"comment energy {energy} disagrees with qc_tmp.log {logged[-1]}")
            k_center, centers = charge_centers(frame[1], frame[2])
            refs = read_vecfle(prov["vecfle"])
            if len(refs) != args.n:
                raise RuntimeError("vecfle length mismatch")
            if float(np.linalg.norm(k_center)) > 5e-3:
                raise RuntimeError(f"K is not at the origin ({k_center})")
            target = np.vstack([v / np.linalg.norm(v) * r for _a, v, r in refs])
            shift = np.linalg.norm(centers - target, axis=1)
            if float(shift.max()) > 5e-3:
                raise RuntimeError(f"water charge centers moved, max {float(shift.max()):.4f} A")
            frames.append(frame)
            energies.append(energy)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    coords = [np.asarray(fr[2], float) for fr in frames]
    duplicates = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if float(np.max(np.abs(coords[i] - coords[j]))) < 1e-4:
                duplicates.append([i + 1, j + 1])
    audit = {
        "expected": count,
        "valid": len(frames),
        "errors": errors,
        "duplicate_frames_1based": duplicates,
        "energy_min_hartree": min(energies) if energies else None,
        "energy_max_hartree": max(energies) if energies else None,
        "stage": "CONSTRAINED_ONLY",
        "radius_A": prov.get("initial_K_to_water_reference_A"),
    }
    (base / "collect_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if errors or duplicates or len(frames) != count:
        raise RuntimeError(f"Invalid constrained batch: errors={len(errors)} duplicates={duplicates}")
    paths = constrained_paths(base, count)
    write_xyz(paths["xyz"], frames)
    e0 = min(energies)
    with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["xyz_frame", "xyz_file", "energy_hartree", "delta_energy_hartree"])
        for i, energy in enumerate(energies, 1):
            writer.writerow([i, f"K_{args.n:02d}w_{i:03d}.xyz", f"{energy:.14f}", f"{(energy - e0):.14f}"])
    if paths["zip"].exists():
        paths["zip"].unlink()
    with zipfile.ZipFile(paths["zip"], "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for i, frame in enumerate(frames, 1):
            name = f"K_{args.n:02d}w_{i:03d}.xyz"
            temp = base / name
            write_xyz(temp, [frame])
            archive.write(temp, arcname=name)
            temp.unlink()
    verify_triplet(base, args.n, count)
    print(f"Collected {count} constrained GWH structures into the three stage files", flush=True)


def verify_triplet(base, n, count):
    paths = constrained_paths(base, count)
    frames = read_xyz(paths["xyz"])
    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with zipfile.ZipFile(paths["zip"]) as archive:
        names = archive.namelist()
        zipped = {name: parse_xyz(archive.read(name).decode("utf-8"), name)[0] for name in names}
    if len(frames) != count or len(rows) != count or len(names) != count:
        raise RuntimeError("triplet lengths differ")
    e0 = min(xyz_energy(fr[0]) for fr in frames)
    for i, (frame, row) in enumerate(zip(frames, rows), 1):
        check_formula(frame, n)
        name = f"K_{n:02d}w_{i:03d}.xyz"
        if int(row["xyz_frame"]) != i or row["xyz_file"] != name or name not in zipped:
            raise RuntimeError(f"frame {i} naming mismatch")
        energy = xyz_energy(frame[0])
        if abs(energy - float(row["energy_hartree"])) > 1e-12:
            raise RuntimeError(f"frame {i} KV energy mismatch")
        if abs((energy - e0) - float(row["delta_energy_hartree"])) > 1e-12:
            raise RuntimeError(f"frame {i} delta is not within this n")
        zf = zipped[name]
        if zf[1] != frame[1] or np.max(np.abs(np.asarray(zf[2]) - np.asarray(frame[2]))) > 1e-9:
            raise RuntimeError(f"ZIP frame {i} is not the multiframe geometry")
    print(f"Triplet OK: {count} frames, ZIP/XYZ/KV agree, delta within n={n}")


def read_frequencies(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.lstrip().startswith(("#", "$")):
            continue
        cols = line.split()
        if len(cols) < 2 or not cols[0].isdigit():
            continue
        for tok in cols[1:]:
            try:
                rows.append(float(tok))
                break
            except ValueError:
                continue
    return rows


def read_internal_frequencies(path, natoms):
    """Read exactly 3N-6 vibrational modes from the Gaussian-like xTB Hessian output."""
    path = Path(path)
    if not path.is_file():
        return []
    freqs = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^\s*Frequencies\s*--\s*(.*)$", line)
        if match:
            freqs.extend(float(x.replace("D", "E")) for x in match.group(1).split())
    required = 3 * natoms - 6
    if len(freqs) != required:
        raise RuntimeError(f"{path}: {len(freqs)} internal frequencies; expected {required}")
    return freqs


def last_log_energy(path):
    found = LOG_EN.findall(Path(path).read_text(encoding="utf-8", errors="replace"))
    return float(found[-1]) if found else None


def opt_hess(start, dest, xtb, level, threads, compute_hessian=True):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    env = xtb_env(threads)
    binary = resolve_xtb(xtb)
    previous = Path(start)
    logs = []
    converged = None
    for attempt in range(4):
        stage = dest if attempt == 0 else dest / f"continuation_{attempt}"
        stage.mkdir(parents=True, exist_ok=True)
        shutil.copy2(previous, stage / "input.xyz")
        rc = run(
            [binary, "input.xyz", "--gfn", "2", "--chrg", "+1", "--uhf", "0", "--alpb", "water", "--opt", level],
            stage,
            "opt.log",
            env,
        )
        log = (stage / "opt.log").read_text(encoding="utf-8", errors="replace")
        phrase = "GEOMETRY OPTIMIZATION CONVERGED" in log
        has_xyz = (stage / "xtbopt.xyz").is_file()
        logs.append({"attempt": attempt, "exit": rc, "converged": phrase, "xtbopt": has_xyz, "directory": str(stage)})
        if phrase and has_xyz and rc == 0:
            converged = stage
            break
        if not has_xyz:
            break
        previous = stage / "xtbopt.xyz"
    if converged is None:
        return {"status": "OPT_FAILED", "attempts": logs, "directory": str(dest), "level": level}
    comment = read_xyz(converged / "xtbopt.xyz")[0][0]
    e_xyz = xyz_energy(comment)
    e_log = last_log_energy(converged / "opt.log")
    if e_xyz is None or e_log is None or abs(e_xyz - e_log) > 1e-6:
        return {
            "status": "ENERGY_MISMATCH",
            "energy_xyz": e_xyz,
            "energy_log": e_log,
            "directory": str(converged),
            "attempts": logs,
            "level": level,
        }
    if not compute_hessian:
        return {
            "status": "OPT_ONLY", "energy_hartree": e_xyz,
            "directory": str(converged), "xtbopt": str(converged / "xtbopt.xyz"),
            "level": level, "optimization_attempts": logs,
        }
    hess = converged / "hess"
    if hess.exists():
        shutil.rmtree(hess)
    hess.mkdir()
    shutil.copy2(converged / "xtbopt.xyz", hess / "input.xyz")
    rc = run([binary, "input.xyz", "--gfn", "2", "--chrg", "+1", "--uhf", "0", "--alpb", "water", "--hess"], hess, "hess.log", env)
    nat = len(read_xyz(converged / "xtbopt.xyz")[0][1])
    if rc or not (hess / "g98.out").is_file():
        return {"status":"HESS_FAILED","exit":rc,"energy_hartree":e_xyz,
                "reason":"nonzero exit or missing g98.out", "directory":str(converged),
                "attempts":logs,"level":level}
    try:
        freq = read_internal_frequencies(hess / "g98.out", nat)
    except (OSError, ValueError, RuntimeError) as exc:
        return {"status":"HESS_FAILED","exit":rc,"energy_hartree":e_xyz,
                "reason":f"could not parse Hessian: {exc}","directory":str(converged),
                "attempts":logs,"level":level}
    if rc or len(freq) != 3 * nat - 6:
        return {
            "status": "HESS_FAILED",
            "exit": rc,
            "energy_hartree": e_xyz,
            "n_frequencies": len(freq),
            "expected_frequencies": 3 * nat - 6,
            "directory": str(converged),
            "attempts": logs,
            "level": level,
        }
    for scratch in ("charges", "wbo", "xtbrestart", "xtbtopo.mol"):
        for folder in (converged, hess):
            f = folder / scratch
            if f.is_file():
                f.unlink()
    lowest = min(freq)
    return {
        "status": "MINIMUM" if lowest > FREQ_IMAG_LIMIT else "IMAGINARY",
        "energy_hartree": e_xyz,
        "lowest_freq_cm1": lowest,
        "n_nonpositive_internal": int(sum(v <= FREQ_IMAG_LIMIT for v in freq)),
        "level": level,
        "optimization_attempts": logs,
        "directory": str(converged),
        "xtbopt": str(converged / "xtbopt.xyz"),
        "vibspectrum": str(hess / "vibspectrum"),
        "g98": str(hess / "g98.out"),
    }


def publish_xyz(frame_dir, result):
    if result.get("status") not in ("MINIMUM", "IMAGINARY"):
        return
    src = Path(result["directory"]) / "xtbopt.xyz"
    if not src.is_file():
        raise RuntimeError(f"converged directory has no xtbopt.xyz: {src}")
    dest = Path(frame_dir) / "optimized.xyz"
    shutil.copy2(src, dest)
    a = np.asarray(read_xyz(src)[0][2], float)
    b = np.asarray(read_xyz(dest)[0][2], float)
    if a.shape != b.shape or float(np.max(np.abs(a - b))) != 0.0:
        raise RuntimeError("published XYZ does not match the converged xtbopt.xyz")
    # A continuation folder must be the one that was published.
    if "continuation_" in src.parent.name:
        parent_xyz = src.parents[1] / "xtbopt.xyz"
        if parent_xyz.is_file():
            stale = np.asarray(read_xyz(parent_xyz)[0][2], float)
            if stale.shape == a.shape and float(np.max(np.abs(stale - a))) < 1e-8:
                result["continuation_same_as_parent"] = True
            else:
                result["continuation_geometry_transferred"] = True


def relax_one(index, frame, out, xtb, threads, n):
    dest = out / f"frame_{index:04d}"
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / "status.json"
    if marker.is_file():
        old = json.loads(marker.read_text(encoding="utf-8"))
        result = old.get("final", {})
        if result.get("status") in ("MINIMUM", "IMAGINARY") and (dest / "optimized.xyz").is_file() and Path(result.get("directory", ""), "hess", "g98.out").is_file():
            return {"index": index, "status": "skip", "final": result["status"]}
    constrained = dest / "constrained.xyz"
    if not constrained.exists():
        write_xyz(constrained, [frame])
    check_formula(frame, n)
    first_path = dest / "free_result.json"
    if first_path.is_file():
        first = json.loads(first_path.read_text(encoding="utf-8"))
        if first.get("status") != "OPT_ONLY" or not Path(first.get("xtbopt", "")).is_file():
            first_path.unlink()
    if not first_path.is_file():
        first = opt_hess(constrained, dest / "free_opt", xtb, "normal", threads, compute_hessian=False)
        if first.get("status") == "OPT_ONLY":
            first_path.write_text(json.dumps(first, indent=2), encoding="utf-8")
    if first.get("status") != "OPT_ONLY" or not Path(first["xtbopt"]).is_file():
        return {"index": index, "status": "FREE_OPT_FAILED"}
    tight_path = dest / "tight_result.json"
    if tight_path.is_file():
        final = json.loads(tight_path.read_text(encoding="utf-8"))
        folder = Path(final.get("directory", ""))
        if final.get("status") not in ("MINIMUM", "IMAGINARY") or not (folder / "xtbopt.xyz").is_file() or not (folder / "hess" / "g98.out").is_file():
            tight_path.unlink()
    if not tight_path.is_file():
        final = opt_hess(Path(first["xtbopt"]), dest / "tight", xtb, "tight", threads)
        if final.get("status") in ("MINIMUM", "IMAGINARY"):
            tight_path.write_text(json.dumps(final, indent=2), encoding="utf-8")
    if final.get("status") in ("MINIMUM", "IMAGINARY"):
        publish_xyz(dest, final)
    marker.write_text(json.dumps({"origin": f"frame_{index:04d}", "free": first, "final": final,
        "protocol": "free normal -> tight -> Hessian -> exclude imaginary; no mode-follow"}, indent=2), encoding="utf-8")
    return {"index": index, "status": final.get("status", "FAILED"), "energy": final.get("energy_hartree"), "nu": final.get("lowest_freq_cm1")}


def find_constrained(base):
    found = sorted(base.glob("constrained_*_multiframe.xyz"))
    if not found:
        legacy = base / "source" / "constrained_multiframe.xyz"
        if legacy.is_file():
            return legacy
        raise RuntimeError("constrained multiframe XYZ is missing; run collect")
    return found[-1]


def relax(args):
    base = work(args.n, args.tag)
    src = find_constrained(base)
    frames = read_xyz(src)
    count = min(len(frames), args.limit or len(frames))
    out = base / "free"
    out.mkdir(parents=True, exist_ok=True)
    pending = []
    for i in range(count):
        marker = out / f"frame_{i+1:04d}" / "status.json"
        if marker.is_file():
            old = json.loads(marker.read_text(encoding="utf-8"))
            if old.get("final", {}).get("status") in ("MINIMUM", "IMAGINARY") and (marker.parent / "optimized.xyz").is_file() and (Path(old["final"].get("directory", "")) / "hess" / "g98.out").is_file():
                continue
        pending.append(i)
    print(f"relax n={args.n}: {count - len(pending)} done, {len(pending)} to run", flush=True)
    if not pending:
        return
    workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(relax_one, i + 1, frames[i], out, args.xtb, args.threads, args.n) for i in pending]
        for fut in as_completed(futures):
            result = fut.result()
            status = result['status']
            data = f"E={result['energy']:.7f} Ha νmin={result['nu']:.1f}" if result.get('energy') is not None and result.get('nu') is not None else status
            print(f"  frame {result['index']:04d} {data}", flush=True)


def candidate_rows(frame_dir, n):
    data = json.loads((frame_dir / "status.json").read_text(encoding="utf-8"))
    final = data["final"]
    rows = []
    if final.get("status") == "MINIMUM":
        rows.append((frame_dir.name, final))
    checked = []
    for label, result in rows:
        if result.get("status") != "MINIMUM":
            continue
        directory = Path(result["directory"])
        xyz = directory / "xtbopt.xyz"
        freq = read_internal_frequencies(directory / "hess" / "g98.out", len(read_xyz(xyz)[0][1]))
        if not freq or min(freq) <= 0.0:
            raise RuntimeError(f"{label}: confirmed minimum must have positive internal frequencies")
        if abs(min(freq) - float(result["lowest_freq_cm1"])) > 1e-6:
            raise RuntimeError(f"{label}: lowest frequency was not taken from its own Hessian")
        frame = read_xyz(xyz)[0]
        check_formula(frame, n)
        energy = xyz_energy(frame[0])
        if abs(energy - float(result["energy_hartree"])) > 1e-6:
            raise RuntimeError(f"{label}: XYZ energy does not match the Hessian geometry")
        checked.append({
            "source": label,
            "energy": energy,
            "frame": frame,
            "lowest_frequency": min(freq),
            "source_path": str(xyz),
        })
    return checked, final.get("status")


def export(args):
    base = work(args.n, args.tag)
    src = base / "free"
    rows, errors, total = [], [], 0
    for marker in sorted(src.glob("frame_*/status.json")):
        total += 1
        try:
            got, status = candidate_rows(marker.parent, args.n)
        except Exception as exc:
            errors.append([marker.parent.name, str(exc)])
            continue
        if not got:
            errors.append([marker.parent.name, status])
        rows.extend(got)
    if not rows:
        raise RuntimeError("No verified minima. Inspect free/tight/Hessian logs; do not invent structures.")
    rows.sort(key=lambda item: (item["energy"], item["source"]))
    out = base / "verified"
    out.mkdir(parents=True, exist_ok=True)
    write_xyz(
        out / "free_sorted_for_cocos.xyz",
        [(f"id={row['source']} E = {row['energy']:.14f} Eh", row["frame"][1], row["frame"][2]) for row in rows],
    )
    with (out / "free_sorted_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sorted_frame", "source", "energy_hartree", "min_freq_cm1", "path"])
        writer.writeheader()
        for j, row in enumerate(rows, 1):
            writer.writerow({
                "sorted_frame": j,
                "source": row["source"],
                "energy_hartree": f"{row['energy']:.14f}",
                "min_freq_cm1": f"{row['lowest_frequency']:.6f}",
                "path": row["source_path"],
            })
    text = (
        f"VERIFIED_FREE_MINIMA={len(rows)}\n"
        f"EXCLUDED_OR_UNRESOLVED_STARTS={len(errors)}\n"
        f"ORIGINAL_STARTS_PROCESSED={total}\n"
    )
    if errors:
        text += "UNRESOLVED\n" + "\n".join(f"{a}\t{b}" for a, b in errors) + "\n"
    (out / "INCOMPLETE.txt").write_text(text, encoding="utf-8")
    print(f"Exported {len(rows)} verified minima from {total} starts; unresolved {len(errors)}", flush=True)


def parse_cocos_groups(log_text, nframes):
    member = {}
    rep_to_group = {}
    group_ctr = 0
    for line in log_text.splitlines():
        added = re.search(r"added structure (\d+)\b", line)
        removed = re.search(r"Removed structure (\d+) as a copy of (\d+)", line)
        if added:
            index = int(added.group(1))
            if index in member:
                raise RuntimeError(f"CoCoS added structure {index} twice")
            member[index] = group_ctr
            rep_to_group[index] = group_ctr
            group_ctr += 1
        elif removed:
            index, group = int(removed.group(1)), int(removed.group(2))
            if group >= group_ctr:
                raise RuntimeError(
                    f"CoCoS copy index {group} is not an existing representative group; "
                    "refusing to guess the partition"
                )
            member[index] = group
    if len(member) != nframes:
        raise RuntimeError(f"CoCoS mapped {len(member)} of {nframes} structures")
    return member, group_ctr


def write_unique(base, frames, member, energies):
    best = {}
    for index, group in member.items():
        energy = energies[index]
        if group not in best or energy < best[group][0]:
            best[group] = (energy, index)
    chosen = sorted(((group, energy, index) for group, (energy, index) in best.items()), key=lambda x: x[1])
    out = base / "unique_minima"
    out.mkdir(parents=True, exist_ok=True)
    write_xyz(out / "unique_minima.xyz", [frames[index] for _g, _e, index in chosen])
    with (out / "unique_minima.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["xyz_frame", "group_id", "sorted_frame", "energy_hartree", "delta_energy_hartree"])
        e0 = chosen[0][1]
        for j, (group, energy, index) in enumerate(chosen, 1):
            writer.writerow([j, group + 1, index + 1, f"{energy:.14f}", f"{(energy - e0):.14f}"])
    print(f"Unique minima (lowest energy in every CoCoS group, second shell kept): {len(chosen)}", flush=True)


def cocos(args):
    base = work(args.n, args.tag)
    inp = base / "verified" / "free_sorted_for_cocos.xyz"
    if not inp.is_file():
        raise RuntimeError("Run export after the frequency check")
    status = (base / "verified" / "INCOMPLETE.txt").read_text(encoding="utf-8")
    if args.tag == "full_100":
        if "ORIGINAL_STARTS_PROCESSED=100\n" not in status:
            raise RuntimeError("full_100 has not processed 100 starts. CoCoS was not run.")
        if "EXCLUDED_OR_UNRESOLVED_STARTS=0\n" not in status:
            print("CoCoS uses only frequency-confirmed minima. Unresolved starts stay listed in INCOMPLETE.txt.", flush=True)
    cocos_path = Path(args.cocos).expanduser()
    if not cocos_path.is_file():
        raise RuntimeError("CoCoS script missing")
    source = cocos_path.read_text(encoding="utf-8")
    if "AccountForMirrorImages = True" not in source:
        raise RuntimeError("CoCoS mirror default was not found; refusing a different comparison program")
    out = base / "cocos"
    critfreq = float(getattr(args, "critfreq", 125.0))
    if not math.isfinite(critfreq) or critfreq <= 0.0:
        raise ValueError("CoCoS --CritFreq must be finite and positive")
    provenance = {"critfreq_cm1": critfreq, "num_samples": int(args.samples),
                  "source_sha256": hashlib.sha256(inp.read_bytes()).hexdigest(),
                  "cocos_sha256": hashlib.sha256(cocos_path.read_bytes()).hexdigest(),
                  "mirror_equivalent": True}
    meta = out / "command.json"
    if (out / "group_members.csv").is_file() and not args.rerun and meta.is_file():
        if json.loads(meta.read_text(encoding="utf-8")) == provenance:
            print(f"CoCoS already present: CritFreq={critfreq:g} cm^-1", flush=True)
            return
    # Different criterion or source invalidates the existing group assignment,
    # but NOT any successful GWH, xTB, Hessian or MDAtom calculations.
    if out.exists():
        shutil.rmtree(out)
    stab = base / "cocos_stability"
    if stab.exists():
        shutil.rmtree(stab)
    out.mkdir(parents=True)
    argv = [
        str(cocos_path),
        "-f",
        str(inp.resolve()),
        "--DoNotUseGlobalOpt",
        "--NumSamples",
        str(args.samples),
        "--DoNotSaveDoubles",
        "--CritFreq", f"{critfreq:.8g}",
    ]
    # Do not pass --DoAccountForMirrorImages: the author default treats mirrors as the same species.
    launcher = "import runpy, sys\nsys.argv = %r\nrunpy.run_path(%r, run_name='__main__')\n" % (argv, str(cocos_path))
    rc = run([sys.executable, "-c", launcher], out, "cocos.log")
    if rc:
        raise RuntimeError("CoCoS failed; see cocos.log")
    log = (out / "cocos.log").read_text(encoding="utf-8", errors="replace")
    frames = read_xyz(inp)
    member, ngroups = parse_cocos_groups(log, len(frames))
    trjs = list(out.glob("*_quick_cleanup.trj"))
    if len(trjs) != 1 or len(read_xyz(trjs[0])) != ngroups:
        raise RuntimeError("CoCoS representative trajectory does not match the parsed groups")
    energies = []
    for frame in frames:
        energy = xyz_energy(frame[0])
        if energy is None:
            raise RuntimeError("verified XYZ is missing an energy")
        energies.append(energy)
    with (out / "group_members.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sorted_frame", "group_id", "energy_hartree"])
        for index, frame_energy in enumerate(energies):
            writer.writerow([index + 1, member[index] + 1, f"{frame_energy:.14f}"])
    first = {}
    for line in log.splitlines():
        added = re.search(r"added structure (\d+)\b", line)
        if added:
            index = int(added.group(1))
            first[member[index]] = index
    if len(first) != ngroups:
        raise RuntimeError("could not recover one CoCoS representative per group")
    with (out / "representatives.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cocos_group", "input_frame_1based", "energy_hartree"])
        for group in range(ngroups):
            index = first[group]
            writer.writerow([group + 1, index + 1, f"{energies[index]:.14f}"])
    write_xyz(out / "representatives_original_xyz.xyz", [frames[first[g]] for g in range(ngroups)])
    write_unique(base, frames, member, energies)
    meta.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"CoCoS groups: {ngroups}; CritFreq={critfreq:g} cm^-1 (mirrors equivalent)", flush=True)
    if args.stability:
        cocos_stability(args, frames, member)


def labels_from_member(member):
    return [member[i] for i in range(len(member))]


def adjusted_rand(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) != len(b):
        raise ValueError("partition length mismatch")
    n = len(a)
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), int)
    for i, j in zip(a, b):
        table[i, j] += 1

    def comb2(x):
        return x * (x - 1) / 2

    sum_ij = comb2(table).sum()
    sum_a = comb2(table.sum(axis=1)).sum()
    sum_b = comb2(table.sum(axis=0)).sum()
    total = comb2(n)
    if total == 0:
        return 1.0
    expected = sum_a * sum_b / total
    maximum = 0.5 * (sum_a + sum_b)
    denom = maximum - expected
    return 1.0 if denom == 0 else float((sum_ij - expected) / denom)


def cocos_stability(args, frames, member):
    """Repeat CoCoS on the reversed trajectory and compare partitions."""
    base = work(args.n, args.tag)
    stab = base / "cocos_stability"
    if stab.exists():
        shutil.rmtree(stab)
    stab.mkdir()
    reversed_xyz = stab / "reversed.xyz"
    write_xyz(reversed_xyz, list(reversed(frames)))
    cocos_path = str(Path(args.cocos).expanduser().resolve())
    argv = [cocos_path, "-f", str(reversed_xyz.resolve()), "--DoNotUseGlobalOpt", "--NumSamples", str(args.samples), "--DoNotSaveDoubles", "--CritFreq", f"{args.critfreq:.8g}"]
    launcher = "import runpy, sys\nsys.argv = %r\nrunpy.run_path(%r, run_name='__main__')\n" % (argv, cocos_path)
    rc = run([sys.executable, "-c", launcher], stab, "cocos.log")
    if rc:
        raise RuntimeError("stability CoCoS failed")
    log = (stab / "cocos.log").read_text(encoding="utf-8", errors="replace")
    rev_member, _ng = parse_cocos_groups(log, len(frames))
    # reversed frame 0 is original frame N-1
    mapped = {}
    n = len(frames)
    for rev_index, group in rev_member.items():
        mapped[n - 1 - rev_index] = group
    score = adjusted_rand(labels_from_member(member), labels_from_member(mapped))
    (stab / "stability.json").write_text(
        json.dumps({"adjusted_rand_vs_forward": score, "samples": args.samples, "n_structures": n, "critfreq_cm1": args.critfreq}, indent=2),
        encoding="utf-8",
    )
    print(f"CoCoS stability adjusted Rand index: {score:.4f}", flush=True)
    if score < 0.95:
        print("WARNING: clustering changes with input order. Inspect cocos_stability before interpreting groups.", flush=True)
    # Explore the author's default (83.4) and a stricter candidate (166.8).
    # A greater CritFreq tightens the equivalence criterion and may yield more groups.
    # Do not select the value that merely produces the desired CN histogram.
    inp = base / "verified" / "free_sorted_for_cocos.xyz"
    current_groups = len(set(member.values()))
    results = [{"critfreq_cm1": args.critfreq, "groups": current_groups,
                "adjusted_rand_vs_selected": 1.0, "status": "selected"}]
    for freq in (83.4, 166.8):
        if abs(freq - args.critfreq) < 1e-8:
            continue
        with tempfile.TemporaryDirectory(prefix="critfreq_", dir=stab) as td:
            td = Path(td)
            cmd = [cocos_path, "-f", str(inp.resolve()), "--DoNotUseGlobalOpt",
                   "--NumSamples", str(args.samples), "--DoNotSaveDoubles",
                   "--CritFreq", f"{freq:.8g}"]
            launcher = "import runpy, sys\nsys.argv = %r\nrunpy.run_path(%r, run_name='__main__')\n" % (cmd, cocos_path)
            rc = run([sys.executable, "-c", launcher], td, "cocos.log")
            if rc:
                raise RuntimeError(f"CoCoS frequency sensitivity failed for {freq} cm^-1")
            log = (td / "cocos.log").read_text(encoding="utf-8", errors="replace")
            alt_member, alt_ngroups = parse_cocos_groups(log, len(frames))
            ari = adjusted_rand(labels_from_member(member), labels_from_member(alt_member))
            results.append({"critfreq_cm1": freq, "groups": alt_ngroups,
                            "adjusted_rand_vs_selected": round(ari, 6), "status": "sensitivity"})
    with (stab / "frequency_sensitivity.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(sorted(results, key=lambda r: r["critfreq_cm1"]))
    print("CoCoS CritFreq sensitivity: " + ", ".join(
        f"{x['critfreq_cm1']:g}:{x['groups']} groups" for x in results), flush=True)


def mdatom(args):
    base = work(args.n, args.tag)
    inp = base / "verified" / "free_sorted_for_cocos.xyz"
    if not inp.is_file():
        raise RuntimeError("Need frequency-verified geometries first")
    script = Path(args.mdatom).expanduser()
    if not script.is_file():
        raise RuntimeError("MDAtom script missing")
    source = script.read_text(encoding="utf-8")
    out = base / "mdatom"
    if (out / "mdatom_counts.csv").is_file() and not args.rerun:
        manifest = out / "command.json"
        if manifest.is_file():
            prior=json.loads(manifest.read_text(encoding="utf-8"))
            if abs(float(prior.get("scale", -1))-float(args.scale)) < 1e-12 and prior.get("script_sha256") == hashlib.sha256(script.read_bytes()).hexdigest():
                print(f"MDAtom already checked: scale={args.scale:.2f}", flush=True)
                return
        print("MDAtom settings changed; recomputing structural descriptors only", flush=True)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    scale = float(getattr(args, "scale", 1.0))
    if not (0.1 <= scale <= 3.0):
        raise ValueError("--scale must be within [0.1, 3.0] to avoid accidental nonsense")
    cmd = [
        sys.executable, str(script.resolve()), "-t", str(inp.resolve()), "-e", "K",
        "--RunWaterAnalysis", "--DoNotUseRounding", "--scale", f"{scale:.16f}",
    ]
    unknown = [flag for flag in ("-t", "-e", "--RunWaterAnalysis", "--DoNotUseRounding", "--scale") if flag not in source and flag.lstrip("-") not in source]
    rc = run(cmd, out, "mdatom.log")
    if rc:
        raise RuntimeError("MDAtom failed; see mdatom.log")
    if rounding_mode(out / "mdatom.log"):
        raise RuntimeError("MDAtom log says rounding is on. --DoNotUseRounding was not applied.")
    nframes = len(read_xyz(inp))
    table = parse_mdatom_output(out, nframes)
    (out / "command.json").write_text(
        json.dumps({
            "command": cmd,
            "exit_code": rc,
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "flag_strings_missing": unknown,
            "scale": scale,
            "operational_rounding": False,
            "note": "#water is the one-decimal value printed in .res, without integer rounding. For full-shell screening use a declared tolerance and assess sensitivity.",
        }, indent=2),
        encoding="utf-8",
    )
    with (out / "mdatom_counts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sorted_frame",
                "coordination_mdal",
                "coordination_raw",
                "radius_mdatom_A",
                "orientation_cos_mdatom",
            ],
        )
        writer.writeheader()
        writer.writerows(table)
    print(f"MDAtom rows: {len(table)}", flush=True)


def parse_water_res(path):
    """Parse local_surrounding_water_anlysis_*.res written by MDAtomLocalEnvironment."""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            raise RuntimeError(f"MDAtom water row has {len(parts)} fields: {stripped}")
        rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return rows


def rounding_mode(log_path):
    """MDAtom --DoNotUseRounding is store_false: the flag absent means rounding on."""
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    on = "We will use rounding for computing stoichiometry" in text
    off = "We will NOT use rounding for computing stoichiometry" in text
    if on == off:
        raise RuntimeError(f"MDAtom log does not state the rounding mode: {log_path}")
    return on


def reported_water_cn(raw):
    """Keep the one-decimal #water value printed by MDAtom. Do not rint 5.5 to 6."""
    return float(f"{float(raw):.1f}")


def cn_equals_n(raw, n, tolerance=0.15):
    """Near-complete weighted shell (example #water=5.9 for n=6).

    Operational, PREDECLARED threshold on the one-decimal .res value,
    NOT a claim of exact integer coordination. Explore sensitivity separately.
    """
    cn=reported_water_cn(raw)
    return -1e-9 <= float(n)-cn <= float(tolerance)+1e-9


def water_rows(out, nframes):
    res_files = sorted(Path(out).glob("local_surrounding_water_anlysis_*.res"))
    if len(res_files) != 1:
        log_path = Path(out) / "mdatom.log"
        log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        produced = [str(p.relative_to(out)) for p in Path(out).rglob("*") if p.is_file()]
        raise RuntimeError(
            "MDAtom finished but the water-analysis table was not found "
            f"({len(res_files)} .res files). Files: {produced}. Log tail: {log[-1500:]}"
        )
    rows = parse_water_res(res_files[0])
    if len(rows) != nframes:
        raise RuntimeError(f"MDAtom returned {len(rows)} rows for {nframes} structures")
    return rows


def parse_mdatom_output(out, nframes):
    """Read #water from local_surrounding_water_anlysis_*.res without integer rounding."""
    parsed = []
    for i, (radius, orient, raw) in enumerate(water_rows(out, nframes), 1):
        parsed.append({
            "sorted_frame": i,
            "coordination_mdal": f"{reported_water_cn(raw):.1f}",
            "coordination_raw": f"{raw:.1f}",
            "radius_mdatom_A": f"{radius:.4f}",
            "orientation_cos_mdatom": f"{orient:.3f}",
        })
    return parsed


def select(args):
    base = work(args.n, args.tag)
    groups = base / "cocos" / "group_members.csv"
    xyz_path = base / "verified" / "free_sorted_for_cocos.xyz"
    if not groups.is_file():
        raise RuntimeError("Run CoCoS first")
    with groups.open(newline="", encoding="utf-8") as handle:
        group_rows = list(csv.DictReader(handle))
    with Path(args.mdatom_csv).open(newline="", encoding="utf-8") as handle:
        md_rows = list(csv.DictReader(handle))
    frames = read_xyz(xyz_path)
    if len(frames) != len(md_rows) or len(frames) != len(group_rows):
        raise RuntimeError("CoCoS and MDAtom must cover every frequency-verified structure")
    md = {}
    for row in md_rows:
        index = int(row["sorted_frame"])
        if index in md:
            raise RuntimeError("repeated MDAtom frame")
        cn = reported_water_cn(row["coordination_mdal"])
        orient = float(row["orientation_cos_mdatom"])
        md[index] = (cn, orient)
    best_cn, audit = {}, []
    for row in group_rows:
        index = int(row["sorted_frame"])
        group = int(row["group_id"])
        energy = float(row["energy_hartree"])
        cn, orient = md[index]
        passed = cn_equals_n(cn, args.n, args.cn_tol)
        audit.append({
            "sorted_frame": index,
            "group_id": group,
            "energy_hartree": energy,
            "coordination_mdal": f"{cn:.1f}",
            "orientation_cos_mdatom": orient,
            "pass_full_first_shell": passed,
            # E.g. n=6, #water=5.5: a distorted octahedron can be a candidate
            # for manual/geometric review without changing the prespecified filter.
            "needs_geometric_review": (not passed and 0.0 <= args.n - cn <= 0.50 + 1e-9),
            "cn_deficit": f"{args.n - cn:.2f}",
            "cn_tolerance": f"{args.cn_tol:.2f}",
        })
        if passed and (group not in best_cn or energy < best_cn[group][0]):
            best_cn[group] = (energy, index)
    out = base / "final"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    chosen = sorted(((group, energy, index) for group, (energy, index) in best_cn.items()), key=lambda x: x[1])
    if chosen:
        write_xyz(out / "models.xyz", [frames[index - 1] for _g, _e, index in chosen])
        with (out / "models.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["xyz_frame", "group_id", "sorted_frame", "waters", "energy_hartree", "delta_energy_hartree",
                      "coordination_mdal", "orientation_cos_mdatom", "radius_mdatom_A", "cn_deficit"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            e0 = chosen[0][1]
            for j, (group, energy, index) in enumerate(chosen, 1):
                cn, orient = md[index]
                writer.writerow({
                    "xyz_frame": j, "group_id": group, "sorted_frame": index, "waters": args.n,
                    "energy_hartree": f"{energy:.14f}", "delta_energy_hartree": f"{(energy - e0):.14f}",
                    "coordination_mdal": f"{cn:.1f}", "orientation_cos_mdatom": f"{orient:.3f}",
                    "radius_mdatom_A": f"{float(next(x['radius_mdatom_A'] for x in md_rows if int(x['sorted_frame']) == index)):.4f}",
                    "cn_deficit": f"{args.n-cn:.2f}",
                })
    with (out / "selection_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]) if audit else ["sorted_frame", "group_id", "energy_hartree", "coordination_mdal", "orientation_cos_mdatom", "pass_full_first_shell"])
        writer.writeheader()
        writer.writerows(audit)
    counts = Counter(row["coordination_mdal"] for row in audit)
    (out / "RESULTS_STATUS.json").write_text(json.dumps({
        "n": args.n,
        "total_verified": len(frames),
        "passed_full_shell": int(sum(row["pass_full_first_shell"] for row in audit)),
        "cocos_groups_total": len(set(row["group_id"] for row in audit)),
        "final_representatives": len(chosen),
        "cn_histogram": {str(k): counts[k] for k in sorted(counts)},
        "native_mdatom_csv": str(Path(args.mdatom_csv).resolve()),
        "cn_tolerance": args.cn_tol,
        "cn_sensitivity_counts": {str(t): sum(cn_equals_n(float(r['coordination_mdal']), args.n, t) for r in audit) for t in (0.05,0.10,0.15,0.20,0.30)},
        "cn_geometric_review_count": sum(bool(row["needs_geometric_review"]) for row in audit),
        "cocos_critfreq_cm1": json.loads((base / "cocos" / "command.json").read_text(encoding="utf-8"))["critfreq_cm1"],
        "note": "Near-complete shell is n-#water <= declared tolerance, using one-decimal #water printed by MDAtom with --DoNotUseRounding --scale 1.0. Example 5.9 of 6 passes at tolerance 0.15. This is an operational classification, not a proof of first-shell occupancy.",
    }, indent=2), encoding="utf-8")
    print(f"CN=n representatives: {len(chosen)} of {len(frames)}; CN histogram {dict(counts)}", flush=True)


def characterize(args):
    """CoCoS, real MDAtom, CN = n selection, and geometric shell sensitivity for one n."""
    base = work(args.n, args.tag)
    verified = base / "verified" / "INCOMPLETE.txt"
    if not verified.is_file():
        raise RuntimeError("Frequency export is missing. Characterize was not run.")
    status_text = verified.read_text(encoding="utf-8")
    if "ORIGINAL_STARTS_PROCESSED=100\n" not in status_text:
        raise RuntimeError(f"n={args.n} has not processed 100 starts")
    cocos_ns = argparse.Namespace(
        n=args.n, tag=args.tag, cocos=args.cocos, samples=args.samples, critfreq=args.critfreq, rerun=False, stability=True,
    )
    cocos(cocos_ns)
    if not (base / "cocos_stability" / "stability.json").is_file():
        frames = read_xyz(base / "verified" / "free_sorted_for_cocos.xyz")
        member = {}
        with (base / "cocos" / "group_members.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                member[int(row["sorted_frame"]) - 1] = int(row["group_id"]) - 1
        cocos_stability(cocos_ns, frames, member)
    mdatom(argparse.Namespace(n=args.n, tag=args.tag, mdatom=args.mdatom, rerun=False, scale=args.scale))
    counts = base / "mdatom" / "mdatom_counts.csv"
    select(argparse.Namespace(n=args.n, tag=args.tag, mdatom_csv=str(counts), cn_tol=args.cn_tol))
    shell = base / "shell"
    if not (shell / "per_structure.csv").is_file():
        validator = ROOT / "scripts" / "validate_k_shell.py"
        rc = run(
            [sys.executable, str(validator), "--input", str(base / "verified" / "free_sorted_for_cocos.xyz"), "--output", str(shell)],
            shell,
            "shell_validate.log",
        )
        if rc:
            raise RuntimeError("geometric shell validation failed")
    record_characterization(args.n)


def record_characterization(n):
    """Refresh one progress-table row and append a chronology entry. Earlier entries stay."""
    base = work(n, "full_100")
    status = json.loads((base / "final" / "RESULTS_STATUS.json").read_text(encoding="utf-8"))
    unique = list(csv.DictReader((base / "unique_minima" / "unique_minima.csv").open(encoding="utf-8")))
    verified = (base / "verified" / "INCOMPLETE.txt").read_text(encoding="utf-8")
    n_verified = verified.split("VERIFIED_FREE_MINIMA=")[1].splitlines()[0]
    stability = json.loads((base / "cocos_stability" / "stability.json").read_text(encoding="utf-8"))
    shell_rows = list(csv.DictReader((base / "shell" / "sensitivity_by_n.csv").open(encoding="utf-8")))
    shell = shell_rows[0] if shell_rows else {}
    energies = [float(row["energy_hartree"]) for row in unique]
    if not energies:
        raise RuntimeError(f"n={n} produced no unique minima")
    span = max(energies) - min(energies)
    row = (
        f"| {n} | DONE 100/100 | DONE 100/100 | DONE {n_verified} | "
        f"DONE {status['cocos_groups_total']} групп | "
        f"DONE {status['total_verified']} | DONE {status['final_representatives']} | PENDING |"
    )
    path = ROOT / "EXPERIMENT_LOG.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.startswith(f"| {n} |"):
            lines[i] = row
            replaced = True
            break
    if not replaced:
        raise RuntimeError("progress table has no row for this n")
    full_ko = shell.get("full_ko_3.2", "")
    note = (
        f"### {datetime.now().strftime('%Y-%m-%d %H:%M')}. n = {n}: CoCoS, MDAtom, оболочка\n"
        f"\n"
        f"Подтверждённых структур {n_verified}. Уникальных минимумов CoCoS {len(unique)}; "
        f"групп {status['cocos_groups_total']}; CritFreq={stability['critfreq_cm1']:g} см⁻¹. Устойчивость к обратному порядку: "
        f"adjusted Rand {stability['adjusted_rand_vs_forward']:.4f}. "
        f"Энергия уникальных минимумов {min(energies):.12f}…{max(energies):.12f} Eh, "
        f"размах внутри n = {span:.6f} Eh.\n"
        f"\n"
        f"MDAtom (--DoNotUseRounding, scale = 1.0, порог n−КЧ ≤ {status['cn_tolerance']:.2f}; #water из .res): "
        f"гистограмма КЧ {status['cn_histogram']}. "
        f"Структур с КЧ = n: {status['passed_full_shell']}. "
        f"Представителей групп с КЧ = n: {status['final_representatives']}; "
        f"пограничных структур для геометрической проверки: {status['cn_geometric_review_count']}. "
        f"Группы без такого члена сохранены в уникальных минимумах и в `selection_audit.csv`.\n"
        f"\n"
        f"Геометрическая диагностика, не КЧ MDAtom: структур, у которых все {n} контактов K–O ≤ 3,2 Å: {full_ko}. "
        f"Средний K–O по проверенным структурам {shell.get('mean_KO_A', '')} Å. "
        f"Файлы: `runs/K_{n:02d}/full_100/cocos`, `mdatom`, `shell`, `unique_minima`, `final`."
    )
    lines.extend(["", note])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Characterized n={n}: groups {status['cocos_groups_total']}, CN=n representatives {status['final_representatives']}", flush=True)


def status(args):
    for n in range(4, 15):
        base = work(n, args.tag)
        if not base.exists():
            print(f"n={n}: PENDING")
            continue
        done = sum(gwh_output_ok(base / "gwh" / f"gwh_opt_str{i:04d}", i, n) for i in range(100))
        rows = list((base / "free").glob("frame_*/status.json"))
        good = sum(json.loads(p.read_text(encoding="utf-8")).get("final",{}).get("status") == "MINIMUM" for p in rows)
        final = base / "final" / "RESULTS_STATUS.json"
        groups = json.loads(final.read_text(encoding="utf-8"))["final_representatives"] if final.is_file() else None
        print(f"n={n:02d} GWH={done}/100 Hessian={len(rows)}/100 minima={good} CN=n groups={groups if groups is not None else '-'}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check_p = sub.add_parser("check")
    check_p.add_argument("--xtb", default="xtb")
    check_p.add_argument("--skip-xtb",action="store_true",help="Verify inputs and hashes only; does not prove xTB works")
    check_p.add_argument("--pack-dir", default=None)
    check_p.set_defaults(func=check)
    specs = [
        ("gwh", gwh), ("collect", collect), ("relax", relax),
        ("export", export), ("cocos", cocos), ("mdatom", mdatom), ("select", select),
        ("characterize", characterize),
    ]
    for name, func in specs:
        item = sub.add_parser(name)
        item.add_argument("--n", required=True, type=int, choices=range(4, 15))
        item.add_argument("--tag", default="pilot_2")
        item.set_defaults(func=func)
        if name == "gwh":
            item.add_argument("--gwh", required=True)
            item.add_argument("--xtb", default="xtb")
            item.add_argument("--numstruct", type=int, default=2)
            item.add_argument("--radius", type=float, default=3.04)
            item.add_argument("--workers", type=int, default=4)
            item.add_argument("--threads", type=int, default=1)
        if name == "relax":
            item.add_argument("--xtb", default="xtb")
            item.add_argument("--limit", type=int)
            item.add_argument("--threads", type=int, default=1)
            item.add_argument("--workers", type=int, default=4)
        if name == "cocos":
            item.add_argument("--cocos", required=True)
            item.add_argument("--samples", type=int, default=50)
            item.add_argument("--critfreq", type=float, default=125.0, help="CoCoS similarity frequency in cm^-1; native default 83.4")
            item.add_argument("--rerun", action="store_true")
            item.add_argument("--stability", action="store_true")
        if name == "mdatom":
            item.add_argument("--mdatom", required=True)
            item.add_argument("--scale", type=float, default=1.0)
            item.add_argument("--rerun", action="store_true")
        if name == "select":
            item.add_argument("--mdatom-csv", required=True)
            item.add_argument("--cn-tol", type=float, default=0.15, help="Prespecified max n-#water for screening (example 5.9 of 6).")
        if name == "characterize":
            item.add_argument("--cocos", required=True)
            item.add_argument("--mdatom", required=True)
            item.add_argument("--samples", type=int, default=50)
            item.add_argument("--critfreq", type=float, default=125.0)
            item.add_argument("--scale", type=float, default=1.0)
            item.add_argument("--cn-tol", type=float, default=0.15)
    status_p = sub.add_parser("status")
    status_p.add_argument("--tag", default="full_100")
    status_p.set_defaults(func=status)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    arguments.func(arguments)
