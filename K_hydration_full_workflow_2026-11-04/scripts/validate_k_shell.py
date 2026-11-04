#!/usr/bin/env python3
"""Independent geometric sensitivity analysis for freely optimized K+(H2O)n XYZ.

Requires: numpy, scipy (Hungarian water identification), matplotlib (optional plots).
Does NOT reproduce MDAtomLocalEnvironment's sigma-weighted coordination number.
No xTB optimization/frequency calculation is performed here.
"""
from __future__ import annotations
import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

KO_CUTS = [round(2.4 + 0.1*i, 2) for i in range(19)]   # 2.4 .. 4.2 A diagnostic only
KH_CUTS = [2.8, 3.2, 3.6]   # diagnostic, not cation coordination
ANGLES = [110, 130, 150]                         # degrees, K+H-O
ENERGY_RE = re.compile(r'\benergy\s*:\s*([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)', re.I)


def xyz_frames(path):
    lines = path.read_text(encoding='utf-8').splitlines()
    i = 0
    frame = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        try:
            nat = int(lines[i].strip())
        except ValueError as e:
            raise ValueError(f'{path}:{i+1} expected atom count') from e
        if i + 2 + nat > len(lines):
            raise ValueError(f'{path}: truncated frame {frame+1}')
        comment = lines[i+1]
        elems = []
        coords = []
        for ln in lines[i+2:i+2+nat]:
            cols = ln.split()
            if len(cols) < 4:
                raise ValueError(f'{path}: bad XYZ line {ln!r}')
            elems.append(cols[0].upper())
            coords.append([float(v.replace('D','E')) for v in cols[1:4]])
        frame += 1
        yield frame, comment, elems, np.asarray(coords, dtype=float)
        i += 2 + nat


def unit(v):
    size = float(np.linalg.norm(v))
    if size < 1e-10:
        raise ValueError('zero-length molecular vector')
    return v / size


def analyze_one(elems, xyz):
    counts = Counter(elems)
    n = counts['O']
    if counts['K'] != 1 or n < 1 or counts['H'] != 2*n or sum(counts.values()) != 3*n+1:
        raise ValueError(f'expected K1 O[n] H[2n], got {dict(counts)}')
    if not np.isfinite(xyz).all():
        raise ValueError('non-finite XYZ coordinate')
    f = xyz[elems.index('K')]
    oxy = xyz[[i for i,e in enumerate(elems) if e == 'O']]
    hyd = xyz[[i for i,e in enumerate(elems) if e == 'H']]
    # Each oxygen has exactly two H slots. Match all H globally with minimum
    # total O-H length; don't assume atoms are ordered O,H,H in the XYZ file.
    slots = np.repeat(oxy, 2, axis=0)
    dist = np.linalg.norm(hyd[:, None, :] - slots[None, :, :], axis=2)
    h_indices, slot_indices = linear_sum_assignment(dist)
    attachments = [[] for _ in range(n)]
    oh_dist = []
    for ih, slot in zip(h_indices, slot_indices):
        k = int(slot // 2)
        attachments[k].append(int(ih))
        oh_dist.append(float(dist[ih, slot]))
    if max(oh_dist) > 1.30 or min(oh_dist) < 0.65:
        raise ValueError(f'O-H assignment suspicious (range {min(oh_dist):.3f}..{max(oh_dist):.3f} A)')

    fo = np.linalg.norm(oxy-f, axis=1)
    closest_fh = []
    fh_angles = []
    cos_theta = []
    for o, (h1_idx, h2_idx) in zip(oxy, attachments):
        hh = hyd[[h1_idx, h2_idx]]
        h_distances = np.linalg.norm(hh-f, axis=1)
        closest_fh.append(float(min(h_distances)))
        angles = []
        for h in hh:
            # K+-H--O (at H): linear H bond is 180 degrees.
            v1 = unit(f-h)
            v2 = unit(o-h)
            angles.append(math.degrees(math.acos(float(np.clip(v1@v2, -1, 1)))))
        fh_angles.append(angles)
        bisector = unit(hh.mean(axis=0)-o)
        rw = (o + hh.mean(axis=0))/2
        radial = unit(rw-f)
        cos_theta.append(float(radial @ bisector))

    result = {
        'n': n,
        'mean_KO_A': float(fo.mean()),
        'min_KO_A': float(fo.min()),
        'max_KO_A': float(fo.max()),
        'min_KH_A': min(closest_fh),
        'max_closest_KH_A': max(closest_fh),
        'mean_cos_unweighted': float(np.mean(cos_theta)),
    }
    for cut in KO_CUTS:
        result[f'cn_ko_{cut:.1f}'] = int(np.sum(fo <= cut))
    for fh_cut in KH_CUTS:
        for angle_cut in ANGLES:
            total = 0
            for o_index, (h1_idx, h2_idx) in enumerate(attachments):
                good = False
                for j,h_idx in enumerate([h1_idx,h2_idx]):
                    if np.linalg.norm(hyd[h_idx] - f) <= fh_cut and fh_angles[o_index][j] >= angle_cut:
                        good = True
                total += int(good)
            result[f'cn_kh_{fh_cut:.1f}_{angle_cut}'] = total
    result['n_KO_gt_3.4'] = int(np.sum(fo > 3.4))
    result['surface_index'] = float(np.linalg.norm(((oxy - f) / fo[:, None]).mean(axis=0)))
    ww = 0
    for o_index, (h1_idx, h2_idx) in enumerate(attachments):
        for h_idx in (h1_idx, h2_idx):
            h = hyd[h_idx]
            for other in range(n):
                if other == o_index:
                    continue
                if np.linalg.norm(h - oxy[other]) <= 2.4:
                    v1 = unit(oxy[o_index] - h)
                    v2 = unit(oxy[other] - h)
                    ang = math.degrees(math.acos(float(np.clip(v1 @ v2, -1, 1))))
                    if ang >= 150.0:
                        ww += 1
                        break
    result['n_ww_hbonds'] = ww
    return result


def write_csv(path, rows):
    if not rows:
        return
    cols = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='One multiframe XYZ or recursive directory of free-optimized XYZ files')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--metadata-csv', type=Path, help='Optional models.csv for mapping xyz_frame to candidate_id; single XYZ file only')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input.is_file():
        inputs = [args.input]
    else:
        inputs = sorted(args.input.rglob('*.xyz'))
    if not inputs:
        parser.error('No XYZ files found')
    if any(('constrained' in str(x).lower() or 'gwh_opt' in str(x).lower() or x.name.startswith('mol_str')) for x in inputs):
        parser.error('Input looks like constrained GWH geometries. Supply FULLY FREELY OPTIMIZED XYZ structures, including rejected candidates.')

    metadata = {}
    if args.metadata_csv:
        if len(inputs) != 1:
            parser.error('--metadata-csv is only allowed with a single multi-frame XYZ')
        with args.metadata_csv.open(encoding='utf-8-sig', newline='') as f:
            metadata = {int(d['xyz_frame']): d for d in csv.DictReader(f)}

    rows, errors = [], []
    for path in inputs:
        try:
            for frame, comment, elems, xyz in xyz_frames(path):
                ident = f'{path.name}:{frame}'
                try:
                    res = analyze_one(elems, xyz)
                    extra = metadata.get(frame, {})
                    mat = ENERGY_RE.search(comment)
                    reported = mat.group(1) if mat else extra.get('energy_hartree','')
                    if extra:
                        if int(extra['waters']) != res['n']:
                            raise ValueError('metadata waters not equal XYZ oxygen count')
                        if mat and abs(float(reported) - float(extra['energy_hartree'])) > 1e-6:
                            raise ValueError('energy disagreement between XYZ and metadata')
                    rows.append({'source':str(path),'frame':frame,
                                 'candidate_id':extra.get('candidate_id', ident),
                                 'group_id': extra.get('group_id',''),
                                 'energy_hartree':reported,
                                 'mdal_reported_cn':extra.get('coordination_mdal',''),
                                 **res})
                except Exception as e:
                    errors.append({'source':str(path),'frame':frame,'error':str(e)})
        except Exception as e:
            errors.append({'source':str(path),'frame':'','error':str(e)})
    if not rows:
        raise SystemExit('No valid geometry; see diagnostics below: '+str(errors[:3]))
    write_csv(args.output/'per_structure.csv',rows)
    write_csv(args.output/'invalid_structures.csv',errors)

    # Each cutoff is a sensitivity choice, NOT a universal shell boundary.
    by_n = defaultdict(list)
    for row in rows:
        by_n[row['n']].append(row)
    summary=[]
    for n, subset in sorted(by_n.items()):
        values={'n':n,'structures':len(subset),
                'mean_KO_A':round(float(np.mean([r['mean_KO_A'] for r in subset])),5),
                'avg_cn_ko_3.0':round(float(np.mean([r['cn_ko_3.0'] for r in subset])),3),
                'avg_cn_kh_3.2_130':round(float(np.mean([r['cn_kh_3.2_130'] for r in subset])),3)}
        for cut in KO_CUTS:
            key=f'cn_ko_{cut:.1f}'
            values[f'full_ko_{cut:.1f}'] = sum(r[key] == n for r in subset)
        for fh in KH_CUTS:
            for a in ANGLES:
                key=f'cn_kh_{fh:.1f}_{a}'
                values[f'full_hb_{fh:.1f}_{a}'] = sum(r[key] == n for r in subset)
        summary.append(values)
    write_csv(args.output/'sensitivity_by_n.csv',summary)
    with (args.output/'readme_results.txt').open('w',encoding='utf-8') as f:
        f.write(f'Input: {args.input}\nAnalyzed geometries: {len(rows)}\nInvalid: {len(errors)}\n')
        f.write('WARNING: these are GEOMETRIC proxy criteria, NOT MDAtomLocalEnvironment sigma-based CN.\n')
        f.write('K–O direct contact diagnostic sweep: 2.4..4.2 A step 0.1 A.\n')
        f.write('K···H–O hydrogen-bond sweep: K+H <= 2.8,3.2,3.6 A and angle K+H-O >= 110,130,150 deg; these are NOT coordination criteria for a cation.\n')
        f.write('No frequency validation, optimizations or CoCoS grouping in this program.\n')
        f.write('Use the FULL K+ freely optimized XYZ catalog, NOT constrained GWH frames. Number of accepted K+ structures is unknown until computations are run.\n')
        f.write('Only analyzed n values appear; absent compositions are NOT zero.\n')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax=plt.subplots(figsize=(8,4.6))
        xx=sorted(by_n)
        for k in ['cn_ko_2.8','cn_ko_3.0','cn_ko_3.2']:
            frac=[sum(r[k]==n for r in by_n[n])/len(by_n[n]) for n in xx]
            ax.plot(xx,frac,marker='o',label=f'K-O <= {k[-3:]} A')
        k='cn_kh_3.2_130'
        frac=[sum(r[k]==n for r in by_n[n])/len(by_n[n]) for n in xx]
        ax.plot(xx,frac,marker='s',linestyle='--',label='K+H <= 3.4 A, angle >= 130 deg (diagnostic)')
        ax.set(xlabel='Number of water molecules n',ylabel='Fraction with all n direct contacts',ylim=(-.05,1.05))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output/'k_full_shell_sensitivity.png',dpi=180)
        plt.close(fig)
    except ImportError:
        pass
    print('Analyzed',len(rows),'geometries; invalid',len(errors))
    print('n values:',', '.join(f'{n}: {len(items)}' for n,items in sorted(by_n.items())))
    print('Saved to',args.output)

if __name__ == '__main__':
    main()
