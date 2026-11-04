#!/usr/bin/env python3
"""Postprocess *all* frequency-verified K(H2O)n geometries without any xTB/GWH.

This script makes an AUDIT and candidates for HUMAN coordination review; it does
not automatically claim a full first shell or assign an accepted coordination.
CoCoS groups are shown as metadata only; no candidate is dropped by CoCoS.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from pathlib import Path
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
EV=627.509474
CUTOFFS=(2.6,2.8,3.0,3.2,3.4,3.6,3.8,4.0)


def all_xyz_frames(path):
    lines=Path(path).read_text(encoding='utf-8',errors='replace').splitlines()
    i=0
    frames=[]
    while i<len(lines):
        if not lines[i].strip():i+=1;continue
        n=int(lines[i]);assert n>1 and i+n+2<=len(lines),f'{path}: truncated frame'
        comment=lines[i+1]
        sym=[];positions=[]
        for raw in lines[i+2:i+2+n]:
            parts=raw.split()
            if len(parts)<4:raise ValueError(f'Malformed coordinate: {raw}')
            sym.append(parts[0].title())
            positions.append([float(x.replace('D','E')) for x in parts[1:4]])
        frames.append((comment,sym,np.array(positions,float)))
        i+=n+2
    return frames


def write_one_xyz(path,frame,n,index):
    comment,sym,xyz=frame
    assert len(sym)==3*n+1
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as out:
        out.write(f'{len(sym)}\nK_(H2O)_{n} verified_frame={index}; '+comment.strip()+'\n')
        for el,(x,y,z) in zip(sym,xyz):out.write(f'{el:2s} {x:.10f} {y:.10f} {z:.10f}\n')


def geometry(frame):
    _,sym,xyz=frame
    counts=Counter(sym)
    n=counts['O']
    if counts!={'K':1,'O':n,'H':2*n} or not (4<=n<=14):
        raise ValueError(f'Unexpected K hydrate composition: {counts}')
    if not np.isfinite(xyz).all():raise ValueError('Nonfinite coordinates')
    c=xyz[sym.index('K')]
    o=xyz[np.array([s=='O' for s in sym])]
    v=o-c
    d=np.linalg.norm(v,axis=1)
    if (d<1).any():raise ValueError('Unphysical K-O distance < 1 A')
    order=np.argsort(d)
    sorted_d=d[order]
    gaps=np.diff(sorted_d)
    u=v/d[:,None]
    tensor=(u.T@u)/n
    eig=np.linalg.eigvalsh(tensor)
    angles=np.degrees(np.arccos(np.clip(u@u.T,-1,1)))
    offdiagonal=angles[np.triu_indices(n,1)]
    surface=float(np.linalg.norm(u.mean(axis=0)))
    maxgap=float(np.max(gaps)) if gaps.size else 0.
    imax=int(np.argmax(gaps)) if gaps.size else 0
    # Largest gap: a CANDIDATE split only; severe asymmetry of valid polyhedra is possible.
    return {
        'n':n, 'distances_A':sorted_d,'gaps_A':gaps, 'rmin_A':float(np.min(d)),
        'rmax_A':float(np.max(d)),'rmean_A':float(np.mean(d)),
        'rstd_A':float(np.std(d)),'cv':float(np.std(d)/np.mean(d)),
        'max_gap_A':maxgap,'gap_after_k':imax+1,'gap_left_A':float(sorted_d[imax]),
        'gap_right_A':float(sorted_d[imax+1]) if n>1 else float(sorted_d[imax]),
        'surface_index':surface,'eig_min':float(eig[0]),'eig_mid':float(eig[1]),'eig_max':float(eig[2]),
        'angle_min_deg':float(offdiagonal.min()),'angle_max_deg':float(offdiagonal.max()),
        'angle_near180_pairs':int(np.sum(offdiagonal>155)),
        'cut_counts':{f'K_O_le_{r:.1f}A':int(np.sum(d<=r)) for r in CUTOFFS},
    }


def load_rows(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))


def csv_write(path,records):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not records:return
    with path.open('w',newline='',encoding='utf-8-sig') as out:
        writer=csv.DictWriter(out,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


def make_review(tag):
    rows=[];short=[];selection=[];missing=[]
    for n in range(4,15):
        base=ROOT/'runs'/f'K_{n:02d}'/tag
        xyzpath=base/'verified'/'free_sorted_for_cocos.xyz'
        metapath=base/'verified'/'free_sorted_metadata.csv'
        mdpath=base/'mdatom'/'mdatom_counts.csv'
        gp=base/'cocos'/'group_members.csv'
        if not xyzpath.exists() or not metapath.exists():
            missing.append(n)
            selection.append({'n':n,'status':'NOT_CALCULATED','n_verified':0,'E_global_min_Ha':'',
                              'best_candidate_id':'','candidate_status':'NOT_AVAILABLE'})
            continue
        frames=all_xyz_frames(xyzpath)
        meta=load_rows(metapath)
        if len(meta)!=len(frames):raise RuntimeError(f'n={n}: metadata vs XYZ count differs')
        md_by={int(x['sorted_frame']):x for x in load_rows(mdpath)} if mdpath.exists() else {}
        gp_by={int(x['sorted_frame']):x['group_id'] for x in load_rows(gp)} if gp.exists() else {}
        r=[]
        for i,(fr,m) in enumerate(zip(frames,meta),1):
            if int(m['sorted_frame'])!=i:raise RuntimeError(f'n={n} row mismatch at sorted frame {i}')
            g=geometry(fr)
            if g['n']!=n:raise RuntimeError(f'n={n} XYZ composition wrong frame {i}')
            E=float(m['energy_hartree'])
            # The geometry original may contain 'energy' but metadata is canonical.
            idm=m['source'];md=md_by.get(i,{})
            vals={
                'n':n, 'sorted_frame':i, 'source':idm,'energy_Ha':E,
                'delta_E_kcal_per_mol':0.,'group_id':gp_by.get(i,''),
                'CN_MDATOM_fractional':md.get('coordination_mdal',''),
                'r_MDATOM_weighted_A':md.get('radius_mdatom_A',''),
                'mean_KO_A':round(g['rmean_A'],6),'min_KO_A':round(g['rmin_A'],6),
                'max_KO_A':round(g['rmax_A'],6),
                'all_KO_sorted_A':';'.join(f'{x:.4f}' for x in g['distances_A']),
                'gaps_sorted_A':';'.join(f'{x:.4f}' for x in g['gaps_A']),
                'largest_gap_A':round(g['max_gap_A'],5),
                'largest_gap_after_k':g['gap_after_k'],
                'gap_between_distances_A':f'{g["gap_left_A"]:.4f}→{g["gap_right_A"]:.4f}',
                'CV_KO':round(g['cv'],6),'surface_index':round(g['surface_index'],6),
                'eig_min':round(g['eig_min'],6),'eig_mid':round(g['eig_mid'],6),'eig_max':round(g['eig_max'],6),
                'angle_min_deg':round(g['angle_min_deg'],3),'angle_max_deg':round(g['angle_max_deg'],3),
                'antipodal_pairs_gt155':g['angle_near180_pairs'],
                **g['cut_counts'],
                'coordination_from_visual_XYZ':'',
                'second_shell_from_visual_XYZ':'',
                'review_status':'REQUIRES_MANUAL_REVIEW',
                'reviewer_comment':'',
                'xyz_source':str(xyzpath.relative_to(ROOT)),
            }
            r.append((vals,fr))
        Emin=min(v['energy_Ha'] for v,_ in r)
        for v,_ in r:v['delta_E_kcal_per_mol']=round((v['energy_Ha']-Emin)*EV,7)
        for v,_ in r:rows.append(v)
        r.sort(key=lambda x:(x[0]['energy_Ha'],x[0]['sorted_frame']))
        # Global min + best per each probed direct-contact cutoff, regardless of CoCoS family.
        chosen={r[0][0]['sorted_frame']:'GLOBAL_ENERGY_MIN'}
        for cut in CUTOFFS:
            key=f'K_O_le_{cut:.1f}A'
            cand=next(((row,fr) for row,fr in r if row[key]==n),None)
            if cand:
                ident=cand[0]['sorted_frame']
                chosen[ident]=chosen.get(ident,'')+f' | candidate all K–O ≤ {cut:.1f} Å'
        # If no all-n candidates among cutoff sweep, retain most nearly complete by max contacts at largest cutoff.
        key=f'K_O_le_{CUTOFFS[-1]:.1f}A'
        if not any(v[key]==n for v,_ in r):
            candidate=max(r,key=lambda x:(x[0][key],-x[0]['energy_Ha']))
            chosen[candidate[0]['sorted_frame']]=chosen.get(candidate[0]['sorted_frame'],'')+' | MOST_CONTACTS_DIAGNOSTIC'
        for v,fr in r:
            if v['sorted_frame'] not in chosen:continue
            path=ROOT/'review'/'output'/f'K_{n:02d}'/f'{v["source"]}_sorted{v["sorted_frame"]:04d}.xyz'
            write_one_xyz(path,fr,n,v['sorted_frame'])
            short.append({**v,'candidate_reason':chosen[v['sorted_frame']],'review_xyz':str(path.relative_to(ROOT))})
        selection.append({
            'n':n,'status':'CALCULATED_REQUIRES_MANUAL_REVIEW','n_verified':len(frames),
            'E_global_min_Ha':round(Emin,12),'best_candidate_id':r[0][0]['source'],
            'candidate_status':'REQUIRES_MANUAL_REVIEW',
        })
    folder=ROOT/'review'/'output'
    folder.mkdir(parents=True,exist_ok=True)
    csv_write(folder/'all_verified_geometry_audit.csv',rows)
    csv_write(folder/'candidate_minima.csv',short)
    csv_write(folder/'selection_summary.csv',selection)
    txt=[
      '# K⁺(H₂O)ₙ — кандидаты для ручного отбора по координации',
      '',
      '> **НЕ АВТОМАТИЧЕСКОЕ ПРИНЯТИЕ**. Ни одному кандидату не присвоен статус ACCEPT без проверки XYZ исследователем.',
      '> Анализ использует все частотно подтверждённые структуры, независимо от принадлежности к группам CoCoS.',
      '> `radius[A]` MDAtom — взвешенное среднее до условных центров H₂O, **не** радиус отсечения.',
      '> Пороговые расстояния K–O — диагностический sweep, а не универсальные границы оболочки.',
      '',
      '| n | Глобальный минимум | E, Ha | Кандидатов к просмотру | Задача |',
      '|---:|---|---:|---:|---|',
    ]
    for v in selection:
      n=v['n']; cnt=sum(x['n']==n for x in short)
      txt.append(f"| {n} | {v['best_candidate_id'] or '—'} | {v['E_global_min_Ha'] or '—'} | {cnt} | {v['candidate_status']} |")
    txt += ['', '## Претенденты (без автоматического утверждения КЧ)', '',
            '| n | frame | E, Ha | ΔE, kcal/mol | КЧ MDAtom | K–O, Å (min…max) | Наибольший разрыв, Å | Диагн. число K–O≤3.2 Å | Статус |',
            '|---:|---|---:|---:|---:|---|---:|---:|---|']
    for v in short:
      txt.append(f"| {v['n']} | `{v['source']}` | {v['energy_Ha']:.12f} | {v['delta_E_kcal_per_mol']:.5f} | {v['CN_MDATOM_fractional'] or '—'} | {v['min_KO_A']:.3f}…{v['max_KO_A']:.3f} | {v['largest_gap_A']:.3f} | {v['K_O_le_3.2A']} | REVIEW XYZ |")
    txt+=['','## Правила ручной проверки','',
       '1. Смотреть XYZ-кандидаты в `review/output/K_XX`, визуально оценивать реальное K–O окружение.',
       '2. Разрывы расстояний, углы, тензор формы и H-связи считать вспомогательными; вытянутый полиэдр не равно внешняя вода.',
       '3. Для каждого n сперва подтвердить подходящую координацию, затем выбрать минимальную электронную энергию **среди подтверждённых**.',
       '4. Если таких кандидатов нет, отмечать `NOT_FOUND_IN_SEARCH`, не утверждать физическую невозможность.',
       '5. Итоговые ACCEPT/REJECT оформлять в CSV `candidate_minima.csv` после ручной проверки; оригинальные расчёты не исправлять.',
       '', '## Полнота',
       f"Всего записей в геометрическом аудите: {len(rows)}; кандидатов для просмотра: {len(short)}; отсутствующие составы: {missing}.",
       'Это **постобработка**; GWH, xTB и Hessian данным скриптом не запускались.', '']
    (folder/'K_coordination_minima_review.md').write_text('\n'.join(txt),encoding='utf-8')
    print(f'Review: verified={len(rows)}; candidates={len(short)}; missing n={missing}')
    return len(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag',default='full_100')
    args=ap.parse_args()
    n=make_review(args.tag)
    if n==0:print('No computed K minima yet; only honest NOT_CALCULATED statuses written.')
if __name__=='__main__':main()
