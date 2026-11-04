"""Synthetic non-chemical verification of geometry and ranking. NOT a result for potassium."""
import csv
import importlib.util
import tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kr',ROOT/'scripts/k_coordination_review.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
# regular tetrahedron around K. Valid synthetic geometry to test math, not a predicted hydrate.
v=np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]],float)
v/=np.linalg.norm(v,axis=1)[:,None]
coords=np.zeros((13,3));sy=['K']+['x']*12
for i,u in enumerate(v):
 o=3.*u; coords[1+3*i]=o
 coords[2+3*i]=o+np.array([0,.8,.5]);coords[3+3*i]=o+np.array([0,-.8,.5])
 sy[1+3*i:1+3*i+3]=['O','H','H']
geo=m.geometry(('synthetic',sy,coords))
assert geo['n']==4 and abs(geo['rmean_A']-3)<1e-9
assert geo['cut_counts']['K_O_le_3.2A']==4
assert geo['max_gap_A']<1e-8 and geo['surface_index']<1e-8
with tempfile.TemporaryDirectory() as td:
 root=Path(td);m.ROOT=root
 base=root/'runs/K_04/full_100';vdir=base/'verified';vdir.mkdir(parents=True)
 lines=['13','id=frame_0001 E = -1.0 Eh']
 for ele,r in zip(sy,coords):lines.append(ele+' '+' '.join(f'{x:.9f}' for x in r))
 (vdir/'free_sorted_for_cocos.xyz').write_text('\n'.join(lines)+'\n')
 with (vdir/'free_sorted_metadata.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['sorted_frame','source','energy_hartree']);writer.writeheader()
  writer.writerow({'sorted_frame':1,'source':'frame_0001','energy_hartree':'-1.0'})
 assert m.make_review('full_100')==1
 data=(root/'review/output/K_coordination_minima_review.md').read_text()
 assert 'REQUIRES_MANUAL_REVIEW' in data and 'NOT_AVAILABLE' in data
 rows=m.load_rows(root/'review/output/candidate_minima.csv')
 assert len(rows)==1 and rows[0]['review_status']=='REQUIRES_MANUAL_REVIEW'
 assert len(m.load_rows(root/'review/output/all_verified_geometry_audit.csv'))==1
print('PASS: synthetic K–O geometry, source-ID integrity, energy ranking, manual review status')
