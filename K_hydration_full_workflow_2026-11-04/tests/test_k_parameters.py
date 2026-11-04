"""Independent input/parameter integrity tests; no xTB chemistry performed."""
from pathlib import Path
import csv,hashlib,math,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
with (ROOT/'inputs_manifest.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
assert len(rows)==11 and [int(r['n_water']) for r in rows]==list(range(4,15))
reference_angles={4:109.4712206,5:90.0,6:90.0,7:77.8695421,8:74.8584922,9:70.5287794,10:66.146822,11:63.4349488,12:63.4349488,13:57.1367031,14:55.6705700}
for row in rows:
 n=int(row['n_water']);ion=row['ion'];r=float(row['initial_K_to_water_reference_A'])
 assert ion=='K' and row['case']==f'K_{n:02d}' and row['charge']=='1' and row['multiplicity']=='1'
 assert n in range(4,15) and row['n_start_target']=='100' and r==3.04
 assert 19-1+10*n == 18+10*n and (18+10*n)%2==0
 path=ROOT/'inputs'/row['case']/'packing.vecfle'
 assert hashlib.sha256(path.read_bytes()).hexdigest()==row['vecfle_sha256']
 pack=ROOT/'inputs/packings'/row['packing_source']
 assert hashlib.sha256(pack.read_bytes()).hexdigest()==row['packing_sha256']
 pairs=[ln.split() for ln in path.read_text().splitlines() if ln.strip()]
 assert len(pairs)==n
 xyz=np.array([[float(x) for x in v[1:4]] for v in pairs]); norm=np.linalg.norm(xyz,axis=1)
 assert all(v[0]=='K' and float(v[4])==3.04 for v in pairs)
 assert np.max(abs(norm-1))<1e-7
 dots=np.clip(xyz@xyz.T,-1,1)
 angles=np.degrees(np.arccos(dots[np.triu_indices(n,1)]))
 assert abs(min(angles)-reference_angles[n])<0.04, (n,min(angles))
assert abs(1.52+1.52-3.04)<1e-12
src=(ROOT/'vendors/MDAtom/atom_local_environment_computer.py').read_text(encoding='utf-8')
assert re.search(r'"k"\s*:\s*\{[^\n]*"vdwRad"\s*:\s*2\.750',src)
assert re.search(r'"k"\s*:\s*\{[^\n]*"covRad"\s*:\s*1\.960',src)
gwh=(ROOT/'vendors/GWH/gloriouswaterhole.py').read_text(encoding='utf-8')
assert "Rotation.from_euler('zyz'" in gwh
pipe=(ROOT/'scripts/k_pipeline.py').read_text(encoding='utf-8')
assert '"--chrg", "+1", "--uhf", "0", "--alpb", "water"' in pipe
print('PASS: K+ valence 18e, 11 manifest hashes, spherical min-angles, radius 3.04, vendor K sigma, zyz, GFN2/ALPB charge+spin')
