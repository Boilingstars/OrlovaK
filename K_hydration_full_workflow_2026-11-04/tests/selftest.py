"""Non-quantum unit tests; do NOT interpret these fixtures as K hydration results."""
import importlib.util,json,math,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def mod(name):
    spec=importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    a=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    return a

p=mod('k_pipeline')
v=mod('validate_k_shell')
for n in range(4,15):
    vec=p.read_vecfle(ROOT/f'inputs/K_{n:02d}/packing.vecfle')
    assert len(vec)==n
    assert all(el=='K' and abs(r-3.04)<1e-9 and abs(float((u*u).sum())-1)<1e-7 for el,u,r in vec)
with tempfile.TemporaryDirectory() as td:
    path=Path(td)/'g98.out'
    path.write_text(''.join(' Frequencies -- '+ ' '.join(f'{20+3*i+j:.3f}' for j in range(3))+'\n' for i in range(11)))
    freq=p.read_internal_frequencies(path,13)
    assert len(freq)==33 and min(freq)>0
    res=Path(td)/'local_surrounding_water_anlysis_K1.res'
    res.write_text('# time radius[A] cos(x) #water\n0.00 3.1534 -0.512 5.5\n1.00 3.1402 -0.501 6.0\n')
    parsed=p.parse_water_res(res)
    assert len(parsed)==2 and parsed[0][2]==5.5
    assert not p.cn_equals_n(parsed[0][2],6) and p.cn_equals_n(parsed[1][2],6)
    assert p.cn_equals_n(5.9,6) and not p.cn_equals_n(5.8,6)
    opts=p.build_parser().parse_args(['characterize','--n','6','--cocos','cocos.py','--mdatom','MDAtom.py'])
    assert opts.scale==1.0 and opts.cn_tol==0.15 and opts.critfreq==125.0
    assert abs(p.build_parser().parse_args(['cocos','--n','6','--cocos','cocos.py']).critfreq - 125.0) < 1.e-10
    assert '--CritFreq' in (ROOT/'vendors/CoCoS/cocos.py').read_text()
# Synthetic geometry used solely to check element-order-agnostic descriptor logic.
symbols=['K'];coords=[[0.,0.,0.]]
for u in ([1,0,0],[-1,0,0],[0,1,0],[0,-1,0]):
    o=[x*3.3 for x in u]
    symbols+=['O','H','H']
    coords += [o,[o[0]+.6,o[1]+.6,o[2]+.6],[o[0]-.6,o[1]-.6,o[2]+.6]]
import numpy as np
result=v.analyze_one(symbols,np.asarray(coords,float))
assert result['n']==4 and abs(result['mean_KO_A']-3.3)<1e-6 and result['cn_ko_3.4']==4
print('PASS: 11 K vecfle, 3N-6 frequencies, MDAtom fractional CN, K geometry')
