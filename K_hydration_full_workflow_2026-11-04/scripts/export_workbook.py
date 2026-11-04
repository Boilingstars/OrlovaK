#!/usr/bin/env python3
"""Export verified K+ MDAtom data to a CSV; never overwrite other researchers' sheets.

The optional Excel template runs/MDAtom_water_TEMPLATE.xlsx is static and contains
no fake quantum-chemical results. Import analysis/ion_tables/K.csv into it manually.
"""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
HEAD=['n','structure','energy_hartree','radius_mdatom_A','cos_x','water_cn']

def main():
    rows=[]
    for n in range(4,15):
        base=ROOT/'runs'/f'K_{n:02d}'/'full_100'
        meta=base/'verified'/'free_sorted_metadata.csv'
        water=base/'mdatom'/'mdatom_counts.csv'
        if not meta.is_file() or not water.is_file():continue
        with meta.open(encoding='utf-8',newline='') as fh:en=list(csv.DictReader(fh))
        with water.open(encoding='utf-8',newline='') as fh:md=list(csv.DictReader(fh))
        if len(en)!=len(md):raise RuntimeError(f'n={n}: MDAtom and energy rows differ')
        for e,m in zip(en,md):
            if int(e['sorted_frame']) != int(m['sorted_frame']):
                raise RuntimeError(f'n={n}: sorted_frame misalignment')
            rows.append([n,e['source'],e['energy_hartree'],m['radius_mdatom_A'],
                         m['orientation_cos_mdatom'],m['coordination_mdal']])
    output=ROOT/'analysis'/'ion_tables'/'K.csv'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8-sig',newline='') as fh:
        writer=csv.writer(fh);writer.writerow(HEAD);writer.writerows(rows)
    print(f'Exported {len(rows)} real K entries to {output}; XLSX template left untouched')
if __name__=='__main__':main()
