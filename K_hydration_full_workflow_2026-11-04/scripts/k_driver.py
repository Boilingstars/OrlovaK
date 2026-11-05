#!/usr/bin/env python3
"""Sequential, resumable K+(H2O)n research driver. Never manufactures results."""
from __future__ import annotations
import argparse,csv,json,os,subprocess,sys,time,zipfile
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PIPE=ROOT/'scripts/k_pipeline.py'
RUN=ROOT/'runs'
LOG=RUN/'full_100.log'
LOCK=RUN/'k_driver.lock'
JOURNAL=ROOT/'EXPERIMENT_LOG.md'


def emit(msg):
    ts=datetime.now().astimezone().isoformat(timespec='seconds')
    entry=f'{ts} {msg}'
    print(entry,flush=True)
    RUN.mkdir(parents=True,exist_ok=True)
    with LOG.open('a',encoding='utf-8') as f:f.write(entry+'\n')


def active_pid(pid):
    if pid<=0:return False
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False
    except PermissionError:return True
    except OSError:return False


def acquire():
    RUN.mkdir(exist_ok=True)
    for _ in range(2):
        try:
            fd=os.open(str(LOCK),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
            with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(str(os.getpid()))
            return
        except FileExistsError:
            try:pid=int(LOCK.read_text().strip())
            except (ValueError,OSError):pid=0
            if active_pid(pid):raise RuntimeError(f'Another K driver is running: PID={pid}. Do not start a duplicate.')
            try:LOCK.unlink()
            except FileNotFoundError:pass
    raise RuntimeError('Could not acquire single driver lock')


def release():
    try:
        if LOCK.is_file() and LOCK.read_text().strip()==str(os.getpid()):LOCK.unlink()
    except OSError:pass


def journal(n,stage,status,details=''):
    if not JOURNAL.is_file():return
    text=JOURNAL.read_text(encoding='utf-8')
    # The journal must keep all earlier entries and only update the status matrix.
    lines=text.splitlines()
    cols={'gwh':2,'collect':2,'relax':3,'export':4,'characterize':5,'mdatom':6,'select':7,'assemble':8}
    # Existing summary table has: n|GWH|free+tight|Hessian|CoCoS|MDAtom|CN=n|Export.
    idx={'gwh':1,'relax':2,'export':3,'characterize':4}.get(stage)
    if idx is not None:
        for i,line in enumerate(lines):
            cells=[s.strip() for s in line.split('|')[1:-1]]
            if len(cells)==8 and cells[0]==str(n):
                if stage=='gwh':cells[1]='DONE 100/100' if status=='DONE' else status
                elif stage=='relax':cells[2]='DONE 100/100' if status=='DONE' else status
                elif stage=='export':cells[3]='DONE' if status=='DONE' else status
                elif stage=='characterize':
                    for k in (4,5,6):
                        if not (status=='DONE' and cells[k].startswith('DONE')): cells[k]='DONE' if status=='DONE' else status
                lines[i]='| '+' | '.join(cells)+' |';break
    addition=f'\n### {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")}: n={n}, {stage}\n- **Статус:** {status}. {details}\n'
    JOURNAL.write_text('\n'.join(lines)+'\n'+addition,encoding='utf-8')


def metrics(n,stage):
    p=RUN/f'K_{n:02d}'/'full_100'
    if stage=='collect':
        with (p/'energies.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        e=min(float(r['energy_hartree']) for r in rows)
        return f'100/100; Emin_GWH={e:.7f} Ha'
    if stage=='export':
        lines=(p/'verified/INCOMPLETE.txt').read_text(encoding='utf-8').splitlines()
        m=dict(t.split('=',1) for t in lines[:3] if '=' in t)
        return f"minima={m.get('VERIFIED_FREE_MINIMA','?')}; excluded={m.get('EXCLUDED_OR_UNRESOLVED_STARTS','?')}"
    if stage=='characterize':
        v=json.loads((p/'final/RESULTS_STATUS.json').read_text(encoding='utf-8'))
        return f"CoCoS={v['cocos_groups_total']}; CN=n groups={v['final_representatives']}"
    if stage=='gwh':
        return '100/100 successful starts (individual GWH energies in terminal and KV)'
    return 'completed'


def command(n,stage,args):
    base=[sys.executable,str(PIPE),stage,'--n',str(n),'--tag','full_100']
    if stage=='gwh':return base+['--numstruct','100','--radius','3.04','--workers',str(args.workers),'--threads','1','--gwh',str(ROOT/'vendors/GWH/gloriouswaterhole.py')]
    if stage=='relax':return base+['--workers',str(args.relax_workers),'--threads',str(args.threads)]
    if stage=='characterize':return base+['--cocos',str(ROOT/'vendors/CoCoS/cocos.py'),'--mdatom',str(ROOT/'vendors/MDAtom/atom_local_environment_computer.py'),'--scale',str(args.scale),'--cn-tol',str(args.cn_tol),'--critfreq',str(args.critfreq)]
    return base


def call(cmd,dry_run=False):
    print('$ '+' '.join(str(a) for a in cmd),flush=True)
    if dry_run:return 0
    return subprocess.run(cmd,cwd=ROOT,check=False).returncode


def run_stage(n,stage,args):
    cmd=command(n,stage,args)
    for k in range(1,args.retries+2):
        emit(f'BEGIN n={n} {stage} attempt={k}')
        rc=call(cmd,args.dry_run)
        if rc==0:
            descr='DRY_RUN' if args.dry_run else metrics(n,stage)
            emit(f'END n={n} {stage} {descr}')
            if not args.dry_run:journal(n,stage,'DONE',descr)
            return True
        emit(f'ERROR n={n} {stage} exit={rc}, retry {k}/{args.retries+1}')
        journal(n,stage,'FAILED',f'Код {rc}; детали в runs/full_100.log и журналах стадии. Сохраняются успешные результаты.')
        if k<=args.retries:time.sleep(min(10,2*k))
    return False


def after_characterize(n):
    # Refresh the workbook using KV provenance, without changing individual-ion inputs.
    c=[sys.executable,str(ROOT/'scripts/export_workbook.py')]
    try:
        rc=subprocess.run(c,cwd=ROOT,check=False).returncode
        if rc:emit(f'WARNING workbook export after n={n}: exit {rc} (science results retained)')
    except Exception as ex:emit(f'WARNING workbook export after n={n}: {ex}')


def _xtb_process_count():
    """Count foreign xtb executables. Never used to kill anything."""
    if os.name == 'nt':
        proc=subprocess.run(
            ['powershell','-NoProfile','-Command',
             '@(Get-Process -Name xtb -ErrorAction SilentlyContinue).Count'],
            capture_output=True,text=True,timeout=25,check=False)
        if proc.returncode != 0:
            raise RuntimeError('xtb process query failed')
        return int(proc.stdout.strip().splitlines()[-1])
    proc=subprocess.run(['pgrep','-x','xtb'],capture_output=True,text=True,timeout=25,check=False)
    if proc.returncode not in (0,1):
        raise RuntimeError('xtb process query failed')
    pids=[p for p in proc.stdout.split() if p.isdigit()]
    return len(pids)


def wait_for_idle_xtb():
    """Defer initial heavy launch while another xTB executable is already running.
    This NEVER terminates another process; checking resource availability is best effort.
    """
    announced=False
    while True:
        try:
            count=_xtb_process_count()
            if count==0:return
            if not announced:
                emit(f'Other xTB processes active: {count}; defer heavy launch without terminating them')
                announced=True
            time.sleep(60)
        except (ValueError, subprocess.TimeoutExpired, IndexError, RuntimeError):
            emit('WARNING: could not confirm machine idle; inspect running xTB processes')
            return


def smoke_test(args):
    """Small real n=4 pilot to verify all external programs, no synthetic chemistry."""
    tag='smoke_3'; n=4; base=RUN/'K_04'/tag
    marker=base/'SMOKE_VERIFIED.json'
    if marker.is_file():
        emit('SMOKE preflight previously verified; reusing it')
        return
    commands=[
        [sys.executable,str(PIPE),'gwh','--n','4','--tag',tag,'--numstruct','3','--radius','3.04',
         '--workers','1','--threads','1','--gwh',str(ROOT/'vendors/GWH/gloriouswaterhole.py')],
        [sys.executable,str(PIPE),'collect','--n','4','--tag',tag],
        [sys.executable,str(PIPE),'relax','--n','4','--tag',tag,'--workers','1','--threads','1'],
    ]
    for cmd in commands:
        emit('SMOKE '+cmd[2])
        if call(cmd):raise RuntimeError('Pilot failed; diagnose and restart. No data fabricated.')
    statuses=sorted((base/'free').glob('frame_*/status.json'))
    if len(statuses)!=3:
        raise RuntimeError(f'Smoke preflight expected 3 free/Hessian results, got {len(statuses)}')
    minima=0
    for path in statuses:
        data=json.loads(path.read_text(encoding='utf-8'))
        state=data.get('final',{}).get('status','UNKNOWN')
        if state not in ('MINIMUM','IMAGINARY'):
            raise RuntimeError(f'Smoke preflight incomplete: {path} = {state}')
        minima+=int(state=='MINIMUM')
    # An all-saddle three-structure pilot is scientifically valid; do not fail.
    # If minima exist, also test the export stage, otherwise CoCoS/MDAtom are first tested with full_100.
    if minima:
        if call([sys.executable,str(PIPE),'export','--n','4','--tag',tag]):
            raise RuntimeError('Pilot verified minima exist, but export failed')
    marker.write_text(json.dumps({'status':'preflight_ok','n_starts':3,'n_minima':minima,
        'note':'Pilot verifies GWH and free normal/tight/Hessian, export if minima; it does not prove CoCoS/MDAtom stages.'},indent=2),encoding='utf-8')
    emit(f'SMOKE preflight passed; minimum={minima} / 3 (zero is allowed)')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--status',action='store_true')
    ap.add_argument('--analysis-only',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--workers',type=int,default=5)
    ap.add_argument('--relax-workers',type=int,default=4)
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--retries',type=int,default=2)
    ap.add_argument('--scale',type=float,default=1.0)
    ap.add_argument('--cn-tol',type=float,default=0.15)
    ap.add_argument('--critfreq',type=float,default=125.0,help='CoCoS: exploratory increased frequency vs native 83.4 cm^-1')
    ap.add_argument('--skip-smoke',action='store_true', help='Only if a real first-use smoke test was previously validated')
    args=ap.parse_args()
    if args.status:
        raise SystemExit(subprocess.run([sys.executable,str(PIPE),'status','--tag','full_100'],cwd=ROOT).returncode)
    acquire()
    try:
        emit('K series driver: one process, n=4..14, no imaginary-mode displacement')
        if not args.dry_run and not args.analysis_only:
            wait_for_idle_xtb()
        if not args.analysis_only:
            if call([sys.executable,str(PIPE),'check'],args.dry_run):raise RuntimeError('Input checks failed')
            if not args.dry_run and not args.skip_smoke:smoke_test(args)
        failures=[]
        stages=(['characterize'] if args.analysis_only else ['gwh','collect','relax','export','characterize'])
        for n in range(4,15):
            for stage in stages:
                if not run_stage(n,stage,args):
                    failures.append((n,stage))
                    # We cannot characterize an incomplete n. Other n continue in the same process.
                    break
            else:
                if not args.dry_run:after_characterize(n)
        if failures and not args.dry_run:
            emit(f'RETRY_BACKLOG failed stage entries={len(failures)}; resuming without repeating verified computations')
            backlog=[]
            for n,first in failures:
                # Resume from first unsuccessful stage. Earlier stages are already complete.
                for stage in stages[stages.index(first):]:
                    if not run_stage(n,stage,args):
                        backlog.append((n,stage))
                        break
                else:
                    after_characterize(n)
            failures=backlog
        if failures:
            emit('UNRESOLVED stages: '+repr(failures)+'; keep verified results, repair and resume same command')
            if not args.dry_run:
                # Preserve useful partial data for review, without falsely declaring full success.
                rc=call([sys.executable,str(ROOT/'scripts/k_coordination_review.py'), '--tag','full_100'])
                if rc:emit('WARNING: partial coordination review failed')
                try:
                    rc=call([sys.executable,str(ROOT/'scripts/k_assemble.py'), '--tag','full_100','--allow-incomplete'])
                    if rc:emit('WARNING: partial assembly failed')
                except Exception as e:
                    emit(f'WARNING: partial assembly not available: {e}')
            raise SystemExit(2)
        if not args.dry_run:
            after_characterize(14)
            # Human-review candidates from ALL confirmed geometries; CoCoS not a filter.
            rc=call([sys.executable,str(ROOT/'scripts/k_coordination_review.py'),'--tag','full_100'])
            if rc:emit('WARNING geometry review not finished; quantum-chemical results retained')
            emit('ASSEMBLE')
            rc=call([sys.executable,str(ROOT/'scripts/k_assemble.py'),'--tag','full_100'])
            if rc:raise SystemExit(rc)
            lines=JOURNAL.read_text(encoding='utf-8').splitlines()
            for i,line in enumerate(lines):
                cells=[c.strip() for c in line.split('|')[1:-1]]
                if len(cells)==8 and cells[0].isdigit() and 4<=int(cells[0])<=14:
                    cells[7]='DONE'
                    lines[i]='| '+' | '.join(cells)+' |'
            JOURNAL.write_text('\n'.join(lines)+'\n\n## Итог\nСерия n=4–14 завершена; реальные сводки в `K_SUMMARY.csv` и `analysis/ion_tables/K.csv`.\n',encoding='utf-8')
        emit('SERIES_DONE' if not args.dry_run else 'DRY_RUN_DONE')
    finally:release()

if __name__=='__main__':main()
