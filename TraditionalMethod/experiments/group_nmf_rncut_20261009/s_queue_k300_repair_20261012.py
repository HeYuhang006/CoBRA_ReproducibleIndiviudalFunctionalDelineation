"""One finite dependent queue: await original capture, then gated repair launcher."""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,sys,time
from s_diagnose_labelcost_20261011 import B,CODE,PY,save

D=B/'initialization_guard_20261012'

def main():
    D.mkdir(exist_ok=True)
    lock=(D/'dependent_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if '--run' not in sys.argv:
        assert not (D/'dependent_queue_launch.json').exists(),'Finite queue already launched'
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        with (D/'dependent_queue.log').open('ab',buffering=0) as f:
            p=subprocess.Popen([PY,str(Path(__file__).resolve()),'--run'],stdin=subprocess.DEVNULL,
                stdout=f,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
        save(D/'dependent_queue_launch.json',dict(pid=p.pid));print(json.dumps(dict(pid=p.pid)))
        return
    save(D/'dependent_queue_status.json',dict(state='waiting_for_original_capture',pid=os.getpid(),
         started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),maximum_wait_seconds=7200))
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        r=json.loads((B/'k300_context_20261012/receipt.json').read_text())
        if r['state']=='capture_finished':break
        time.sleep(30)
    else:
        save(D/'dependent_queue_status.json',dict(state='expired_without_restarting_capture'))
        return
    if (D/'receipt.json').exists():
        save(D/'dependent_queue_status.json',dict(state='existing_candidate_receipt_no_duplicate'))
        return
    p=subprocess.run([PY,str(CODE/'s_fix_k300_initialization_20261012.py')],stdin=subprocess.DEVNULL,
                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    save(D/'dependent_queue_status.json',dict(state='candidate_dispatched' if p.returncode==0 else 'launch_gate_rejected',
        returncode=p.returncode,output=p.stdout,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(p.stdout,flush=True)

if __name__=='__main__':main()
