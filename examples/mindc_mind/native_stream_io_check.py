#!/usr/bin/env python3
"""Source-lowered MIND stream-I/O diagnostic transport, not compiler admission.

Build the two real MIND source-image drivers with the adjacent context harness
--build-only. This runner requires their exact hashes and the captured prefix.
Executable fixtures are MIND. Python only supplies bytes and observes processes.
"""
import argparse, hashlib, json, os, resource, subprocess, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
CASES=HERE/'native_stream_io_cases.json'
def sha(b):return hashlib.sha256(b).hexdigest()
def limits():
    resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(20,21))
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('native','cpu','prefix','evidence'):ap.add_argument('--'+name,type=Path,required=True)
    for name in ('native','cpu','prefix','cases'):ap.add_argument('--'+name+'-sha256',required=True)
    ap.add_argument('--cases',type=Path,default=CASES,help='explicit versioned catalogue; the original contract remains the default')
    ap.add_argument('--reference',type=Path);ap.add_argument('--reference-sha256')
    a=ap.parse_args();e=a.evidence.resolve();e.mkdir(parents=True,exist_ok=False)
    files=[a.native.resolve(),a.cpu.resolve(),a.prefix.resolve(),a.cases.resolve(),Path(__file__).resolve(),HERE/'self_host_native_emission_context_smoke.py']
    if a.reference:files.append(a.reference.resolve())
    captured={str(p):p.read_bytes() for p in files};inputs={p:sha(b) for p,b in captured.items()}
    for name in ('native','cpu','prefix'):
        assert inputs[str(getattr(a,name).resolve())]==getattr(a,name+'_sha256')
    assert inputs[str(a.cases.resolve())]==a.cases_sha256
    if a.reference:assert inputs[str(a.reference.resolve())]==a.reference_sha256
    (e/'INPUTS.json').write_text(json.dumps(inputs,indent=2)+'\n')
    raw=captured[str(a.cases.resolve())];(e/'CASES.json').write_bytes(raw);(e/'CASES.json').chmod(0o444)
    import self_host_native_emission_context_smoke as base
    prefix=captured[str(a.prefix.resolve())];cases=json.loads(raw)['cases'];plans=[]
    for row in cases:
        src=prefix+row['source'].encode()
        for geo in row['geometries']:
            for route in row['routes']:
                name=row['id']+'-'+geo+'-'+route
                wire=base.packet(10,source=src,user=len(prefix),route=int(route=='force-home'),request=base.GEOMETRIES[geo])
                (e/(name+'.packet')).write_bytes(wire)
                plans.append(dict(id=name,row=row['id'],geometry=geo,route=route,packet_sha256=sha(wire)))
    (e/'PLAN.json').write_text(json.dumps(plans,indent=2)+'\n')
    env={k:v for k,v in os.environ.items() if not k.startswith('LD_') and k not in ('GLIBC_TUNABLES','MINDC_NATIVE_ELF','MINDC_SO')}
    env.update(LC_ALL='C',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    receipts=[];results=[]
    def run(label,exe,data):
        start=time.monotonic();timeout=False
        try:q=subprocess.run([str(exe)],input=data,capture_output=True,env=env,preexec_fn=limits,timeout=30);rc,so,se=q.returncode,q.stdout,q.stderr
        except subprocess.TimeoutExpired as ex:rc,so,se,timeout=None,ex.stdout or b'',ex.stderr or b'',True
        (e/(label+'.stdout')).write_bytes(so);(e/(label+'.stderr')).write_bytes(se)
        item=dict(label=label,executable=str(exe),stdin_sha256=sha(data),exit=rc,timeout=timeout,stdout_sha256=sha(so),stderr_sha256=sha(se),seconds=time.monotonic()-start)
        receipts.append(item);(e/'executions.json').write_text(json.dumps(receipts,indent=2)+'\n');return rc,so,se,timeout
    byid={r['id']:r for r in cases}
    for plan in plans:
        row=byid[plan['row']];name=plan['id'];wire=(e/(name+'.packet')).read_bytes();assert sha(wire)==plan['packet_sha256'];outputs=[]
        for label,driver in [('native',a.native),('cpu',a.cpu)]:
            rc,so,se,to=run(name+'-'+label+'-compile',driver,wire);outputs.append(so);good=rc==0 and not se and not to
            if row['expected']=='refuse':good=good and so==__import__('struct').pack('<4q',*row['diagnostic'])
            else:
                good=good and so[:32]==bytes(32) and so[32:36]==b'\x7fELF'
                if good:
                    elf=e/(name+'-'+label+'.elf');elf.write_bytes(so[32:]);elf.chmod(0o700)
                    er,eo,ee,et=run(name+'-'+label+'-runtime',elf,bytes.fromhex(row['stdin_hex']))
                    good=er==row['exit'] and eo==bytes.fromhex(row['stdout_hex']) and ee==bytes.fromhex(row['stderr_hex']) and not et
            results.append(dict(id=name,backend=label,expected=row['expected'],pass_=bool(good)))
        results.append(dict(id=name,backend='parity',pass_=outputs[0]==outputs[1]))
        if row['id']=='no-io' and a.reference:
            rr,ro,re,rt=run(name+'-historical-reference',a.reference,wire)
            results.append(dict(id=name,backend='historical-byte-preservation',pass_=rr==0 and not re and not rt and ro==outputs[0]))
        (e/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    for p,h in inputs.items():assert sha(Path(p).read_bytes())==h,('input drift',p)
    summary=dict(cases=len(cases),packets=len(plans),observations=len(results),passed=sum(x['pass_'] for x in results),failed=[x for x in results if not x['pass_']],processes=len(receipts),authority='development diagnostic only; no qualified pin or consumer acceptance')
    (e/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return int(bool(summary['failed']))
if __name__=='__main__':raise SystemExit(main())
