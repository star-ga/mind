#!/usr/bin/env python3
"""Author: STARGA Inc. Exact four-argument I/O parity and raw-ABI controls.

Enumerate before execution. All executable assertions are MIND. This Python
runner only builds, isolates inputs/files, records outcomes and compares them.
A baseline run records predeclared old native defects; it is not qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from self_host_native_cf_smoke import v2_header, image

OFFSETS = [('sub', '0 - 1', True), ('negative', '-1', True),
           ('parentheses', '(0 - 1)', True), ('fold', '1 - 2', True),
           ('computed', 'offset()', False), ('other_negative', '0 - 2', False),
           ('positioned', '0', False)]
PREFIX = 'fn offset() -> i64 { 0 - 1 }\n'

def cases():
    rows = []
    def add(name, source, rc=0, stdout=b'', stdin=b'', before=None, after=None,
            baseline=None, native_only=False, refuse=False):
        rows.append(dict(name=name, source=source, expected_exit=rc,
                         expected_stdout_hex=stdout.hex(), stdin_hex=stdin.hex(),
                         before=before or {}, after=after or {},
                         baseline_exit=rc if baseline is None else baseline,
                         native_only=native_only, refuse=refuse))
    for op in ['read', 'write']:
        for spelling, off, old_raw in OFFSETS:
            for guard, buf, count, want in [('invalid_fd','p','1',1), ('null','0','1',0),
                                            ('zero','p','0',0), ('negative_count','p','0 - 1',0)]:
                src = PREFIX + f'fn main() -> i64 {{ let p:i64=__mind_alloc(8); let got:i64=__mind_{op}(0 - 1,{buf},{count},{off}); return 0 - got; }}\n'
                add(f'{op}_{spelling}_{guard}', src, want, baseline=9 if old_raw else want)
    for spelling, off in [('literal','0 - 1'), ('computed','offset()')]:
        common = '''let fds:i64=__mind_alloc(8); if __mind_sys_pipe2(fds,2048)!=0 {return 90;}
 let rd:i64=__mind_load_i32(fds); let wr:i64=__mind_load_i32(fds+4);
 let p:i64=__mind_alloc(8192); let mut i:i64=0;
 while i<8192 {__mind_store_i8(p+i,65);i=i+1;}
'''
        close = '__mind_sys_close(rd);__mind_sys_close(wr);'
        fill = '''let mut fill:i64=1;
 while fill>0 {fill=__mind_write(wr,p,4096,0 - 2);}
 if fill!=0 - 1 {return 91;}
'''
        add('eagain_read_'+spelling, PREFIX+'fn main()->i64 {\n'+common+
            f'let got:i64=__mind_read(rd,p,1,{off});'+close+'return 0-got;}',
            1, baseline=11 if spelling=='literal' else 1)
        add('eagain_write_'+spelling, PREFIX+'fn main()->i64 {\n'+common+fill+
            f'let got:i64=__mind_write(wr,p,1,{off});'+close+'return 0-got;}',
            1, baseline=11 if spelling=='literal' else 1)
        add('short_write_'+spelling, PREFIX+'fn main()->i64 {\n'+common+fill+
            'if __mind_read(rd,p,4096,0 - 2)!=4096 {return 92;}\n'+
            f'let got:i64=__mind_write(wr,p,8192,{off});'+close+
            'if got!=4096 {return 93;} return 0;}')
        add('short_read_'+spelling, PREFIX+'fn main()->i64 {let p:i64=__mind_alloc(8);'+
            f'if __mind_read(0,p,8,{off})!=1 {{return 94;}}'+
            'if __mind_load_i8(p)!=65 {return 95;} if __mind_write(1,p,1,0 - 2)!=1 {return 96;} return 0;}',
            stdout=b'A', stdin=b'A')
        add('eof_'+spelling, PREFIX+'fn main()->i64 {let p:i64=__mind_alloc(8);'+
            f'if __mind_read(0,p,8,{off})!=0 {{return 97;}} return 0;}}')
        add('stream_write_'+spelling, PREFIX+'fn main()->i64 {let p:i64=__mind_alloc(8);'+
            '__mind_store_i8(p,65);__mind_store_i8(p+1,66);__mind_store_i8(p+2,67);'+
            f'if __mind_write(1,p,3,{off})!=3 {{return 98;}} return 0;}}',stdout=b'ABC')
    add('positioned_cursor_short', '''fn main()->i64 {
 let path:i64=__mind_alloc(8);__mind_store_i64(path,111516299194217);
 let fd:i64=__mind_sys_openat(0 - 100,path,2,0);if fd<0 {return 100;}
 let p:i64=__mind_alloc(8);__mind_store_i8(p,90);
 if __mind_sys_lseek(fd,1,0)!=1 {return 101;}
 if __mind_write(fd,p,1,1)!=1 {return 102;}
 if __mind_sys_lseek(fd,0,1)!=1 {return 103;}
 if __mind_read(fd,p,8,3)!=1 {return 104;}
 if __mind_load_i8(p)!=68 {return 105;}
 if __mind_sys_lseek(fd,0,1)!=1 {return 106;}
 if __mind_read(fd,p,1,0 - 1)!=1 {return 107;}
 if __mind_load_i8(p)!=90 {return 108;}
 if __mind_sys_lseek(fd,0,1)!=2 {return 109;}
 if __mind_write(1,p,1,0 - 1)!=1 {return 110;}
 if __mind_sys_close(fd)!=0 {return 111;} return 0;
}''',stdout=b'Z',before={'iofile':'41424344'},after={'iofile':'415a4344'})
    add('raw_errno_preserved','fn main()->i64 {if __mind_sys_close(0 - 1)!=0 - 9 {return 112;} return 0;}')
    for op in ['read','write']:
        add('legacy_three_'+op, f'fn main()->i64 {{let p:i64=__mind_alloc(8); return 0 - __mind_{op}(0 - 1,p,1);}}',9,native_only=True)
        for n in [0,1,2,5]:
            args=','.join('0' for _ in range(n))
            add(f'arity_{op}_{n}',f'fn main()->i64 {{return __mind_{op}({args});}}',refuse=True)
    for row in rows:
        row['source_sha256']=hashlib.sha256(row['source'].encode()).hexdigest()
    return rows

def limit():
    resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(100,110))

def context_cases():
    """Additional packet; preserve the original 80-case contract unchanged."""
    rows=[]
    for op in ['read','write']:
        for n in [0,5]:
            call=f'__mind_{op}('+','.join('0' for _ in range(n))+')'
            bodies=[('statement',call+'; return 0;'),
                    ('let','let got:i64='+call+'; return got;'),
                    ('nested','return 1 + '+call+';'),
                    ('condition','if '+call+'==0 {return 0;} return 1;')]
            for shape,body in bodies:
                source='fn main()->i64 {'+body+'}\n'
                rows.append(dict(name=f'arity_{op}_{n}_{shape}',source=source,
                                 source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                 refuse=True))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--enumerate',action='store_true');p.add_argument('--baseline',action='store_true')
    p.add_argument('--contexts',action='store_true',help='separate 16-case malformed-call context supplement')
    p.add_argument('--native',type=Path);p.add_argument('--mlir',type=Path);p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();e=a.evidence;e.mkdir(parents=True,exist_ok=True)
    data=context_cases() if a.contexts else cases();assert len(data)==(16 if a.contexts else 80)
    packet=json.dumps(data,indent=2)+'\n';manifest=e/'cases.json'
    if a.enumerate:
        manifest.write_text(packet);print('PRELISTED',len(data),hashlib.sha256(packet.encode()).hexdigest());return 0
    assert manifest.read_text()==packet,'enumeration drift';assert a.native and a.mlir
    records=[]
    def run(name,cmd,stdin=b'',cwd=None):
        q=subprocess.run(list(map(str,cmd)),input=stdin,capture_output=True,cwd=cwd or e,timeout=150,preexec_fn=limit)
        (e/(name+'.stdout')).write_bytes(q.stdout);(e/(name+'.stderr')).write_bytes(q.stderr)
        rec=dict(name=name,argv=list(map(str,cmd)),exit=q.returncode,stdout_sha256=hashlib.sha256(q.stdout).hexdigest(),stderr_sha256=hashlib.sha256(q.stderr).hexdigest())
        (e/(name+'.json')).write_text(json.dumps(rec,indent=2)+'\n');records.append(rec);return q
    failures=[]
    for c in data:
        name=c['name'];src=c['source'].encode();assert hashlib.sha256(src).hexdigest()==c['source_sha256']
        (e/(name+'.mind')).write_bytes(src)
        nb=run(name+'.native-build',[a.native],v2_header()+image(src))
        if c['refuse']:
            if a.baseline:
                # Observe malformed-input acceptance without executing it.
                print('BASELINE arity',name,'exit',nb.returncode,'bytes',len(nb.stdout),flush=True)
                continue
            ok=nb.returncode in (2,3) and nb.stderr and not nb.stdout
            if not ok:failures.append((name,'native refusal',nb.returncode,len(nb.stdout),nb.stderr[-120:]))
            else:print('PASS',name,'diagnosed refusal',flush=True)
            continue
        if nb.returncode!=0 or not nb.stdout.startswith(b'\x7fELF') or nb.stderr:
            failures.append((name,'native build',nb.returncode,nb.stderr[-120:]));continue
        ne=e/(name+'.native.elf');ne.write_bytes(nb.stdout);ne.chmod(0o500)
        legs=[('native',ne)]
        if not c['native_only']:
            me=e/(name+'.mlir.elf');mb=run(name+'.mlir-build',[a.mlir,'build',e/(name+'.mind'),'--backend','mlir','--release','--out',me])
            if mb.returncode!=0 or not me.exists():failures.append((name,'MLIR build',mb.returncode,mb.stderr[-120:]));continue
            legs.append(('mlir',me))
        observed=[]
        for leg,exe in legs:
            work=e/(name+'.'+leg+'.work');work.mkdir(exist_ok=True)
            for f,b in c['before'].items():(work/f).write_bytes(bytes.fromhex(b))
            q=run(name+'.'+leg+'-run',[exe],bytes.fromhex(c['stdin_hex']),work)
            files={x.name:x.read_bytes().hex() for x in work.iterdir() if x.is_file()}
            got=(q.returncode,q.stdout,q.stderr,files)
            expected_rc=c['baseline_exit'] if a.baseline and leg=='native' else c['expected_exit']
            wanted=(expected_rc,bytes.fromhex(c['expected_stdout_hex']),b'',c['after'])
            if got!=wanted:failures.append((name,leg,got,wanted))
            observed.append(got)
        if len(observed)==2 and not a.baseline and observed[0]!=observed[1]:failures.append((name,'backend mismatch'))
        print(('BASELINE' if a.baseline else 'CHECK'),name,observed[0][0],flush=True)
    (e/'results.json').write_text(json.dumps(records,indent=2)+'\n')
    (e/'failures.txt').write_text(repr(failures)+'\n')
    label='baseline observation' if a.baseline else ('malformed-call context refusal' if a.contexts else 'four-argument native/MLIR I/O parity')
    print('FAIL' if failures else 'PASS',label,len(data),'cases',flush=True)
    return int(bool(failures))
if __name__=='__main__':raise SystemExit(main())
