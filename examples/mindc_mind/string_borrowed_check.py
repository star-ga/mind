#!/usr/bin/env python3
"""Source-bound borrowed String gate. MIND body and C shim are separate legs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import struct
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def sha(data): return hashlib.sha256(data).hexdigest()
def catalogue():
    rows=[]
    def add(name,mode,n,cap,b,capacity,prefix):
        rows.append(dict(name=name,words=[mode,n,cap,b],stdout=(struct.pack('<q',capacity)+prefix).hex()))
    for n,c in [(0,16),(1,16),(3,16),(15,16),(16,32),(17,32),(31,32),(32,64),(33,64),(255,256),(4096,8192)]:
        add('borrowed-'+str(n),1,n,0,66,c,b'a'*n+b'B')
    for n,c in [(0,0),(1,1),(3,8),(16,16),(32,32)]:
        want={(0,0):16,(1,1):2,(3,8):8,(16,16):32,(32,32):64}[(n,c)]
        add('owned-'+str(n)+'-'+str(c),2,n,c,66,want,b'a'*n+b'B')
    for n,c in [(3,16),(32,64)]: add('view-'+str(n),3,n,0,66,c,b'a'*n+b'B')
    for b in [0,255,256,-1]: add('byte-'+str(b),2,3,8,b,8,b'aaa'+bytes([b&255]))
    add('sequence',4,0,0,0,512,bytes(i&255 for i in range(257)))
    add('empty-nonnull',6,0,0,66,16,b'B')
    add('slice-negative',7,32,0,66,64,b'a'*32+b'B')
    add('slice-interior',8,32,0,66,64,b'a'*32+b'B')
    for name,n,c,want in [
        ('negative-len',-1,0,0),('negative-cap',0,-1,0),('owned-overrun',3,2,0),
        ('max-len',2**63-1,0,0),('max-owned',2**63-1,2**63-1,0),
        ('near-max-borrowed',2**63-2,0,2**63-1),('near-max-owned',2**63-2,2**63-2,2**63-1),
        ('spare-max',2**63-2,2**63-1,2**63-1),('half',2**62-1,2**62-1,2**63-2),
        ('above-half',2**62,2**62,2**62+1),('borrowed-half',2**62,0,2**62+1),
        ('empty',0,0,16),('borrowed16',16,0,32),('owned-spare',15,16,16)]:
        add('capacity-'+name,5,n,c,0,want,b'')
    return rows

DRIVER = r'''
fn sb_out(s:String)->i64 {
    let p:i64=__mind_alloc(8);
    __mind_store_i64(p,cow_string_cap(s));
    if __mind_write(1,p,8,0-1)!=8 {return 91;}
    if __mind_write(1,cow_string_addr(s),cow_string_len(s),0-1)!=cow_string_len(s) {return 92;}
    return 0;
}
fn sb_verify(s:String,n:i64,b:i64)->i64 {
    if cow_string_len(s)!=n+1 {return 80;}
    let mut i:i64=0;
    while i<n {if cow_string_get_byte(s,i)!=97 {return 81;} i=i+1;}
    if cow_string_get_byte(s,n)!=(b&255) {return 82;}
    return 0;
}
fn main()->i64 {
    let p:i64=__mind_alloc(32);
    if __mind_read(0,p,32,0-1)!=32 {return 90;}
    let mode:i64=__mind_load_i64(p);
    let n:i64=__mind_load_i64(p+8);
    let cap:i64=__mind_load_i64(p+16);
    let b:i64=__mind_load_i64(p+24);
    if mode==5 {
        let c:i64=cow_string_push_capacity(n,cap);
        __mind_store_i64(p,c);
        if __mind_write(1,p,8,0-1)!=8 {return 91;}
        return 0;
    }
    if mode==4 {
        let mut s:String=cow_string_new();
        let mut i:i64=0;
        while i<257 {s=cow_string_push_byte(s,i); i=i+1;}
        i=0;
        while i<257 {if cow_string_get_byte(s,i)!=(i&255) {return 84;} i=i+1;}
        return sb_out(s);
    }
    if mode==6 {
        let s:String=String{addr:1,len:0,cap:0};
        let t:String=cow_string_push_byte(s,b);
        if cow_string_addr(t)==1 {return 85;}
        return sb_out(t);
    }
    if mode==1 {
        let original:String="LONG_LITERAL";
        let borrowed:String=String{addr:cow_string_addr(original),len:n,cap:0};
        let t:String=cow_string_push_byte(borrowed,b);
        let mut i:i64=0;
        while i<4096 {if cow_string_get_byte(original,i)!=97 {return 83;} i=i+1;}
        if cow_string_addr(t)==cow_string_addr(original) {return 85;}
        let rc:i64=sb_verify(t,n,b);
        if rc!=0 {return rc;}
        return sb_out(t);
    }
    if mode==2 {
        let a:i64=__mind_alloc(cap+8);
        let mut i:i64=0;
        while i<cap+8 {__mind_store_i8(a+i,97); i=i+1;}
        let s:String=String{addr:a,len:n,cap:cap};
        let t:String=cow_string_push_byte(s,b);
        if n<cap {
            if cow_string_addr(t)!=a {return 85;}
            i=n+1;
            while i<cap+8 {if __mind_load_i8(a+i)!=97 {return 86;} i=i+1;}
        }
        let rc:i64=sb_verify(t,n,b);
        if rc!=0 {return rc;}
        return sb_out(t);
    }
    if mode==3 || mode==7 || mode==8 {
        let a:i64=__mind_alloc(80);
        let mut i:i64=0;
        while i<80 {__mind_store_i8(a+i,97); i=i+1;}
        let parent:String=String{addr:a,len:40,cap:80};
        let mut s:String=String{addr:a+8,len:n,cap:0};
        if mode==7 {
            let small:String=String{addr:a,len:32,cap:80};
            s=cow_string_slice_from(small,0-1);
        }
        if mode==8 {s=cow_string_slice(parent,8,40);}
        let t:String=cow_string_push_byte(s,b);
        i=0;
        while i<80 {if __mind_load_i8(a+i)!=97 {return 83;} i=i+1;}
        let rc:i64=sb_verify(t,n,b);
        if rc!=0 {return rc;}
        return sb_out(t);
    }
    return 93;
}
'''.replace('LONG_LITERAL','a'*4096)

def embedded(source):
    # Every public String helper is renamed, so no weak C compatibility symbol
    # can replace the MIND implementation. The String layout itself is unchanged.
    return re.sub(r'\bstring_', 'cow_string_', source.decode()).encode()+DRIVER.encode()

def limits():
    resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

def run(ev,name,args,data=None):
    t=time.monotonic()
    p=subprocess.run(list(map(str,args)),input=data,capture_output=True,cwd=ev,timeout=300,preexec_fn=limits)
    (ev/(name+'.stdout')).write_bytes(p.stdout);(ev/(name+'.stderr')).write_bytes(p.stderr)
    (ev/(name+'.json')).write_text(json.dumps(dict(argv=list(map(str,args)),exit=p.returncode,seconds=time.monotonic()-t,stdin_sha256=sha(data) if data is not None else None,stdout_sha256=sha(p.stdout),stderr_sha256=sha(p.stderr)),indent=2)+'\n')
    return p

def build(ev,tag,src,seed,mindc):
    source=ev/(tag+'.mind');source.write_bytes(src)
    import sys
    sys.path.insert(0,str(HERE))
    from self_host_native_cf_smoke import v2_header
    packet=v2_header()+struct.pack('<qq',0,len(src))+src
    p=run(ev,tag+'-native-build',[seed],packet)
    assert p.returncode==0 and not p.stderr and p.stdout.startswith(b'\x7fELF'),(tag,p.returncode,p.stderr[-1000:])
    native=ev/(tag+'-native.elf');native.write_bytes(p.stdout);native.chmod(0o500)
    oracle=ev/(tag+'-mlir.elf')
    p=run(ev,tag+'-mlir-build',[mindc,'build',source,'--backend','mlir','--release','--out',oracle])
    assert p.returncode==0 and oracle.exists(),(tag,p.returncode,p.stderr[-1200:])
    return native,oracle

def mutation_specs():
    return [
        ('skip-copy', '__mind_store_i8(fresh + i, __mind_load_i8(s.addr + i));', '__mind_store_i8(fresh + i, 0);', 'borrowed-3'),
        ('fixed16', 'let needed: i64 = len + 1;', 'if cap == 0 { return 16; }\n    let needed: i64 = len + 1;', 'capacity-borrowed16'),
        ('borrowed-reuse', 'let fresh: i64 = __mind_alloc(new_cap);', 'let fresh: i64 = s.addr;', 'view-32'),
        ('wrong-tail', '__mind_store_i8(new_addr + s.len, b);', '__mind_store_i8(new_addr + s.len - 1, b);', 'owned-3-8'),
        ('overflow-double', 'next = needed;', 'return next * 2;', 'capacity-above-half'),
    ]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--evidence',required=True,type=Path);ap.add_argument('--enumerate',action='store_true');ap.add_argument('--seed',type=Path);ap.add_argument('--mindc',type=Path);ap.add_argument('--seed-sha256');ap.add_argument('--mindc-sha256')
    args=ap.parse_args();ev=args.evidence
    if args.enumerate:
        ev.mkdir(parents=True,exist_ok=False)
        data=(json.dumps(catalogue(),indent=2)+'\n').encode();(ev/'CASES.json').write_bytes(data);(ev/'CASES.json').chmod(0o444)
        (ev/'CASES.sha256').write_text(sha(data)+'\n')
        mutations=(json.dumps(mutation_specs(),indent=2)+'\n').encode()
        (ev/'MUTATIONS.json').write_bytes(mutations);(ev/'MUTATIONS.json').chmod(0o444)
        (ev/'MUTATIONS.sha256').write_text(sha(mutations)+'\n')
        print('FROZEN',len(catalogue()),sha(data),'mutations',sha(mutations));return
    if not args.seed or not args.mindc or not args.seed_sha256 or not args.mindc_sha256:
        ap.error('execution requires both compiler paths and expected SHA256 values')
    assert sha(args.seed.read_bytes())==args.seed_sha256, 'native compiler hash mismatch'
    assert sha(args.mindc.read_bytes())==args.mindc_sha256, 'CPU compiler hash mismatch'
    data=(ev/'CASES.json').read_bytes();assert sha(data)==(ev/'CASES.sha256').read_text().strip()
    assert json.loads(data)==catalogue(),'catalogue changed'
    std=(ROOT/'std/string.mind').read_bytes();src=embedded(std)
    (ev/'IDENTITIES.json').write_text(json.dumps({'std':sha(std),'embedded':sha(src),'harness':sha(Path(__file__).read_bytes()),'seed':sha(args.seed.read_bytes()),'mindc':sha(args.mindc.read_bytes())},indent=2)+'\n')
    native,oracle=build(ev,'fixed',src,args.seed,args.mindc)
    results=[]
    for row in catalogue():
        for leg,exe in [('native',native),('mlir',oracle)]:
            p=run(ev,row['name']+'-'+leg,[exe],struct.pack('<qqqq',*row['words']))
            ok=(p.returncode,p.stdout,p.stderr)==(0,bytes.fromhex(row['stdout']),b'')
            results.append(dict(name=row['name'],leg=leg,exit=p.returncode,ok=ok));print(row['name'],leg,p.returncode,ok,flush=True)
    (ev/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    assert all(x['ok'] for x in results),'wrong outcomes'
    print('PASS',len(results),'MIND-body outcomes',flush=True)
    mutdata=(ev/'MUTATIONS.json').read_bytes()
    assert sha(mutdata)==(ev/'MUTATIONS.sha256').read_text().strip()
    assert json.loads(mutdata)==[list(x) for x in mutation_specs()]
    mutants=[]
    for name,before,after,case in mutation_specs():
        assert std.decode().count(before)==1,(name,'mutation not exact')
        changed=std.decode().replace(before,after).encode()
        a,b=build(ev,'mutant-'+name,embedded(changed),args.seed,args.mindc)
        row=next(x for x in catalogue() if x['name']==case)
        for leg,exe in [('native',a),('mlir',b)]:
            p=run(ev,'mutant-'+name+'-'+leg+'-run',[exe],struct.pack('<qqqq',*row['words']))
            # Only a normal wrong result detects a mutation: no signal, timeout,
            # build failure, or arbitrary runtime trap is credited.
            detected=p.stderr==b'' and p.returncode in (0,80,81,82,83,84,85,86) and (p.returncode,p.stdout,p.stderr)!=(0,bytes.fromhex(row['stdout']),b'')
            mutants.append(dict(name=name,leg=leg,case=case,exit=p.returncode,detected=detected))
            print('MUTATION',name,leg,p.returncode,detected,flush=True)
    (ev/'MUTATION_RESULTS.json').write_text(json.dumps(mutants,indent=2)+'\n')
    assert all(x['detected'] for x in mutants),'undetected or abnormal mutant'
    assert (ROOT/'std/string.mind').read_bytes()==std, 'MIND source changed during gate'
    assert sha(args.seed.read_bytes())==args.seed_sha256 and sha(args.mindc.read_bytes())==args.mindc_sha256, 'compiler changed during gate'
    print('PASS',len(mutants),'normal mutation detections')

if __name__=='__main__':main()
