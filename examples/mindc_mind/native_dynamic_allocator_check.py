#!/usr/bin/env python3
"""External diagnostic transport for the MIND allocator emitter/runtime.

MIND emits the runtime without an assembler dependency. This test reassembles
the independent reference, and a bounded C instrument checks emitted machine
code ownership/canaries. Neither is a product runtime or compiler fallback.
"""
import argparse,csv,hashlib,json,os,re,resource,shutil,struct,subprocess
from pathlib import Path
from native_dynamic_writer_check import encoder

HERE=Path(__file__).resolve().parent
CASE_SHA='ecfa89f8568d718ef14e532d639cdd94af15a86f5829b54f51759f56306e9d00'
MUT_SHA='ec18aae3979d086770873aec970d61c8f5b6805b34a8dd837e87eef8be7bfc5f'
def sha(data): return hashlib.sha256(data).hexdigest()
def limit():
    resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_CPU,(180,200))
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=Path,required=True);p.add_argument('--seed-sha256',required=True)
    p.add_argument('--mindc',type=Path,required=True);p.add_argument('--mindc-sha256',required=True);p.add_argument('--evidence',type=Path,required=True)
    a=p.parse_args();e=a.evidence.resolve();e.mkdir(parents=True,exist_ok=False)
    assert sha(a.seed.read_bytes())==a.seed_sha256 and sha(a.mindc.read_bytes())==a.mindc_sha256
    case=HERE/'native_dynamic_allocator_cases.tsv';assert sha(case.read_bytes())==CASE_SHA
    mutations=HERE/'native_dynamic_allocator_mutations.tsv';assert sha(mutations.read_bytes())==MUT_SHA
    rows=list(csv.DictReader([x for x in case.read_text().splitlines() if not x.startswith('#')],delimiter='\t'));assert len(rows)==39
    print('FROZEN',CASE_SHA,len(rows),flush=True)
    env=os.environ.copy()
    for k in list(env):
        if k.startswith('LD_') or k in ('GLIBC_TUNABLES','MINDC_NATIVE_ELF'):env.pop(k)
    env.update(LC_ALL='C',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    records=[]
    def run(label,argv,data=None,cwd=None,timeout=240):
        q=subprocess.run(list(map(str,argv)),input=data,capture_output=True,cwd=cwd,env=env,timeout=timeout,preexec_fn=limit)
        stem=f'{len(records):03d}-{label}';(e/(stem+'.stdout')).write_bytes(q.stdout);(e/(stem+'.stderr')).write_bytes(q.stderr)
        records.append(dict(id=stem,argv=list(map(str,argv)),exit=q.returncode,stdout_sha256=sha(q.stdout),stderr_sha256=sha(q.stderr)))
        (e/'executions.json').write_text(json.dumps(records,indent=2)+'\n');return q
    td=HERE/'testdata/native_dynamic_allocator'
    sources=[HERE/'native_dynamic_allocator.mind',HERE/'native_dynamic_layout.mind',HERE/'native_dynamic_allocator_driver.mind',case,mutations,Path(__file__),td/'allocator.s',td/'gen_mind.py',td/'check.c',td/'probe.s',HERE.parent.parent/'std/string.mind',HERE/'main.mind']
    (e/'SOURCE_HASHES.json').write_text(json.dumps({str(x):sha(x.read_bytes()) for x in sources},indent=2)+'\n')
    (e/'TOOLCHAIN.json').write_text(json.dumps({'native':str(a.seed),'native_sha256':a.seed_sha256,'mlir':str(a.mindc),'mlir_sha256':a.mindc_sha256},indent=2)+'\n')
    diagnostics={}
    for tool in ['gcc','as','objcopy','nm','python3']:
        path=Path(shutil.which(tool)).resolve();diagnostics[tool]={'path':str(path),'sha256':sha(path.read_bytes())}
        q=run(tool+'-version',[path,'--version']);assert q.returncode==0
    (e/'DIAGNOSTIC_TOOLS.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    base=(HERE.parent.parent/'std/string.mind').read_text()+'\n'
    main_source=(HERE/'main.mind').read_text()
    base+='\n'.join(encoder(main_source,n) for n in ['nb_push','nb_le','nb_i64le'])
    base+=(HERE/'native_dynamic_layout.mind').read_text()+'\n'
    body=(HERE/'native_dynamic_allocator.mind').read_text();driver=(HERE/'native_dynamic_allocator_driver.mind').read_text()
    source=(base+body+'\n'+driver).encode();(e/'emitter.mind').write_bytes(source)
    packet=struct.pack('<6q',int.from_bytes(b'MINDLAY\x01','little'),268435456,134217728,4096,0,len(source))+source
    (e/'input.packet').write_bytes(packet)
    q=run('native-build',[a.seed],packet);assert q.returncode==0 and not q.stderr and q.stdout.startswith(b'\x7fELF'),(q.returncode,q.stderr)
    native=e/'emitter.native';native.write_bytes(q.stdout);native.chmod(0o700)
    project=e/'mlir';project.mkdir();(project/'main.mind').write_bytes(source)
    (project/'Mind.toml').write_text('[package]\nname="dynamic_allocator_check"\nversion="0.1.0"\nauthors=["STARGA Inc."]\n[build]\nentry="main.mind"\n')
    mlir=project/'emitter.mlir'
    q=run('mlir-build',[a.mindc,'--allow-nondeterministic','build','--backend','mlir','--target','cpu','--emit','binary','--optimize','debug','--no-cache','--out',mlir],cwd=project)
    assert q.returncode==0 and mlir.is_file(),(q.returncode,q.stderr)
    q=run('assemble-reference',['as','--64',td/'allocator.s','-o',e/'reference.o']);assert q.returncode==0 and not q.stderr
    q=run('extract-reference',['objcopy','-O','binary','--only-section=.text',e/'reference.o',e/'reference.bin']);assert q.returncode==0 and not q.stderr
    q=run('reference-symbols',['nm','-n',e/'reference.o']);assert q.returncode==0 and not q.stderr
    symbols={x.split()[2]:int(x.split()[0],16) for x in q.stdout.decode().splitlines() if len(x.split())==3}
    q=run('build-independent-instrument',['gcc','-O2','-Wall','-Wextra','-Werror','-fPIE','-pie',td/'check.c',td/'probe.s','-o',e/'instrument']);assert q.returncode==0,(q.returncode,q.stderr)
    template=(e/'reference.bin').read_bytes()
    for row in rows:
        name=row['name'];arena=int(row['arena_bytes']);overflow=int(row['overflow_bytes'])
        request=(1,4096,4096,4096,0,arena,overflow,65536,4096,67108864)
        inp=struct.pack('<10q',*request);results=[]
        for leg,exe in [('native',native),('mlir',mlir)]:
            q=run(name+'-emit-'+leg,[exe],inp);assert (q.returncode,q.stderr)==(0,b''),(name,leg,q.returncode,q.stderr)
            results.append(q.stdout)
        assert results[0]==results[1],(name,'emitter byte mismatch')
        data=results[0];meta=struct.unpack('<8q',data[:64]);blob=data[64:]
        assert meta==(1,symbols['prefix'],symbols['rt_alloc'],symbols['rt_free'],symbols['rt_realloc'],len(template),0,0)
        ar=16777216;st=ar+arena-4096;ob=(ar+arena+4096+1048575)//1048576*1048576;oe=ob+overflow
        expected=template
        for k,v in {1:ar,2:ar+8,3:st,4:ob,5:ar+72,6:ob+8,7:oe,8:ar+64}.items():
            expected=expected.replace(bytes([k])+b'\x11'*7,struct.pack('<q',v))
        assert blob==expected,(name,'MIND/reference bytes')
        target=e/(name+'.runtime');target.write_bytes(data)
        q=run(name+'-runtime',[e/'instrument',target,name,arena,overflow])
        err=b'error: native arena exhausted\n' if int(row['exit'])==70 else b''
        assert (q.returncode,q.stdout,q.stderr)==(int(row['exit']),b'',err),(name,q.returncode,q.stdout,q.stderr)
        print('PASS',name,flush=True)
    # Additional invalid requests are explicit normal emitter refusals. Their
    # expected empty64-byte metadata is independent of the implementation.
    for name,idx,value in [('bad_policy',0,0),('small_code',1,1),('small_arena',5,4096),('negative_overflow',6,-1),('unaligned_overflow',6,1)]:
        req=[1,4096,4096,4096,0,8192,1048576,65536,4096,67108864];req[idx]=value
        for leg,exe in [('native',native),('mlir',mlir)]:
            q=run(name+'-'+leg,[exe],struct.pack('<10q',*req));assert (q.returncode,q.stdout,q.stderr)==(2,bytes(64),b''),(name,leg,q.returncode,q.stderr)
    # Actual mutated MIND emitter sources are compiled, then their emitted code
    # executes the fixed diagnostic case. Build failures/signals never detect.
    asm=(td/'allocator.s').read_text()
    edits={
      'repeat_slots':('    cmp [r9 + 536], rax\n    jne init_bad','    cmp [r9 + 536], rax'),
      'foreign_adopt':('ra_foreign:                     # foreign memory ownership stays with caller\n    xor eax, eax','ra_foreign:                     # MUTATION\n    call do_alloc'),
      'reuse_zero':('zero_mark:                      # zero [rdi, rcx) below the region high water,','zero_mark:\n    ret                         # MUTATION'),
      'replace_mapping':('    mov r10d, 0x100022','    mov r10d, 0x32'),
    }
    mutation_rows=list(csv.DictReader(mutations.read_text().splitlines(),delimiter='\t'));assert len(mutation_rows)==5
    for m in mutation_rows:
        mid=m['id'];mutbody=body
        if mid=='widen_end':
            needle=', ob + 8, oe, a + 64)';assert body.count(needle)==1;mutbody=body.replace(needle,', ob + 8, oe + 1048576, a + 64)')
        else:
            old,new=edits[mid];assert asm.count(old)==1
            path=e/(mid+'.s');path.write_text(asm.replace(old,new))
            q=run(mid+'-assemble',['as','--64',path,'-o',e/(mid+'.o')]);assert q.returncode==0 and not q.stderr
            q=run(mid+'-extract',['objcopy','-O','binary','--only-section=.text',e/(mid+'.o'),e/(mid+'.bin')]);assert q.returncode==0 and not q.stderr
            q=run(mid+'-symbols',['nm','-n',e/(mid+'.o')]);assert q.returncode==0 and not q.stderr
            ms={x.split()[2]:int(x.split()[0],16) for x in q.stdout.decode().splitlines() if len(x.split())==3}
            q=run(mid+'-generate',['python3',td/'gen_mind.py',e/(mid+'.bin'),'nda_blob_p']);assert q.returncode==0 and not q.stderr
            head=body[:body.index('// '+str(len(template))+' bytes,')]
            head,count=re.subn(r'(pub fn nda_code_size\(\) -> i64\s*\{\s*)'+str(len(template))+r'(\s*\})',lambda v:v[1]+str((e/(mid+'.bin')).stat().st_size)+v[2],head);assert count==1
            for off,sym in [(8,'prefix'),(16,'rt_alloc'),(24,'rt_free'),(32,'rt_realloc')]:
                needle=f'metadata + {off}, {symbols[sym]}';assert head.count(needle)==1
                head=head.replace(needle,f'metadata + {off}, {ms[sym]}')
            mutbody=head+q.stdout.decode()
        msource=(base+mutbody+'\n'+driver).encode();(e/(mid+'.mind')).write_bytes(msource)
        packet=struct.pack('<6q',int.from_bytes(b'MINDLAY\x01','little'),268435456,134217728,4096,0,len(msource))+msource
        q=run(mid+'-build',[a.seed],packet);assert q.returncode==0 and not q.stderr and q.stdout.startswith(b'\x7fELF'),(mid,'mutation build failed')
        exe=e/(mid+'.elf');exe.write_bytes(q.stdout);exe.chmod(0o700)
        req=(1,4096,4096,4096,0,8192,1048576,65536,4096,67108864)
        q=run(mid+'-emit',[exe],struct.pack('<10q',*req));assert q.returncode==0 and not q.stderr
        artifact=e/(mid+'.runtime');artifact.write_bytes(q.stdout)
        q=run(mid+'-detect',[e/'instrument',artifact,m['case'],8192,1048576])
        expected=int(m['expected_original_exit']);err=b'error: native arena exhausted\n' if expected==70 else b''
        assert q.returncode in (0,3) and (q.returncode,q.stdout,q.stderr)!=(expected,b'',err),(mid,'not a normal wrong-result detection',q.returncode,q.stderr)
        print('DETECTED',mid,'normal exit',q.returncode,flush=True)
    # Re-emit the untouched historical static blob, then compare its exact
    # bytes with the original independently assembled source and fixed layout.
    old_asm=HERE/'testdata/native_alloc_bounded/rt/alloc_rt.s'
    assert sha(old_asm.read_bytes())=='cd0a3909452d2668d1d22054c00e051654e6725e6508d7acede750cabaeeaf89'
    q=run('static-assemble',['as','--64',old_asm,'-o',e/'static.o']);assert q.returncode==0 and not q.stderr
    q=run('static-extract',['objcopy','-O','binary','--only-section=.text',e/'static.o',e/'static.bin']);assert q.returncode==0 and not q.stderr
    old=(HERE.parent.parent/'std/string.mind').read_text()+'\n'
    old+='\n'.join(encoder(main_source,n) for n in ['nb_push','nb_le','nb_i64le'])
    names=re.findall(r'^pub fn (nb_rt_blob_p\d+)\(',main_source,re.M);assert len(names)==7
    old+='\n'.join(encoder(main_source,n) for n in names)
    vals={1:16777216,2:16777224,3:285208576,4:419438592,5:16777232,6:419438600}
    old+='\nfn main()->i64 { let code:String=nb_rt_blob_p1(string_new(),'+','.join(str(vals[k]) for k in range(1,7))+'); if __mind_write(1,string_addr(code),string_len(code),0-1)!=string_len(code) {return 3;} 0 }\n'
    old=old.encode();(e/'static-emitter.mind').write_bytes(old)
    packet=struct.pack('<6q',int.from_bytes(b'MINDLAY\x01','little'),268435456,134217728,4096,0,len(old))+old
    q=run('static-build',[a.seed],packet);assert q.returncode==0 and not q.stderr and q.stdout.startswith(b'\x7fELF')
    exe=e/'static-emitter.elf';exe.write_bytes(q.stdout);exe.chmod(0o700)
    expected=(e/'static.bin').read_bytes()
    for k,v in vals.items():expected=expected.replace(bytes([k])+b'\x11'*7,struct.pack('<q',v))
    q=run('static-emit',[exe]);assert (q.returncode,q.stdout,q.stderr)==(0,expected,b'')
    print('PASS original static runtime bytes',sha(expected),flush=True)
    assert sha(case.read_bytes())==CASE_SHA and sha(mutations.read_bytes())==MUT_SHA
    (e/'VERDICT').write_text('PASS 39 runtime cases,5 emitter refusals,5 actual source mutations,original static bytes\n')
    print('PASS 39 runtime cases;5 emitter refusals;5 source mutations;original static bytes',flush=True)
if __name__=='__main__':main()
