#!/usr/bin/env python3
"""Source-bound fixed libc I/O thunk diagnostics; no production admission."""
# Author: STARGA Inc.
import argparse, fcntl, hashlib, json, os, resource, shutil, struct, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TD=HERE/'testdata/native_libc_io'
MODULES=['main.mind','native_dynamic.mind','native_dynamic_layout.mind','native_dynamic_image.mind','native_dynamic_allocator.mind','native_libc_startup.mind','native_libc_io.mind','native_libc_io_fixture.mind']
CONTRACTS={'runtime_cases.json':'234b13ad3006ee78b98278c053b4e03f870401c552b55294362c6912fb34a12c','metadata_cases.json':'cc8e8b33d691df37e030551717215aaf43962cd5351554e240a5d2ccfb2ea452','mutations.json':'be86d6f6bb84056d265267486f68bf6f673a12c566fc37d25b488bbb461b679e'}

def sha(b):return hashlib.sha256(b).hexdigest()
def pack(*v):return struct.pack('<'+'q'*len(v),*v)
def word(b,index):return struct.unpack_from('<q',b,index*8)[0]

def expected_bundle(code,symbols):
    source=(TD/'generated_source.mind').read_bytes()
    names=['read','pread','write','pwrite']
    wire=b'MCABI001'+struct.pack('<I',4)
    for i,name in enumerate(names):
        wire+=struct.pack('<I',len(name))+name.encode()+bytes([0,1,0])+struct.pack('<I',3+i%2)
        wire+=bytes([7,5,33 if i<2 else 32,2,8])+(bytes([7]) if i%2 else b'')
    so=192;wo=so+len(source);co=wo+len(wire);io=co+len(code);fo=io+160;vo=fo+96
    version=b'GLIBC_2.2.5';soname=b'libc.so.6';interp=b'/lib64/ld-linux-x86-64.so.2';sn=vo+len(version);ip=sn+len(soname);total=ip+len(interp)
    header=pack(int.from_bytes(b'NIOBND01','little'),total,1,1,so,len(source),wo,len(wire),co,len(code),symbols['io_read_entry'],symbols['io_write_entry'],io,4,fo,4,vo,len(version),sn,len(soname),ip,len(interp),0,0)
    imports=b''.join(pack(so+source.index(('fn '+n).encode())+3,len(n),i,0,0) for i,n in enumerate(names))
    fixes=b''.join(pack(symbols[n]+2,index,op) for n,index,op in [('io_read_positioned',1,0),('io_read_sequential',0,0),('io_write_positioned',3,1),('io_write_sequential',2,1)])
    return header+source+wire+code+imports+fixes+version+soname+interp

def metadata_input(case,bundle):
    kind=case['kind'];mode=0;value=0;offset=0
    if kind=='roundtrip':mode=1
    elif kind=='owned':mode=7
    elif kind=='profile':mode=3;value=case['value']
    elif kind=='length':mode=4;value=case['value']
    elif kind=='truncated':mode=5
    elif kind=='extended':mode=6
    else:
        mode=2
        if kind=='header_xor':offset=case['word']*8
        elif kind=='fixup_xor':offset=word(bundle,14)+case['row']*24+case['word']*8
        elif kind=='payload_xor':
            region=case['region']
            offset=word(bundle,{'source':4,'wire':6,'code':8,'version':16,'soname':18,'interpreter':20}.get(region,4))
            if region=='import_name':offset=word(bundle,12)
        else:raise AssertionError(kind)
    return pack(0,mode,value,offset,0,0,0)

def runtime_input(case):
    return pack(case['op'],case['fd'],case['count'],case['offset'],case['buffer_mode'],case['preset_errno'],0,0,0,0)+bytes.fromhex(case['initial_buffer_hex'])+bytes.fromhex(case['extra_stdin_hex'])

def runtime_expected(case):
    return bytes.fromhex(case['expected_stdout_prefix_hex'])+pack(case['expected_result'],case['expected_errno'],case['expected_cursor'])+bytes.fromhex(case['expected_buffer_hex'])

def mutation_sources(product):
    text=product.decode()
    edits=[
        ('null_guard',[('__mind_store_i8(p+3,116);','__mind_store_i8(p+3,144);'),('__mind_store_i8(p+4,42);','__mind_store_i8(p+4,144);')]),
        ('count_guard',[('__mind_store_i8(p+8,126);','__mind_store_i8(p+8,144);'),('__mind_store_i8(p+9,37);','__mind_store_i8(p+9,144);')]),
        ('offset_selection',[('__mind_store_i8(p+15,120);','__mind_store_i8(p+15,116);')]),
        ('wrong_import',[('nio_fixup(out,fixups_at,1,38,0,0)','nio_fixup(out,fixups_at,1,38,2,0)'),('nio_fixup(out,fixups_at,3,88,2,1)','nio_fixup(out,fixups_at,3,88,0,1)')]),
        ('source_snapshot',[('nio_copy(out+__mind_load_i64(out+32),string_addr(source),string_len(source))','nio_zero(out+__mind_load_i64(out+32),string_len(source))')]),
        ('plan_comparison',[('same=nio_same(bundle,fresh,n);','same=1;')]),
    ]
    result={}
    for name,changes in edits:
        changed=text
        for old,new in changes:
            assert changed.count(old)==1,(name,old)
            changed=changed.replace(old,new,1)
        result[name]=changed.encode()
    return result

def limits(size_limit=0):
    resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(600,620))
    if size_limit:resource.setrlimit(resource.RLIMIT_FSIZE,(size_limit,size_limit))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--enumerate',action='store_true');p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--seed',type=Path);p.add_argument('--seed-sha256');p.add_argument('--mindc',type=Path);p.add_argument('--mindc-sha256')
    p.add_argument('--reuse',type=Path);p.add_argument('--mutations',action='store_true')
    p.add_argument('--reuse-mutants',type=Path,help='Trusted prior local build receipts; exact packet/compiler/output binding required')
    a=p.parse_args();e=a.evidence.resolve()
    for name,h in CONTRACTS.items():assert sha((TD/name).read_bytes())==h,('CONTRACT_DRIFT',name)
    contract=(json.dumps(CONTRACTS,indent=2)+'\n').encode()
    if a.enumerate:
        e.mkdir(parents=True,exist_ok=False);(e/'CASES.json').write_bytes(contract);(e/'CASES.json').chmod(0o444)
        print('FROZEN',sha(contract));return
    assert (e/'CASES.json').read_bytes()==contract
    (e/'execution.started').open('x').close()
    assert a.seed and a.mindc
    a.seed=a.seed.resolve();a.mindc=a.mindc.resolve()
    assert sha(a.seed.read_bytes())==a.seed_sha256 and sha(a.mindc.read_bytes())==a.mindc_sha256
    deps=[HERE/n for n in MODULES]+sorted((ROOT/'std').rglob('*.mind'))+sorted(x for x in TD.rglob('*') if x.is_file())+[Path(__file__).resolve(),HERE/'_stdlib_manifest.py',HERE/'testdata/stdlib_manifest.txt',ROOT/'runtime-support/mind_intrinsics.c',Path('/lib/x86_64-linux-gnu/libc.so.6').resolve(),Path('/lib64/ld-linux-x86-64.so.2').resolve()]
    captured={str(x):x.read_bytes() for x in deps}
    inputs={x:sha(b) for x,b in captured.items()};inputs.update({str(a.seed):a.seed_sha256,str(a.mindc):a.mindc_sha256})
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('LD_') or key in ['GLIBC_TUNABLES','MINDC_NATIVE_ELF','MINDC_SO']:env.pop(key)
    env.update(MIND_STDLIB_PATH=str(ROOT/'std'),LC_ALL='C',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    tools={}
    for name,default in [('AS','as'),('OBJCOPY','objcopy'),('NM','nm'),('MLIR_OPT','mlir-opt'),('MLIR_TRANSLATE','mlir-translate'),('CLANG','clang'),('CC','cc')]:
        path=Path(shutil.which(env.get(name,default),path=env.get('PATH'))).resolve();tools[name]=str(path);inputs[str(path)]=sha(path.read_bytes())
    (e/'INPUTS.json').write_text(json.dumps(inputs,indent=2)+'\n');(e/'BUILD_ENVIRONMENT.json').write_text(json.dumps(dict(stdlib=env['MIND_STDLIB_PATH'],tools=tools),indent=2)+'\n')
    (e/'harness.snapshot.py').write_bytes(captured[str(Path(__file__).resolve())])
    def intact():
        assert (e/'CASES.json').read_bytes()==contract
        for name,digest in inputs.items():assert sha(Path(name).read_bytes())==digest,('INPUT_DRIFT',name)
        assert sorted(str(x) for x in (ROOT/'std').rglob('*.mind'))==sorted(x for x in captured if x.startswith(str(ROOT/'std')+'/'))
    rows=[]
    def run(name,args,data=None,cwd=None,pass_fds=(),preexec=limits,timeout=700):
        q=subprocess.run(list(map(str,args)),input=data,capture_output=True,cwd=cwd,env=env,pass_fds=pass_fds,preexec_fn=preexec,timeout=timeout)
        stem=f'{len(rows):04d}-{name}';(e/(stem+'.stdout')).write_bytes(q.stdout);(e/(stem+'.stderr')).write_bytes(q.stderr)
        rows.append(dict(name=name,argv=list(map(str,args)),cwd=str(cwd or Path.cwd()),stdin_sha256=None if data is None else sha(data),exit=q.returncode,stdout_sha256=sha(q.stdout),stderr_sha256=sha(q.stderr),pass_fds=list(pass_fds)))
        (e/'executions.json').write_text(json.dumps(rows,indent=2)+'\n');return q
    def assemble(name,path):
        obj=e/(name+'.o');raw=e/(name+'.bin');q=run(name+'-as',[tools['AS'],'--64',path,'-o',obj]);assert q.returncode==0 and not q.stderr
        q=run(name+'-copy',[tools['OBJCOPY'],'-O','binary','-j','.text',obj,raw]);assert q.returncode==0 and not q.stderr
        q=run(name+'-nm',[tools['NM'],'-n',obj]);assert q.returncode==0 and not q.stderr
        symbols={x.split()[2]:int(x.split()[0],16) for x in q.stdout.decode().splitlines() if len(x.split())==3}
        return raw.read_bytes(),symbols
    code,symbols=assemble('thunk-reference',TD/'thunks.s');bundle=expected_bundle(code,symbols)
    (e/'EXPECTED_BUNDLE.bin').write_bytes(bundle)
    body,_=assemble('runtime-main',TD/'runtime_main.s')
    cs=[];ps=[];body=bytearray(body)
    for kind in [1,2,3]:
        marker=struct.pack('<Q',0x3333333333333300+kind);assert body.count(marker)==1;at=body.index(marker);cs.append((at,kind));body[at:at+8]=bytes(8)
    for index in [5,6]:
        marker=struct.pack('<I',0x44444400+index)
        while marker in body:
            at=body.index(marker);assert body[at-2:at]==b'\xff\x15';ps.append((at,index));body[at:at+4]=bytes(4)
    image_input=pack(1,0,0,0,len(body),len(cs),len(ps))+body+b''.join(pack(*r) for r in cs)+b''.join(pack(*r) for r in ps)
    (e/'image.input').write_bytes(image_input)
    sys.path.insert(0,str(HERE));from _stdlib_manifest import seed_modules
    prefix=b'\n'.join(captured[str(ROOT/'std'/(n+'.mind'))] for n in seed_modules())+b'\n';source=b'\n'.join(captured[str(HERE/n)] for n in MODULES);combined=prefix+source
    packet=pack(int.from_bytes(b'MINDLAY\x01','little'),536870912,134217728,4096,len(prefix),len(combined))+combined
    (e/'build.packet').write_bytes(packet);(e/'assembled.mind').write_bytes(combined)
    intact()
    if a.reuse:
        old=a.reuse.resolve();out=json.loads((old/'OUTPUTS.json').read_text());old_inputs=json.loads((old/'INPUTS.json').read_text())
        assert (old/'build.packet').read_bytes()==packet and (old/'mlir/main.mind').read_bytes()==source
        runner=str(Path(__file__).resolve())
        assert {k:v for k,v in old_inputs.items() if k!=runner}=={k:v for k,v in inputs.items() if k!=runner},'REUSE_DEPENDENCY_DRIFT'
        assert (old/'CASES.json').read_bytes()==contract
        assert sha((old/'harness.snapshot.py').read_bytes())==old_inputs[runner]
        native=Path(out['native']['path']);oracle=Path(out['mlir']['path'])
        for side,path in [('native',native),('mlir',oracle)]:assert sha(path.read_bytes())==out[side]['sha256']
        (e/'REUSE.json').write_text(json.dumps(dict(directory=str(old),manifest_sha256=sha((old/'OUTPUTS.json').read_bytes()),producer_harness_sha256=old_inputs[runner],runner_harness_sha256=inputs[runner],packet_sha256=sha(packet)),indent=2)+'\n')
    else:
        q=run('native-build',[a.seed],packet);assert (q.returncode,q.stderr)==(0,b'') and q.stdout.startswith(b'\x7fELF'),('native-build',q.returncode,q.stderr[-500:])
        native=e/'fixture.native';native.write_bytes(q.stdout);native.chmod(0o700)
        project=e/'mlir';project.mkdir();(project/'main.mind').write_bytes(source);(project/'Mind.toml').write_text('[package]\nname="io-thunk-diagnostic"\nversion="0.0.0"\n[build]\nentry="main.mind"\n')
        oracle=project/'fixture.mlir';q=run('mlir-build',[a.mindc,'--allow-nondeterministic','build','--backend','mlir','--target','cpu','--emit','binary','--optimize','release','--no-cache','--out',oracle],cwd=project);assert q.returncode==0 and oracle.exists(),q.stderr[-1000:]
    output_hashes={side:dict(path=str(path),sha256=sha(path.read_bytes())) for side,path in [('native',native),('mlir',oracle)]};(e/'OUTPUTS.json').write_text(json.dumps(output_hashes,indent=2)+'\n')
    intact()
    images={}
    for side,exe in [('native',native),('mlir',oracle)]:
        q=run('bundle-'+side,[exe],bytes(56),timeout=30);assert (q.returncode,q.stdout,q.stderr)==(0,bytes(32)+pack(len(bundle))+bundle,b''),('bundle',side,q.returncode,q.stdout[:64],q.stderr)
        q=run('image-'+side,[exe],image_input,timeout=30);assert q.returncode==0 and not q.stderr and q.stdout.startswith(b'\x7fELF'),('image',side,q.returncode,q.stderr)
        image=e/('runtime-'+side+'.elf');image.write_bytes(q.stdout);image.chmod(0o700);images[side]=image
    assert images['native'].read_bytes()==images['mlir'].read_bytes(),'EMITTED_ELF_DIFFERENCE'
    # Verify actual GOT destinations for all four product calls. Writer sorts
    # imports lexicographically; these records retain original source indexes.
    elf=images['native'].read_bytes();phoff=struct.unpack_from('<Q',elf,32)[0];phnum=struct.unpack_from('<H',elf,56)[0]
    ph=[struct.unpack_from('<IIQQQQQQ',elf,phoff+i*56) for i in range(phnum)]
    dynamic=next(x for x in ph if x[0]==2);tags=dict(struct.unpack_from('<QQ',elf,dynamic[2]+i*16) for i in range(dynamic[5]//16))
    # GLOB_DAT relocations have explicit per-import GOT offsets in .rela.dyn.
    def file_offset(va):
        for x in ph:
            if x[0]==1 and x[3]<=va<x[3]+x[5]:return x[2]+va-x[3]
        raise AssertionError(('UNMAPPED',va))
    rel=file_offset(tags[7]);relocs=[struct.unpack_from('<QQq',elf,rel+i*24) for i in range(tags[8]//24)]
    names=sorted(['read','pread','write','pwrite','__libc_start_main','__errno_location','lseek'])
    io_at=365+2360
    for label,index in [('io_read_positioned',1),('io_read_sequential',0),('io_write_positioned',3),('io_write_sequential',2)]:
        at=io_at+symbols[label]+2;disp=struct.unpack_from('<i',elf,4096+at)[0];target=0x401000+at+4+disp
        name=['read','pread','write','pwrite'][index];assert target==relocs[names.index(name)][0],('GOT',label)
    metadata=json.loads((TD/'metadata_cases.json').read_text())
    for case in metadata:
        inp=metadata_input(case,bundle);(e/(case['name']+'.input')).write_bytes(inp)
        want=pack(case['expect'],0,0,0,0)
        if case['expect']==0:want=bytes(32)+pack(len(bundle))+bundle
        for side,exe in [('native',native),('mlir',oracle)]:
            q=run(case['name']+'-'+side,[exe],inp,timeout=30);assert (q.returncode,q.stdout,q.stderr)==(0,want,b''),(case['name'],side,q.returncode,q.stdout[:48],q.stderr)
        print('PASS metadata',case['name'],flush=True)
    # Reference uses the existing C runtime through MLIR, not these new thunks.
    refdir=e/'reference';refdir.mkdir();(refdir/'main.mind').write_bytes((TD/'reference.mind').read_bytes());(refdir/'Mind.toml').write_text('[package]\nname="io-existing-reference"\nversion="0.0.0"\n[build]\nentry="main.mind"\n')
    reference=refdir/'reference';q=run('reference-build',[a.mindc,'--allow-nondeterministic','build','--backend','mlir','--target','cpu','--emit','binary','--optimize','release','--no-cache','--out',reference],cwd=refdir);assert q.returncode==0 and reference.exists(),q.stderr[-1000:]
    runtime=json.loads((TD/'runtime_cases.json').read_text())
    def execute_runtime(case,exe,name,expect_correct=True):
        directory=e/name;directory.mkdir();path=directory/'file';path.write_bytes(bytes.fromhex(case['initial_file_hex']))
        setup=case['setup'];read_fd=None;write_fd=None
        if setup=='file':fd=os.open(path,os.O_RDWR);os.lseek(fd,case['initial_cursor'],os.SEEK_SET)
        else:
            read_fd,write_fd=os.pipe2(os.O_NONBLOCK);fcntl.fcntl(write_fd,fcntl.F_SETPIPE_SZ,4096)
            if setup=='pipe_data':os.write(write_fd,b'abc')
            if setup=='pipe_full':assert os.write(write_fd,b'P'*4096)==4096
            fd=read_fd if setup in ['pipe_data','pipe_empty'] else write_fd
        assert fd!=10 and 10 not in [read_fd,write_fd], 'PRIVATE_FD10_ALREADY_USED'
        os.dup2(fd,10,inheritable=True)
        try:q=run(name,[exe],runtime_input(case),cwd=directory,pass_fds=(10,),preexec=lambda:limits(case['file_size_limit']),timeout=25)
        finally:os.close(10)
        side_effects=True
        if setup=='file':os.close(fd)
        else:
            try:pipe_after=os.read(read_fd,8192)
            except BlockingIOError:pipe_after=b''
            (directory/'pipe-after').write_bytes(pipe_after)
            expected_pipe={'pipe_full':b'P'*4096,'pipe_empty':b'','pipe_write':b'A'*case['expected_result'] if case['expected_result']>0 else b'','pipe_data':b'' if case['expected_result']>0 else b'abc'}[setup]
            side_effects=pipe_after==expected_pipe
            os.close(read_fd);os.close(write_fd)
        correct=(q.returncode,q.stdout,q.stderr)==(case['expected_exit'],runtime_expected(case),bytes.fromhex(case['expected_stderr_hex']))
        actual=path.read_bytes();side_effects=side_effects and actual==bytes.fromhex(case['expected_file_hex'])
        if expect_correct:
            assert correct and side_effects,(case['name'],name,q.returncode,q.stdout[:40],runtime_expected(case)[:40],q.stderr,'side_effects',side_effects)
        else:
            assert q.returncode==0 and not q.stderr and len(q.stdout)==len(runtime_expected(case)),(name,'crash or abnormal output is not mutation detection')
        (directory/'result.json').write_text(json.dumps(dict(file_sha256=sha(actual),stdout_sha256=sha(q.stdout),stderr_sha256=sha(q.stderr),exit=q.returncode),indent=2)+'\n')
        return q,correct and side_effects
    for case in runtime:
        for side,exe in [('native',images['native']),('mlir',images['mlir']),('reference',reference)]:execute_runtime(case,exe,case['name']+'-'+side)
        print('PASS runtime',case['name'],flush=True)
    if a.mutations:
        product=captured[str(HERE/'native_libc_io.mind')]
        changed=mutation_sources(product)
        plans=json.loads((TD/'mutations.json').read_text())
        frozen=[dict(row,source_sha256=sha(changed[row['name']])) for row in plans]
        (e/'MUTANTS_FROZEN.json').write_text(json.dumps(frozen,indent=2)+'\n')
        print('FROZEN MUTANTS',sha((e/'MUTANTS_FROZEN.json').read_bytes()),flush=True)
        assert combined.count(product)==1
        for row in plans:
            name=row['name'];altered=combined.replace(product,changed[name]);(e/(name+'.mutant.mind')).write_bytes(changed[name])
            changed_packet=packet[:40]+pack(len(altered))+altered
            (e/(name+'.build.packet')).write_bytes(changed_packet)
            reused=None
            if a.reuse_mutants:
                prior=a.reuse_mutants.resolve();old_packet=prior/(name+'.build.packet')
                if old_packet.exists() and old_packet.read_bytes()==changed_packet:
                    assert (prior/'build.packet').read_bytes()==packet,'MUTANT_BASE_PACKET_DRIFT'
                    old_inputs=json.loads((prior/'INPUTS.json').read_text())
                    assert old_inputs[str(a.seed)]==a.seed_sha256,'MUTANT_COMPILER_DRIFT'
                    old_rows=json.loads((prior/'executions.json').read_text())
                    old_index,receipt=next((i,r) for i,r in enumerate(old_rows) if r['name']==name+'-build')
                    assert receipt['argv']==[str(a.seed)] and receipt['stdin_sha256']==sha(changed_packet)
                    assert receipt['exit']==0 and receipt['stderr_sha256']==sha(b'')
                    reused=(prior/(f'{old_index:04d}-'+name+'-build.stdout')).read_bytes()
                    assert sha(reused)==receipt['stdout_sha256'] and reused.startswith(b'\x7fELF')
                    assert (prior/(name+'.fixture')).read_bytes()==reused
                    assert (prior/(name+'.mutant.mind')).read_bytes()==changed[name]
                    (e/(name+'.REUSE.json')).write_text(json.dumps(dict(prior=str(prior),packet_sha256=sha(changed_packet),compiler_sha256=a.seed_sha256,executable_sha256=sha(reused),receipt=receipt),indent=2)+'\n')
            if reused is None:
                q=run(name+'-build',[a.seed],changed_packet)
                assert q.returncode==0 and not q.stderr and q.stdout.startswith(b'\x7fELF'),(name,'build failure is not detection',q.returncode,q.stderr[-500:])
                reused=q.stdout
            exe=e/(name+'.fixture');exe.write_bytes(reused);exe.chmod(0o700)
            if name in ['source_snapshot','plan_comparison']:
                case=next(c for c in metadata if c['name']==row['case'])
                inp=metadata_input(case,bundle)
                want=bytes(32)+pack(len(bundle))+bundle if case['expect']==0 else pack(case['expect'],0,0,0,0)
                q=run(name+'-metadata',[exe],inp,timeout=30)
                assert q.returncode==0 and not q.stderr and len(q.stdout)>=40 and word(q.stdout,0)==0,(name,'must accept normally')
                assert q.stdout!=want,(name,'escaped detection')
            else:
                q=run(name+'-image',[exe],image_input,timeout=30)
                assert q.returncode==0 and not q.stderr and q.stdout.startswith(b'\x7fELF'),(name,'image failure is not detection')
                image=e/(name+'.elf');image.write_bytes(q.stdout);image.chmod(0o700)
                case=next(c for c in runtime if c['name']==row['case'])
                q,correct=execute_runtime(case,image,'mutant-'+name,False)
                assert not correct,(name,'escaped detection')
            print('MUTATION DETECTED',name,'normal incorrect result',flush=True)

    intact()
    for row in output_hashes.values():assert sha(Path(row['path']).read_bytes())==row['sha256'],'FIXTURE_DRIFT'
    artifacts={str(p.relative_to(e)):sha(p.read_bytes()) for p in e.rglob('*') if p.is_file()}
    (e/'ARTIFACTS.json').write_text(json.dumps(artifacts,indent=2)+'\n')
    print('PASS fixed libc I/O component',len(metadata),'metadata pairs',len(runtime),'runtime triples; no production admission',flush=True)
if __name__=='__main__':main()
