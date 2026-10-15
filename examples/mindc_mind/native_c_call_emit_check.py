#!/usr/bin/env python3
"""Fixed scalar call emission: frozen metadata refusals and real ABI execution.

The external C/assembly fixture is a diagnostic instrument. Product emitters
and their transport are MIND; this harness is not a compiler runtime dependency.
"""
import argparse,ctypes,hashlib,json,mmap,os,resource,struct,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=None
ARGS=None
sys.path.insert(0,str(HERE))
from native_c_call_plan_check import words,ty,sha
from native_dynamic_writer_check import encoder
from _stdlib_manifest import seed_blob

def cases():
 rows=[]
 def add(name,tags=(),values=None,ret=8,result=42,**kw):
  tags=list(tags); values=list(values if values is not None else range(1,len(tags)+1))
  row=dict(name=name,tags=tags,values=values,ret=ret,result=result,cc=0,safety=1,varargs=0,
   count=len(tags),frame=4096,dst=-4096,import_id=17,slots=[-8*(i+1) for i in range(len(tags))],error=0)
  row.update(kw);rows.append(row)
 add('empty')
 for tag,value in [(1,-128),(2,255),(3,-32768),(4,65535),(5,-2147483648),(6,4294967295),(7,-123456789),(8,123456789),(9,-19),(10,19),(12,3.75),(13,1),(14,0x12345678)]:
  add('width-'+str(tag),[tag],[value],ret=tag,result=value)
 for n in (6,7,8,256): add('gp-'+str(n),[7]*n)
 for n in (8,9,10,256): add('sse-'+str(n),[12]*n)
 add('mixed-sse-first',[12]*9+[7]*7)
 add('mixed-gp-first',[7]*7+[12]*9)
 add('mixed-alternating',[7,12]*9)
 add('mixed-narrow',[1,2,3,4,5,6,13,14,12],[-1,255,-1234,60000,-1234567,4000000000,1,0x1234,2.75])
 add('void',[7],ret=0,result=333,dst=123)
 add('same-slot',[7,7],[23,23],slots=[-8,-8])
 add('stack-alignment',ret=8,result=8)
 for tag,value in [(1,-128),(2,128),(3,-32767),(4,32769),(5,-2147483647),(6,2147483649),(13,1)]:
  add('excess-result-'+str(tag),ret=tag,result=value)
 for frame in (0,15,17,2147483648,-1): add('bad-frame-'+str(frame),frame=frame,error=9)
 for index in (-1,256): add('bad-import-'+str(index),import_id=index,error=10)
 for slot in (0,-7,-4097,-9223372036854775808): add('bad-slot-'+str(slot),[7],slots=[slot],error=11)
 for dst in (0,-7,-4097): add('bad-dst-'+str(dst),dst=dst,error=12)
 add('f32-param',[11],error=6)
 add('f32-result',ret=11,error=6)
 add('varargs',varargs=1,error=3)
 add('bad-cc',cc=2,error=1)
 add('bad-safety',safety=2,error=2)
 add('count257',count=257,error=4)
 add('count-negative',count=-1,error=4)
 add('void-arg',[0],error=5)
 return rows

def packet(r):
 record=lambda t: ty((32,2)) if t==14 else ty((t,))
 data=words([r[k] for k in ('cc','safety','varargs','count','frame','dst','import_id')])+record(r['ret'])
 if 0<=r['count']<=256: data+=b''.join(record(t) for t in r['tags'])+words(r['slots'])
 return data

def freeze():
 rows=cases();out=OUT/'cases';out.mkdir(exist_ok=False)
 payload=(json.dumps(rows,indent=2)+'\n').encode();(out/'CASES.json').write_bytes(payload);(out/'CASES.json').chmod(0o444)
 for r in rows:(out/(r['name']+'.input')).write_bytes(packet(r))
 print('FROZEN',len(rows),sha(payload))

def limits():
 resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(180,180))

def run(label,argv,*,data=None,cwd=None):
 env=os.environ.copy()
 for key in list(env):
  if key.startswith('LD_') or key in ('GLIBC_TUNABLES','MINDC_NATIVE_ELF','MINDC_SO'):
   env.pop(key)
 env.update(LC_ALL='C',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 p=subprocess.run(list(map(str,argv)),input=data,capture_output=True,cwd=cwd,timeout=240,preexec_fn=limits,env=env)
 stem=OUT/label;stem.parent.mkdir(parents=True,exist_ok=True)
 stem.with_suffix('.stdout').write_bytes(p.stdout);stem.with_suffix('.stderr').write_bytes(p.stderr)
 stem.with_suffix('.json').write_text(json.dumps(dict(argv=list(map(str,argv)),exit=p.returncode,stdout_sha256=sha(p.stdout),stderr_sha256=sha(p.stderr)))+'\n')
 assert p.returncode==0,(label,p.returncode,p.stderr[:200])
 return p

def build():
 out=OUT/'build';out.mkdir(exist_ok=False)
 src=(HERE/'main.mind').read_text()
 names=['nb_push','nb_le','nb_u32','nb_i32le_signed','nb_load_rax','nb_store_rax','nb_fp_movsd_load_xmmk','nb_fp_movsd_store_xmm0']
 user='\n'.join(encoder(src,n) for n in names).encode()
 for name in ('native_c_call_plan.mind','native_c_call_emit.mind','native_c_call_emit_driver.mind'):user+=b'\n'+(HERE/name).read_bytes()
 prefix=seed_blob();combined=prefix+user
 compiler=ARGS.seed.resolve()
 shell=ARGS.mindc.resolve()
 assert sha(compiler.read_bytes())==ARGS.seed_sha256,'SEED_IDENTITY'
 assert sha(shell.read_bytes())==ARGS.mindc_sha256,'MINDC_IDENTITY'
 owned=out/'compiler.elf';owned.write_bytes(compiler.read_bytes());owned.chmod(0o500)
 inp=words([int.from_bytes(b'MINDLAY\x01','little'),268435456,134217728,4096,len(prefix),len(combined)])+combined
 (out/'packet').write_bytes(inp)
 r=run('build/native',[owned],data=inp)
 assert r.stdout.startswith(b'\x7fELF') and not r.stderr
 native=out/'native.elf';native.write_bytes(r.stdout);native.chmod(0o500)
 project=out/'mlir';project.mkdir();(project/'main.mind').write_bytes(user)
 (project/'Mind.toml').write_text('[package]\nname="c_call_emit_test"\nversion="0.1.0"\nauthors=["STARGA Inc."]\n[build]\nentry="main.mind"\n')
 run('build/oracle',[shell,'build','--backend','mlir','--target','cpu','--emit','binary','--release','--no-cache','--out',project/'oracle.elf'],cwd=project)
 (out/'INPUTS.json').write_text(json.dumps({'main':sha(src.encode()),'user':sha(user),'packet':sha(inp),'native':sha(native.read_bytes()),'oracle':sha((project/'oracle.elf').read_bytes()),'library':sha((HERE/'native_c_call_emit.mind').read_bytes()),'seed':ARGS.seed_sha256,'mindc':ARGS.mindc_sha256,'harness':sha(Path(__file__).read_bytes()),'stdlib':sha(prefix)},indent=2)+'\n')

CT={0:'void',1:'int8_t',2:'uint8_t',3:'int16_t',4:'uint16_t',5:'int32_t',6:'uint32_t',7:'int64_t',8:'uint64_t',9:'intptr_t',10:'uintptr_t',12:'double',13:'_Bool',14:'void*'}
def value(t,v):
 if t==14:return f'(void*)(uintptr_t){v}ULL'
 if t==12:return repr(float(v))
 return str(v)

def fixture(rows):
 c=['#include <stdint.h>','#include <stdlib.h>'];asm=['.text']
 for i,r in enumerate(rows):
  if r['error']:continue
  symbol='fixture_'+str(i);r['symbol']=symbol
  if r['name']=='stack-alignment':
   asm += [f'.globl {symbol}',f'{symbol}:','mov %rsp,%rax','and $15,%rax','ret'];continue
  if r['name'].startswith('excess-result-'):
   width={1:8,2:8,3:16,4:16,5:32,6:32,13:8}[r['ret']]
   bits=(0x76543210fedcba98 & ~((1<<width)-1)) | (r['result'] & ((1<<width)-1))
   asm += [f'.globl {symbol}',f'{symbol}:',f'movabs ${bits},%rax','ret'];continue
  params=', '.join(f'{CT[t]} a{j}' for j,t in enumerate(r['tags'])) or 'void'
  body=' '.join(f'if(a{j} != {value(t,v)}) abort();' for j,(t,v) in enumerate(zip(r['tags'],r['values'])))
  result='return;' if r['ret']==0 else 'return '+value(r['ret'],r['result'])+';'
  c += [f'{CT[r["ret"]]} {symbol}({params}) {{ {body} {result} }}']
 asm+=['.section .note.GNU-stack,"",@progbits']
 (OUT/'fixture.c').write_text('\n'.join(c)+'\n');(OUT/'fixture.S').write_text('\n'.join(asm)+'\n')
 run('fixture-build',[ARGS.clang,'-shared','-fPIC','-O2',OUT/'fixture.c',OUT/'fixture.S','-o',OUT/'fixture.so'])

def execute():
 rows=cases();assert json.loads((OUT/'cases/CASES.json').read_text())==rows
 fixture(rows);receipts=[]
 for i,r in enumerate(rows):
  data=packet(r);assert (OUT/'cases'/(r['name']+'.input')).read_bytes()==data
  previous=None
  for side,exe in [('native',OUT/'build/native.elf'),('mlir',OUT/'build/mlir/oracle.elf')]:
   p=run('observed/'+side+'-'+r['name'],[exe],data=data);blob=p.stdout
   assert not p.stderr
   error,size,version,fixup,index,reserve=struct.unpack('<6q',blob[:48]);code=blob[48:]
   assert error==r['error'] and size==len(code)
   if error:assert blob==words([error,0,0,0,0,0])
   else:
    ints=sum(t!=12 for t in r['tags']);floats=sum(t==12 for t in r['tags'])
    stack=8*(max(0,ints-6)+max(0,floats-8));expected=(stack+15)//16*16
    assert (version,index,reserve)==(1,r['import_id'],expected)
    assert code[fixup-2:fixup+4]==b'\xff\x15\0\0\0\0'
    file=OUT/'observed'/(side+'-'+r['name']+'.call');file.write_bytes(code)
    desc=dict(r,code=str(file),fixup=fixup)
    path=OUT/'observed'/(side+'-'+r['name']+'.execution.json');path.write_text(json.dumps(desc))
    run('runtime/'+side+'-'+r['name'],[sys.executable,__file__,'--child',path])
   if previous is not None:assert previous==blob,'BACKEND_BYTES'
   previous=blob;receipts.append(dict(name=r['name'],side=side,error=error,code_sha256=sha(code),pass_=True))
   (OUT/'RESULTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
 print('PASS',len(receipts),'outcomes',sum(not r['error'] for r in rows)*2,'actual C calls')

def child(path):
 r=json.loads(Path(path).read_text());library=ctypes.CDLL(str(OUT/'fixture.so'))
 addr=ctypes.cast(getattr(library,r['symbol']),ctypes.c_void_p).value
 i32=lambda n:struct.pack('<i',n)
 u64=lambda n:struct.pack('<Q',n&((1<<64)-1))
 code=bytearray(b'\x55\x48\x89\xe5\x48\x81\xec'+i32(r['frame']))
 for t,v,disp in zip(r['tags'],r['values'],r['slots']):
  bits=struct.pack('<d',v) if t==12 else u64(v)
  code+=b'\x48\xb8'+bits+b'\x48\x89\x85'+i32(disp)
 call_start=len(code);call=Path(r['code']).read_bytes();code+=call
 # Assert post-call RSP exactly equals frame bottom before restoring it.
 code+=b'\x48\x89\xe0\x48\x29\xe8\x48\x3d'+i32(-r['frame'])+b'\x74\x02\x0f\x0b'
 if r['ret']==0:code+=b'\x48\xb8'+u64(333)
 else:code+=b'\x48\x8b\x85'+i32(r['dst'])
 code+=b'\xc9\xc3';got=len(code);code+=u64(addr)
 disp=call_start+r['fixup'];code[disp:disp+4]=i32(got-(disp+4))
 mem=mmap.mmap(-1,len(code),prot=mmap.PROT_READ|mmap.PROT_WRITE|mmap.PROT_EXEC);mem.write(code)
 fun=ctypes.CFUNCTYPE(ctypes.c_uint64)(ctypes.addressof(ctypes.c_char.from_buffer(mem)))
 actual=fun();expected=struct.unpack('<Q',struct.pack('<d',r['result']))[0] if r['ret']==12 else r['result']&((1<<64)-1)
 assert actual==expected,(actual,expected)
 print('PASS',r['name'])

def mutations():
 """Compile three changed MIND sources; crashes never count as detection."""
 global HERE,OUT
 original_here,original_out=HERE,OUT
 source=(HERE/'native_c_call_emit.mind').read_text()
 tests=[
  {'id':'import-bound','old':'if import_id > 255 { return 10; }','new':'if import_id > 256 { return 10; }','case':'bad-import-256','mode':'driver'},
  {'id':'zero-metadata','old':'__mind_store_i64(metadata + 24, 0);','new':'__mind_store_i64(metadata + 24, 777);','case':'bad-frame-0','mode':'driver'},
  {'id':'narrow-result','old':'return nb_store_rax(nb_push(nce_op3(code, 72, 15, 190), 192), disp);','new':'return nb_store_rax(code, disp);','case':'excess-result-1','mode':'execution'},
 ]
 out=OUT/'mutations';out.mkdir(exist_ok=False)
 payload=(json.dumps(tests,indent=2)+'\n').encode()
 (out/'CASES.json').write_bytes(payload);(out/'CASES.json').chmod(0o444)
 print('MUTATIONS_FROZEN',sha(payload),flush=True)
 rows=cases();receipts=[]
 try:
  for test in tests:
   dest=out/test['id'];dest.mkdir();inputs=dest/'inputs';inputs.mkdir()
   assert source.count(test['old'])==1,'MUTATION_SITE_DRIFT'
   for name in ['main.mind','native_c_call_plan.mind','native_c_call_emit_driver.mind']:
    (inputs/name).write_bytes((original_here/name).read_bytes())
   changed=source.replace(test['old'],test['new'])
   (inputs/'native_c_call_emit.mind').write_text(changed)
   HERE=inputs;OUT=dest;build()
   r=next(x for x in rows if x['name']==test['case'])
   p=run('mutant-driver',[dest/'build/native.elf'],data=packet(r))
   baseline=(original_out/'observed'/('native-'+test['case']+'.stdout')).read_bytes()
   assert p.stdout!=baseline and not p.stderr
   receipt={'id':test['id'],'source_sha256':sha(changed.encode()),'driver_exit':p.returncode}
   if test['mode']=='execution':
    error,size,version,fixup,index,reserve=struct.unpack('<6q',p.stdout[:48])
    assert error==0 and size==len(p.stdout)-48
    (dest/'fixture.so').write_bytes((original_out/'fixture.so').read_bytes())
    observed=dest/'observed';observed.mkdir();code=observed/'mutation.call'
    code.write_bytes(p.stdout[48:])
    desc=dict(r,symbol='fixture_'+str(rows.index(r)),code=str(code),fixup=fixup)
    path=observed/'execution.json';path.write_text(json.dumps(desc))
    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--child',str(path)],
                          capture_output=True,timeout=15,preexec_fn=limits)
    (dest/'execution.stdout').write_bytes(result.stdout)
    (dest/'execution.stderr').write_bytes(result.stderr)
    assert result.returncode==1 and b'AssertionError' in result.stderr,'MUTANT_NOT_NORMALLY_DETECTED'
    receipt['execution_exit']=result.returncode
   receipt['detected']=True;receipts.append(receipt)
   (out/'RESULTS.json').write_text(json.dumps(receipts,indent=2)+'\n')
 finally:
  HERE,OUT=original_here,original_out
 assert (HERE/'native_c_call_emit.mind').read_text()==source
 print('SOURCE_MUTATIONS',len(receipts),'normally detected')

def main():
 global OUT,ARGS
 if len(sys.argv)==3 and sys.argv[1]=='--child':
  path=Path(sys.argv[2]).resolve();OUT=path.parent.parent
  child(path);return 0
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--enumerate',action='store_true')
 parser.add_argument('--mutations',action='store_true')
 parser.add_argument('--evidence',type=Path,required=True)
 parser.add_argument('--seed',type=Path)
 parser.add_argument('--seed-sha256')
 parser.add_argument('--mindc',type=Path)
 parser.add_argument('--mindc-sha256')
 parser.add_argument('--clang',default='clang')
 ARGS=parser.parse_args();OUT=ARGS.evidence.resolve()
 if ARGS.enumerate:
  OUT.mkdir(parents=True,exist_ok=False);freeze();return 0
 assert ARGS.seed and ARGS.seed_sha256 and ARGS.mindc and ARGS.mindc_sha256,'explicit compiler identities required'
 assert json.loads((OUT/'cases/CASES.json').read_text())==cases(),'CASE_DRIFT'
 source_names=['main.mind','native_c_call_plan.mind','native_c_call_emit.mind','native_c_call_emit_driver.mind']
 sources={n:sha((HERE/n).read_bytes()) for n in source_names}
 build();execute()
 if ARGS.mutations:mutations()
 assert {n:sha((HERE/n).read_bytes()) for n in source_names}==sources,'SOURCE_DRIFT'
 assert sha(ARGS.seed.read_bytes())==ARGS.seed_sha256,'SEED_DRIFT'
 assert sha(ARGS.mindc.read_bytes())==ARGS.mindc_sha256,'MINDC_DRIFT'
 (OUT/'SOURCE_HASHES.json').write_text(json.dumps(sources,indent=2)+'\n')
 (OUT/'FIXTURE_HASHES.json').write_text(json.dumps({n:sha((OUT/n).read_bytes()) for n in ['fixture.c','fixture.S','fixture.so']},indent=2)+'\n')
 return 0

if __name__=='__main__':
 raise SystemExit(main())
