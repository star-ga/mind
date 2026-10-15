#!/usr/bin/env python3
"""Author: STARGA Inc. Diagnostic host orchestration for the MIND MCABI001 registry.
No C ABI admission or native linking is enabled by this gate. Each oracle request
runs in a fresh bounded process. Expectations are declared before execution.
"""
from pathlib import Path
import argparse, ctypes, hashlib, json, os, resource, struct, subprocess, sys
HERE=Path(__file__).resolve().parent
SCALARS=['i8','u8','i16','u16','i32','u32','i64','u64','isize','usize','f32','f64','bool']
def ty(t):
 if t is None:return b'\0'
 if t.startswith('*const '):return bytes([32])+ty(t[7:])
 if t.startswith('*mut '):return bytes([33])+ty(t[5:])
 return bytes([SCALARS.index(t)+1])
def wire(rows):
 b=b'MCABI001'+struct.pack('<I',len(rows))
 for name,cc,unsafe,varargs,ret,params in rows:
  n=name.encode();b+=struct.pack('<I',len(n))+n+bytes([cc,unsafe,varargs])+struct.pack('<I',len(params))+ty(ret)+b''.join(map(ty,params))
 return b
def cases():
 out=[]
 def ok(name,source,rows):out.append((name,source.encode(),0,wire(rows)))
 def no(name,source,code):out.append((name,source.encode(),code,b''))
 ok('empty','',[])
 ok('static','fn main() -> i64 { 7 }',[])
 ok('golden','extern "C" callconv(.sysv) { safe fn probe(a:u32,p:*const u8)->i32; }',[('probe',1,0,0,'i32',['u32','*const u8'])])
 for s in SCALARS:ok('type_'+s,f'extern "C" {{ fn probe(x:{s})->{s}; }}',[('probe',0,1,0,s,[s])])
 for i,cc in enumerate(['c','sysv','win64','aapcs']):ok('cc_'+cc,f'extern "C" callconv(.{cc}) {{ fn f(); }}',[('f',i,1,0,None,[])])
 ok('pointers','extern "C" { unsafe fn p(x:*const *mut i32,y:*mut u8)->*const *const u8; }',[('p',0,1,0,'*const *const u8',['*const *mut i32','*mut u8'])])
 ok('varargs','extern "C" { fn a(x:i32,...)->i64; safe fn b(...); }',[('a',0,1,1,'i64',['i32']),('b',0,0,1,None,[])])
 ok('order','extern "C" { fn z(x:u8,) -> i32 fn a() } extern "C" { safe fn m(); }',[('z',0,1,0,'i32',['u8']),('a',0,1,0,None,[]),('m',0,0,0,None,[])])
 no('comments','// extern "C" { fn fake(); }\nextern "C" { /* extern */ fn real(); }',2)
 ok('line_comments','// extern "C" { fn fake(); }\nextern "C" { // extern\n fn real(); }',[('real',0,1,0,None,[])])
 deep='*const '*32+'u8';ok('pointer_depth32',f'extern "C" {{ fn p(x:{deep}); }}',[('p',0,1,0,None,[deep])])
 params=','.join(f'x{i}:i32' for i in range(256));ok('params256',f'extern "C" {{ fn p({params}); }}',[('p',0,1,0,None,['i32']*256)])
 longname='x'*255;ok('name255',f'extern "C" {{ fn {longname}(); }}',[(longname,0,1,0,None,[])])
 for name,t in [('void','void'),('ptr_void','*const void'),('alias','Alias'),('reference','&i64'),('array','[u8;4]'),('tuple','(i32,i32)'),('fnptr','fn(i32)->i32'),('missing_ptr_qual','*u8'),('str','String')]:
  no('reject_'+name,f'extern "C" {{ fn f(x:{t}); }}',3)
 for name,s in [('duplicate','extern "C" { fn f(); fn f(); }'),('conflict','extern "C" { fn f()->i32; } extern "C" { fn f()->u32; }')]:no(name,s,5)
 no('reserved','extern "C" { fn __mind_imposter(); }',6)
 no('local_collision','extern "C" { fn f(); } fn f() -> i64 { 1 }',9)
 ok('parameter_name256','extern "C" { fn probe('+('x'*256)+':i32); }',[('probe',0,1,0,None,['i32'])])
 no('duplicate_parameter','extern "C" { fn probe(x:i32,x:u32); }',5)
 no('import_shadow','import dependency as u64; extern "C" { fn probe(x:u64); }',3)
 no('import_path_shadow','import dependency.u64; extern "C" { fn probe(x:*const u64); }',3)
 no('import_collision','import dependency as probe; extern "C" { fn probe(); }',9)
 no('alias_shadow','type u64=i32; extern "C" { fn probe(x:u64); }',3)
 no('alias_shadow_pointer','type u8=i32; extern "C" { fn probe(x:*const u8); }',3)
 no('alias_shadow_after','extern "C" { fn probe()->u64; } type u64=i32;',3)
 no('const_collision','const probe:i64=7; extern "C" { fn probe(); }',9)
 no('type_collision','type probe=i32; extern "C" { fn probe(); }',9)
 ok('unused_shadow','type u64=i32; extern "C" { fn probe(x:i32); }',[('probe',0,1,0,None,['i32'])])
 no('nested_scope','module hidden { extern "C" { fn f(); } }',7)
 no('unknown_abi','extern "Rust" { fn f(); }',4)
 no('unknown_cc','extern "C" callconv(.bogus) { fn f(); }',4)
 no('pointer_depth33','extern "C" { fn f(x:'+'*mut '*33+'u8); }',10)
 no('params257','extern "C" { fn f('+','.join(f'x{i}:i64' for i in range(257))+'); }',10)
 no('name256','extern "C" { fn '+'x'*256+'(); }',2)
 for name,s in [
 ('no_close','extern "C" { fn f();'),('no_open','extern "C" fn f();'),('bad_name','extern "C" { fn 5(); }'),
 ('missing_colon','extern "C" { fn f(x i32); }'),('double_comma','extern "C" { fn f(x:i32,,y:i32); }'),
 ('varargs_tail','extern "C" { fn f(...,x:i32); }'),('spaced_dots','extern "C" { fn f(. . .); }'),
 ('function_body','extern "C" { fn f() {} }'),('trailing_junk','extern "C" { fn f() garbage; }'),
 ('malformed_callconv','extern "C" callconv(sysv) { fn f(); }'),('no_function','extern "C" { xyz }')]:no(name,s,2)
 return out

def limits(oracle=False):
 n=(1536<<20) if oracle else (2<<30)
 resource.setrlimit(resource.RLIMIT_AS,(n,n));resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(90,95))
def oracle_child(so):
 limits(True)
 src=sys.stdin.buffer.read();buf=ctypes.create_string_buffer(src+b'\0'*8);out=(ctypes.c_int64*4)()
 lib=ctypes.CDLL(str(so));f=lib.selftest_nb_cabi;f.argtypes=[ctypes.c_int64]*3;f.restype=ctypes.c_int64
 rc=f(ctypes.addressof(buf),len(src),ctypes.addressof(out))
 if rc:
  assert out[0]==0 and out[1]==0 and out[2]==rc,(rc,list(out))
  sys.stderr.buffer.write(struct.pack('<qq',out[2],out[3]));return rc
 assert out[2]==0 and 12<=out[1]<=64<<20,list(out)
 sys.stdout.buffer.write(ctypes.string_at(out[0],out[1]));return 0

def main():
 p=argparse.ArgumentParser();p.add_argument('--oracle-child',type=Path);p.add_argument('--oracle',type=Path);p.add_argument('--native',type=Path);p.add_argument('--evidence',type=Path);p.add_argument('--enumerate',action='store_true');a=p.parse_args()
 if a.oracle_child:return oracle_child(a.oracle_child)
 c=cases();assert c,'empty registry corpus';assert a.evidence;a.evidence.mkdir(parents=True,exist_ok=True)
 manifest=[dict(name=n,source_hex=s.hex(),source_sha256=hashlib.sha256(s).hexdigest(),expected_rc=r,expected_stdout_hex=b.hex()) for n,s,r,b in c]
 packet=json.dumps(manifest,indent=2)+'\n';dst=a.evidence/'cases.json'
 if a.enumerate:dst.write_text(packet);print(f'PRELISTED {len(c)} cases SHA256 {hashlib.sha256(packet.encode()).hexdigest()}');return 0
 assert dst.read_text()==packet,'Case enumeration drift';assert a.oracle and a.native
 rows=[]
 for n,s,expected,out in c:
  result=[]
  for leg,cmd,data in [('native',[str(a.native)],struct.pack('<q',len(s))+s),('oracle',[sys.executable,str(Path(__file__).resolve()),'--oracle-child',str(a.oracle)],s)]:
   q=subprocess.run(cmd,input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=100,preexec_fn=lambda:limits())
   base=a.evidence/(n+'.'+leg);base.with_suffix(base.suffix+'.stdout').write_bytes(q.stdout);base.with_suffix(base.suffix+'.stderr').write_bytes(q.stderr)
   row={'name':n,'leg':leg,'argv':cmd,'exit':q.returncode,'stdout_sha256':hashlib.sha256(q.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(q.stderr).hexdigest(),'expected_exit':expected}
   (a.evidence/(n+'.'+leg+'.json')).write_text(json.dumps(row,indent=2)+'\n');rows.append(row);result.append((q.returncode,q.stdout,q.stderr))
   assert q.returncode==expected,(n,leg,'exit',q.returncode,expected,q.stderr[:500])
   assert q.stdout==out,(n,leg,'wire',q.stdout[:100],out[:100])
   if expected==0:assert q.stderr==b'',(n,leg,q.stderr)
   else:assert len(q.stderr)==16 and struct.unpack('<qq',q.stderr)[0]==expected,(n,leg,q.stderr)
  assert result[0]==result[1],(n,'native/oracle mismatch',result)
  print('PASS',n,flush=True)
 (a.evidence/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(f'ALL PASS {len(c)} registry cases; native/oracle exit/stdout/stderr exact. Metadata only.');return 0
if __name__=='__main__':raise SystemExit(main())
