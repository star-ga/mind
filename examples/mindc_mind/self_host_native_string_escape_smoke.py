#!/usr/bin/env python3
"""Source-bound native/MLIR string values and bounded decoder diagnostics.

Author: STARGA Inc. Product compilation and decoding execute MIND. Python is
an external transport/observation harness, not a compiler runtime dependency.
Run --enumerate first, then supply exact native and MLIR compiler hashes.
"""
import argparse,hashlib,json,os,resource,struct,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.dont_write_bytecode=True
from self_host_native_cf_smoke import v2_header,image,STD_MODULES,BLOB,std_blob
from native_dynamic_writer_check import encoder

def sha(data): return hashlib.sha256(data).hexdigest()

def original_cases():
 rows=[]
 write='let s:String=VALUE; if __mind_write(1,string_addr(s),string_len(s),0-1)!=string_len(s) {return 71;} return 0;'
 def literal(name,raw,want):
  source='import std.string;\nfn main()->i64 {'+write.replace('VALUE','"'+raw+'"')+'}\n'
  rows.append(dict(name=name,source=source,stdout_hex=want.hex(),refuse=False,plain=('\\' not in raw)))
 literal('plain','hello',b'hello');literal('empty','',b'')
 for name,raw,want in [('newline',r'\n',b'\n'),('tab',r'\t',b'\t'),('return',r'\r',b'\r'),('nul',r'\0',b'\0'),('backslash',r'\\',b'\\'),('quote',r'\"',b'"'),('apostrophe',r"\'",b"'")]:literal(name,raw,want)
 literal('utf8-nul','π'+r'\0'+'界'+r'\n', 'π'.encode()+b'\0'+'界'.encode()+b'\n')
 literal('all',r'A\nB\tC\rD\0E\\F\"G\'H',b'A\nB\tC\rD\0E\\F"G\'H')
 literal('repeated',r'\n\n\\\\\"\"\0\0',b'\n\n\\\\""\0\0')
 for n in (255,256,257,4095,4096,4097,65535):literal('leaf-'+str(n),'a'*n+r'\n'+'z',b'a'*n+b'\nz')
 for name,decl,value in [('constant','const VALUE: String = "left\\nright"\n','VALUE'),('helper','fn value()->String { "left\\nright" }\n','value()'),('branch','fn value(n:i64)->String { if n==1 {return "left\\nright";} "wrong" }\n','value(1)')]:
  rows.append(dict(name=name,source='import std.string;\n'+decl+'fn main()->i64 {'+write.replace('VALUE',value)+'}\n',stdout_hex=b'left\nright'.hex(),refuse=False,plain=False))
 multi='import std.string;\nfn main()->i64 {'+write.replace('VALUE','"A\\n"').replace('return 0;','')+write.replace('VALUE','"B\\0C"').replace('let s:','let t:').replace('string_addr(s)','string_addr(t)').replace('string_len(s)','string_len(t)')+'}\n'
 rows.append(dict(name='multiple',source=multi,stdout_hex=b'A\nB\0C'.hex(),refuse=False,plain=False))
 for name,raw in [('unknown',r'\q'),('hex',r'\x41'),('unicode',r'\u{41}'),('trailing','abc\\'),('escaped-final-quote','abc\\"')]:
  rows.append(dict(name=name,source='import std.string;\nfn main()->i64 {let s:String="'+raw+'"; return string_len(s);}\n',refuse=True))
 return rows

def helper_cases():
 rows=[]
 def add(name,mode=0,data=b'',lo=0,hi=None,off=0,n=0,want=b''):
  header=struct.pack('<7q',mode,lo,len(data) if hi is None else hi,off,n,len(data),0)
  rows.append(dict(name=name,input_hex=(header+data).hex(),stdout_hex=want.hex()))
 for name,src,expected,lo,hi in [('empty',b'',b'',0,0),('all',b'\\n\\t\\r\\0\\\\\\"\\\'',b'\n\t\r\0\\"\'',0,14),('prefix-suffix',b'!a\\nb!',b'a\nb',1,5),('unknown',b'a\\q',None,0,3),('trailing',b'a\\',None,0,2),('bad-lo',b'',None,-1,0),('inverted',b'ab',None,2,1)]:
  add(name,data=src,lo=lo,hi=hi,want=struct.pack('<q',-1 if expected is None else len(expected))+(expected or b''))
 for off,n,want in [(0,0,1),(0,8388608,1),(0,8388609,0),(8388608,0,1),(8388608,1,0),(8388607,1,1),(8388607,2,0),(-1,0,0),(0,-1,0),(9223372036854775807,1,0),(1,9223372036854775807,0)]:
  add('room-'+str(off)+'-'+str(n),mode=1,off=off,n=n,want=struct.pack('<q',want))
 for n in (4097,8388608):add('copy-'+str(n),mode=2,n=n,want=struct.pack('<4q',n,1,n,3))
 return rows

def cases():
 rows=original_cases()
 # The initial diagnostic mislabeled this valid escaped-quote source as a
 # refusal. Its unchanged input returns length4. See the correction record.
 for r in rows:
  if r['name']=='escaped-final-quote':
   r.update(refuse=False,plain=False,exit=4,stdout_hex='')
 write='let s:String=VALUE; if __mind_write(1,string_addr(s),string_len(s),0-1)!=string_len(s) {return 71;} return 0;'
 def add(name,decl,body,want):
  rows.append(dict(name=name,source='import std.string;\n'+decl+'\nfn main()->i64 {'+body+'}\n',refuse=False,plain=False,stdout_hex=want.hex()))
 add('const-semicolon','const VALUE:String="x\\tY";',write.replace('VALUE','VALUE'),b'x\tY')
 add('const-empty','const VALUE:String=""',write,b'')
 add('const-utf8','const VALUE:String="π\\0界"',write,'π'.encode()+b'\0'+'界'.encode())
 add('local-shadows-const','const VALUE:String="wrong"','let VALUE:String="local\\n";'+write,b'local\n')
 add('param-shadows-const','const VALUE:String="wrong"\nfn emit(VALUE:String)->i64 {'+write+'}','return emit("param\\n");',b'param\n')
 for name,decl in [('const-duplicate','const VALUE:String="a"\nconst VALUE:String="b"'),('const-nonscalar','const VALUE:i64="a"'),('const-expression','const VALUE:String="a"+"b"'),('const-lowercase','const value:String="a"')]:
  rows.append(dict(name=name,source='import std.string;\n'+decl+'\nfn main()->i64 {return string_len('+('value' if name=='const-lowercase' else 'VALUE')+');}\n',refuse=True,native_only=True))
 # Independent review found six silent wrong results in these exact binding
 # shapes. Keep the no-global controls and the original failing inputs.
 for kind in ('for','tuple','match'):
  for suffix,decl in [('none',''),('integer','const VALUE:i64=91;\n'),('string','const VALUE:String="global";\n')]:
   if kind=='for': body='for VALUE in 7..8 { return VALUE; } 3'
   elif kind=='tuple': body='let (VALUE, other) = (7, 9); return VALUE;'
   else:
    decl+='enum E { V(i64), N }\n'
    body='let e:E=E::V(7); match e { E::V(VALUE) => { return VALUE; }, E::N => { return 3; } } 4'
   rows.append(dict(name='binder-'+kind+'-'+suffix,source='import std.string;\n'+decl+'fn main()->i64 { '+body+' }\n',refuse=False,plain=False,exit=7,stdout_hex=''))
 for name,body in [('same-line','let VALUE = 1 + 6; return VALUE;'),('newline','let VALUE = 1\n + 6; return VALUE;'),('mut','let mut VALUE:i64=6; VALUE=VALUE+1; return VALUE;')]:
  rows.append(dict(name='binder-'+name,source='import std.string;\nconst VALUE:i64=91;\nfn main()->i64 {'+body+'}\n',refuse=False,plain=False,exit=7,stdout_hex=''))
 rows.append(dict(name='constructor-global-value',source='import std.string;\nconst VALUE:i64=7;\nenum E { V(i64), N }\nfn main()->i64 {let e:E=E::V(VALUE); match e {E::V(local)=>{return local;},E::N=>{return 3;}} 4}\n',refuse=False,plain=False,exit=7,stdout_hex=''))
 for name,decl,body in [
  ('payload','enum E { V(i64), N }','let e:E=E::V(7); match e {E::V(VALUE) + + {return VALUE;},E::N=>{return 3;}} 4'),
  ('payload-const','const VALUE:i64=91; enum E { V(i64), N }','let e:E=E::V(7); match e {E::V(VALUE) + + {return VALUE;},E::N=>{return 3;}} 4'),
  ('scalar','','match 7 {7 + + {return 7;},_=>{return 3;}} 4'),
  ('wildcard','','match 7 {_ + + {return 7;}} 4'),
  ('qualified','enum E { V, N }','let e:E=E::V; match e {E::V + + {return 7;},_=>{return 3;}} 4')]:
  rows.append(dict(name='bad-arrow-'+name,source='import std.string;\n'+decl+'\nfn main()->i64 {'+body+'}\n',refuse=True))
 return rows

def limits():
 resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
 resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 resource.setrlimit(resource.RLIMIT_CPU,(500,505))

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--enumerate',action='store_true');p.add_argument('--evidence',type=Path,required=True)
 p.add_argument('--native',type=Path);p.add_argument('--native-sha256');p.add_argument('--mlir',type=Path);p.add_argument('--mlir-sha256')
 args=p.parse_args();out=args.evidence.resolve();out.mkdir(parents=True,exist_ok=True)
 contract=(json.dumps({'source':cases(),'helpers':helper_cases()},indent=2)+'\n').encode();manifest=out/'CASES.json'
 if args.enumerate:
  with manifest.open('xb') as f:f.write(contract)
  manifest.chmod(0o444);print('FROZEN',len(cases()),len(helper_cases()),sha(contract));return 0
 assert manifest.read_bytes()==contract,'CASE_DRIFT'
 with (out/'execution.started').open('x') as f:f.write('One execution per frozen evidence directory.\n')
 assert args.native and args.mlir,'MISSING_COMPILER'
 args.native=args.native.resolve();args.mlir=args.mlir.resolve()
 for path,digest in [(args.native,args.native_sha256),(args.mlir,args.mlir_sha256)]:
  assert path and digest and sha(path.read_bytes())==digest,'COMPILER_IDENTITY'
 rows=[]
 std=HERE.parents[1]/'std'
 deps=[HERE/'main.mind',HERE/'string_escape_driver.mind',Path(__file__).resolve(),args.native,args.mlir,HERE/'self_host_native_cf_smoke.py',HERE/'native_dynamic_writer_check.py',HERE/'_stdlib_manifest.py']
 deps.extend(std/(name+'.mind') for name in STD_MODULES)
 inputs={str(path.resolve()):sha(path.read_bytes()) for path in deps}
 (out/'INPUTS.json').write_text(json.dumps(inputs,indent=2)+'\n')
 assert BLOB==std_blob(),'IMPORTED_STDLIB_DRIFT'
 (out/'STDLIB.sha256').write_text(sha(BLOB)+'\n')
 def intact():
  assert manifest.read_bytes()==contract,'CASE_DRIFT'
  for name,digest in inputs.items():assert sha(Path(name).read_bytes())==digest,('INPUT_DRIFT',name)
 env=os.environ.copy()
 for key in list(env):
  if key.startswith('LD_') or key in ('GLIBC_TUNABLES','MINDC_NATIVE_ELF','MINDC_SO'):env.pop(key)
 env['MINDC_BUILD_TOOL_TIMEOUT_SECS']='300'
 # Both legs use this exact repository's standard library. An inherited path
 # must not silently compare the native prefix against a different std tree.
 env['MIND_STDLIB_PATH']=str(std)
 def run(name,argv,data=None):
  intact()
  before=time.monotonic();q=subprocess.run(list(map(str,argv)),input=data,capture_output=True,cwd=out,timeout=600,preexec_fn=limits,env=env)
  (out/(name+'.stdout')).write_bytes(q.stdout);(out/(name+'.stderr')).write_bytes(q.stderr)
  (out/(name+'.json')).write_text(json.dumps(dict(argv=list(map(str,argv)),exit=q.returncode,elapsed=time.monotonic()-before,stdin_sha256=None if data is None else sha(data),stdout_sha256=sha(q.stdout),stderr_sha256=sha(q.stderr)),indent=2)+'\n')
  intact()
  print(name,q.returncode,len(q.stdout),flush=True);return q
 def build(name,src,raw=False):
  f=out/(name+'.mind');f.write_bytes(src)
  packet=v2_header()+(struct.pack('<2q',0,len(src))+src if raw else image(src.replace(b'import std.string;\n',b'')))
  (out/(name+'.packet')).write_bytes(packet)
  q=run(name+'-native-build',[args.native],packet)
  return f,q
 failures=[]
 for r in cases():
  name=r['name'];f,nb=build(name,r['source'].encode())
  if r['refuse']:
   ok=nb.returncode in (2,3) and not nb.stdout and bool(nb.stderr)
   rows.append(dict(name=name,leg='native-refusal',passed=bool(ok)))
   if not r.get('native_only'):
    me=out/(name+'.mlir.elf');q=run(name+'-mlir-build',[args.mlir,'build',f,'--backend','mlir','--release','--out',me])
    rows.append(dict(name=name,leg='mlir-refusal',passed=q.returncode>0 and not me.exists()))
   continue
  ok=nb.returncode==0 and not nb.stderr and nb.stdout.startswith(b'\x7fELF')
  if not ok:rows.append(dict(name=name,leg='native-build',passed=False));continue
  ne=out/(name+'.native.elf');ne.write_bytes(nb.stdout);ne.chmod(0o500)
  me=out/(name+'.mlir.elf');mb=run(name+'-mlir-build',[args.mlir,'build',f,'--backend','mlir','--release','--out',me])
  for leg,exe in [('native',ne),('mlir',me)]:
   if not exe.exists() or (leg=='mlir' and mb.returncode!=0):rows.append(dict(name=name,leg=leg+'-build',passed=False));continue
   q=run(name+'-'+leg+'-run',[exe]);rows.append(dict(name=name,leg=leg,passed=(q.returncode,q.stdout,q.stderr)==(r.get('exit',0),bytes.fromhex(r['stdout_hex']),b'')))
 text=(HERE/'main.mind').read_text()
 names=['load_byte','char_escape_value','string_escape_value','nb_rodata_addr','nb_arena_addr','nb_string_step','nb_string_step_good','nb_string_step_advance','nb_string_decoded_length','nb_string_decoded_byte','nb_rod_room','nb_rod_decoded_copy']
 src=('\n'.join(encoder(text,n) for n in names)+'\n').encode()+(HERE/'string_escape_driver.mind').read_bytes()
 f,q=build('helpers',src,raw=True);assert q.returncode==0 and not q.stderr and q.stdout.startswith(b'\x7fELF')
 ne=out/'helpers.native.elf';ne.write_bytes(q.stdout);ne.chmod(0o500)
 me=out/'helpers.mlir.elf';q=run('helpers-mlir-build',[args.mlir,'build',f,'--backend','mlir','--release','--out',me]);assert q.returncode==0
 for r in helper_cases():
  for leg,exe in [('native',ne),('mlir',me)]:
   q=run(r['name']+'-'+leg,[exe],bytes.fromhex(r['input_hex']));rows.append(dict(name=r['name'],leg=leg,passed=(q.returncode,q.stdout,q.stderr)==(0,bytes.fromhex(r['stdout_hex']),b'')))
 (out/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n')
 failures=[r for r in rows if not r['passed']]
 print('FAIL' if failures else 'PASS',len(rows),'outcomes',failures,flush=True)
 return int(bool(failures))
if __name__=='__main__':raise SystemExit(main())
