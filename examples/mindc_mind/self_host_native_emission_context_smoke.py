#!/usr/bin/env python3
"""Independent diagnostic transport for native emission contexts.

Executable fixture logic is MIND. This harness records exact builds, inputs,
streams and hashes; component success does not authorize production admission.
The original source runtime contract and static byte contract remain separate.
"""
import argparse, hashlib, json, os, resource, shutil, struct, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
POINTER_CATALOGUE=HERE/'native_pointer_origin_cases.json'
DRIVER=r"""
// STARGA Inc. Diagnostic-only transport; no compiler admission or proof authority.
pub fn ect_read(p:i64,n:i64)->i64 {
 let mut i:i64=0;
 while i<n { let k:i64=__mind_read(0,p+i,n-i,0-1); if k<=0{return 0;} i=i+k; }
 1
}
pub fn ect_write(p:i64,n:i64)->i64 {
 let mut i:i64=0;
 while i<n { let k:i64=__mind_write(1,p+i,n-i,0-1); if k<=0{return 0;} i=i+k; }
 1
}
pub fn ect_word(v:i64)->i64 {
 let p:i64=__mind_alloc(8); __mind_store_i64(p,v);
 let r:i64=ect_write(p,8); __mind_free(p); r
}
pub fn ect_emit(op:i64,ctx:i64)->String {
 let s:String=string_new();
 if op==0{return nb_emit_alloc_ctx(s,0-24,ctx);}
 if op==1{return nb_emit_free_at(s,0-16,__mind_load_i64(ctx+16));}
 if op==2{return nb_emit_realloc_at(s,0-16,0-24,__mind_load_i64(ctx+16));}
 if op==3{return nb_emit_argc_ctx(s,0-8,ctx);}
 nb_emit_argv_ctx(s,0-16,0-8,ctx)
}
pub fn ect_route(src:i64,n:i64,user:i64)->i64 {
 let ts:Vec=lex(src,n); let ast:i64=parse(ts,src);
 let items:i64=ast_child0(ast); let ni:i64=ast_aux(ast);
 let fns:i64=__mind_alloc((ni+1)*8); let nf:i64=nb_collect_fns(items,ni,0,fns,0);
 let route:i64=__mind_alloc((nf+1)*8); let mut count:i64=0; let mut i:i64=0;
 while i<nf { let lo:i64=ast_span_lo(ast_child0(__mind_load_i64(fns+i*8)));
  if lo>=user {__mind_store_i64(route+8+count*8,lo);count=count+1;}
  i=i+1;
 }
 __mind_store_i64(route,count); __mind_free(fns); route
}
pub fn ect_parts(src:i64,n:i64,hash:i64,user:i64,forced:i64,req:i64)->i64 {
 let mut route:i64=0;
 if forced==1 {route=ect_route(src,n,user);}
 let parts:i64=nec_lower_parts(src,n,hash,user,route,req);
 __mind_free(route);
 if parts==0 {ect_word(0);return 0;}
 // Probe ownership after construction: caller source and request are changed.
 __mind_store_i8(src,0); __mind_store_i64(req,0);
 if nb_parts_patch_local(parts)!=1 {nb_parts_free(parts);__mind_free(parts);ect_word(0);return 0;}
 let head:i64=__mind_alloc(80);
 __mind_store_i64(head,1);
 __mind_store_i64(head+8,__mind_load_i64(parts+16));
 __mind_store_i64(head+16,__mind_load_i64(parts+40));
 __mind_store_i64(head+24,__mind_load_i64(parts+64));
 __mind_store_i64(head+32,__mind_load_i64(parts+112));
 __mind_store_i64(head+40,__mind_load_i64(parts+120));
 __mind_store_i64(head+48,__mind_load_i64(parts+128));
 __mind_store_i64(head+56,__mind_load_i64(parts+136));
 __mind_store_i64(head+64,__mind_load_i64(parts+168));
 __mind_store_i64(head+72,__mind_load_i64(parts+152));
 ect_write(head,80);
 ect_write(__mind_load_i64(parts+8),__mind_load_i64(parts+16));
 ect_write(__mind_load_i64(parts+32),__mind_load_i64(parts+40));
 ect_write(__mind_load_i64(parts+56),__mind_load_i64(parts+64));
 ect_write(__mind_load_i64(parts+80),__mind_load_i64(parts+112)*8);
 ect_write(__mind_load_i64(parts+88),__mind_load_i64(parts+112)*16);
 ect_write(__mind_load_i64(parts+96),__mind_load_i64(parts+120)*24);
 ect_write(__mind_load_i64(parts+104),__mind_load_i64(parts+120)*8);
 ect_write(__mind_load_i64(parts+144),__mind_load_i64(parts+152));
 __mind_free(head); nb_parts_free(parts); __mind_free(parts); 0
}
pub fn ect_image(src:i64,n:i64,hash:i64,user:i64,forced:i64,req:i64)->i64 {
 let mut route:i64=0;if forced==1{route=ect_route(src,n,user);}
 let policy:i64=__mind_alloc(48);let diag:i64=__mind_alloc(32);
 __mind_store_i64(policy,__mind_load_i64(req));let mut i:i64=1;
 while i<6{__mind_store_i64(policy+i*8,__mind_load_i64(req+(i+4)*8));i=i+1;}
 let image:String=nsi_compile(src,n,hash,user,route,policy,diag);
 ect_write(diag,32);ect_write(string_addr(image),string_len(image));
 __mind_free(string_addr(image));__mind_free(diag);__mind_free(policy);__mind_free(route);0
}
pub fn main()->i64 {
 let h:i64=__mind_alloc(128);
 if ect_read(h,128)!=1{return 71;}
 let mode:i64=__mind_load_i64(h);let arg:i64=__mind_load_i64(h+8);
 let arg2:i64=__mind_load_i64(h+16);let n:i64=__mind_load_i64(h+24);
 let user:i64=__mind_load_i64(h+32);let forced:i64=__mind_load_i64(h+40);
 if n<0{return 71;}if n>8388608{return 71;}
 let src:i64=__mind_alloc(n+16);let hash:i64=__mind_alloc(32);
 if ect_read(src,n)!=1{return 71;}if ect_read(hash,32)!=1{return 71;}
 let mut z:i64=0;while z<16{__mind_store_i8(src+n+z,0);z=z+1;}
 let eof:i64=__mind_alloc(1);if __mind_read(0,eof,1,0-1)!=0{return 72;}
 let req:i64=h+48;
 if mode==5{return ect_parts(src,n,hash,user,forced,req);}
 if mode==10{return ect_image(src,n,hash,user,forced,req);}
 if mode==7{ect_word(nec_new(0));ect_word(nec_check(0));return 0;}
 if mode==9{ect_word(nb_ctx_entry_ok(0,arg));return 0;}
 let ctx:i64=nec_new(req);
 if mode==6 {let mut ok:i64=0;if ctx!=0{ok=1;}ect_word(ok);nec_free(ctx);return 0;}
 if ctx==0{return 73;}
 if mode==0 {ect_word(nec_check(ctx));ect_write(ctx,416);}
 if mode==1 {if arg<0{return 71;}if arg>=52{return 71;}
  __mind_store_i64(ctx+arg*8,__mind_load_i64(ctx+arg*8)+arg2);ect_word(nec_check(ctx));}
 if mode==2 {ect_word(nb_ctx_entry_ok(ctx,arg));}
 if mode==3 {ect_word(nb_ctx_intrinsic_ok(ctx,arg));}
 if mode==4 {let s:String=ect_emit(arg,ctx);ect_write(string_addr(s),string_len(s));__mind_free(string_addr(s));}
 if mode==8 {__mind_store_i64(req,0);ect_word(nec_check(ctx));}
 nec_free(ctx);0
}
"""
GEOMETRIES={'A': [1, 4096, 4096, 4096, 4096, 67108864, 67108864, 67108864, 4096, 268435456], 'B': [1, 8192, 4096, 8192, 8193, 33554432, 1048576, 16777216, 8192, 134217728]}
CORRECTIONS=[{'id': 'tamper-B-14', 'original_expected_check': 0, 'expected_check': 1, 'before_execution': True, 'reason': 'Owned request literal8193 to8194 changes no rounded plan field; this is another coherent valid context. nec_check recomputes consistency, not original-source authorization.'}]
RUNTIME_SUPPLEMENT=[{'id': 'negative-main-return', 'source': 'fn main()->i64{return 0-1;}\n', 'exit': 255, 'argv': [], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'float-helper', 'source': 'fn half(x:f64)->f64{return x/2.0;} fn main()->i64{if half(8.0)==4.0{return 0;}return 91;}\n', 'exit': 0, 'argv': [], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'main-i32-argc', 'source': 'fn main(argc:i32)->i64{return argc as i64;}\n', 'exit': 3, 'argv': ['one', 'two'], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'main-raw-pointer', 'source': 'fn main(argc:i64,argv:*const *const u8)->i64{let p:i64=argv as i64;if p==0{return 92;}if __mind_load_i64(p)==0{return 93;}return argc;}\n', 'exit': 3, 'argv': ['one', 'two'], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'literal-escape', 'source': 'fn main()->i64{let s:String="a\\nb";if string_len(s)!=3{return 94;}return __mind_load_i8(string_addr(s)+1);}\n', 'exit': 10, 'argv': [], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'global-literal', 'source': 'const MESSAGE:String="ok";fn main()->i64{return __mind_load_i8(string_addr(MESSAGE));}\n', 'exit': 111, 'argv': [], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'main-unit-refusal', 'source': 'fn main(){let x:i64=1;}\n', 'refuse': True, 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'main-bool-refusal', 'source': 'fn main(x:bool)->i64{return 0;}\n', 'refuse': True, 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'main-f32-refusal', 'source': 'fn main()->f32{return 1.0;}\n', 'refuse': True, 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'unused-extern-refusal', 'source': 'extern "C"{fn getpid()->i32;}fn main()->i64{return 17;}\n', 'refuse': True, 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}]
PRIMITIVE_SHADOW_CASES=[{'id': 'return-alias-shadow', 'source': 'type i64=f64; fn main()->i64{return 1.0;}\n', 'expected': 'normal source admission refusal', 'geometries': ['A', 'B']}, {'id': 'return-struct-shadow', 'source': 'struct i64{value:f64} fn main()->i64{return i64{value:1.0};}\n', 'expected': 'normal source admission refusal', 'geometries': ['A', 'B']}, {'id': 'return-enum-shadow', 'source': 'enum i64{A,B} fn main()->i64{return i64.A;}\n', 'expected': 'normal source admission refusal', 'geometries': ['A', 'B']}, {'id': 'return-import-shadow', 'source': 'import std.math as i64; fn main()->i64{return 0;}\n', 'expected': 'normal source admission refusal', 'geometries': ['A', 'B']}, {'id': 'param-alias-shadow', 'source': 'type i32=u64; fn main(argc:i32)->i64{return 0;}\n', 'expected': 'normal source admission refusal', 'geometries': ['A', 'B']}, {'id': 'pointer-pointee-shadow', 'source': 'type u8=f64; fn main(argc:i64,argv:*const u8)->i64{return 0;}\n', 'expected': 'normal source admission refusal', 'geometries': ['A', 'B']}]
V2_SOURCE_CASES=[{'id': 'v2-array-field-unique-identifier', 'source': 'struct EmissionContextArrayFixture{a:[i64;3]} fn main()->i64{let b:EmissionContextArrayFixture=EmissionContextArrayFixture{a:[3,5,7]};return b.a[1];}\n', 'exit': 5, 'routes': ['automatic', 'force-home'], 'argv': [], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B']}, {'id': 'v2-pointer-nested-unused', 'source': 'fn main(argc:i64,argv:*const *const u8)->i64{return argc;}\n', 'exit': 3, 'argv': ['one', 'two'], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'v2-pointer-single-unused', 'source': 'fn main(argc:i64,argv:*const u8)->i64{return argc;}\n', 'exit': 3, 'argv': ['one', 'two'], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'v2-pointer-helper-param', 'source': 'fn keep(p:*const u8,n:i64)->i64{return n;} fn main(argc:i64,argv:*const u8)->i64{return keep(argv,argc);}\n', 'exit': 3, 'argv': ['one', 'two'], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'v2-pointer-mut-unused', 'source': 'fn main(argc:i64,argv:*mut u8)->i64{return argc;}\n', 'exit': 3, 'argv': ['one', 'two'], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'v2-pointer-missing-qualifier', 'source': 'fn main(argc:i64,argv:*bogus u8)->i64{return argc;}\n', 'refuse': True, 'argv': [], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}, {'id': 'v2-pointer-missing-pointee', 'source': 'fn main(argc:i64,argv:*const)->i64{return argc;}\n', 'refuse': True, 'argv': [], 'stdout_hex': '', 'stderr_hex': '', 'geometries': ['A', 'B'], 'routes': ['automatic']}]

def enumerate_cases():
    A=[1,4096,4096,4096,4096,67108864,67108864,67108864,4096,268435456]
    B=[1,8192,4096,8192,8193,33554432,1048576,16777216,8192,134217728]
    # Explicit semantic expectations, written before execution. Empty stdout/stderr.
    sources=[
     ('scalar', 'fn main() -> i64 { return 42; }',42,[]),
     ('local-calls','fn twice(x:i64)->i64{return x*2;} fn main()->i64{return twice(21);}',42,[]),
     ('nested-return','fn find(n:i64)->i64{let mut i:i64=0;while i<n{if i==4{return 19;}i=i+1;}return 3;} fn main()->i64{return find(9);}',19,[]),
     ('loop-carry','fn main()->i64{let mut i:i64=0;let mut s:i64=0;while i<8{if i%2==0{s=s+i;}else{s=s+2*i;}i=i+1;}return s;}',44,[]),
     ('shadow','fn main()->i64{let x:i64=7;if x>0{let x:i64=90;let y:i64=x+1;}return x;}',7,[]),
     ('main-argc','fn main(argc:i64)->i64{return argc;}',3,['one','two']),
     ('main-argc-argv','fn main(argc:i64,argv:i64)->i64{if argv==0{return 81;}return argc;}',3,['one','two']),
     ('intrinsic-argc','fn main()->i64{return __mind_argc();}',3,['one','two']),
     ('alloc-free','fn main()->i64{let p:i64=__mind_alloc(16);__mind_store_i64(p,37);let x:i64=__mind_load_i64(p);__mind_free(p);return x;}',37,[]),
     ('realloc','fn main()->i64{let p:i64=__mind_alloc(16);__mind_store_i64(p,23);let q:i64=__mind_realloc(p,48);let x:i64=__mind_load_i64(q);__mind_free(q);return x;}',23,[]),
     ('literal-cow','fn main()->i64{let s:String="ab";let t:String=string_push_byte(s,99);if string_len(t)!=3{return 81;}if __mind_load_i8(string_addr(t)+2)!=99{return 82;}if __mind_load_i8(string_addr(s))!=97{return 83;}return 0;}',0,[]),
     ('record-allocate','struct Pair{a:i64,b:i64} fn main()->i64{let p:Pair=Pair{a:11,b:17};return p.a+p.b;}',28,[]),
     ('array-field','struct Box{a:[i64;3]} fn main()->i64{let b:Box=Box{a:[3,5,7]};return b.a[1];}',5,[]),
     ('vector','fn main()->i64{let v:Vec=vec_new();let w:Vec=vec_push(v,31);return vec_get(w,0);}',31,[]),
     ('argc-after-alloc','fn main(argc:i64,argv:i64)->i64{let p:i64=__mind_alloc(32);__mind_store_i64(p,17);if __mind_argc()!=argc{return 84;}if __mind_argv(0)!=__mind_load_i64(argv){return 85;}__mind_free(p);return argc;}',3,['one','two']),
     ('library','fn helper()->i64{return 17;}',0,[]),
    ]
    refusals=[
     ('missing-main','fn helper()->i64{return 3;}'),
     ('duplicate-main','fn main()->i64{return 1;} fn main()->i64{return 2;}'),
     ('float-main','fn main()->f64{return 1.0;}'),
     ('three-main-args','fn main(a:i64,b:i64,c:i64)->i64{return a+b+c;}'),
     ('float-main-arg','fn main(a:f64)->i64{return 0;}'),
     ('unknown-call','fn main()->i64{return absent(1);}'),
     ('user-extern','extern "C" { fn getpid()->i32; } fn main()->i64{return 0;}'),
     ('read-stream','fn main()->i64{return __mind_read(0,0,0,0-1);}'),
     ('write-stream','fn main()->i64{return __mind_write(1,0,0,0-1);}'),
     ('legacy-read','fn main()->i64{return __mind_read(0,0,0);}'),
     ('legacy-write','fn main()->i64{return __mind_write(1,0,0);}'),
     ('open','fn main()->i64{return __mind_open(0);}'),
     ('syscall','fn main()->i64{return __mind_sys_getpid();}'),
     ('poison-item','@bad fn main()->i64{return 0;}'),
    ]
    rows=[]
    for name,source,rc,args in sources:rows.append({'id':'source-'+name,'kind':'source','source':source+'\n','exit':rc,'argv':args,'stdout_hex':'','stderr_hex':'','static_layouts':['default','bounded'],'dynamic_geometries':[] if name=='library' else ['A','B'],'routes':['automatic','force-home'] if name in ('scalar','local-calls','loop-carry','literal-cow','record-allocate','array-field') else ['automatic']})
    for name,source in refusals:rows.append({'id':'dynamic-refuse-'+name,'kind':'dynamic-refusal','source':source+'\n','expected':'normal diagnosed refusal, no executable run','geometries':['A','B']})
    for name,request in [('A',A),('B',B)]:
     rows.append({'id':'context-'+name,'kind':'context','request':request,'expected_check':1,'owned_source_mutation_after_construct':True})
     for i in range(52):rows.append({'id':f'tamper-{name}-{i}','kind':'tamper','request':request,'word':i,'delta':1,'expected_check':0})
     for k in [-1,0,1,2,3,9223372036854775807]:rows.append({'id':f'entry-{name}-{k}','kind':'entry','request':request,'entry':k,'expected':int(k in [0,2])})
     for k in [-1,0,*range(1,55),1008,1009,9223372036854775807]:rows.append({'id':f'intrinsic-{name}-{k}','kind':'intrinsic','request':request,'intrinsic':k,'expected':int(k in set(range(1,8))|{10,11}|set(range(13,20))|{51,52,53})})
     for op in ['alloc','free','realloc','argc','argv']:rows.append({'id':f'emitter-{name}-{op}','kind':'emitter','request':request,'op':op,'expectation':'exact independently encoded instruction bytes; then runtime coverage via source cases'})
    for k in [-1,0,1,2,3]:rows.append({'id':f'entry-static-{k}','kind':'entry-static','entry':k,'expected':int(k in [0,1])})
    for word,value in [(0,0),(0,2),(1,0),(2,0),(3,0),(4,-1),(5,4096),(5,8193),(6,1),(7,0),(8,0),(9,1),(1,9223372036854775807)]:
     req=A.copy();req[word]=value;rows.append({'id':f'invalid-request-{word}-{value}','kind':'invalid-request','request':req,'expected_new':0})
    rows.append({'id':'null-context','kind':'null-context','expected_new':0,'expected_check':0})
    return rows

def sha(b):
    return hashlib.sha256(b).hexdigest()

def limited():
    resource.setrlimit(resource.RLIMIT_AS,(2<<30,2<<30))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))

def packet(mode=0,arg=0,arg2=0,source=b'',user=0,route=0,request=None):
    return struct.pack('<16q',mode,arg,arg2,len(source),user,route,*(request or [0]*10))+source+hashlib.sha256(source).digest()

def expected_emitter(r,op):
    state=16777216+r[5]-4096
    call=lambda slot:b'\x49\xbb'+struct.pack('<Q',state+slot)+b'\x41\xff\x13'
    ld=lambda opcode,n:bytes.fromhex(opcode)+struct.pack('<i',n)
    rows=[ld('488bb5',-24)+call(536),ld('488bbd',-16)+call(544),
          ld('488bbd',-16)+ld('488bb5',-24)+call(552),
          b'\x48\xb8'+struct.pack('<Q',16777224)+bytes.fromhex('488b00')+ld('488985',-8),
          b'\x48\xb8'+struct.pack('<Q',16777232)+bytes.fromhex('488b00')+ld('488b8d',-16)+bytes.fromhex('488b04c8')+ld('488985',-8)]
    return rows[op]

def context_words(request):
    # Independent integer layout calculation, frozen before baseline output is read.
    _,code,meta,relro,literal,arena,overflow,stack,guard,reserve=request
    page=lambda n:(n+4095)//4096*4096
    cm,mm,rm,lm=map(page,(code,meta,relro,literal))
    mo=4096+cm;ro=mo+mm;re=ro+rm;context=4194304+re
    arena_end=16777216+arena;ob=(arena_end+guard+1048575)//1048576*1048576
    oe=ob+overflow;sb=oe+guard;top=sb+stack;fg=ob-arena_end
    initial=(context+4096-4194304)+lm+arena+stack
    phnum=14 if literal else 13;interp=64+phnum*56;note=(interp+256+3)//4*4
    plan=[1,4194304,4096,4198400,cm,mo,4194304+mo,mm,ro,4194304+ro,rm,
          context,re if literal else 0,8388608,lm,16777216,arena,arena_end-4096,
          ob,oe,sb,top,fg,guard,initial,initial+overflow,
          initial+overflow+fg+guard+reserve,re+lm,phnum,reserve,interp,note]
    header=[1,1,arena_end-4096,16777224,16777232,8388608,8388608,16777216,top,0]
    return [1]+header+request+plan

def inspect_parts(data,source):
    assert len(data)>=80,'refused or truncated parts'
    ok,cn,rn,nn,nf,nc,entry,arity,patched,sn=struct.unpack('<10q',data[:80])
    assert ok==1 and patched==1 and cn>0 and rn>=0 and nn==52 and nf>0 and nc>=0
    assert 0<=entry<nf and 0<=arity<=2 and sn==len(source)
    assert len(data)==80+cn+rn+nn+nf*24+nc*32+sn
    code=data[80:80+cn];off=80+cn+rn
    note=data[off:off+nn];off+=nn
    assert note[-32:]==hashlib.sha256(source).digest(),'note/source binding'
    starts=struct.unpack(f'<{nf}q',data[off:off+nf*8]);off+=nf*8
    names=struct.unpack(f'<{nf*2}q',data[off:off+nf*16]);off+=nf*16
    calls=struct.unpack(f'<{nc*3}q',data[off:off+nc*24]);off+=nc*24
    owners=struct.unpack(f'<{nc}q',data[off:off+nc*8]);off+=nc*8
    assert data[off:]==source,'caller source mutation escaped into owned parts'
    assert starts[0]==0 and list(starts)==sorted(set(starts)) and starts[-1]<cn
    symbols=[]
    for lo,hi in zip(names[::2],names[1::2]):
        assert 0<=lo<hi<=sn;symbols.append(source[lo:hi])
    assert symbols[0]==b'main'
    for i,owner in enumerate(owners):
        assert 0<=owner<nf
        at,lo,hi=calls[i*3:i*3+3];begin=starts[owner];end=cn if owner+1==nf else starts[owner+1]
        assert 1<=at<=end-begin-4 and code[begin+at-1]==0xe8
        assert 0<=lo<hi<=sn and source[lo:hi] in symbols
        target=starts[symbols.index(source[lo:hi])]
        assert struct.unpack('<i',code[begin+at:begin+at+4])[0]==target-(begin+at+4)
    return dict(code_sha256=sha(code),n_functions=nf,n_calls=nc,arity=arity)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--enumerate',action='store_true')
    ap.add_argument('--evidence',type=Path,required=True)
    ap.add_argument('--seed',type=Path);ap.add_argument('--seed-sha256')
    ap.add_argument('--mindc',type=Path);ap.add_argument('--mindc-sha256')
    ap.add_argument('--main-sha256');ap.add_argument('--context-sha256')
    ap.add_argument('--source-image-sha256',default='751661cd0d688641e0fdba333519aada55a7a44612b1039429a8eb0daa0f4db0')
    ap.add_argument('--build-only',action='store_true',help='record diagnostic artifacts; execute no case catalogue')
    a=ap.parse_args();e=a.evidence.resolve()
    cases=enumerate_cases();casebytes=(json.dumps(cases,indent=2)+'\n').encode()
    pointer_bytes=POINTER_CATALOGUE.read_bytes();pointer_rows=json.loads(pointer_bytes)['cases']
    assert len({r['id'] for r in pointer_rows})==len(pointer_rows)
    assert all(r['expected'] in ('accept','refuse') for r in pointer_rows)
    if a.enumerate:
        e.mkdir(parents=True,exist_ok=False)
        (e/'cases.json').write_bytes(casebytes);(e/'cases.json').chmod(0o444)
        (e/'cases.sha256').write_text(sha(casebytes)+'\n')
        (e/'expectation-corrections.json').write_text(json.dumps(CORRECTIONS,indent=2)+'\n')
        (e/'driver.mind').write_text(DRIVER)
        (e/'pointer-origin-cases.json').write_bytes(pointer_bytes)
        (e/'pointer-origin-cases.json').chmod(0o444)
        (e/'pointer-origin-cases.sha256').write_text(sha(pointer_bytes)+'\n')
        (e/'runtime-supplement.json').write_text(json.dumps(RUNTIME_SUPPLEMENT,indent=2)+'\n')
        (e/'primitive-shadow-cases.json').write_text(json.dumps(PRIMITIVE_SHADOW_CASES,indent=2)+'\n')
        print('FROZEN',len(cases),sha(casebytes));return
    assert (e/'cases.json').read_bytes()==casebytes
    assert (e/'cases.sha256').read_text().strip()==sha(casebytes)
    assert json.loads((e/'expectation-corrections.json').read_text())==CORRECTIONS
    assert (e/'driver.mind').read_text()==DRIVER
    assert (e/'pointer-origin-cases.json').read_bytes()==pointer_bytes
    assert (e/'pointer-origin-cases.sha256').read_text().strip()==sha(pointer_bytes)
    assert json.loads((e/'runtime-supplement.json').read_text())==RUNTIME_SUPPLEMENT
    assert json.loads((e/'primitive-shadow-cases.json').read_text())==PRIMITIVE_SHADOW_CASES
    (e/'execution.started').open('x').close()
    assert a.seed and a.mindc and a.main_sha256 and a.context_sha256
    seed,mindc=a.seed.resolve(),a.mindc.resolve()
    assert sha(seed.read_bytes())==a.seed_sha256
    assert sha(mindc.read_bytes())==a.mindc_sha256
    paths=[HERE/'main.mind',HERE/'native_dynamic_layout.mind',HERE/'native_emission_context.mind',
           HERE/'native_dynamic.mind',HERE/'native_dynamic_image.mind',HERE/'native_dynamic_allocator.mind',
           HERE/'native_libc_startup.mind',HERE/'native_libc_io.mind',HERE/'native_source_image.mind']
    assert sha(paths[0].read_bytes())==a.main_sha256
    assert sha(paths[2].read_bytes())==a.context_sha256
    assert sha(paths[-1].read_bytes())==a.source_image_sha256
    stdpaths=sorted((ROOT/'std').rglob('*.mind'))
    deps=paths+[POINTER_CATALOGUE,Path('/lib64/ld-linux-x86-64.so.2').resolve(),Path('/lib/x86_64-linux-gnu/libc.so.6').resolve()]+stdpaths+[Path(__file__).resolve(),HERE/'_stdlib_manifest.py',HERE/'testdata/stdlib_manifest.txt',ROOT/'runtime-support/mind_intrinsics.c']
    captured={str(f.resolve()):f.read_bytes() for f in deps}
    inputs={f:sha(b) for f,b in captured.items()};inputs.update({str(seed):a.seed_sha256,str(mindc):a.mindc_sha256})
    sys.path.insert(0,str(HERE));from _stdlib_manifest import seed_modules
    modules=seed_modules();prefix=b'\n'.join(captured[str((ROOT/'std'/f'{m}.mind').resolve())] for m in modules)+b'\n'
    source=b'\n'.join(captured[str(f.resolve())] for f in paths)+b'\n'+DRIVER.encode()
    combined=prefix+source
    buildpacket=struct.pack('<6q',int.from_bytes(b'MINDLAY\x01','little'),805306368,402653184,1048576,len(prefix),len(combined))+combined
    (e/'build.packet').write_bytes(buildpacket);(e/'source.mind').write_bytes(source);(e/'prefix.mind').write_bytes(prefix)
    (e/'harness.snapshot.py').write_bytes(Path(__file__).read_bytes())
    env=os.environ.copy()
    for k in list(env):
        if k.startswith('LD_') or k in ('GLIBC_TUNABLES','MINDC_NATIVE_ELF','MINDC_SO'):env.pop(k)
    env.update(LC_ALL='C',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MIND_STDLIB_PATH=str(ROOT/'std'))
    toolenv={}
    for key,default in [('MLIR_OPT','mlir-opt'),('MLIR_TRANSLATE','mlir-translate'),('CLANG','clang'),('CC','cc')]:
        selected=env.get(key,default);found=shutil.which(selected,path=env.get('PATH'));assert found,(key,selected)
        path=Path(found).resolve();inputs[str(path)]=sha(path.read_bytes());env[key]=str(path);toolenv[key]=str(path)
    (e/'INPUTS.json').write_text(json.dumps(inputs,indent=2)+'\n')
    (e/'ENVIRONMENT.json').write_text(json.dumps(dict(tools=toolenv,stdlib=env['MIND_STDLIB_PATH'],modules=modules,prefix_sha256=sha(prefix)),indent=2)+'\n')
    def intact():
        assert (e/'cases.json').read_bytes()==casebytes
        assert (e/'pointer-origin-cases.json').read_bytes()==pointer_bytes
        assert sorted((ROOT/'std').rglob('*.mind'))==stdpaths
        for f,d in inputs.items():assert sha(Path(f).read_bytes())==d,('input drift',f)
    receipts=[]
    def run(label,argv,data=None,cwd=None,timeout=300):
        start=time.monotonic();q=subprocess.run(list(map(str,argv)),input=data,capture_output=True,cwd=cwd,env=env,preexec_fn=limited,timeout=timeout)
        stem=f'{len(receipts):04d}-{label}';(e/(stem+'.stdout')).write_bytes(q.stdout);(e/(stem+'.stderr')).write_bytes(q.stderr)
        receipts.append(dict(label=label,argv=list(map(str,argv)),cwd=str(cwd or Path.cwd()),exit=q.returncode,stdin_sha256=None if data is None else sha(data),stdout_sha256=sha(q.stdout),stderr_sha256=sha(q.stderr),elapsed=time.monotonic()-start))
        (e/'executions.json').write_text(json.dumps(receipts,indent=2)+'\n');return q
    intact();q=run('native-build',[seed],buildpacket,timeout=600)
    assert q.returncode==0 and q.stdout.startswith(b'\x7fELF') and not q.stderr,('native build',q.returncode,q.stderr)
    native=e/'context.native';native.write_bytes(q.stdout);native.chmod(0o700)
    project=e/'mlir';project.mkdir();(project/'main.mind').write_bytes(source)
    (project/'Mind.toml').write_text('[package]\nname="emission-context-diagnostic"\nversion="0.0.0"\nauthors=["STARGA Inc."]\n[build]\nentry="main.mind"\n')
    mlir=project/'context.mlir'
    q=run('mlir-build',[mindc,'build','--backend','mlir','--target','cpu','--emit','binary','--release','--no-cache','--out',mlir],cwd=project,timeout=600)
    assert q.returncode==0 and mlir.is_file(),('MLIR build',q.returncode,q.stderr)
    artifacts={str(f):sha(f.read_bytes()) for f in (native,mlir)}
    (e/'ARTIFACTS.json').write_text(json.dumps(artifacts,indent=2)+'\n');intact()
    if a.build_only:
        (e/'BUILD_ONLY.json').write_text(json.dumps(dict(case_execution=False,artifacts=artifacts),indent=2)+'\n')
        print('BUILD_ONLY: artifacts recorded; no catalogue outcomes claimed');return
    failures=[];outcomes=[]
    def check(label,wire,expected=None,src=None):
        pair=[]
        for leg,exe in [('native',native),('mlir',mlir)]:
            q=run(label+'-'+leg,[exe],wire);pair.append(q.stdout)
            try:
                assert q.returncode==0 and not q.stderr,('process',q.returncode,q.stderr)
                if src is not None:detail=inspect_parts(q.stdout,src)
                elif expected is not None:assert q.stdout==expected,('bytes',dict(actual_length=len(q.stdout),actual_sha256=sha(q.stdout),actual_prefix=q.stdout[:32].hex(),expected_length=len(expected),expected_sha256=sha(expected),expected_prefix=expected[:32].hex()))
                else:detail={}
                outcomes.append(dict(id=label,leg=leg,pass_=True))
            except AssertionError as ex:
                failures.append(dict(id=label,leg=leg,error=str(ex)));outcomes.append(dict(id=label,leg=leg,pass_=False))
        if pair[0]!=pair[1]:failures.append(dict(id=label,leg='parity',error='stdout differs'))
        (e/'OUTCOMES.json').write_text(json.dumps(outcomes,indent=2)+'\n');(e/'FAILURES.json').write_text(json.dumps(failures,indent=2)+'\n')
    for row in cases:
        k=row['kind'];rid=row['id'];r=row.get('request');mode=0;arg=0;arg2=0;ex=None
        if k=='source':
            for geo in row['dynamic_geometries']:
                for route in row['routes']:
                    src=prefix+row['source'].encode();check(rid+'-'+geo+'-'+route,packet(5,source=src,user=len(prefix),route=int(route=='force-home'),request=GEOMETRIES[geo]),src=src)
            continue
        if k=='dynamic-refusal':
            for geo in row['geometries']:
                src=prefix+row['source'].encode();check(rid+'-'+geo,packet(5,source=src,user=len(prefix),request=GEOMETRIES[geo]),struct.pack('<q',0))
            continue
        if k=='context':
            check(rid+'-exact-layout',packet(0,request=r),struct.pack('<53q',*context_words(r)))
            check(rid+'-caller-mutation',packet(8,request=r),struct.pack('<q',1));continue
        if k=='tamper':
            mode=1;arg=row['word'];arg2=row['delta'];ex=1 if rid=='tamper-B-14' else row['expected_check']
        elif k=='entry':mode=2;arg=row['entry'];ex=row['expected']
        elif k=='intrinsic':mode=3;arg=row['intrinsic'];ex=row['expected']
        elif k=='emitter':
            op=['alloc','free','realloc','argc','argv'].index(row['op']);check(rid,packet(4,op,request=r),expected_emitter(r,op));continue
        elif k=='entry-static':mode=9;arg=row['entry'];ex=row['expected']
        elif k=='invalid-request':mode=6;ex=row['expected_new']
        elif k=='null-context':check(rid,packet(7),bytes(16));continue
        else:raise AssertionError(('unhandled catalogue kind',k))
        check(rid,packet(mode,arg,arg2,request=r),struct.pack('<q',ex))
    # Phase2 begins only after every original component observation is recorded.
    (e/'COMPONENT_PHASE.json').write_text(json.dumps(dict(outcomes=len(outcomes),failures=len(failures)),indent=2)+'\n')
    (e/'V2_SOURCE_CASES.json').write_text(json.dumps(V2_SOURCE_CASES,indent=2)+'\n')
    runtime_rows=[]
    for row in cases:
        if row['kind']=='source' and row['dynamic_geometries']:
            runtime_rows.append(dict(row,geometries=row['dynamic_geometries']))
        elif row['kind']=='dynamic-refusal':runtime_rows.append(dict(row,refuse=True,routes=['automatic']))
    runtime_rows+=RUNTIME_SUPPLEMENT
    runtime_rows+=V2_SOURCE_CASES
    runtime_rows+=[dict(row,refuse=True,routes=['automatic']) for row in PRIMITIVE_SHADOW_CASES]
    # Version2 explicitly adds unused checked declaration admission. Original
    # unused-extern refusal stays in the result ledger as a changed contract.
    runtime_rows.append(dict(id='v2-unused-checked-extern-acceptance',source='extern "C"{fn getpid()->i32;}fn main()->i64{return 0;}\n',exit=0,argv=[],stdout_hex='',stderr_hex='',geometries=['A','B'],routes=['automatic']))
    original_runtime_ids={row['id'] for row in runtime_rows}
    for row in pointer_rows:
        runtime_rows.append(dict(id='pointer-v1-'+row['id'],source=row['source'],refuse=row['expected']=='refuse',exit=row.get('exit',0),argv=[],stdout_hex='',stderr_hex='',geometries=['A','B'],routes=['automatic','force-home']))
    runtime_outcomes=[]
    for row in runtime_rows:
        for geo in row['geometries']:
            for route in row['routes']:
                src=prefix+row['source'].encode();wire=packet(10,source=src,user=len(prefix),route=int(route=='force-home'),request=GEOMETRIES[geo]);images=[]
                for leg,exe in [('native',native),('mlir',mlir)]:
                    label=row['id']+'-'+geo+'-'+route+'-'+leg
                    q=run('image-'+label,[exe],wire);item=dict(id=label,case_id=row['id'],phase='original-runtime' if row['id'] in original_runtime_ids else 'pointer-supplement',pass_=False)
                    try:
                        assert q.returncode==0 and not q.stderr and len(q.stdout)>=32,('image process',q.returncode,q.stderr)
                        diag=struct.unpack('<4q',q.stdout[:32]);image=q.stdout[32:];images.append(image)
                        item.update(diagnostic=diag,image_sha256=sha(image))
                        if row.get('refuse'):
                            assert diag[0]>0 and not image,('expected refusal',diag,len(image))
                        else:
                            assert diag==(0,0,0,0) and image.startswith(b'\x7fELF'),('expected image',diag,len(image))
                            phoff=struct.unpack_from('<Q',image,32)[0];phentsize,phnum=struct.unpack_from('<HH',image,54)
                            assert phentsize==56 and phoff+phnum*56<=len(image)
                            for j in range(phnum):
                                typ,flags=struct.unpack_from('<II',image,phoff+j*56)
                                assert not (typ==1 and flags&3==3),'RWX load segment'
                                if typ==0x6474e551:assert flags==6,'executable stack'
                            target=e/('image-'+label+'.elf');target.write_bytes(image);target.chmod(0o700)
                            z=run('runtime-'+label,[target,*row['argv']],timeout=15)
                            assert (z.returncode,z.stdout,z.stderr)==(row['exit'],bytes.fromhex(row['stdout_hex']),bytes.fromhex(row['stderr_hex'])),('runtime',z.returncode,z.stdout,z.stderr)
                        item['pass_']=True
                    except AssertionError as ex:item['error']=str(ex);failures.append(dict(id=label,leg='runtime',error=str(ex)))
                    runtime_outcomes.append(item)
                    (e/'RUNTIME_OUTCOMES.json').write_text(json.dumps(runtime_outcomes,indent=2)+'\n')
                    (e/'FAILURES.json').write_text(json.dumps(failures,indent=2)+'\n')
                if len(images)==2 and images[0]!=images[1]:failures.append(dict(id=row['id']+'-'+geo+'-'+route,leg='image-parity',error='native/CPU emitted different images'))
    original_runtime=[x for x in runtime_outcomes if x['phase']=='original-runtime']
    pointer_outcomes=[x for x in runtime_outcomes if x['phase']=='pointer-supplement']
    (e/'RUNTIME_PHASE.json').write_text(json.dumps(dict(outcomes=len(original_runtime),pass_count=sum(x['pass_'] for x in original_runtime)),indent=2)+'\n')
    (e/'POINTER_OUTCOMES.json').write_text(json.dumps(pointer_outcomes,indent=2)+'\n')
    (e/'POINTER_PHASE.json').write_text(json.dumps(dict(outcomes=len(pointer_outcomes),pass_count=sum(x['pass_'] for x in pointer_outcomes),catalogue_sha256=sha(pointer_bytes)),indent=2)+'\n')
    intact()
    for f,d in artifacts.items():assert sha(Path(f).read_bytes())==d
    result=dict(status='FAIL' if failures else 'COMPONENT_CONTROLS_PASS',outcomes=len(outcomes),failures=len(failures),
        remaining=['Full compiler closure, static ctx0 byte comparison and compiled mutations are separately recorded gates; this component run does not replace them'],
        qualification='No seed or pin qualification; this is a component diagnostic')
    (e/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));sys.exit(bool(failures))

if __name__=='__main__':main()
