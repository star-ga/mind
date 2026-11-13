"""Original 38 adapter controls and four MIND source mutants; enumeration only."""
import hashlib,json,struct
sha=lambda data:hashlib.sha256(data).hexdigest()

def prepare(out,prefix,parts,driver):
    def replace_fn(source,name,body):
        start=source.index('pub fn '+name+'('); brace=source.index('{',start); depth=1; end=brace+1
        while depth:
            depth += (source[end]=='{')-(source[end]=='}'); end+=1
        return source[:brace+1]+'\n '+body+'\n'+source[end-1:]

    stub='let mut s:String=string_new();\n s=string_push_byte(s,127);s=string_push_byte(s,69);\n s=string_push_byte(s,76);s=string_push_byte(s,70);\n let mut i:i64=0;while i<15{s=string_push_byte(s,65+i);i=i+1;}s'
    unit = replace_fn(driver,'nsd_compile_once',stub)
    variants = {'unit':unit,
     'short-read':replace_fn(driver,'nsd_read_once','let count:i64=if n>97{97}else{n}; __mind_read(0,p,count,0-1)'),
     'U27':replace_fn(unit,'nsd_write_once','let count:i64=if n>2{2}else{n}; __mind_write(fd,p,count,0-1)'),
     'U28':replace_fn(unit,'nsd_read_once','0-1'),
     'U29':replace_fn(unit,'nsd_write_once','if fd==1{return 0-1;} __mind_write(fd,p,n,0-1)'),
     'U30':replace_fn(driver,'nsd_compile_once','__mind_store_i64(diagnostic,1);'+stub),
     'U31':replace_fn(driver,'nsd_compile_once','string_new()'),
     'U32':replace_fn(driver,'nsd_compile_once',stub.replace('s=string_push_byte(s,127)','s=string_push_byte(s,78)')),
     'U33':replace_fn(unit,'nsd_install_sigpipe_policy','0-1'),
     'U34':replace_fn(unit,'nsd_read_once','n+1'),
     'U35':replace_fn(unit,'nsd_read_once','if n==1{return 0-1;} __mind_read(0,p,n,0-1)'),
     'U36':replace_fn(unit,'nsd_write_once','if fd==1{return n+1;} __mind_write(fd,p,n,0-1)'),
    }
    variants['M01']=unit.replace('let eof_read:i64=nsd_read_once(control+48,1);','let eof_read:i64=0;')
    variants['M02']=unit.replace('if source_length!=nested_length-16{return nsd_fail(control,0,0,64);}','')
    variants['M03']=replace_fn(variants['U30'],'nsd_diag_zero','1')
    a=variants['U27']; start=a.index('pub fn nsd_write_exact('); end=a.index('pub fn nsd_set_malformed(')
    part=a[start:end]; assert part.count('i=i+k;')==1
    variants['M04']=a[:start]+part.replace('i=i+k;','return 1;')+a[end:]
    (out/'variants').mkdir()
    for name,source in variants.items():
        if name=='short-read': source=b'\n'.join(parts).decode()+'\n'+source
        (out/'variants'/(name+'.mind')).write_text(source)

    def frame(body,user=None):
        src=prefix+body
        return struct.pack('<8s5q',b'MINDNSI1',32,0,len(src)+16,len(prefix) if user is None else user,len(src))+src
    def word(wire,offset,value):
        return wire[:offset]+struct.pack('<q',value)+wire[offset+8:]

    valid=frame(b'pub fn main()->i64{0}\n')
    rows=[]
    (out/'inputs').mkdir()
    def add(name,wire,exitcode=0,kind='image',variant='real',runtime=0,args=None,control=None):
        filename='inputs/'+name+'.bin'; (out/filename).write_bytes(wire)
        rows.append(dict(id=name,file=filename,sha256=sha(wire),exit=exitcode,kind=kind,variant=variant,
                         runtime=runtime,args=args or [],control=control or name,
                         stderr='' if exitcode==0 else {64:'malformed input\n',65:'source/image refused\n',70:'allocation refused\n',74:'input/output failure\n'}[exitcode]))
    add('C01',valid)
    add('C02',frame(b'pub fn main()->i64{7}\n'),runtime=7)
    add('C03',frame(b'pub fn main(argc:i64,argv:i64)->i64{let p:i64=__mind_load_i64(argv+8);if __mind_load_i8(p)==97{return argc;}99}\n'),runtime=3,args=['alpha','beta'])
    add('C04',valid,control='same-request-repeat')
    add('C05',valid,control='cross-backend-image-identity')
    add('C06a',frame(b'pub fn unused()->i64{2}\npub fn main()->i64{0}\n'),control='source-order-note-a')
    add('C06b',frame(b'pub fn main()->i64{0}\npub fn unused()->i64{2}\n'),control='source-order-note-b')
    add('C09',b'X'+valid[1:],64,'empty')
    for name,offset,value in [('C10',8,31),('C11',16,1),('C12',24,-1),('C13',24,16),('C14',24,4194321),('C17',24,len(valid)-32+1),('C18',40,0),('C19',32,-1),('C20',32,len(valid)-48)]:add(name,word(valid,offset,value),64,'empty')
    add('C15',valid[:31],64,'empty')
    add('C16',valid[:-1],64,'empty')
    add('C21',valid+b'X',64,'empty')
    add('C22',frame(b'pub fn main()->i64{unknown_source_call()}\n'),65,'empty')
    add('C23',frame(b'pub fn other()->i64{0}\n'),65,'empty')
    add('C24',valid,control='exact-source-digest-note')
    add('C25',frame(b'pub fn main()->i64{0} \n'),control='whitespace-changes-note')
    add('C26',valid,variant='short-read',control='bounded-short-read-seam-real-image')
    fake=b'\x7fELF'+bytes(range(65,80))
    for n in range(27,37):
        name='U'+str(n); code=0 if n==27 else 65 if n in (30,31,32) else 74
        add(name,valid,code,'fake' if n==27 else 'empty',variant=name)
        if n==33:rows[-1]['stderr']=''
    add('P01',valid,74,'closed-stdout',variant='unit',control='actual-closed-stdout-default-SIGPIPE')
    add('P02',valid[:31],64,'closed-stderr',variant='unit',control='actual-closed-stderr-default-SIGPIPE');rows[-1]['stderr']=''
    add('P03',valid,74,'closed-stdout-ignored',variant='unit',control='actual-closed-stdout-inherited-SIGPIPE-ignore')
    mutations=[dict(id='M01',case='C21',expected_exit=64),dict(id='M02',case='C17',expected_exit=64),dict(id='M03',case='U30',expected_exit=65),dict(id='M04',case='U27',expected_exit=0,expected_stdout_hex=fake.hex())]
    (out/'CASES.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'MUTATIONS.json').write_text(json.dumps(mutations,indent=2)+'\n')
    return rows, mutations
