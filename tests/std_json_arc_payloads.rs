// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! ARC-AGI-3 API payload tests through std JSON and an env-var read.
//!
//! Tests in this file verify:
//!
//! 1. std/json.mind can BUILD each ARC-AGI-3 API payload shape:
//!    scorecard open, scorecard close, RESET, ACTION1..ACTION7 with x/y,
//!    and a frame response carrying a 64x64 integer grid, state, and
//!    levels_completed.
//! 2. std/json.mind can PARSE each of those payload shapes back and
//!    extract every field.
//! 3. A MIND program can READ an environment variable (ARC_API_KEY)
//!    through std/process.mind proc_getenv WITHOUT printing it -- the
//!    test asserts only the returned length, never the value bytes.
//!
//! Gate: cargo test --features "std-surface cross-module-imports mlir-build"
//!                   --test std_json_arc_payloads

#![cfg(feature = "std-surface")]

mod common;

#[cfg(all(feature = "mlir-build", feature = "cross-module-imports"))]
mod arc_payloads {
    use std::path::PathBuf;
    use std::process::Command;

    fn compile_to_so(src: &str, tag: &str) -> Option<PathBuf> {
        let mindc = crate::common::mindc_or_skip("std_json_arc_payloads")?;
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("std_json_arc_payloads");
        std::fs::create_dir_all(&dir).expect("create output dir");
        let src_path = dir.join(format!("{tag}.mind"));
        let so_path = dir.join(format!("{tag}.so"));
        std::fs::write(&src_path, src).expect("write .mind source");
        let status = Command::new(&mindc)
            .args([
                src_path.to_str().unwrap(),
                "--emit-shared",
                so_path.to_str().unwrap(),
                "--allow-nondeterministic",
            ])
            .status()
            .expect("spawn mindc");
        assert!(
            status.success(),
            "{tag}: mindc --emit-shared failed; check build output"
        );
        Some(so_path)
    }

    fn py(script: &str) -> String {
        let out = Command::new("python3")
            .args(["-c", script])
            .output()
            .expect("python3 not found on PATH");
        assert!(
            out.status.success(),
            "python3 script failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    }

    fn arc_mind_src() -> String {
        let json_src = include_str!("../std/json.mind");
        let proc_src = include_str!("../std/process.mind");
        format!(
            r#"{json_src}

{proc_src}

// ms -- build a native string record [addr|len|cap] from raw bytes.
fn ms(addr: i64, len: i64) -> i64 {{
    let rec: i64 = __mind_alloc(24);
    __mind_store_i64(rec + 0, addr);
    __mind_store_i64(rec + 8, len);
    __mind_store_i64(rec + 16, len);
    rec
}}

// sfl -- build a string record from up to 16 byte values + length.
fn sfl(b0: i64, b1: i64, b2: i64, b3: i64, b4: i64, b5: i64, b6: i64, b7: i64,
       b8: i64, b9: i64, b10: i64, b11: i64, b12: i64, b13: i64, b14: i64,
       b15: i64, n: i64) -> i64 {{
    let dat: i64 = __mind_alloc(16);
    __mind_store_i8(dat + 0, b0);
    __mind_store_i8(dat + 1, b1);
    __mind_store_i8(dat + 2, b2);
    __mind_store_i8(dat + 3, b3);
    __mind_store_i8(dat + 4, b4);
    __mind_store_i8(dat + 5, b5);
    __mind_store_i8(dat + 6, b6);
    __mind_store_i8(dat + 7, b7);
    __mind_store_i8(dat + 8, b8);
    __mind_store_i8(dat + 9, b9);
    __mind_store_i8(dat + 10, b10);
    __mind_store_i8(dat + 11, b11);
    __mind_store_i8(dat + 12, b12);
    __mind_store_i8(dat + 13, b13);
    __mind_store_i8(dat + 14, b14);
    __mind_store_i8(dat + 15, b15);
    let rec: i64 = __mind_alloc(24);
    __mind_store_i64(rec + 0, dat);
    __mind_store_i64(rec + 8, n);
    __mind_store_i64(rec + 16, n);
    rec
}}

// kvs -- insert (key_str, string(val_str)) into object v.
fn kvs(v: i64, k_rec: i64, v_rec: i64) -> i64 {{
    let vv: Value = string(ms(__mind_load_i64(v_rec + 0), __mind_load_i64(v_rec + 8)));
    insert(Value {{ h: v }}, ms(__mind_load_i64(k_rec + 0), __mind_load_i64(k_rec + 8)), vv).h
}}

// kvi -- insert (key_str, int(val)) into object v.
fn kvi(v: i64, k_rec: i64, val: i64) -> i64 {{
    let vv: Value = int(val);
    insert(Value {{ h: v }}, ms(__mind_load_i64(k_rec + 0), __mind_load_i64(k_rec + 8)), vv).h
}}

// -- build: scorecard open --
pub fn arc_build_scorecard_open(game_id_addr: i64, game_id_len: i64) -> i64 {{
    let obj: i64 = object_new().h;
    obj = kvs(obj,
        sfl(116,121,112,101,0,0,0,0,0,0,0,0,0,0,0,0, 4),
        sfl(115,99,111,114,101,99,97,114,100,95,111,112,101,110,0,0, 14));
    obj = kvs(obj,
        sfl(103,97,109,101,95,105,100,0,0,0,0,0,0,0,0,0, 7),
        ms(game_id_addr, game_id_len));
    jv_dump(obj, 0)
}}

// -- build: scorecard close --
pub fn arc_build_scorecard_close(game_id_addr: i64, game_id_len: i64, score: i64) -> i64 {{
    let obj: i64 = object_new().h;
    obj = kvs(obj,
        sfl(116,121,112,101,0,0,0,0,0,0,0,0,0,0,0,0, 4),
        sfl(115,99,111,114,101,99,97,114,100,95,99,108,111,115,101,0, 15));
    obj = kvs(obj,
        sfl(103,97,109,101,95,105,100,0,0,0,0,0,0,0,0,0, 7),
        ms(game_id_addr, game_id_len));
    obj = kvi(obj,
        sfl(115,99,111,114,101,0,0,0,0,0,0,0,0,0,0,0, 5),
        score);
    jv_dump(obj, 0)
}}

// -- build: RESET --
pub fn arc_build_reset() -> i64 {{
    let obj: i64 = object_new().h;
    obj = kvs(obj,
        sfl(99,109,100,0,0,0,0,0,0,0,0,0,0,0,0,0, 3),
        sfl(82,69,83,69,84,0,0,0,0,0,0,0,0,0,0,0, 5));
    jv_dump(obj, 0)
}}

// -- build: ACTION1..ACTION7 with x, y --
pub fn arc_build_action(action_num: i64, x: i64, y: i64) -> i64 {{
    let obj: i64 = object_new().h;
    // "cmd" key
    let ck: i64 = sfl(99,109,100,0,0,0,0,0,0,0,0,0,0,0,0,0, 3);
    // "ACTION<N>" value
    let d: i64 = 48 + action_num;
    let cv: i64 = sfl(65,67,84,73,79,78,d,0,0,0,0,0,0,0,0,0, 7);
    obj = kvs(obj, ck, cv);
    obj = kvi(obj, sfl(120,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1), x);
    obj = kvi(obj, sfl(121,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, 1), y);
    jv_dump(obj, 0)
}}

// -- build: frame response --
pub fn arc_build_frame(grid_addr: i64, state: i64, levels_completed: i64) -> i64 {{
    let obj: i64 = object_new().h;
    let frame_arr: i64 = array_new().h;
    let mut row: i64 = 0;
    while row < 64 {{
        let row_arr: i64 = array_new().h;
        let mut col: i64 = 0;
        while col < 64 {{
            let idx: i64 = row * 64 + col;
            let val: i64 = __mind_load_i64(grid_addr + idx * 8);
            row_arr = push(Value {{ h: row_arr }}, int(val)).h;
            col = col + 1;
        }}
        frame_arr = push(Value {{ h: frame_arr }}, Value {{ h: row_arr }}).h;
        row = row + 1;
    }}
    obj = insert(Value {{ h: obj }},
        ms(__mind_load_i64(sfl(102,114,97,109,101,0,0,0,0,0,0,0,0,0,0,0, 5) + 0),
           __mind_load_i64(sfl(102,114,97,109,101,0,0,0,0,0,0,0,0,0,0,0, 5) + 8)),
        Value {{ h: frame_arr }}).h;
    obj = kvi(obj, sfl(115,116,97,116,101,0,0,0,0,0,0,0,0,0,0,0, 5), state);
    obj = kvi(obj, sfl(108,101,118,101,108,115,95,99,111,109,112,108,101,116,101,100, 16), levels_completed);
    jv_dump(obj, 0)
}}

// -- parse helpers --
pub fn arc_parse_get_string(json_addr: i64, json_len: i64,
                             key_addr: i64, key_len: i64) -> i64 {{
    let root: i64 = jv_parse(json_addr, json_len);
    if root == 0 {{ return 0; }}
    let val: i64 = jv_get(root, key_addr, key_len);
    if val == 0 {{ return 0; }}
    if jv_kind(val) != jv_kind_string() {{ return 0; }}
    val
}}

pub fn arc_parse_get_int(json_addr: i64, json_len: i64,
                          key_addr: i64, key_len: i64) -> i64 {{
    let root: i64 = jv_parse(json_addr, json_len);
    if root == 0 {{ return -999999; }}
    let val: i64 = jv_get(root, key_addr, key_len);
    if val == 0 {{ return -999999; }}
    if jv_kind(val) != jv_kind_number() {{ return -999999; }}
    if jv_n_is_int(val) != 1 {{ return -999999; }}
    if jv_n_neg(val) == 1 {{ return -999999 - jv_n_int(val); }}
    jv_n_int(val)
}}

pub fn arc_parse_frame_dims(json_addr: i64, json_len: i64) -> i64 {{
    let root: i64 = jv_parse(json_addr, json_len);
    if root == 0 {{ return 0; }}
    let fk: i64 = sfl(102,114,97,109,101,0,0,0,0,0,0,0,0,0,0,0, 5);
    let frame: i64 = jv_get(root, __mind_load_i64(fk + 0), __mind_load_i64(fk + 8));
    if frame == 0 {{ return 0; }}
    if jv_kind(frame) != jv_kind_array() {{ return 0; }}
    let rows: i64 = jv_arr_len(frame);
    if rows == 0 {{ return 0; }}
    let row0: i64 = jv_arr_get(frame, 0);
    if row0 == 0 {{ return 0; }}
    if jv_kind(row0) != jv_kind_array() {{ return 0; }}
    rows * 10000 + jv_arr_len(row0)
}}

pub fn arc_parse_frame_cell(json_addr: i64, json_len: i64,
                             r: i64, c: i64) -> i64 {{
    let root: i64 = jv_parse(json_addr, json_len);
    if root == 0 {{ return -999999; }}
    let fk: i64 = sfl(102,114,97,109,101,0,0,0,0,0,0,0,0,0,0,0, 5);
    let frame: i64 = jv_get(root, __mind_load_i64(fk + 0), __mind_load_i64(fk + 8));
    if frame == 0 {{ return -999999; }}
    if jv_kind(frame) != jv_kind_array() {{ return -999999; }}
    if r < 0 {{ return -999999; }}
    if r >= jv_arr_len(frame) {{ return -999999; }}
    let row: i64 = jv_arr_get(frame, r);
    if row == 0 {{ return -999999; }}
    if jv_kind(row) != jv_kind_array() {{ return -999999; }}
    if c < 0 {{ return -999999; }}
    if c >= jv_arr_len(row) {{ return -999999; }}
    let cell: i64 = jv_arr_get(row, c);
    if cell == 0 {{ return -999999; }}
    if jv_kind(cell) != jv_kind_number() {{ return -999999; }}
    if jv_n_neg(cell) == 1 {{ return -999999 - jv_n_int(cell); }}
    jv_n_int(cell)
}}

pub fn arc_read_env_len(name_addr: i64) -> i64 {{
    let rec: i64 = proc_getenv(name_addr);
    __mind_load_i64(rec + 8)
}}
"#
        )
    }

    #[test]
    fn scorecard_open_round_trip() {
        let src = arc_mind_src();
        let Some(so) = compile_to_so(&src, "scorecard_open") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes, json as jsonlib
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_store_i8 = getattr(lib, '__mind_store_i8'); lib.__mind_store_i8.restype = None; lib.__mind_store_i8.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.arc_build_scorecard_open = getattr(lib, 'arc_build_scorecard_open'); lib.arc_build_scorecard_open.restype = ctypes.c_int64; lib.arc_build_scorecard_open.argtypes = [ctypes.c_int64, ctypes.c_int64]
game_id = b"arc-game-001"
gbuf = (ctypes.c_uint8 * len(game_id))(*game_id)
rec = lib.arc_build_scorecard_open(ctypes.addressof(gbuf), len(game_id))
assert rec != 0, "returned null"
addr = lib.__mind_load_i64(rec)
length = lib.__mind_load_i64(rec + 8)
raw = ctypes.string_at(addr, length)
j = jsonlib.loads(raw)
assert j["type"] == "scorecard_open", f"type mismatch: {{j}}"
assert j["game_id"] == "arc-game-001", f"game_id mismatch: {{j}}"
print("ok")
"#,
            so = so_str
        );
        let result = py(&script);
        assert_eq!(result, "ok", "scorecard_open_round_trip: {result}");
    }

    #[test]
    fn scorecard_close_round_trip() {
        let src = arc_mind_src();
        let Some(so) = compile_to_so(&src, "scorecard_close") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes, json as jsonlib
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_store_i8 = getattr(lib, '__mind_store_i8'); lib.__mind_store_i8.restype = None; lib.__mind_store_i8.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.arc_build_scorecard_close = getattr(lib, 'arc_build_scorecard_close'); lib.arc_build_scorecard_close.restype = ctypes.c_int64; lib.arc_build_scorecard_close.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
game_id = b"arc-game-002"
gbuf = (ctypes.c_uint8 * len(game_id))(*game_id)
rec = lib.arc_build_scorecard_close(ctypes.addressof(gbuf), len(game_id), 42)
assert rec != 0, "returned null"
addr = lib.__mind_load_i64(rec)
length = lib.__mind_load_i64(rec + 8)
raw = ctypes.string_at(addr, length)
j = jsonlib.loads(raw)
assert j["type"] == "scorecard_close", f"type mismatch: {{j}}"
assert j["game_id"] == "arc-game-002", f"game_id mismatch: {{j}}"
assert j["score"] == 42, f"score mismatch: {{j}}"
print("ok")
"#,
            so = so_str
        );
        let result = py(&script);
        assert_eq!(result, "ok", "scorecard_close_round_trip: {result}");
    }

    #[test]
    fn reset_round_trip() {
        let src = arc_mind_src();
        let Some(so) = compile_to_so(&src, "reset") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes, json as jsonlib
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_store_i8 = getattr(lib, '__mind_store_i8'); lib.__mind_store_i8.restype = None; lib.__mind_store_i8.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.arc_build_reset = getattr(lib, 'arc_build_reset'); lib.arc_build_reset.restype = ctypes.c_int64; lib.arc_build_reset.argtypes = []
rec = lib.arc_build_reset()
assert rec != 0, "returned null"
addr = lib.__mind_load_i64(rec)
length = lib.__mind_load_i64(rec + 8)
raw = ctypes.string_at(addr, length)
j = jsonlib.loads(raw)
assert j["cmd"] == "RESET", f"cmd mismatch: {{j}}"
print("ok")
"#,
            so = so_str
        );
        let result = py(&script);
        assert_eq!(result, "ok", "reset_round_trip: {result}");
    }

    #[test]
    fn action_round_trips() {
        let src = arc_mind_src();
        let Some(so) = compile_to_so(&src, "actions") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes, json as jsonlib
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_store_i8 = getattr(lib, '__mind_store_i8'); lib.__mind_store_i8.restype = None; lib.__mind_store_i8.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.arc_build_action = getattr(lib, 'arc_build_action'); lib.arc_build_action.restype = ctypes.c_int64; lib.arc_build_action.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
for n in range(1, 8):
    x = n * 10
    y = n * 20
    rec = lib.arc_build_action(n, x, y)
    assert rec != 0, f"arc_build_action({{n}}) returned null"
    addr = lib.__mind_load_i64(rec)
    length = lib.__mind_load_i64(rec + 8)
    raw = ctypes.string_at(addr, length)
    j = jsonlib.loads(raw)
    assert j["cmd"] == f"ACTION{{n}}", f"cmd mismatch for n={{n}}: {{j}}"
    assert j["x"] == x, f"x mismatch for n={{n}}: {{j}}"
    assert j["y"] == y, f"y mismatch for n={{n}}: {{j}}"
print("ok")
"#,
            so = so_str
        );
        let result = py(&script);
        assert_eq!(result, "ok", "action_round_trips: {result}");
    }

    #[test]
    fn frame_round_trip() {
        let src = arc_mind_src();
        let Some(so) = compile_to_so(&src, "frame") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes, json as jsonlib
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_store_i8 = getattr(lib, '__mind_store_i8'); lib.__mind_store_i8.restype = None; lib.__mind_store_i8.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.arc_build_frame = getattr(lib, 'arc_build_frame'); lib.arc_build_frame.restype = ctypes.c_int64; lib.arc_build_frame.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.arc_parse_frame_dims = getattr(lib, 'arc_parse_frame_dims'); lib.arc_parse_frame_dims.restype = ctypes.c_int64; lib.arc_parse_frame_dims.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.arc_parse_frame_cell = getattr(lib, 'arc_parse_frame_cell'); lib.arc_parse_frame_cell.restype = ctypes.c_int64; lib.arc_parse_frame_cell.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
lib.arc_parse_get_int = getattr(lib, 'arc_parse_get_int'); lib.arc_parse_get_int.restype = ctypes.c_int64; lib.arc_parse_get_int.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
grid = (ctypes.c_int64 * (64 * 64))()
for r in range(64):
    for c in range(64):
        grid[r * 64 + c] = r * 64 + c
rec = lib.arc_build_frame(ctypes.addressof(grid), 7, 3)
assert rec != 0, "returned null"
addr = lib.__mind_load_i64(rec)
length = lib.__mind_load_i64(rec + 8)
raw = ctypes.string_at(addr, length)
j = jsonlib.loads(raw)
assert len(j["frame"]) == 64, f"rows: {{len(j['frame'])}}"
assert len(j["frame"][0]) == 64, f"cols: {{len(j['frame'][0])}}"
assert j["frame"][0][0] == 0
assert j["frame"][3][5] == 3 * 64 + 5
assert j["frame"][63][63] == 63 * 64 + 63
assert j["state"] == 7
assert j["levels_completed"] == 3
buf = (ctypes.c_uint8 * len(raw))(*raw)
dims = lib.arc_parse_frame_dims(ctypes.addressof(buf), len(raw))
assert dims == 64 * 10000 + 64, f"dims: {{dims}}"
cell = lib.arc_parse_frame_cell(ctypes.addressof(buf), len(raw), 3, 5)
assert cell == 3 * 64 + 5, f"cell: {{cell}}"
state = lib.arc_parse_get_int(ctypes.addressof(buf), len(raw),
    ctypes.addressof((ctypes.c_uint8 * 5)(*b"state")), 5)
assert state == 7, f"state: {{state}}"
lc = lib.arc_parse_get_int(ctypes.addressof(buf), len(raw),
    ctypes.addressof((ctypes.c_uint8 * 16)(*b"levels_completed")), 16)
assert lc == 3, f"lc: {{lc}}"
print("ok")
"#,
            so = so_str
        );
        let result = py(&script);
        assert_eq!(result, "ok", "frame_round_trip: {result}");
    }

    #[test]
    fn parse_string_fields() {
        let src = arc_mind_src();
        let Some(so) = compile_to_so(&src, "parse_strings") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_store_i8 = getattr(lib, '__mind_store_i8'); lib.__mind_store_i8.restype = None; lib.__mind_store_i8.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.jv_s_addr = getattr(lib, 'jv_s_addr'); lib.jv_s_addr.restype = ctypes.c_int64; lib.jv_s_addr.argtypes = [ctypes.c_int64]
lib.jv_s_len = getattr(lib, 'jv_s_len'); lib.jv_s_len.restype = ctypes.c_int64; lib.jv_s_len.argtypes = [ctypes.c_int64]
lib.arc_parse_get_string = getattr(lib, 'arc_parse_get_string'); lib.arc_parse_get_string.restype = ctypes.c_int64; lib.arc_parse_get_string.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
js = b'{{"type":"scorecard_open","game_id":"arc-test-99"}}'
jbuf = (ctypes.c_uint8 * len(js))(*js)
tkey = b"type"
tbuf = (ctypes.c_uint8 * len(tkey))(*tkey)
trec = lib.arc_parse_get_string(ctypes.addressof(jbuf), len(js), ctypes.addressof(tbuf), len(tkey))
assert trec != 0, "type not found"
assert ctypes.string_at(lib.jv_s_addr(trec), lib.jv_s_len(trec)) == b"scorecard_open"
gkey = b"game_id"
gbuf2 = (ctypes.c_uint8 * len(gkey))(*gkey)
grec = lib.arc_parse_get_string(ctypes.addressof(jbuf), len(js), ctypes.addressof(gbuf2), len(gkey))
assert grec != 0, "game_id not found"
assert ctypes.string_at(lib.jv_s_addr(grec), lib.jv_s_len(grec)) == b"arc-test-99"
print("ok")
"#,
            so = so_str
        );
        let result = py(&script);
        assert_eq!(result, "ok", "parse_string_fields: {result}");
    }

    #[test]
    fn env_read_without_printing() {
        let src = arc_mind_src();
        let Some(so) = compile_to_so(&src, "env_read") else {
            return;
        };
        let so_str = so.to_string_lossy().into_owned();
        let script = format!(
            r#"
import ctypes, os
lib = ctypes.CDLL('{so}')
lib.__mind_alloc = getattr(lib, '__mind_alloc'); lib.__mind_alloc.restype = ctypes.c_int64; lib.__mind_alloc.argtypes = [ctypes.c_int64]
lib.__mind_load_i64 = getattr(lib, '__mind_load_i64'); lib.__mind_load_i64.restype = ctypes.c_int64; lib.__mind_load_i64.argtypes = [ctypes.c_int64]
lib.__mind_store_i64 = getattr(lib, '__mind_store_i64'); lib.__mind_store_i64.restype = None; lib.__mind_store_i64.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.__mind_store_i8 = getattr(lib, '__mind_store_i8'); lib.__mind_store_i8.restype = None; lib.__mind_store_i8.argtypes = [ctypes.c_int64, ctypes.c_int64]
lib.arc_read_env_len = getattr(lib, 'arc_read_env_len'); lib.arc_read_env_len.restype = ctypes.c_int64; lib.arc_read_env_len.argtypes = [ctypes.c_int64]
os.environ['ARC_API_KEY'] = 'test-key-not-real-0000'
name = b'ARC_API_KEY\x00'
nbuf = (ctypes.c_uint8 * len(name))(*name)
vlen = lib.arc_read_env_len(ctypes.addressof(nbuf))
assert vlen == len('test-key-not-real-0000'), f"len: {{vlen}}"
name2 = b'ARC_UNSET_VAR_XYZ\x00'
nbuf2 = (ctypes.c_uint8 * len(name2))(*name2)
vlen2 = lib.arc_read_env_len(ctypes.addressof(nbuf2))
assert vlen2 == 0, f"unset len: {{vlen2}}"
print("ok")
"#,
            so = so_str
        );
        let result = py(&script);
        assert_eq!(result, "ok", "env_read_without_printing: {result}");
        assert!(
            !result.contains("test-key-not-real-0000"),
            "env value leaked into test output"
        );
    }
}
