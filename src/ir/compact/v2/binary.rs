// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License").

//! MIC-B v2 binary format encoder and decoder.
//!
//! Wire format:
//! ```text
//! [0..4)   : magic "MICB"
//! [4]      : version 0x02
//! [5..]    : tables (ULEB128 encoded)
//!
//! Tables:
//!   1. String table: interned strings
//!   2. Symbol table: symbolic dimension names
//!   3. Type table: tensor types
//!   4. Value table: args, params, nodes
//!   5. Output: single value ID
//! ```

use std::collections::HashMap;
use std::io::{Read, Write};

use super::map_limits::{
    check_map_bytes_len, check_map_entry_count, check_map_key, check_map_nesting_depth,
    check_map_string_len,
};
use super::types::{DType, Graph, Map, MapValue, Opcode, TensorType, Value};
use super::varint::{sleb128_read, sleb128_write, uleb128_read, uleb128_write};
use super::{MICB_MAGIC, MICB_VERSION};

/// Error type for MIC-B operations.
#[derive(Debug, Clone)]
pub struct MicbError {
    pub message: String,
}

impl std::fmt::Display for MicbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MICB error: {}", self.message)
    }
}

impl std::error::Error for MicbError {}

impl From<std::io::Error> for MicbError {
    fn from(e: std::io::Error) -> Self {
        Self {
            message: e.to_string(),
        }
    }
}

/// Serialize a Graph to MIC-B v2 binary format.
///
/// The output is deterministic: same Graph always produces same bytes.
pub fn emit_micb<W: Write>(graph: &Graph, w: &mut W) -> Result<(), MicbError> {
    let mut encoder = MicbEncoder::new(graph);
    encoder.encode(w)
}

/// Parse MIC-B v2 binary format into a Graph.
///
/// DoS-bounded: the reader is capped at [`MAX_MICB_INPUT`] bytes, and the total
/// bytes of decoded string clones are capped relative to the string table, so
/// untrusted input cannot drive unbounded memory use. Both limits are far above
/// any legitimate artifact and never fire on valid input.
pub fn parse_micb<R: Read>(r: &mut R) -> Result<Graph, MicbError> {
    let mut limited = LimitedReader::new(r);
    let mut decoder = MicbDecoder::new();
    decoder.decode(&mut limited)
}

/// MIC-B encoder with string table interning.
struct MicbEncoder<'a> {
    graph: &'a Graph,
    strings: Vec<String>,
    string_map: HashMap<String, usize>,
}

impl<'a> MicbEncoder<'a> {
    fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            strings: Vec::new(),
            string_map: HashMap::new(),
        }
    }

    fn intern(&mut self, s: &str) -> usize {
        if let Some(&idx) = self.string_map.get(s) {
            return idx;
        }
        let idx = self.strings.len();
        self.strings.push(s.to_string());
        self.string_map.insert(s.to_string(), idx);
        idx
    }

    fn build_string_table(&mut self) {
        // Intern strings in deterministic order:
        // 1. Symbols
        // 2. Type dimension tokens
        // 3. Value names
        // 4. MAP keys and string values (canonical order, §3.4)

        for sym in &self.graph.symbols {
            self.intern(sym);
        }

        for t in &self.graph.types {
            for dim in &t.shape {
                self.intern(dim);
            }
        }

        for v in &self.graph.values {
            match v {
                Value::Arg(name, _) | Value::Param(name, _) => {
                    self.intern(name);
                }
                Value::Node(Opcode::Custom(name), _) => {
                    self.intern(name);
                }
                _ => {}
            }
        }

        // MAP strings appended after graph strings in canonical order.
        let canonical_map = self.graph.map.canonicalize();
        if !canonical_map.is_empty() {
            self.intern_map_strings(&canonical_map);
        }
    }

    fn encode<W: Write>(&mut self, w: &mut W) -> Result<(), MicbError> {
        // Build string table
        self.build_string_table();

        // Magic + version
        w.write_all(&MICB_MAGIC)?;
        w.write_all(&[MICB_VERSION])?;

        // String table
        uleb128_write(w, self.strings.len() as u64)?;
        for s in &self.strings {
            let bytes = s.as_bytes();
            uleb128_write(w, bytes.len() as u64)?;
            w.write_all(bytes)?;
        }

        // Symbol table
        uleb128_write(w, self.graph.symbols.len() as u64)?;
        for sym in &self.graph.symbols {
            let idx = self.string_map[sym];
            uleb128_write(w, idx as u64)?;
        }

        // Type table
        uleb128_write(w, self.graph.types.len() as u64)?;
        for t in &self.graph.types {
            w.write_all(&[t.dtype.to_byte()])?;
            uleb128_write(w, t.shape.len() as u64)?;
            for dim in &t.shape {
                let idx = self.string_map[dim];
                uleb128_write(w, idx as u64)?;
            }
        }

        // Value table
        uleb128_write(w, self.graph.values.len() as u64)?;
        for v in &self.graph.values {
            self.encode_value(w, v)?;
        }

        // Output
        uleb128_write(w, self.graph.output as u64)?;

        // MAP section (§3.4): only emit when non-empty (§5 rule 10 / §2 rule 3).
        // String table was already populated by build_string_table() above.
        let canonical_map = self.graph.map.canonicalize();
        if !canonical_map.is_empty() {
            uleb128_write(w, 0x4D)?; // map_marker
            self.encode_map_entries(w, &canonical_map)?;
        }

        Ok(())
    }

    /// Intern all keys and string values in canonical order so the string table
    /// is stable before encoding MAP entries.
    fn intern_map_strings(&mut self, map: &Map) {
        for (key, value) in map.iter_canonical() {
            self.intern(key);
            match value {
                MapValue::String(s) => {
                    self.intern(s);
                }
                MapValue::Nested(inner) => {
                    self.intern_map_strings(inner);
                }
                MapValue::Int(_) | MapValue::Bytes(_) => {}
            }
        }
    }

    /// Encode MAP entries: count then each entry recursively.
    fn encode_map_entries<W: Write>(&self, w: &mut W, map: &Map) -> Result<(), MicbError> {
        uleb128_write(w, map.len() as u64)?;
        for (key, value) in map.iter_canonical() {
            let key_idx = self.string_map[key];
            uleb128_write(w, key_idx as u64)?;
            self.encode_map_value(w, value)?;
        }
        Ok(())
    }

    fn encode_map_value<W: Write>(&self, w: &mut W, value: &MapValue) -> Result<(), MicbError> {
        match value {
            MapValue::String(s) => {
                w.write_all(&[0])?; // value_tag = 0
                let idx = self.string_map[s.as_str()];
                uleb128_write(w, idx as u64)?;
            }
            MapValue::Int(i) => {
                w.write_all(&[1])?; // value_tag = 1
                sleb128_write(w, *i)?;
            }
            MapValue::Bytes(b) => {
                w.write_all(&[2])?; // value_tag = 2
                uleb128_write(w, b.len() as u64)?;
                w.write_all(b)?;
            }
            MapValue::Nested(inner) => {
                w.write_all(&[3])?; // value_tag = 3
                self.encode_map_entries(w, inner)?;
            }
        }
        Ok(())
    }

    fn encode_value<W: Write>(&self, w: &mut W, value: &Value) -> Result<(), MicbError> {
        match value {
            Value::Arg(name, type_idx) => {
                w.write_all(&[0])?; // tag
                let name_idx = self.string_map[name];
                uleb128_write(w, name_idx as u64)?;
                uleb128_write(w, *type_idx as u64)?;
            }
            Value::Param(name, type_idx) => {
                w.write_all(&[1])?; // tag
                let name_idx = self.string_map[name];
                uleb128_write(w, name_idx as u64)?;
                uleb128_write(w, *type_idx as u64)?;
            }
            Value::Node(opcode, inputs) => {
                w.write_all(&[2])?; // tag
                self.encode_opcode(w, opcode)?;
                uleb128_write(w, inputs.len() as u64)?;
                for inp in inputs {
                    uleb128_write(w, *inp as u64)?;
                }
            }
        }
        Ok(())
    }

    fn encode_opcode<W: Write>(&self, w: &mut W, opcode: &Opcode) -> Result<(), MicbError> {
        w.write_all(&[opcode.to_byte()])?;

        // Encode opcode parameters
        match opcode {
            Opcode::Softmax(axis) => {
                sleb128_write(w, *axis)?;
            }
            Opcode::Transpose(perm) => {
                uleb128_write(w, perm.len() as u64)?;
                for p in perm {
                    sleb128_write(w, *p)?;
                }
            }
            Opcode::Sum(axes) | Opcode::Mean(axes) | Opcode::Max(axes) => {
                uleb128_write(w, axes.len() as u64)?;
                for a in axes {
                    sleb128_write(w, *a)?;
                }
            }
            Opcode::Concat(axis) => {
                sleb128_write(w, *axis)?;
            }
            Opcode::Split(axis, n) => {
                sleb128_write(w, *axis)?;
                uleb128_write(w, *n as u64)?;
            }
            Opcode::Gather(axis) => {
                sleb128_write(w, *axis)?;
            }
            Opcode::Custom(name) => {
                let idx = self.string_map[name];
                uleb128_write(w, idx as u64)?;
            }
            _ => {} // No additional parameters
        }

        Ok(())
    }
}

// ─── DoS hardening ───────────────────────────────────────────────────────────

/// Absolute cap on any ULEB128 element count or byte-length read by the
/// MIC-B parser.  16 million is comfortably above any realistic graph
/// (the residual-block test uses ~tens of entries) while staying far below
/// the gigabytes of allocation that a single crafted varint could otherwise
/// trigger on a stream reader (where there is no known total length to
/// compare against).  A well-formed artifact always has counts below this
/// cap, so this limit is never reached on valid input — every existing
/// round-trip test passes byte-identically.
const MAX_MICB_ELEMENTS: usize = 16_000_000;

/// Read one ULEB128 value from `r` and reject it if it exceeds
/// [`MAX_MICB_ELEMENTS`].  Mirrors the `read_count` helper in the mic@3
/// parser (`src/ir/compact/v3/parse.rs`) adapted for the stream case where
/// no total-input length is available.
#[inline]
fn read_bounded_count<R: Read>(r: &mut R) -> Result<usize, MicbError> {
    let n = uleb128_read(r)? as usize;
    if n > MAX_MICB_ELEMENTS {
        return Err(MicbError {
            message: format!(
                "element count {} exceeds MAX_MICB_ELEMENTS {} (possible DoS blob)",
                n, MAX_MICB_ELEMENTS
            ),
        });
    }
    Ok(n)
}

/// Maximum accepted MIC-B input size (bytes).  Mirrors `MAX_MIC3_INPUT` in
/// the mic@3 parser (`src/ir/compact/v3/parse.rs`) so both binary
/// serializations are bounded by the same policy.
///
/// `MAX_MICB_ELEMENTS` bounds each *individual* count but says nothing about
/// the *total* input, so a blob well under any single cap could previously
/// drive unbounded work.  MIC-B encodes tensor graphs (the residual-block
/// round-trip test uses ~tens of entries), so 10 MiB is ~three orders of
/// magnitude above any legitimate artifact and is never reached on valid
/// input — the byte-identity of every ACCEPTED parse is unchanged.
pub const MAX_MICB_INPUT: usize = 10 * 1024 * 1024;

/// Cap on the total bytes of decoded string CLONES, as a multiple of the
/// string-table size.  The decoder clones a string-table entry per wire
/// reference (a 1-byte ULEB index), so a small blob that references a large
/// entry many times expands into hundreds of megabytes of retained heap — an
/// allocation bomb.  A total-input bound alone does NOT close this: with a
/// 10 MiB budget split between a large entry and many references, the product
/// is still unbounded in practice.  Mirrors `DECODE_AMPLIFICATION_FACTOR` in
/// the mic@3 parser, keyed here to the string table (the quantity actually
/// being amplified) rather than the whole input, which is strictly tighter.
const MICB_CLONE_AMPLIFICATION_FACTOR: usize = 64;

/// Absolute floor for the per-parse clone budget so a small-but-legitimate
/// graph that re-references short identifiers is never rejected.  Mirrors
/// `MIN_DECODE_BUDGET` in the mic@3 parser.
///
/// deferred: this floor means a crafted blob can still drive ~64 MiB of
/// retained clones before being refused (measured: 78 MB peak RSS for the
/// 1 MB bomb, down from 339 MB and now bounded instead of unbounded). That is
/// the same residual the mic@3 parser already ships, and it is a bounded spike
/// rather than an OOM — upgrade path: intern the string table into `Rc<str>`
/// so `Value::Arg`/`Opcode::Custom`/`MapValue::String`/symbols/shape share one
/// allocation and the per-reference clone disappears entirely, making the
/// budget unnecessary. Not done here because it changes the public
/// `compact::v2::types` signatures (a wire-format-adjacent API break), which
/// needs its own change with the round-trip byte-identity corpus re-run.
const MICB_MIN_CLONE_BUDGET: usize = 64 * 1024 * 1024;

/// A `Read` adapter that refuses to yield more than [`MAX_MICB_INPUT`] bytes.
///
/// `parse_micb` accepts a generic `R: Read` with no known total length, so the
/// mic@3 trick of checking `data.len()` up front is unavailable.  Counting at
/// the read boundary gives the same guarantee for a stream: the parser can
/// never consume — and therefore never allocate proportionally to — more than
/// the cap, whatever the underlying reader claims to hold.
struct LimitedReader<'a, R: Read> {
    inner: &'a mut R,
    consumed: usize,
}

impl<'a, R: Read> LimitedReader<'a, R> {
    fn new(inner: &'a mut R) -> Self {
        Self { inner, consumed: 0 }
    }
}

impl<R: Read> Read for LimitedReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        // Never hand the parser a byte past the cap: clamp this read to the
        // remaining allowance.
        let remaining = MAX_MICB_INPUT - self.consumed;
        if remaining == 0 {
            if buf.is_empty() {
                return Ok(0);
            }
            // The allowance is spent and the parser wants more. Two cases must
            // be told apart, because the decoder treats EOF as meaningful (an
            // artifact ending right after the output varint has no MAP
            // section, §3.4): an input of EXACTLY the cap is legitimate and
            // must still see a genuine EOF, while an input that continues past
            // the cap is the oversize case and must be refused. Probing the
            // inner reader for one byte distinguishes them.
            let mut probe = [0u8; 1];
            return match self.inner.read(&mut probe)? {
                0 => Ok(0), // input was exactly the cap: report real EOF
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("MIC-B input too large: exceeds MAX_MICB_INPUT {MAX_MICB_INPUT} bytes"),
                )),
            };
        }
        let cap = buf.len().min(remaining);
        let n = self.inner.read(&mut buf[..cap])?;
        self.consumed += n;
        Ok(n)
    }
}

// ─── MIC-B decoder ───────────────────────────────────────────────────────────

/// MIC-B decoder.
struct MicbDecoder {
    strings: Vec<String>,
    /// Remaining decoded-string-clone budget in bytes for the in-progress
    /// decode.  Sized from the string table once it is known (see
    /// [`MICB_CLONE_AMPLIFICATION_FACTOR`]) and charged down at every site
    /// that clones a string-table entry.  A `Cell` because the map / opcode
    /// decoders take `&self`.
    clone_budget: std::cell::Cell<usize>,
}

impl MicbDecoder {
    fn new() -> Self {
        Self {
            strings: Vec::new(),
            // Sized for real once the string table is known; until then the
            // floor applies, so a malformed header can never borrow budget.
            clone_budget: std::cell::Cell::new(MICB_MIN_CLONE_BUDGET),
        }
    }

    /// Clone string-table entry `idx`, charging its length against this parse's
    /// clone budget.
    ///
    /// Every site that materialises a `String` from a 1-byte wire index goes
    /// through here, so the total retained string bytes are bounded no matter
    /// how many references the blob contains. The range check stays at each
    /// call site so its specific diagnostic is preserved; the `get` here is
    /// defence in depth so a future call site that forgets it cannot panic the
    /// parser on untrusted input.
    #[inline]
    fn clone_string(&self, idx: usize) -> Result<String, MicbError> {
        let s = self.strings.get(idx).ok_or_else(|| MicbError {
            message: format!("string index {idx} out of bounds"),
        })?;
        match self.clone_budget.get().checked_sub(s.len()) {
            Some(rest) => {
                self.clone_budget.set(rest);
                Ok(s.clone())
            }
            None => Err(MicbError {
                message: format!(
                    "decoded string clones exceed MIC-B budget: cloning string-table \
                     entry {} ({} bytes) would exceed the remaining {} bytes \
                     (possible allocation bomb)",
                    idx,
                    s.len(),
                    self.clone_budget.get()
                ),
            }),
        }
    }

    fn decode<R: Read>(&mut self, r: &mut R) -> Result<Graph, MicbError> {
        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != MICB_MAGIC {
            return Err(MicbError {
                message: format!("invalid magic: expected {:?}, got {:?}", MICB_MAGIC, magic),
            });
        }

        // Version
        let mut version = [0u8; 1];
        r.read_exact(&mut version)?;
        if version[0] != MICB_VERSION {
            return Err(MicbError {
                message: format!(
                    "unsupported version: expected {}, got {}",
                    MICB_VERSION, version[0]
                ),
            });
        }

        // String table
        let n_strings = read_bounded_count(r)?;
        self.strings = Vec::with_capacity(n_strings);
        for _ in 0..n_strings {
            let len = read_bounded_count(r)?;
            let mut buf = vec![0u8; len];
            r.read_exact(&mut buf)?;
            let s = String::from_utf8(buf).map_err(|_| MicbError {
                message: "invalid UTF-8 in string table".into(),
            })?;
            self.strings.push(s);
        }

        // Size this parse's decoded-string-clone budget now that the string
        // table is known. Keyed to the table's own byte size: re-referencing
        // short identifiers stays far inside the budget, while a blob that
        // references a large entry thousands of times trips it.
        let strings_bytes: usize = self.strings.iter().map(|s| s.len()).sum();
        self.clone_budget.set(
            strings_bytes
                .saturating_mul(MICB_CLONE_AMPLIFICATION_FACTOR)
                .max(MICB_MIN_CLONE_BUDGET),
        );

        // Symbol table
        let n_symbols = read_bounded_count(r)?;
        let mut symbols = Vec::with_capacity(n_symbols);
        for _ in 0..n_symbols {
            let idx = uleb128_read(r)? as usize;
            if idx >= self.strings.len() {
                return Err(MicbError {
                    message: format!("symbol string index {} out of bounds", idx),
                });
            }
            symbols.push(self.clone_string(idx)?);
        }

        // Type table
        let n_types = read_bounded_count(r)?;
        let mut types = Vec::with_capacity(n_types);
        for _ in 0..n_types {
            let mut dtype_byte = [0u8; 1];
            r.read_exact(&mut dtype_byte)?;
            let dtype = DType::from_byte(dtype_byte[0]).ok_or_else(|| MicbError {
                message: format!("unknown dtype byte: {}", dtype_byte[0]),
            })?;

            let rank = read_bounded_count(r)?;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                let idx = uleb128_read(r)? as usize;
                if idx >= self.strings.len() {
                    return Err(MicbError {
                        message: format!("type dim string index {} out of bounds", idx),
                    });
                }
                shape.push(self.clone_string(idx)?);
            }

            types.push(TensorType::new(dtype, shape));
        }

        // Value table
        let n_values = read_bounded_count(r)?;
        let mut values = Vec::with_capacity(n_values);
        for vid in 0..n_values {
            let value = self.decode_value(r, vid, types.len())?;
            values.push(value);
        }

        // Output
        let output = uleb128_read(r)? as usize;
        if output >= values.len() && !values.is_empty() {
            return Err(MicbError {
                message: format!("output {} out of bounds (max {})", output, values.len() - 1),
            });
        }

        // §3.4 detection rule: after output varint, peek one byte.
        // EOF → no MAP (empty). 0x4D → MAP follows. Any other → parse error.
        let map = self.decode_map_section(r)?;

        Ok(Graph {
            symbols,
            types,
            values,
            output,
            map,
        })
    }

    fn decode_value<R: Read>(
        &self,
        r: &mut R,
        current_id: usize,
        n_types: usize,
    ) -> Result<Value, MicbError> {
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag)?;

        match tag[0] {
            0 | 1 => {
                // Arg or Param
                let name_idx = uleb128_read(r)? as usize;
                let type_idx = uleb128_read(r)? as usize;

                if name_idx >= self.strings.len() {
                    return Err(MicbError {
                        message: format!("value name string index {} out of bounds", name_idx),
                    });
                }
                if type_idx >= n_types {
                    return Err(MicbError {
                        message: format!("value type index {} out of bounds", type_idx),
                    });
                }

                let name = self.clone_string(name_idx)?;
                if tag[0] == 0 {
                    Ok(Value::Arg(name, type_idx))
                } else {
                    Ok(Value::Param(name, type_idx))
                }
            }
            2 => {
                // Node
                let opcode = self.decode_opcode(r)?;
                let n_inputs = read_bounded_count(r)?;
                let mut inputs = Vec::with_capacity(n_inputs);
                for _ in 0..n_inputs {
                    let inp = uleb128_read(r)? as usize;
                    if inp >= current_id {
                        return Err(MicbError {
                            message: format!(
                                "forward reference: input {} >= current id {}",
                                inp, current_id
                            ),
                        });
                    }
                    inputs.push(inp);
                }
                Ok(Value::Node(opcode, inputs))
            }
            _ => Err(MicbError {
                message: format!("unknown value tag: {}", tag[0]),
            }),
        }
    }

    /// §3.4 detection rule: peek one byte after output varint.
    /// EOF → empty MAP. 0x4D → MAP section follows. Any other → parse error.
    fn decode_map_section<R: Read>(&self, r: &mut R) -> Result<Map, MicbError> {
        let mut sentinel = [0u8; 1];
        match r.read_exact(&mut sentinel) {
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                // EOF = no MAP section (§2 rule 3, §3.4).
                return Ok(Map::new());
            }
            Err(e) => {
                return Err(MicbError {
                    message: e.to_string(),
                });
            }
            Ok(()) => {}
        }
        if sentinel[0] != 0x4D {
            return Err(MicbError {
                message: format!(
                    "unexpected byte 0x{:02X} after output varint: expected 0x4D (MAP) or EOF",
                    sentinel[0]
                ),
            });
        }
        // MAP marker confirmed — read entries at nesting depth 0.
        self.decode_map_entries(r, 0)
    }

    /// Decode one MAP level.
    ///
    /// Every §3.2/§3.5 rule applied here comes from `super::map_limits`, which
    /// the TEXT parser calls at the matching points. That is the whole contract:
    /// both serializations encode the same `Map`, so anything the text side
    /// would refuse must not be reachable through the binary side — otherwise
    /// `binary → Map → emit_mic2 → parse_mic2` stops being the identity (MAP
    /// keys are emitted raw and unescaped, so a key carrying a newline re-emits
    /// as two entries). `parse_micb` is a public API documented as accepting
    /// untrusted input, so the caller cannot be assumed to pre-validate.
    fn decode_map_entries<R: Read>(&self, r: &mut R, depth: usize) -> Result<Map, MicbError> {
        let count = read_bounded_count(r)?;
        // §3.5 entry count, on the DECLARED count: refuse before the loop rather
        // than after decoding a budget's worth of an absurd claim.
        check_map_entry_count(count).map_err(|message| MicbError { message })?;

        let mut map = Map::new();
        for _ in 0..count {
            // §3.5 entry count again, now recursively: a level whose declared
            // count is legal can still blow the subtree budget through nested
            // maps. Mirrors the text parser's per-entry check exactly.
            check_map_entry_count(map.recursive_entry_count() + 1)
                .map_err(|message| MicbError { message })?;

            let key_idx = uleb128_read(r)? as usize;
            let key = self.strings.get(key_idx).ok_or_else(|| MicbError {
                message: format!("MAP key string index {key_idx} out of bounds"),
            })?;
            // §3.2 grammar + §3.5 key bounds, checked on the BORROWED table
            // entry so a rejected key never charges the clone budget.
            check_map_key(key).map_err(|message| MicbError { message })?;

            let key = self.clone_string(key_idx)?;
            let value = self.decode_map_value(r, depth)?;
            map.insert_unique(key, value)
                .map_err(|e| MicbError { message: e })?;
        }
        Ok(map)
    }

    fn decode_map_value<R: Read>(&self, r: &mut R, depth: usize) -> Result<MapValue, MicbError> {
        let mut tag = [0u8; 1];
        r.read_exact(&mut tag)?;
        match tag[0] {
            0 => {
                let idx = uleb128_read(r)? as usize;
                let s = self.strings.get(idx).ok_or_else(|| MicbError {
                    message: format!("MAP string value index {idx} out of bounds"),
                })?;
                // §3.5 string size, on the borrowed entry so an oversize value
                // never charges the clone budget.
                check_map_string_len(s.len()).map_err(|message| MicbError { message })?;
                Ok(MapValue::String(self.clone_string(idx)?))
            }
            1 => {
                let i = sleb128_read(r)?;
                Ok(MapValue::Int(i))
            }
            2 => {
                let len = read_bounded_count(r)?;
                // §3.5 bytes size, checked BEFORE the allocation it authorises.
                check_map_bytes_len(len).map_err(|message| MicbError { message })?;
                let mut buf = vec![0u8; len];
                r.read_exact(&mut buf)?;
                Ok(MapValue::Bytes(buf))
            }
            3 => {
                // §3.5 nesting, checked at the descend point with the level of
                // the map we are descending FROM — the same placement the text
                // parser uses, so the two agree on the deepest legal map rather
                // than differing by one level. This is also the stack bound on
                // untrusted input: `decode_map_entries` and `decode_map_value`
                // are mutually recursive through this arm, and `MAX_MICB_INPUT`
                // bounds total SIZE while saying nothing about DEPTH.
                check_map_nesting_depth(depth).map_err(|message| MicbError { message })?;
                let inner = self.decode_map_entries(r, depth + 1)?;
                Ok(MapValue::Nested(inner))
            }
            other => Err(MicbError {
                message: format!("unknown MAP value tag: {other}"),
            }),
        }
    }

    fn decode_opcode<R: Read>(&self, r: &mut R) -> Result<Opcode, MicbError> {
        let mut opcode_byte = [0u8; 1];
        r.read_exact(&mut opcode_byte)?;

        match opcode_byte[0] {
            0 => Ok(Opcode::Matmul),
            1 => Ok(Opcode::Add),
            2 => Ok(Opcode::Sub),
            3 => Ok(Opcode::Mul),
            4 => Ok(Opcode::Div),
            5 => Ok(Opcode::Relu),
            6 => {
                let axis = sleb128_read(r)?;
                Ok(Opcode::Softmax(axis))
            }
            7 => Ok(Opcode::Sigmoid),
            8 => Ok(Opcode::Tanh),
            9 => Ok(Opcode::Gelu),
            10 => Ok(Opcode::LayerNorm),
            11 => {
                let n = read_bounded_count(r)?;
                let mut perm = Vec::with_capacity(n);
                for _ in 0..n {
                    perm.push(sleb128_read(r)?);
                }
                Ok(Opcode::Transpose(perm))
            }
            12 => Ok(Opcode::Reshape),
            13 => {
                let n = read_bounded_count(r)?;
                let mut axes = Vec::with_capacity(n);
                for _ in 0..n {
                    axes.push(sleb128_read(r)?);
                }
                Ok(Opcode::Sum(axes))
            }
            14 => {
                let n = read_bounded_count(r)?;
                let mut axes = Vec::with_capacity(n);
                for _ in 0..n {
                    axes.push(sleb128_read(r)?);
                }
                Ok(Opcode::Mean(axes))
            }
            15 => {
                let n = read_bounded_count(r)?;
                let mut axes = Vec::with_capacity(n);
                for _ in 0..n {
                    axes.push(sleb128_read(r)?);
                }
                Ok(Opcode::Max(axes))
            }
            16 => {
                let axis = sleb128_read(r)?;
                Ok(Opcode::Concat(axis))
            }
            17 => {
                let axis = sleb128_read(r)?;
                let n = uleb128_read(r)? as usize;
                Ok(Opcode::Split(axis, n))
            }
            18 => {
                let axis = sleb128_read(r)?;
                Ok(Opcode::Gather(axis))
            }
            255 => {
                let idx = uleb128_read(r)? as usize;
                if idx >= self.strings.len() {
                    return Err(MicbError {
                        message: format!("custom opcode name index {} out of bounds", idx),
                    });
                }
                Ok(Opcode::Custom(self.clone_string(idx)?))
            }
            _ => Err(MicbError {
                message: format!("unknown opcode byte: {}", opcode_byte[0]),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::compact::v2::types::GraphEq;
    use std::io::Cursor;

    #[test]
    fn test_roundtrip_residual() {
        let graph = Graph::residual_block();

        let mut buf = Vec::new();
        emit_micb(&graph, &mut buf).expect("encode failed");

        let mut cursor = Cursor::new(&buf);
        let parsed = parse_micb(&mut cursor).expect("decode failed");

        assert!(graph.eq(&parsed));
    }

    #[test]
    fn test_determinism() {
        let graph = Graph::residual_block();

        let mut buf1 = Vec::new();
        let mut buf2 = Vec::new();

        emit_micb(&graph, &mut buf1).unwrap();
        emit_micb(&graph, &mut buf2).unwrap();

        assert_eq!(buf1, buf2);
    }

    #[test]
    fn test_magic_check() {
        let bad_magic = vec![0x00, 0x00, 0x00, 0x00, 0x02];
        let mut cursor = Cursor::new(&bad_magic);
        assert!(parse_micb(&mut cursor).is_err());
    }

    #[test]
    fn test_version_check() {
        let bad_version = vec![0x4D, 0x49, 0x43, 0x42, 0x99];
        let mut cursor = Cursor::new(&bad_version);
        assert!(parse_micb(&mut cursor).is_err());
    }

    #[test]
    fn test_empty_graph() {
        let graph = Graph::new();

        let mut buf = Vec::new();
        emit_micb(&graph, &mut buf).expect("encode failed");

        // Should be small: magic(4) + version(1) + counts
        assert!(buf.len() < 20);

        let mut cursor = Cursor::new(&buf);
        let parsed = parse_micb(&mut cursor).expect("decode failed");
        assert!(graph.eq(&parsed));
    }

    #[test]
    fn test_size_comparison() {
        use crate::ir::compact::v2::emit::emit_mic2;

        let graph = Graph::residual_block();

        // Text format
        let text = emit_mic2(&graph);
        let text_size = text.len();

        // Binary format
        let mut buf = Vec::new();
        emit_micb(&graph, &mut buf).unwrap();
        let binary_size = buf.len();

        // Binary should be smaller
        assert!(
            binary_size < text_size,
            "binary ({}) should be smaller than text ({})",
            binary_size,
            text_size
        );

        println!(
            "Residual block: text={} bytes, binary={} bytes, ratio={:.2}x",
            text_size,
            binary_size,
            text_size as f64 / binary_size as f64
        );
    }

    #[test]
    fn test_string_table_dedup() {
        use crate::ir::compact::v2::types::Value;

        let mut graph = Graph::new();

        // Same dim used multiple times
        graph.add_type(TensorType::new(
            DType::F32,
            vec!["128".into(), "128".into()],
        ));
        graph.add_type(TensorType::new(DType::F32, vec!["128".into()]));
        graph.add_value(Value::arg("x", 0));

        let mut buf = Vec::new();
        emit_micb(&graph, &mut buf).unwrap();

        // String "128" should only appear once in table
        let mut cursor = Cursor::new(&buf);
        let parsed = parse_micb(&mut cursor).unwrap();
        assert!(graph.eq(&parsed));
    }
}
