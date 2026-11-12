// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Composes the input image consumed by the frozen pure-MIND compiler.
//!
//! Wire layout: `[8B user_lo LE][8B src_len LE][std_blob ++ user_region]`.
//! `user_lo` marks the first user byte; `src_len` covers the complete source image.
//! Standard-library modules retain manifest order and a trailing newline each.
//! User sources retain resolver order with a newline between sources only, keeping
//! a single source byte-identical to the original input layout.
//!
//! Order is semantic: the frozen compiler resolves duplicate bare names by their
//! position. The caller must reject collisions before emitting the image.

/// The std seed blob: every seed module's bytes, each followed by `\n`.
///
/// A newtype, not a bare `Vec<u8>`, because the seam is a correctness boundary
/// rather than a convention. Everything at or after `user_lo` is a prune ROOT to
/// the frozen compiler; everything before it is prunable std. If user text could
/// be passed where a std blob is expected, it would land below the seam and
/// silently stop being a root — changing which bodies the compiler keeps, with no
/// diagnostic. Making the only constructor [`compose_std_blob`] also guarantees
/// the per-module trailing `\n`: a blob assembled by hand without it would glue
/// its last std line to the first user line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StdBlob(Vec<u8>);

impl StdBlob {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// Compose the std seed blob: each module's bytes followed by `\n`.
pub fn compose_std_blob(modules: &[Vec<u8>]) -> StdBlob {
    let mut blob = Vec::new();
    for m in modules {
        blob.extend_from_slice(m);
        blob.push(b'\n');
    }
    StdBlob(blob)
}

/// The user region: sources in the caller's order, joined by a single `\n`.
///
/// The joiner goes BETWEEN sources only, so a single source is emitted verbatim —
/// that is what keeps the single-file path byte-identical, and a trailing newline
/// there would be exactly the silent one-byte drift the baselines exist to catch.
///
/// An EMPTY source is not special-cased, and the consequence is worth stating
/// because it is a bytes policy: `["a", ""]` composes `a\n` and `["", "a"]`
/// composes `\na`. Each empty element still contributes its joiner. That is
/// harmless to the frozen parser but it is observable, so a caller that could
/// produce empty sources must refuse them BEFORE composing rather than rely on
/// this function to absorb them. The bridge never does: its sources come from a
/// resolved file set, and an unreadable file is a hard refusal upstream.
pub fn compose_user_region(sources: &[Vec<u8>]) -> Vec<u8> {
    let mut region = Vec::new();
    for (i, src) in sources.iter().enumerate() {
        if i > 0 {
            region.push(b'\n');
        }
        region.extend_from_slice(src);
    }
    region
}

/// A composed image plus the two offsets the frozen driver needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeImage {
    pub bytes: Vec<u8>,
    pub user_lo: i64,
    pub src_len: i64,
}

/// Compose the full stdin image from a std blob and an ordered user source list.
pub fn compose_image(std_blob: &StdBlob, sources: &[Vec<u8>]) -> NativeImage {
    let user_region = compose_user_region(sources);
    let mut combined = Vec::with_capacity(std_blob.len() + user_region.len());
    combined.extend_from_slice(std_blob.as_bytes());
    let user_lo = combined.len() as i64;
    combined.extend_from_slice(&user_region);
    let src_len = combined.len() as i64;

    let mut bytes = Vec::with_capacity(16 + combined.len());
    bytes.extend_from_slice(&user_lo.to_le_bytes());
    bytes.extend_from_slice(&src_len.to_le_bytes());
    bytes.extend_from_slice(&combined);
    NativeImage {
        bytes,
        user_lo,
        src_len,
    }
}

/// Magic tagging a v2 (explicit-layout) image header: `"MINDLAY\x01"` as a
/// little-endian i64. Mirrors `nb_layout_magic()` in `examples/mindc_mind/main.mind`.
pub const LAYOUT_MAGIC: i64 = i64::from_le_bytes(*b"MINDLAY\x01");

/// Fixed arena base of every native image (`nb_arena_addr()` = 16 MiB).
pub const ARENA_ADDR: u64 = 0x100_0000;

/// The memory layout of an emitted native image.
///
/// One value decides every layout-dependent byte the frozen compiler emits: the
/// arena PT_LOAD size, the stack PT_LOAD address and size, the entry stack
/// pointer and the immediate of each inlined allocation guard. The default
/// profile is not represented here: when no layout is requested the bridge sends
/// the v1 header, so default images are byte-identical to before this type
/// existed. Field meaning and the validation rules mirror `nb_layout_check` in
/// `examples/mindc_mind/main.mind`, which re-validates inside the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeLayout {
    /// Bump-arena PT_LOAD size (bytes, whole pages, >= 1 MiB).
    pub arena_bytes: u64,
    /// Call-stack PT_LOAD size (bytes, whole pages, >= 64 KiB).
    pub stack_bytes: u64,
    /// Unmapped gap between the arena end and the stack bottom (bytes, >= 1 page).
    pub guard_bytes: u64,
}

/// Largest single region a layout may request (1 TiB), so sums cannot overflow.
const LAYOUT_REGION_MAX: u64 = 1 << 40;
/// The stack top must stay below 2^46, far inside the 47-bit user address space.
const LAYOUT_TOP_MAX: u64 = 1 << 46;

impl NativeLayout {
    /// The BOUNDED profile: every native image -- including the compiler seed --
    /// loads and runs under `RLIMIT_AS` = 2 GiB.
    ///
    /// Sized from measured peaks of the self-host compile (the largest input
    /// the seed handles): the arena high-water mark is below 537 MiB (the
    /// self-compile succeeds under a 537 MiB arena cap) and the stack peak touch
    /// is 228,597,760 bytes, so 768 MiB arena and 384 MiB stack leave at least
    /// 30% and 43% headroom. The 1 MiB guard is unmapped (no address-space cost)
    /// and exceeds the largest frame of the compiler seed (15,216 bytes); a frame
    /// larger than the guard could still step over it. Reserved load total is
    /// about 1.15 GiB (measured VmPeak 1,182,300 kB for the self-compile),
    /// leaving room under the 2 GiB cap for the kernel's initial stack and vDSO.
    pub const BOUNDED: NativeLayout = NativeLayout {
        arena_bytes: 768 << 20,
        stack_bytes: 384 << 20,
        guard_bytes: 1 << 20,
    };

    /// Look up a named profile. `default` means "no explicit layout" (`None`).
    pub fn from_profile(name: &str) -> Result<Option<NativeLayout>, String> {
        match name {
            "default" => Ok(None),
            "bounded" => Ok(Some(NativeLayout::BOUNDED)),
            other => Err(format!(
                "unknown native profile `{other}` (expected `default` or `bounded`)"
            )),
        }
    }

    /// Check the layout exactly as the compiler will, with a readable reason.
    pub fn validate(&self) -> Result<(), String> {
        let named = [
            ("arena", self.arena_bytes, 1 << 20),
            ("stack", self.stack_bytes, 64 << 10),
            ("guard", self.guard_bytes, 4096),
        ];
        for (name, v, min) in named {
            if v < min {
                return Err(format!("{name} size {v} is below the minimum {min} bytes"));
            }
            if v > LAYOUT_REGION_MAX {
                return Err(format!("{name} size {v} exceeds the 1 TiB region limit"));
            }
            if v % 4096 != 0 {
                return Err(format!(
                    "{name} size {v} is not a whole number of 4096-byte pages"
                ));
            }
        }
        if self.stack_top() > LAYOUT_TOP_MAX {
            return Err(format!(
                "stack top {:#x} exceeds the {LAYOUT_TOP_MAX:#x} address limit",
                self.stack_top()
            ));
        }
        Ok(())
    }

    /// Stack PT_LOAD base: the arena end plus the unmapped guard gap.
    pub fn stack_addr(&self) -> u64 {
        ARENA_ADDR + self.arena_bytes + self.guard_bytes
    }

    /// One past the highest stack byte (the entry stack pointer region top).
    pub fn stack_top(&self) -> u64 {
        self.stack_addr() + self.stack_bytes
    }

    /// The (vaddr, memsz) of the arena and stack PT_LOADs an image built with
    /// this layout must carry.
    pub fn expected_bss_segments(&self) -> [(u64, u64); 2] {
        [
            (ARENA_ADDR, self.arena_bytes),
            (self.stack_addr(), self.stack_bytes),
        ]
    }
}

/// Compose the stdin image, with the v2 header when a layout is requested.
///
/// `None` is exactly [`compose_image`] (the v1 header), so the default path
/// cannot drift. `Some` prefixes `[magic][arena][stack][guard]` to the v1 header.
pub fn compose_image_with_layout(
    std_blob: &StdBlob,
    sources: &[Vec<u8>],
    layout: Option<&NativeLayout>,
) -> NativeImage {
    let img = compose_image(std_blob, sources);
    let Some(l) = layout else {
        return img;
    };
    let mut bytes = Vec::with_capacity(32 + img.bytes.len());
    bytes.extend_from_slice(&LAYOUT_MAGIC.to_le_bytes());
    for v in [l.arena_bytes, l.stack_bytes, l.guard_bytes] {
        bytes.extend_from_slice(&(v as i64).to_le_bytes());
    }
    bytes.extend_from_slice(&img.bytes);
    NativeImage {
        bytes,
        user_lo: img.user_lo,
        src_len: img.src_len,
    }
}

/// The (vaddr, memsz) of every RW PT_LOAD of a 64-bit little-endian ELF, in
/// program-header order; `None` when the header table is malformed.
pub fn rw_load_segments(elf: &[u8]) -> Option<Vec<(u64, u64)>> {
    let rd16 = |o: usize| -> Option<u64> {
        Some(u16::from_le_bytes(elf.get(o..o + 2)?.try_into().ok()?) as u64)
    };
    let rd32 = |o: usize| -> Option<u64> {
        Some(u32::from_le_bytes(elf.get(o..o + 4)?.try_into().ok()?) as u64)
    };
    let rd64 =
        |o: usize| -> Option<u64> { Some(u64::from_le_bytes(elf.get(o..o + 8)?.try_into().ok()?)) };
    let phoff = rd64(32)? as usize;
    let phentsize = rd16(54)? as usize;
    let phnum = rd16(56)? as usize;
    if phentsize != 56 {
        return None;
    }
    let mut out = Vec::new();
    for i in 0..phnum {
        let ph = phoff.checked_add(i.checked_mul(phentsize)?)?;
        // PT_LOAD with PF_W set.
        if rd32(ph)? == 1 && rd32(ph + 4)? & 2 != 0 {
            out.push((rd64(ph + 16)?, rd64(ph + 40)?));
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_magic_matches_the_compiler_constant() {
        // nb_layout_magic() in examples/mindc_mind/main.mind.
        assert_eq!(LAYOUT_MAGIC, 97180662784477517);
    }

    #[test]
    fn no_layout_is_exactly_the_v1_image() {
        let blob = compose_std_blob(&[b"s".to_vec()]);
        let user = vec![b"fn main() -> i64 { return 42; }".to_vec()];
        assert_eq!(
            compose_image_with_layout(&blob, &user, None),
            compose_image(&blob, &user)
        );
    }

    #[test]
    fn bounded_layout_prefixes_the_v2_header() {
        let blob = compose_std_blob(&[b"s".to_vec()]);
        let user = vec![b"u".to_vec()];
        let v1 = compose_image(&blob, &user);
        let v2 = compose_image_with_layout(&blob, &user, Some(&NativeLayout::BOUNDED));
        assert_eq!(&v2.bytes[32..], &v1.bytes[..]);
        let w = |k: usize| i64::from_le_bytes(v2.bytes[k * 8..k * 8 + 8].try_into().unwrap());
        assert_eq!(w(0), LAYOUT_MAGIC);
        assert_eq!(w(1), 768 << 20);
        assert_eq!(w(2), 384 << 20);
        assert_eq!(w(3), 1 << 20);
    }

    #[test]
    fn bounded_profile_fits_two_gib_with_room() {
        let l = NativeLayout::BOUNDED;
        l.validate().unwrap();
        assert_eq!(l.stack_addr(), 823_132_160);
        assert!(l.arena_bytes + l.stack_bytes < (2u64 << 30) - (512 << 20));
    }

    #[test]
    fn layout_validation_refuses_each_bad_shape() {
        let ok = NativeLayout::BOUNDED;
        for bad in [
            NativeLayout {
                arena_bytes: 4096,
                ..ok
            },
            NativeLayout {
                stack_bytes: 4096,
                ..ok
            },
            NativeLayout {
                guard_bytes: 0,
                ..ok
            },
            NativeLayout {
                arena_bytes: ok.arena_bytes + 1,
                ..ok
            },
            NativeLayout {
                stack_bytes: 1 << 41,
                ..ok
            },
            NativeLayout {
                guard_bytes: 1 << 41,
                ..ok
            },
        ] {
            assert!(bad.validate().is_err(), "{bad:?} must refuse");
        }
    }

    #[test]
    fn profile_names() {
        assert_eq!(NativeLayout::from_profile("default").unwrap(), None);
        assert_eq!(
            NativeLayout::from_profile("bounded").unwrap(),
            Some(NativeLayout::BOUNDED)
        );
        assert!(NativeLayout::from_profile("tiny").is_err());
    }

    /// The load-bearing property: with ONE source the joiner contributes nothing,
    /// so the composed bytes are exactly the pre-existing single-file bytes.
    #[test]
    fn one_source_composes_the_legacy_layout_exactly() {
        let blob = compose_std_blob(&[b"a".to_vec(), b"bb".to_vec()]);
        assert_eq!(blob.as_bytes(), b"a\nbb\n");
        let user = b"fn main() -> i64 { return 7; }".to_vec();

        let img = compose_image(&blob, std::slice::from_ref(&user));

        // Byte-for-byte reconstruction of what the bridge did before this module.
        let mut legacy = blob.as_bytes().to_vec();
        let legacy_user_lo = legacy.len() as i64;
        legacy.extend_from_slice(&user);
        let legacy_src_len = legacy.len() as i64;
        let mut expect = Vec::new();
        expect.extend_from_slice(&legacy_user_lo.to_le_bytes());
        expect.extend_from_slice(&legacy_src_len.to_le_bytes());
        expect.extend_from_slice(&legacy);

        assert_eq!(img.bytes, expect, "single-source bytes must not move");
        assert_eq!(img.user_lo, legacy_user_lo);
        assert_eq!(img.src_len, legacy_src_len);
    }

    /// `src_len` is the WHOLE image length, not the user length.
    #[test]
    fn src_len_covers_std_and_user() {
        let blob = compose_std_blob(&[b"xy".to_vec()]);
        let img = compose_image(&blob, &[b"abc".to_vec()]);
        assert_eq!(img.user_lo, 3, "std blob is 'xy' + newline");
        assert_eq!(img.src_len, 6, "3 std + 3 user, NOT 3");
        assert_eq!(&img.bytes[16..], b"xy\nabc");
    }

    /// Sources are joined by exactly one newline, between only — no trailer.
    #[test]
    fn sources_are_joined_by_a_single_newline_with_no_trailer() {
        let img = compose_image(
            &compose_std_blob(&[]),
            &[b"one".to_vec(), b"two".to_vec(), b"three".to_vec()],
        );
        assert_eq!(&img.bytes[16..], b"one\ntwo\nthree");
        assert_eq!(img.user_lo, 0);
        assert_eq!(img.src_len, 13);
    }

    /// Order is preserved EXACTLY, asserted against expected bytes.
    ///
    /// Asserting merely that two orders DIFFER would be tautological: a function
    /// that reversed its input would satisfy it. The frozen compiler resolves
    /// duplicate bare names last-definition-wins, so which order actually lands in
    /// the image decides which program runs.
    #[test]
    fn order_is_preserved_exactly() {
        let a = b"AAA".to_vec();
        let b = b"BBB".to_vec();
        let empty = compose_std_blob(&[]);
        assert_eq!(
            &compose_image(&empty, &[a.clone(), b.clone()]).bytes[16..],
            b"AAA\nBBB"
        );
        assert_eq!(&compose_image(&empty, &[b, a]).bytes[16..], b"BBB\nAAA");
    }

    /// The empty-source policy is PINNED rather than left to discovery: an empty
    /// element still contributes its joiner, so `["a", ""]` composes `a\n`. That
    /// is observable, so a caller which could produce empty sources must refuse
    /// them before composing.
    #[test]
    fn an_empty_source_still_contributes_its_joiner() {
        let empty = compose_std_blob(&[]);
        assert_eq!(
            &compose_image(&empty, &[b"a".to_vec(), b"".to_vec()]).bytes[16..],
            b"a\n"
        );
        assert_eq!(
            &compose_image(&empty, &[b"".to_vec(), b"a".to_vec()]).bytes[16..],
            b"\na"
        );
    }

    /// An empty source list is a legal composition of zero user bytes: user_lo
    /// and src_len coincide. The bridge refuses earlier, but the function must
    /// not panic or silently produce a trailing joiner.
    #[test]
    fn no_sources_yields_an_empty_user_region() {
        let blob = compose_std_blob(&[b"s".to_vec()]);
        let img = compose_image(&blob, &[]);
        assert_eq!(img.user_lo, 2);
        assert_eq!(img.src_len, 2);
        assert_eq!(&img.bytes[16..], b"s\n");
    }
}
