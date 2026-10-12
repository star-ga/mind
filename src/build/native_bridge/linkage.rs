// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Std seed linkage for the native bridge.
//!
//! The frozen compiler receives `std blob ++ user region` and compiles it as one
//! program. This module prepares what admission needs to reason about the std half
//! of that program — per-module names and one lowering of the seed — and fences
//! each std function a program links. It never changes the image: the seed set,
//! its order and its bytes are exactly what `stage1.elf` was minted from.

use std::collections::BTreeSet;

use crate::ir::frozen_profile::{
    FrozenProfileRejection, profile_frozen_admits_linked, profile_frozen_admits_program,
    profile_frozen_admits_source,
};
use crate::ir::native_closure::{
    StdSeed, reserved_names_from_text, reserved_struct_names_from_text,
};

/// Stack for the seed lowering thread: the seed is ~400 KB of source, lowered
/// recursively, so it gets the same headroom a deep user program would.
const SEED_LOWERING_STACK: usize = 256 * 1024 * 1024;

/// Build the [`StdSeed`] for one native build.
///
/// The seed is parsed and lowered on a FRESH THREAD. Lowering keeps its working
/// state in thread-locals; running it on the bridge's own thread would leave the
/// seed's state behind for the program's lowering that follows, so admission of
/// the same program could depend on whether the seed had been lowered first. A
/// seed that does not parse or lower (or a thread that cannot be spawned) yields
/// `lowered: None`, which reserves every seed name — refusal only, never a guess.
pub(super) fn std_seed_for_linkage(modules: &[&str], sources: &[Vec<u8>], blob: &[u8]) -> StdSeed {
    let structs = sources
        .iter()
        .flat_map(|src| reserved_struct_names_from_text(&String::from_utf8_lossy(src)))
        .collect();
    let names = modules
        .iter()
        .zip(sources)
        .map(|(module, src)| {
            (
                (*module).to_string(),
                reserved_names_from_text(&String::from_utf8_lossy(src)),
            )
        })
        .collect();
    let text = match std::str::from_utf8(blob) {
        Ok(t) => t.to_string(),
        Err(_) => {
            return StdSeed {
                modules: names,
                structs,
                lowered: None,
            };
        }
    };
    let lowered = std::thread::Builder::new()
        .name("native-std-seed".into())
        .stack_size(SEED_LOWERING_STACK)
        .spawn(move || lower_seed_dropping_refused(crate::parser::parse(&text).ok()?))
        .ok()
        .and_then(|handle| handle.join().ok())
        .flatten();
    StdSeed {
        modules: names,
        structs,
        lowered,
    }
}

/// How many seed functions lowering may refuse before the seed is given up on.
const MAX_UNLOWERABLE_SEED_FNS: usize = 64;

/// Lower the seed, DROPPING each top-level function lowering refuses.
///
/// Lowering is all-or-nothing per program, and a few seed functions use constructs
/// it cannot represent (measured: an inline struct-field read). Giving up on the
/// whole seed for one such function would refuse every program that calls ANY std
/// function. Instead the refused function is removed from the seed AST and the
/// rest is lowered again. A dropped function has no IR body, so it is never a
/// linkable name: a program that reaches it is refused `closure.unresolved_edge`,
/// exactly as before linkage existed. Dropping therefore only ever turns a
/// would-be admission into a refusal, never the reverse.
///
/// Lowering panics and refusals without a source position give up (`None`).
fn lower_seed_dropping_refused(
    mut ast: crate::ast::Module,
) -> Option<(crate::ast::Module, crate::ir::IRModule)> {
    use crate::eval::materialization::MaterializationRefusal as Refusal;
    for _ in 0..=MAX_UNLOWERABLE_SEED_FNS {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let lowered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::eval::lower_to_ir(&ast)
        }));
        std::panic::set_hook(previous);
        let at = match lowered {
            Ok(Ok(ir)) => return Some((ast, ir)),
            Ok(Err(
                Refusal::UnsupportedLoweringOperation { start, .. }
                | Refusal::InvalidLoweringReceiver { start, .. },
            )) => start,
            Ok(Err(_)) | Err(_) => return None,
        };
        // The top-level function whose span holds `at`. A refusal outside every
        // function gives up rather than dropping an innocent one.
        let victim = ast.items.iter().position(|item| {
            matches!(item, crate::ast::Node::FnDef(_, span)
                if span.start() <= at && at < span.end())
        })?;
        ast.items.remove(victim);
    }
    None
}

/// Fence one linked std seed function with the same predicates a user body
/// passes: the source-level width checks, the f64 slot proof and the IR construct
/// allowlist.
/// A linked name whose body cannot be found is refused, never assumed proven.
pub(super) fn fence_std_seed_fn(
    seed: &StdSeed,
    name: &str,
    linked: &BTreeSet<String>,
) -> Result<(), FrozenProfileRejection> {
    let missing = || FrozenProfileRejection {
        construct: "std.linked_body_unavailable",
    };
    let source = seed.fn_source(name).ok_or_else(missing)?;
    profile_frozen_admits_source(&source)?;
    super::float_slots::admit_f64_slots(&source).map_err(|r| FrozenProfileRejection {
        construct: r.construct,
    })?;
    let ir = seed.fn_ir(name).ok_or_else(missing)?;
    profile_frozen_admits_linked(&ir, linked)
}

/// The whole-program half of the fence: `user` together with the IR of every
/// linked std seed function, so a fact that holds only across a call (an i64
/// argument, a returned value, the bytes a call tree allocates) is proven at both
/// ends. Runs after every linked body passed [`fence_std_seed_fn`].
pub(super) fn admit_linked_program(
    seed: &StdSeed,
    user: &crate::ir::IRModule,
    linked: &BTreeSet<String>,
) -> Result<(), FrozenProfileRejection> {
    let irs = linked
        .iter()
        .map(|name| {
            seed.fn_ir(name).ok_or(FrozenProfileRejection {
                construct: "std.linked_body_unavailable",
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    profile_frozen_admits_program(user, &irs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The embedded seed must lower, or every build silently loses std linkage and
    /// falls back to reserving all 657 seed names.
    #[test]
    fn the_embedded_seed_lowers_for_linkage() {
        let modules = crate::build::native_bridge::frozen_std_seed_modules();
        let sources = super::super::seed::std_seed_sources(&modules).expect("embedded seed");
        let blob = crate::build::native_image::compose_std_blob(&sources);
        // Name the failing stage instead of only observing `lowered == None`.
        let text = std::str::from_utf8(blob.as_bytes()).expect("seed is UTF-8");
        crate::parser::parse(text).expect("the std seed parses");
        let seed = std_seed_for_linkage(&modules, &sources, blob.as_bytes());
        assert_eq!(seed.modules.len(), 21);
        let (ast, _) = seed
            .lowered
            .as_ref()
            .expect("the embedded std seed must lower");
        let all = seed.all_names();
        assert!(
            all.contains("vec_push") && all.contains("bsig0"),
            "{}",
            all.len()
        );
        let kept: BTreeSet<String> = ast
            .items
            .iter()
            .filter_map(|item| match item {
                crate::ast::Node::FnDef(f, _) => Some(f.name.clone()),
                _ => None,
            })
            .collect();
        let dropped: Vec<&String> = all.difference(&kept).collect();
        // Pinned so a lowering change that drops more of the seed is noticed: each
        // dropped function is a std function no native program can link.
        assert_eq!(
            dropped,
            [
                "args_first_positional_idx",
                "args_flag_value",
                "args_flag_value_idx",
                "args_has_flag",
                "args_push",
                "equals_value",
                "is_double_dash"
            ],
            "seed functions lowering refuses (std.cli struct-field reads)"
        );
        assert!(seed.fn_ir("vec_push").is_some());
        assert!(seed.fn_source("vec_push").is_some());
        assert!(seed.fn_ir("no_such_seed_fn").is_none());
    }

    /// The seed struct names are derived from the seed text, so a seed that gains a
    /// struct refuses a program declaring it without a code change. Pinned to the
    /// set measured to crash the frozen compiler when a program redeclares it.
    #[test]
    fn the_seed_struct_names_come_from_the_seed_text() {
        let modules = crate::build::native_bridge::frozen_std_seed_modules();
        let sources = super::super::seed::std_seed_sources(&modules).expect("embedded seed");
        let blob = crate::build::native_image::compose_std_blob(&sources);
        let seed = std_seed_for_linkage(&modules, &sources, blob.as_bytes());
        let want: BTreeSet<String> = [
            "Args", "Box", "File", "Map", "String", "TermSize", "Text", "Value", "Vec",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(seed.structs, want);
    }

    /// A linked name with no body refuses by name instead of passing the fence.
    #[test]
    fn a_linked_name_without_a_body_is_refused() {
        let seed = StdSeed {
            modules: Vec::new(),
            structs: BTreeSet::new(),
            lowered: None,
        };
        let err = fence_std_seed_fn(&seed, "anything", &BTreeSet::new()).unwrap_err();
        assert_eq!(err.construct, "std.linked_body_unavailable");
    }
}
