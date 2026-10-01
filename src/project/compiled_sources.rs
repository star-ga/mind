// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! The outcome of compiling a project's sources to native objects.
//!
//! `compile_sources` used to return a bare tuple. It grew a fourth member — the
//! CAUSE a module fell back — and at that width a positional return stops being
//! readable: `(Vec<PathBuf>, bool, Vec<String>, Option<FallbackReason>)` names
//! nothing, and the two `Vec`-shaped members are distinguishable only by
//! position at the call site. Naming the members is what makes the caller's
//! `if !entry_native_compiled` check obviously about the ENTRY rather than
//! about whichever `bool` came third.

use std::path::PathBuf;

use crate::diagnostics::capability::FallbackReason;

/// What compiling a project's sources produced.
#[derive(Debug, Clone, Default)]
pub struct CompiledSources {
    /// The native object files to link, in compile order.
    pub objects: Vec<PathBuf>,
    /// The std substrate archive (see `substrate_link::SubstrateArchive`), to
    /// link after EVERY object — the native_sources C objects included, since
    /// an archive only resolves references from objects that precede it.
    pub substrate_archive: Option<PathBuf>,
    /// Whether the ENTRY module was natively compiled.
    ///
    /// `false` means it was embedded as a runtime-JIT fallback, and every
    /// caller that reports success fails loud on it instead: the artifact is a
    /// launcher deferring to the installed mind-runtime, which may exit 0
    /// without executing the program.
    pub entry_native_compiled: bool,
    /// The names of every source that fell back, entry or not.
    pub fallback_sources: Vec<String>,
    /// WHY they fell back, when the compile step could say.
    ///
    /// `None` is not "no fallback" — `fallback_sources` answers that. It is
    /// "the cause was not classified", which every refusal site must read as
    /// the fail-closed cause (a real failure), never as a host-capability gap.
    pub fallback_reason: Option<FallbackReason>,
}
