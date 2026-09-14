// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0

use crate::runtime_interface::{MindRuntime, NoOpRuntime};

/// Top-level evaluation context used by the compiler front-end.
pub struct Evaluator {
    pub runtime: Box<dyn MindRuntime>,
}

impl Default for Evaluator {
    fn default() -> Self {
        Self {
            runtime: Box::new(NoOpRuntime),
        }
    }
}

impl Evaluator {
    /// Construct an evaluator with the default no-op runtime.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct an evaluator with an explicit runtime implementation.
    pub fn with_runtime(runtime: Box<dyn MindRuntime>) -> Self {
        Self { runtime }
    }
}
