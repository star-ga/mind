// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

//! Publish bytes to a path, then make that path safe to execute.
//!
//! A concurrent spawn can fork while the writer is open and inherit the
//! writable descriptor. On Linux the later exec fails with "Text file busy"
//! until that inherited descriptor closes. The exclusive lock taken on the
//! writer is held by the inherited descriptor. The read-only lock waits for it.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use fs2::FileExt;

pub(crate) fn write_atomic(dest: &Path, data: &[u8]) -> Result<()> {
    let tmp = dest.with_extension("tmp");
    {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp)
            .with_context(|| format!("write tmp {}", tmp.display()))?;
        file.write_all(data)
            .with_context(|| format!("write tmp {}", tmp.display()))?;
        file.sync_all()
            .with_context(|| format!("sync tmp {}", tmp.display()))?;
        file.lock_exclusive()
            .with_context(|| format!("lock tmp {}", tmp.display()))?;
    }
    fs::rename(&tmp, dest)
        .with_context(|| format!("rename {} -> {}", tmp.display(), dest.display()))?;
    let reader = OpenOptions::new()
        .read(true)
        .open(dest)
        .with_context(|| format!("reopen {}", dest.display()))?;
    reader
        .lock_exclusive()
        .with_context(|| format!("lock {}", dest.display()))?;
    Ok(())
}
