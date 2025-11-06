use crate::ast::Module;

#[derive(Debug, thiserror::Error)]
pub enum TypeError {
    #[error("type checking not implemented (Phase 1 scaffold)")]
    Unimplemented,
}

/// Phase 1: accept everything (stub) so pipeline compiles.
pub fn check(_m: &Module) -> Result<(), TypeError> {
    Ok(())
}
