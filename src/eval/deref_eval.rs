// Copyright 2025 STARGA Inc.

pub(super) fn unsupported() -> super::EvalError {
    super::EvalError::UnsupportedMsg(
        "dereference `*p` / assignment-through-dereference `*p = v` is not supported: \
         the reference/place ABI is unimplemented"
            .to_string(),
    )
}
