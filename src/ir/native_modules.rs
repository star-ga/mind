// Copyright 2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.

// Native delivery modules. Included into the IR root so the declarations
// stay parent-scoped without growing that file.
#[cfg(feature = "std-surface")]
pub mod native_c_abi;
pub mod native_closure;
#[cfg(feature = "std-surface")]
pub mod native_lane_profile;
