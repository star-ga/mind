// Copyright 2025-2026 STARGA Inc.
// Licensed under the Apache License, Version 2.0.
// Part of the MIND project (Machine Intelligence Native Design).

// mind_growth.h — the region frame functions that the growth paths depend on:
// mind_region_retarget (a growth calls it before it frees a buffer) and the
// enter, track and exit functions of a `region { }` block.
// A part of mind_intrinsics.c, not a standalone header: that file includes it
// once, after the region frame stack, and mindc splices it into the same
// single source.
#ifndef MIND_GROWTH_H
#define MIND_GROWTH_H

// mind_region_retarget(old, new) — a growth path replaced buffer `old` by
// `new`. Every live frame entry that tracks `old` now tracks `new`, so the
// region exit frees the new buffer once and never frees the old one again.
// Scans frames in a fixed order; no-op outside a region or when old == new.
static void mind_region_retarget(int64_t old_ptr, int64_t new_ptr) {
    if (old_ptr == 0 || old_ptr == new_ptr) return;
    for (int d = 0; d < mind_region_depth; d++) {
        MindRegionFrame *f = &mind_region_stack[d];
        for (int64_t i = 0; i < f->len; i++) {
            if (f->ptrs[i] == old_ptr) {
                f->ptrs[i] = new_ptr;
            }
        }
    }
}

// __mind_region_enter() — push a new allocation-tracking frame.
//
// Returns 0 (unit i64). Called at the entry of every `region { }` block.
MIND_EXPORT int64_t __mind_region_enter(void) {
    if (mind_region_depth >= MIND_REGION_STACK_DEPTH) {
        /* region nesting overflow — abort deterministically */
        abort();
    }
    MindRegionFrame *f = &mind_region_stack[mind_region_depth];
    f->ptrs = (int64_t *)malloc((size_t)MIND_REGION_FRAME_INIT_CAP * sizeof(int64_t));
    f->len  = 0;
    f->cap  = MIND_REGION_FRAME_INIT_CAP;
    mind_region_depth++;
    return 0;
}

// __mind_region_track(ptr) — record an allocation made inside a region.
//
// `ptr` is the i64 opaque address returned by `__mind_alloc`. Appended to
// the top frame's list. Returns `ptr` unchanged (so the call can be
// composed with the alloc result in a single SSA chain).
//
// No-ops when called outside a region (mind_region_depth == 0) so that
// user code that calls `__mind_alloc` directly from outside any region does
// not fault.
MIND_EXPORT int64_t __mind_region_track(int64_t ptr) {
    if (mind_region_depth == 0 || ptr == 0) return ptr;
    MindRegionFrame *f = &mind_region_stack[mind_region_depth - 1];
    if (f->len >= f->cap) {
        int64_t new_cap = f->cap * 2;
        f->ptrs = (int64_t *)realloc(f->ptrs, (size_t)new_cap * sizeof(int64_t));
        f->cap  = new_cap;
    }
    f->ptrs[f->len++] = ptr;
    return ptr;
}

// __mind_region_exit() — free every allocation in the top frame, then pop.
//
// Returns 0 (unit i64). Called at the exit of every `region { }` block.
// The order of freeing is LIFO (last allocated, first freed) — deterministic
// and consistent with the evidence-chain contract.
MIND_EXPORT int64_t __mind_region_exit(void) {
    if (mind_region_depth == 0) return 0;
    mind_region_depth--;
    MindRegionFrame *f = &mind_region_stack[mind_region_depth];
    /* Free in reverse insertion order — LIFO */
    for (int64_t i = f->len - 1; i >= 0; --i) {
        if (f->ptrs[i] != 0) {
            free((void *)(uintptr_t)f->ptrs[i]);
        }
    }
    free(f->ptrs);
    f->ptrs = NULL;
    f->len  = 0;
    f->cap  = 0;
    return 0;
}

#endif  /* MIND_GROWTH_H */
