// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Part of the MIND project (Machine Intelligence Native Design).

//! Membership set for the top-level SSA ids `verify_module` has seen defined.

use std::collections::BTreeSet;

use crate::ir::ValueId;

/// Ids below this bound live in inline words, so the common module (a few dozen
/// top-level ids) verifies without touching the heap.
const INLINE_IDS: usize = 256;

/// Ids at or past this bound spill to an ordered set. Lowering mints ids
/// densely from 0, so real modules stay in the bitmap; a crafted mic@3 id
/// cannot size it (the bitmap tops out at 128 KiB).
const DENSE_LIMIT: usize = 1 << 20;

/// Set semantics identical to `BTreeSet<ValueId>` for `insert` / `contains`,
/// without a tree node per definition: verification runs twice per compile,
/// and the tree's allocation and teardown were most of its cost.
pub(super) struct ValueIdSet {
    inline: [u64; INLINE_IDS / 64],
    words: Vec<u64>,
    /// Indices of the `words` that hold at least one id, in first-set order, so
    /// an ordered copy costs O(ids) — not O(bitmap) for one id near the limit.
    #[cfg(any(test, feature = "std-surface"))]
    occupied: Vec<usize>,
    spill: BTreeSet<ValueId>,
}

impl ValueIdSet {
    /// An empty set pre-sized for ids below `bound` (the module's `next_id`).
    pub(super) fn with_bound(bound: usize) -> Self {
        let heap_ids = bound.min(DENSE_LIMIT).saturating_sub(INLINE_IDS);
        Self {
            inline: [0; INLINE_IDS / 64],
            words: Vec::with_capacity(heap_ids.div_ceil(64)),
            #[cfg(any(test, feature = "std-surface"))]
            occupied: Vec::new(),
            spill: BTreeSet::new(),
        }
    }

    pub(super) fn contains(&self, id: ValueId) -> bool {
        let index = id.0;
        if index < INLINE_IDS {
            return self.inline[index / 64] & (1u64 << (index % 64)) != 0;
        }
        if index >= DENSE_LIMIT {
            return self.spill.contains(&id);
        }
        self.words
            .get((index - INLINE_IDS) / 64)
            .is_some_and(|word| word & (1u64 << (index % 64)) != 0)
    }

    /// Adds `id`; returns whether it was absent (the `BTreeSet::insert` contract).
    pub(super) fn insert(&mut self, id: ValueId) -> bool {
        let index = id.0;
        if index >= DENSE_LIMIT {
            return self.spill.insert(id);
        }
        // INLINE_IDS is a multiple of 64, so the bit position is the same in
        // either half.
        let bit = 1u64 << (index % 64);
        let word = if index < INLINE_IDS {
            &mut self.inline[index / 64]
        } else {
            let slot = (index - INLINE_IDS) / 64;
            if slot >= self.words.len() {
                self.words.resize(slot + 1, 0);
            }
            #[cfg(any(test, feature = "std-surface"))]
            if self.words[slot] == 0 {
                self.occupied.push(slot);
            }
            &mut self.words[slot]
        };
        let absent = *word & bit == 0;
        *word |= bit;
        absent
    }

    /// The ids as an ordered set: the seed the nested-region validators thread
    /// through a top-level `While` / `If` / `Region` (a copy, as before).
    #[cfg(any(test, feature = "std-surface"))]
    pub(super) fn to_btree_set(&self) -> BTreeSet<ValueId> {
        let mut ids = Vec::new();
        // Inline word `i` holds ids from `64 * i`; heap word `slot` continues
        // after the inline ones.
        let inline = self.inline.iter().copied().enumerate();
        let heap = self
            .occupied
            .iter()
            .map(|&slot| (INLINE_IDS / 64 + slot, self.words[slot]));
        for (index, word) in inline.chain(heap) {
            let mut bits = word;
            while bits != 0 {
                ids.push(ValueId(index * 64 + bits.trailing_zeros() as usize));
                bits &= bits - 1;
            }
        }
        ids.extend(self.spill.iter().copied());
        ids.into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_btreeset_across_the_dense_and_spill_ranges() {
        let ids = [
            0,
            63,
            64,
            5,
            63,
            INLINE_IDS - 1,
            INLINE_IDS,
            INLINE_IDS + 70,
            INLINE_IDS,
            DENSE_LIMIT - 1,
            DENSE_LIMIT,
            DENSE_LIMIT + 7,
            DENSE_LIMIT,
            usize::MAX,
            0,
        ];
        let mut set = ValueIdSet::with_bound(16);
        let mut reference = BTreeSet::new();
        for id in ids {
            assert_eq!(
                set.insert(ValueId(id)),
                reference.insert(ValueId(id)),
                "{id}"
            );
        }
        assert_eq!(set.to_btree_set(), reference);
        for probe in [
            0,
            1,
            5,
            63,
            64,
            65,
            INLINE_IDS - 1,
            INLINE_IDS,
            INLINE_IDS + 69,
        ]
        .into_iter()
        .chain([
            INLINE_IDS + 70,
            4096,
            DENSE_LIMIT - 1,
            DENSE_LIMIT,
            DENSE_LIMIT + 6,
        ])
        .chain([DENSE_LIMIT + 7, usize::MAX - 1, usize::MAX])
        {
            assert_eq!(
                set.contains(ValueId(probe)),
                reference.contains(&ValueId(probe)),
                "{probe}"
            );
        }
    }

    #[test]
    fn a_small_module_stays_off_the_heap() {
        let mut set = ValueIdSet::with_bound(INLINE_IDS);
        for id in 0..INLINE_IDS {
            assert!(set.insert(ValueId(id)));
        }
        assert_eq!(set.words.capacity(), 0);
    }

    #[test]
    fn an_ordered_copy_scales_with_ids_not_with_the_bitmap() {
        let mut set = ValueIdSet::with_bound(0);
        set.insert(ValueId(DENSE_LIMIT - 1));
        set.insert(ValueId(INLINE_IDS + 3));
        assert_eq!(set.occupied.len(), 2);
        let expected = BTreeSet::from([ValueId(INLINE_IDS + 3), ValueId(DENSE_LIMIT - 1)]);
        assert_eq!(set.to_btree_set(), expected);
    }

    #[test]
    fn a_hostile_bound_does_not_size_the_bitmap() {
        let set = ValueIdSet::with_bound(usize::MAX);
        assert!(set.words.capacity() <= (DENSE_LIMIT - INLINE_IDS) / 64);
    }
}
