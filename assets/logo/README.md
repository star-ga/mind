# MIND logo assets

| File | What it is | Format | Use for |
|------|-----------|--------|---------|
| `mind-mark.svg` | Waveform mark only, no wordmark | **True vector** (3 paths) | Anything that scales: title cards, decks, thumbnails, press, favicons |
| `mind-logo.svg` | Full lockup — mark over `MIND` wordmark | **Raster** (499×541 PNG in an SVG wrapper) | Legacy. Prefer `mind-mark.svg` + live text. |

## Colors

| Token | Hex | Where |
|-------|-----|-------|
| Mark | `#4F46E5` | The waveform. Indigo-600. |
| Wordmark | `#293857` | The `MIND` lettering in the lockup. Dark slate. |

⚠ `#293857` is chosen for **light** backgrounds. On a dark/black surface it is
near-invisible — use white or `#4F46E5` for the wordmark instead of the lockup's
baked-in color.

## Why `mind-mark.svg` exists

Both pre-existing assets in the ecosystem — this directory's `mind-logo.svg` and
`mindlang.dev/public/favicon.svg` — are PNGs base64-embedded inside an `<image>`
tag. They carry an `.svg` extension but do not scale: the site favicon is
323×289 and the lockup's payload is 499×541, so both go soft above roughly
their native size. Rendering a 1080p title card from either meant upscaling a
raster.

`mind-mark.svg` is a real vector trace of the mark: 3 `<path>` elements, one
flat `#4F46E5` fill, no embedded bitmap. Verified against the source bitmap at
0.44% ink-pixel deviation (antialiasing on curve edges, not shape error).

## Regenerating

The trace was produced with `vtracer` (binary mode, spline) from the indigo
channel of the lockup's embedded PNG, then recolored to the exact brand hex and
given a `viewBox`. If the mark ever changes, re-trace from the highest-resolution
source available rather than editing the paths by hand — they are machine-generated
and not hand-maintainable.
