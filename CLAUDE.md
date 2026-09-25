# CLAUDE.md

Context for Claude Code (or any future assistant) working in this repo.
Read this before making changes — several parts of `blur_video.py` encode
hard-won fixes for non-obvious ffmpeg behavior, and it's easy to
accidentally regress them while refactoring.

## What this project is

`blur_video.py` is a single-file Python CLI that wraps `ffmpeg` to
obscure (blur/pixelate/median/solid-fill) a rectangular region — or the
whole frame — of one or more MP4 files. It supports time-windowed
effects, soft-edge feathering, a batch parameter-file mode for applying
several passes to every file, a short test-clip preview mode, dry-run,
and per-file error handling that skips problems instead of aborting a
whole batch.

User-facing docs live in `README.md` (quick start) and `blur_video.1`
(the full man page — the authoritative reference for every switch,
the parameter-file format, exit codes, and caveats). Keep both in sync
with the code; the man page in particular should be treated as the
spec, not an afterthought.

## Architecture at a glance

- **CLI**: `argparse`, with a custom `-H`/`-h` help (`add_help=False`)
  and `allow_abbrev=False` (needed so `-D` never gets ambiguously
  matched against `-DX`/`-DY`).
- **Prerequisites**: `ensure_module()` auto-`pip install`s
  `opencv-python` (video probing) and `imageio-ffmpeg` (bundles a
  portable ffmpeg binary) on demand. Dry-run mode (`-N`) never probes a
  video at all, so it doesn't need `opencv-python` — don't add a probe
  call to the dry-run path without checking that constraint still holds.
- **File discovery**: `-F` (glob) / `-R` (recursive walk) / `-C`
  (cwd only), `.mp4` extension only, case-insensitive.
- **Effects**: `effect_params()` computes a scale-independent "intensity"
  (sigma / radius / block size / fill color) from the *original* box
  size; `effect_filter_string()` turns that into a concrete ffmpeg
  filter expression for a *given* crop size — which differs between the
  plain case (the box itself) and the feathered case (the box grown by
  the fade width). Keep this split intact: don't compute filter strings
  directly from the original box size, or feathered passes will use the
  wrong dimensions for pixelation's scale targets and solid fill's
  drawbox size.
- **Region resolution**: `-A` (whole frame) is represented internally as
  `region = None` and substituted to `(0, 0, width, height)` at
  filter-build time in `build_blur_args()`. Every pass — including
  whole-frame — goes through the identical
  crop → effect → [feather blend] → overlay pipeline. There is
  deliberately no separate "-vf" fast path anymore; it was removed
  because relying on every effect filter (`scale`, `boxblur`, `median`)
  supporting ffmpeg's `enable` timeline option directly was an
  unnecessary risk, whereas `overlay`'s `enable` is used everywhere and
  well-tested. Don't reintroduce a fast path without re-verifying
  timeline support per filter.
- **Feathering** (`-D`, or the per-line `D` field under `-I`):
  `attach_feather()` computes a grown crop plus a soft rectangular mask
  (built from `color` + `drawbox` + `boxblur`, blended via
  `maskedmerge`). This is the most fragile part of the codebase — see
  "Known ffmpeg gotchas" below before touching it.
- **Batch mode** (`-I`): `load_param_file()` parses and fully validates
  an 8-field-per-line file (`X;DX;Y;DY;M;D;S;E`) *before* any file is
  opened or any output written, so a malformed line aborts immediately
  with a specific line number rather than failing partway through a
  batch. All passes for one input file are chained into a single
  `filter_complex`/encode, not multiple sequential re-encodes.
- **Test mode** (`-T`): without `-I`, one original + one blurred 5s
  clip. With `-I`, one original + one blurred clip *per line*, each
  anchored at **that line's own `S`** (not a shared start time) so the
  preview shows the actual moment in the video where that pass applies.
  This was a deliberate correction during development — an earlier
  version anchored all previews at a single shared start time, which
  was wrong.
- **ffmpeg execution**: `run_ffmpeg()` streams `-progress pipe:1` for the
  live percent-complete indicator, and `run_ffmpeg_for_output()` wraps
  it to silently delete the output file on any failure or Ctrl-C
  (`except BaseException`, so `KeyboardInterrupt` is caught here too).
  All ffmpeg subprocess calls go through these two functions — don't
  add an ad hoc `subprocess.Popen` call for ffmpeg elsewhere.

## Known ffmpeg gotchas fixed here (do not regress these)

1. **stdout/stderr pipe deadlock.** Never open `stdout=PIPE` and
   `stderr=PIPE` as two separate pipes while only reading one. ffmpeg
   fills the unread pipe's OS buffer and blocks on it, which freezes
   the *entire* process — the output file stops growing and even
   Ctrl-C can't cleanly unstick it, because ffmpeg itself is wedged.
   This exact bug shipped once and was reported as "the script hangs
   and won't respond to Ctrl-C." Fix in place:
   `stderr=subprocess.STDOUT`, a single continuously-drained pipe.

2. **`maskedmerge` + an infinite-duration `color` source hangs forever.**
   The feather mask is built from `color=c=black:s=WxH[...]`. The
   `color` source filter defaults to infinite duration, and
   `maskedmerge` waits for *every* input to finish before it will
   finish itself. Always give the color source an explicit `:d=<secs>`
   (the code uses `max(duration, pass_end, 0.1)`). Confirmed by direct
   testing: without this, ffmpeg runs indefinitely with zero output.

3. **A filter-pad label used as input to two different filters directly
   is rejected** by (at least) ffmpeg 7.0.2 — and the resulting error
   message is misleading (`Invalid stream specifier: <label>`, which
   looks like a `-map` problem but isn't). Always explicit `split` a
   crop's output before feeding it to two downstream filters (used
   where the feathered crop feeds both the effect and the mask-merge).

4. **`boxblur`'s chroma-plane radius limit is roughly half the
   luma-plane limit** (chroma is half-resolution under standard 4:2:0
   subsampling, which is what virtually all real MP4s use). Passing one
   shared radius that's valid for luma can be silently rejected on
   chroma with `Invalid chroma_param radius value N, must be >= 0 and
   <= M`. Always build box blur filter strings through `boxblur_expr()`,
   which computes and clamps `luma_radius` and `chroma_radius`
   independently — never call `boxblur=` directly with one shared value.

5. **ffmpeg's own `-progress` machine-readable output can flood a naive
   "last N lines" error buffer**, burying the real error under a final
   block of `key=N/A` lines (`out_time_us=N/A`, `speed=N/A`, etc.).
   `_is_progress_line()` filters these known keys out of the rolling
   error-tail buffer (`run_ffmpeg`'s `tail` deque) so genuine
   diagnostics survive to be shown to the user. If you add new
   `-progress`-driven parsing, extend `_PROGRESS_KEYS` rather than
   assuming the tail buffer is clean.

## How this was tested during development

There is no automated test suite yet. Verification was done manually:

- `python3 -m py_compile blur_video.py` for a fast syntax check after
  every edit — cheap, do this first, always.
- A small synthetic test video generated with ffmpeg's own `testsrc` +
  `sine` sources (has both video and audio, and produces verbose
  libx264 log output — deliberately, since verbose logging is exactly
  what originally triggered the pipe-deadlock bug above).
- Every manual test run was wrapped in `timeout N` — given the history
  of genuine hangs in this codebase, **never invoke this script during
  development without a timeout guard**, or a regression can stall a
  whole session.
- Each of the five modes (G/P/B/M/S) tested individually, each combined
  with feathering, whole-frame (`-A`), the `-I` batch file (including
  per-line `M`/`D`), and `-T` combined with `-I`.
- Rejection paths tested explicitly: `-D` with `-A`, `-A` with `-I`, and
  an invalid `-M` mode letter.

If you add a real test suite, prioritize (in this order): the
`run_ffmpeg` pipe-handling path (hardest to get right, easiest to
silently break), `boxblur_expr()` at extreme strengths on small regions,
and `load_param_file()`'s line-validation error messages.

## Style / conventions

- Single-file script, stdlib `argparse`, no external CLI framework —
  keep it that way unless there's a strong reason to split it up.
- Per-file errors are caught, logged to `blur_errors.log` (created
  lazily via `LazyFileHandler` — only on the *first* real error, so a
  clean run leaves no log file behind), and the run continues to the
  next file. `KeyboardInterrupt` is explicitly re-raised past this
  per-file `except Exception` so Ctrl-C aborts the whole run rather than
  just skipping the current file.
- Output filenames never overwrite: `unique_output_path()` appends
  `_00`, `_01`, ... until a name is free.

## Open items / possible future work (not started)

- `-R`/`-C` only match `.mp4`; other containers (`.mov`, `.mkv`) aren't
  picked up even though ffmpeg itself could handle them.
- Processing is strictly sequential — no parallelism across files.
- No automated tests (see above).
