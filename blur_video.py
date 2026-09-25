#!/usr/bin/env python3
"""
blur_video.py — Obscure a rectangular region (or the whole frame) of one or
more MP4 videos, optionally restricted to a time window, using ffmpeg.

Prerequisites are checked and auto-installed via pip when possible:
  - opencv-python   (to read video width/height/fps/duration; not needed in
                      dry-run mode, since dry runs never open/probe a video)
  - imageio-ffmpeg  (bundles a portable ffmpeg binary, so no separate
                      system install of ffmpeg is required)

USAGE EXAMPLES
--------------
Pixelate a fixed region for the whole video:
    python blur_video.py -F movie.mp4 -X 100 -Y 50 -DX 300 -DY 200 -M P 60

Gaussian-blur the whole frame from 10s to 20s, recursively over subfolders:
    python blur_video.py -R -A -S 10 -E 20 -M G 100 -O ./out

Test settings on a 5-second clip before committing to a full run:
    python blur_video.py -F movie.mp4 -X 100 -Y 50 -DX 300 -DY 200 -M B 70 -T

Apply several blur regions to every matched file, using a parameter file:
    python blur_video.py -C -I passes.txt

Soft-edged (feathered) blur, 12px fade band around the box:
    python blur_video.py -F movie.mp4 -X 100 -Y 50 -DX 300 -DY 200 -M G 80 -D 12

SWITCHES
--------
  -F mm         Input filename or wildcard pattern (e.g. "movie*.mp4")
  -R            Recursively process all .mp4 files in cwd and subfolders
  -C            Process all .mp4 files in the current folder
  -O ff         Output folder (default: same folder as each source file)
  -X xx / -Y yy Upper-left corner of the blur region
  -DX xx1/-DY yy1  Width/height of the blur region (clamped to frame edges)
  -A            Blur the entire frame (mutually exclusive with -X/-Y/-DX/-DY, -I)
  -S ss / -E ee Start/end time (seconds) of the blur effect
  -T            Test mode: extract a 5s clip. Without -I: one original clip +
                one blurred clip, starting at -S (or 0). With -I: one
                original clip + one blurred clip PER LINE in the parameter
                file, each starting at THAT LINE's own S value (so each pass
                previews the moment it actually applies to in the real video).
  -M mode gg    Blur effect and strength (gg is 0-100). mode is one of:
                  G  Gaussian blur    gg -> blur radius (100 = fully flattened)
                  P  Pixelate/mosaic  gg -> block coarseness (100 = huge blocks)
                  B  Box blur         gg -> blur radius, blockier than Gaussian
                  M  Median filter    gg -> smoothing radius (capped for speed)
                  S  Solid fill       gg -> fill brightness, 0=black, 100=white
                                      (always fully opaque, not a "strength")
                Default if omitted: G 50. Example: -M P 60
  -I ii         Read blur parameters from text file ii instead of the command
                line. Each line is one blur pass: X;DX;Y;DY;M;D;S;E — all
                passes are applied, in order, to every matched file. When -I
                is used, -X/-DX/-Y/-DY/-M/-D/-S/-E on the command line are
                all ignored; each line supplies its own mode/strength (M,
                same "G70"/"P60"/"S0" form as below) and fade (D).
  -D dd         Fade: grow the blur box by dd pixels on every side, with the
                effect strength ramping linearly from full (at the original
                box edge) down to 0 (at dd pixels out) — a soft edge instead
                of a hard rectangle. Not usable with -A. Ignored when -I is
                given (use each line's own D field instead).
  -N            Dry run: show what would happen, without opening/probing any
                video or writing any files
  -H            Show help and exit (all other switches are ignored)

Exactly one of -F / -R / -C is required. Either -A, or -X and -Y (with
optional -DX/-DY), or -I, must be given.

PARAMETER FILE FORMAT (-I)
---------------------------
Each non-blank, non-'#' line is one blur pass: X;DX;Y;DY;M;D;S;E
  X, Y    upper-left corner (required, whole numbers)
  DX, DY  width/height of the box (blank = extend to the frame edge)
  M       mode letter immediately followed by strength 0-100, no separator,
          e.g. "G70", "P45", "B60", "M20", "S0" (blank = "G50")
  D       fade width in pixels, as -D above (blank or 0 = no fade)
  S, E    start/end time in seconds (blank = 0 / end of video)

Problem files (unreadable, permission-denied, empty, or not actually valid
video despite the .mp4 name/extension) and problem output locations
(folders that can't be created or aren't writable) are skipped with a
message logged to blur_errors.log; the run continues with the remaining
files. blur_errors.log is only created if an error actually occurs.

Press Ctrl-C at any time to stop. The file currently being written is
removed automatically; anything already finished is left in place.
"""

import argparse
import glob
import importlib
import logging
import os
import subprocess
import sys
from collections import deque

VIDEO_EXT = ".mp4"
VALID_MODES = "GPBMS"

# Populated in main() before any per-file work happens.
cv2 = None
imageio_ffmpeg = None


# --------------------------------------------------------------------------
# Prerequisite handling
# --------------------------------------------------------------------------
def ensure_module(module_name, pip_name=None):
    """Import module_name, pip-installing pip_name (or module_name) if missing."""
    pip_name = pip_name or module_name
    try:
        return importlib.import_module(module_name)
    except ImportError:
        print(f"Required module '{module_name}' not found. "
              f"Attempting to install '{pip_name}' via pip...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", pip_name]
            )
        except subprocess.CalledProcessError as exc:
            sys.exit(f"ERROR: failed to install '{pip_name}': {exc}")
        try:
            return importlib.import_module(module_name)
        except ImportError:
            sys.exit(f"ERROR: '{module_name}' still not importable after install.")


# --------------------------------------------------------------------------
# Logging — the error log file is only created the first time an error is
# actually logged (lazy open), so a clean run leaves no log file behind.
# --------------------------------------------------------------------------
class LazyFileHandler(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._fh = None

    def emit(self, record):
        try:
            if self._fh is None:
                self._fh = open(self.filename, "a", encoding="utf-8")
            self._fh.write(self.format(record) + "\n")
            self._fh.flush()
        except OSError:
            pass  # never let logging failures crash the run

    def close(self):
        if self._fh:
            self._fh.close()
        super().close()


def setup_logger():
    logger = logging.getLogger("blur_video")
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    log_path = os.path.join(os.getcwd(), "blur_errors.log")
    fh = LazyFileHandler(log_path)
    fh.setLevel(logging.ERROR)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
    logger.addHandler(fh)
    return logger


# --------------------------------------------------------------------------
# Parameter file (-I) parsing and validation
# --------------------------------------------------------------------------
def parse_mode_field(field_str, lineno):
    """Parse an M field like 'G70', 'P45', 'S0' into (mode, strength).
    Blank means the default, G at strength 50."""
    if not field_str:
        return "G", 50.0
    mode_char = field_str[0].upper()
    if mode_char not in VALID_MODES:
        raise ValueError(
            f"line {lineno}: M field '{field_str}' has unknown mode '{field_str[0]}' "
            f"(must be one of G,P,B,M,S)"
        )
    try:
        value = float(field_str[1:])
    except ValueError:
        raise ValueError(
            f"line {lineno}: M field '{field_str}' must be a mode letter followed by "
            f"a number 0-100, e.g. 'G70'"
        )
    if not (0.0 <= value <= 100.0):
        raise ValueError(f"line {lineno}: M field '{field_str}' value must be 0-100, got {value}")
    return mode_char, value


def load_param_file(path):
    """Parse and validate a -I parameter file.

    Each non-blank, non-'#' line must have exactly 8 semicolon-separated
    fields, in order: X;DX;Y;DY;M;D;S;E
    DX, DY, D, S, E may be left blank (frame edge / no fade / 0 / end of
    video, respectively). Raises ValueError with a human-readable
    explanation on any formatting problem.
    """
    if not os.path.isfile(path):
        raise ValueError(f"parameter file '{path}' does not exist")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw_lines = fh.readlines()
    except OSError as exc:
        raise ValueError(f"cannot read parameter file '{path}': {exc}") from exc

    passes = []
    for lineno, raw in enumerate(raw_lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        fields = [f.strip() for f in line.split(";")]
        if len(fields) != 8:
            raise ValueError(
                f"line {lineno}: expected 8 fields (X;DX;Y;DY;M;D;S;E), "
                f"found {len(fields)}: '{line}'"
            )
        x_s, dx_s, y_s, dy_s, m_s, d_s, s_s, e_s = fields

        try:
            x = int(x_s)
            y = int(y_s)
        except ValueError:
            raise ValueError(f"line {lineno}: X and Y must be whole numbers: '{line}'")
        if x < 0 or y < 0:
            raise ValueError(f"line {lineno}: X and Y must not be negative: '{line}'")

        try:
            dx = int(dx_s) if dx_s else None
            dy = int(dy_s) if dy_s else None
        except ValueError:
            raise ValueError(f"line {lineno}: DX and DY must be whole numbers (or blank): '{line}'")
        if (dx is not None and dx <= 0) or (dy is not None and dy <= 0):
            raise ValueError(f"line {lineno}: DX and DY must be positive: '{line}'")

        mode, strength = parse_mode_field(m_s, lineno)

        try:
            fade = int(d_s) if d_s else 0
        except ValueError:
            raise ValueError(f"line {lineno}: D (fade) must be a whole number (or blank): '{line}'")
        if fade < 0:
            raise ValueError(f"line {lineno}: D (fade) must not be negative: '{line}'")

        try:
            s = float(s_s) if s_s else None
            e = float(e_s) if e_s else None
        except ValueError:
            raise ValueError(f"line {lineno}: S and E must be numbers (or blank): '{line}'")
        if s is not None and e is not None and e <= s:
            raise ValueError(f"line {lineno}: E must be greater than S: '{line}'")

        passes.append({"x": x, "y": y, "dx": dx, "dy": dy,
                        "mode": mode, "strength": strength, "fade": fade,
                        "start": s, "end": e})

    if not passes:
        raise ValueError(f"parameter file '{path}' contains no parameter lines")
    return passes


# --------------------------------------------------------------------------
# File discovery
# --------------------------------------------------------------------------
def discover_files(args):
    files = []
    if args.file_pattern:
        files = [f for f in glob.glob(args.file_pattern) if os.path.isfile(f)]
    elif args.recursive:
        for root, _dirs, fnames in os.walk(os.getcwd()):
            for fname in fnames:
                if fname.lower().endswith(VIDEO_EXT):
                    files.append(os.path.join(root, fname))
    elif args.current_folder:
        files = [f for f in glob.glob(os.path.join(os.getcwd(), f"*{VIDEO_EXT}"))]
    return sorted(files)


def unique_output_path(path):
    """If path exists, append _00, _01, ... before the extension until unique."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 0
    while True:
        candidate = f"{base}_{i:02d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        i += 1


def check_output_dir(directory):
    """Ensure the output directory exists and is genuinely writable."""
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise OSError(f"cannot create output folder '{directory}': {exc}") from exc
    probe = os.path.join(directory, ".blur_video_write_test.tmp")
    try:
        with open(probe, "wb"):
            pass
        os.remove(probe)
    except OSError as exc:
        raise PermissionError(f"output folder '{directory}' is not writable: {exc}") from exc


def make_output_path(input_path, output_dir, suffix="_blurred", create_dir=True):
    directory = output_dir if output_dir else os.path.dirname(input_path) or "."
    if create_dir:
        check_output_dir(directory)
    base = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]
    candidate = os.path.join(directory, f"{base}{suffix}{ext}")
    return unique_output_path(candidate)


# --------------------------------------------------------------------------
# Input file validation (cheap — no video decoding)
# --------------------------------------------------------------------------
def check_input_file(path):
    """Existence / permission / non-empty checks that don't require opening
    or decoding the video. Raises with a clear message on any problem."""
    if not os.path.isfile(path):
        raise FileNotFoundError("input file not found (it may have been moved or deleted)")
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise OSError(f"cannot access input file: {exc}") from exc
    if size == 0:
        raise ValueError("input file is empty (0 bytes)")
    try:
        with open(path, "rb") as fh:
            fh.read(4)
    except PermissionError as exc:
        raise PermissionError("input file is not readable (permission denied)") from exc
    except OSError as exc:
        raise OSError(f"cannot read input file: {exc}") from exc
    return size


# --------------------------------------------------------------------------
# Video inspection (only used outside dry-run)
# --------------------------------------------------------------------------
def get_video_info(path):
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("could not open video (unreadable, corrupt, "
                                "or not actually a video file)")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV could not read this file: {exc}") from exc

    if width <= 0 or height <= 0:
        raise RuntimeError("could not determine video dimensions "
                            "(file may not actually be a video)")
    duration = frame_count / fps if fps > 0 else 0.0
    return width, height, fps, duration


# --------------------------------------------------------------------------
# Region / timing resolution
# --------------------------------------------------------------------------
def clamp_region(x, y, dx, dy, width, height):
    if x > width or y > height:
        raise ValueError(f"region start ({x},{y}) is outside the frame ({width}x{height})")
    if dx is None:
        dx = width - x
    if dy is None:
        dy = height - y
    if x + dx > width:
        dx = width - x
    if y + dy > height:
        dy = height - y
    if dx <= 0 or dy <= 0:
        raise ValueError("blur region has zero or negative area")
    return x, y, dx, dy


def resolve_region(args, width, height):
    """Returns None for whole-frame blur, or a clamped (x, y, dx, dy)."""
    if args.blur_all:
        return None
    if args.x is None or args.y is None:
        raise ValueError("region blur requires -X and -Y (or use -A)")
    return clamp_region(args.x, args.y, args.dx, args.dy, width, height)


def resolve_time_window_values(start, end, duration):
    start = start if start is not None else 0.0
    end = end if end is not None else duration
    start = max(0.0, min(start, duration))
    end = max(0.0, min(end, duration))
    if end <= start:
        raise ValueError(f"invalid time window: start={start}s, end={end}s")
    return start, end


def resolve_time_window(args, duration):
    return resolve_time_window_values(args.start, args.end, duration)


# --------------------------------------------------------------------------
# Blur-effect parameter mapping (0-100 strength -> concrete filter settings)
# --------------------------------------------------------------------------
def boxblur_expr(radius, w, h):
    """Build a boxblur filter expression with luma and chroma radii each
    clamped to what ffmpeg actually allows for that plane's dimensions.
    ffmpeg rejects a radius larger than roughly half the plane's smaller
    dimension, and the chroma plane is conservatively assumed to be half
    resolution (standard 4:2:0 subsampling), so its safe radius is about
    half of luma's — passing one shared radius for both (as if chroma had
    the same limit as luma) can get silently rejected by ffmpeg on chroma
    even though the luma radius alone would have been fine."""
    luma_cap = max(1, min(w, h) // 2)
    chroma_cap = max(1, luma_cap // 2)
    lr = max(1, min(radius, luma_cap))
    cr = max(1, min(radius, chroma_cap))
    return f"boxblur=luma_radius={lr}:luma_power=1:chroma_radius={cr}:chroma_power=1"


def effect_params(mode, strength, region_w, region_h):
    """Compute the scale-independent 'intensity' of an effect, based on the
    ORIGINAL (unfeathered) box's own size. Returns a small dict of filter
    parameters, or None if the effect is a no-op at this strength (every
    mode except S, solid fill, can be a no-op at strength 0)."""
    mode = mode.upper()
    strength = max(0.0, min(100.0, strength))

    if mode == "G":  # Gaussian blur: sigma scales with region size
        if strength <= 0:
            return None
        return {"sigma": min((strength / 100.0) * max(region_w, region_h), 1024.0)}

    if mode == "B":  # Box blur: radius scales with region size
        if strength <= 0:
            return None
        radius = int(round((strength / 100.0) * max(region_w, region_h) / 2))
        return {"radius": max(1, radius)}

    if mode == "M":  # Median filter: radius capped low, this filter is slow
        if strength <= 0:
            return None
        radius = int(round((strength / 100.0) * 15))
        return {"radius": max(1, min(radius, 127))}

    if mode == "P":  # Pixelate: strength -> block size (pixels per block)
        if strength <= 0:
            return None
        block = int(round((strength / 100.0) * (min(region_w, region_h) / 2)))
        return {"block": max(1, block)}

    if mode == "S":  # Solid fill: strength -> fill brightness, never a no-op
        gray = max(0, min(255, int(round(strength / 100.0 * 255))))
        return {"color": f"0x{gray:02x}{gray:02x}{gray:02x}"}

    raise ValueError(f"unknown blur mode '{mode}'")


def effect_filter_string(mode, params, crop_w, crop_h):
    """Build the ffmpeg filter expression to apply to a crop of size
    crop_w x crop_h, given the size-independent params from effect_params().
    Returns None (pass the crop through unchanged) for a strength-0 no-op."""
    if params is None:
        return None
    mode = mode.upper()
    if mode == "G":
        return f"gblur=sigma={params['sigma']:.3f}"
    if mode == "B":
        return boxblur_expr(params["radius"], crop_w, crop_h)
    if mode == "M":
        return f"median=radius={params['radius']}"
    if mode == "P":
        block = params["block"]
        small_w = max(1, round(crop_w / block))
        small_h = max(1, round(crop_h / block))
        return f"scale={small_w}:{small_h}:flags=neighbor,scale={crop_w}:{crop_h}:flags=neighbor"
    if mode == "S":
        return f"drawbox=x=0:y=0:w={crop_w}:h={crop_h}:color={params['color']}:t=fill"
    raise ValueError(f"unknown blur mode '{mode}'")


def resolve_passes(args, blur_passes_from_file, width, height, duration):
    """Build the final list of passes to apply: each is
    {region, mode, effect_params, start, end, fade}."""
    if blur_passes_from_file is not None:
        resolved = []
        for p in blur_passes_from_file:
            region = clamp_region(p["x"], p["y"], p["dx"], p["dy"], width, height)
            start, end = resolve_time_window_values(p["start"], p["end"], duration)
            params = effect_params(p["mode"], p["strength"], region[2], region[3])
            resolved.append({"region": region, "mode": p["mode"], "effect_params": params,
                              "start": start, "end": end, "fade": p["fade"]})
        return resolved

    region = resolve_region(args, width, height)
    region_w, region_h = (width, height) if region is None else (region[2], region[3])
    params = effect_params(args.mode, args.strength, region_w, region_h)
    start, end = resolve_time_window(args, duration)
    return [{"region": region, "mode": args.mode, "effect_params": params,
             "start": start, "end": end, "fade": args.fade}]


def attach_feather(passes, width, height):
    """For every pass with a fade width set, attach a "feather" description:
    a slightly larger crop, plus the geometry of a soft (linearly-ramped)
    mask used to blend the effect back to the original over `fade` pixels.
    Whole-frame passes (region None) have no edge to feather. Mutates
    passes in place."""
    for p in passes:
        fade = p.get("fade") or 0
        if not fade or p["region"] is None:
            continue
        x, y, dx, dy = p["region"]
        cx0 = max(0, x - fade)
        cy0 = max(0, y - fade)
        cx1 = min(width, x + dx + fade)
        cy1 = min(height, y + dy + fade)
        p["feather"] = {
            "cx0": cx0, "cy0": cy0,
            "big_w": cx1 - cx0, "big_h": cy1 - cy0,
            "inner_x": x - cx0, "inner_y": y - cy0,
            "inner_w": dx, "inner_h": dy,
            # boxblur's radius produces an exactly linear ramp ~2x its
            # radius wide, so half the requested fade width gives a ramp
            # that spans (approximately) the requested number of pixels.
            "ramp": max(1, fade // 2),
        }


# --------------------------------------------------------------------------
# ffmpeg command construction
# --------------------------------------------------------------------------
def build_blur_args(passes, duration, width, height):
    """passes: list of {region: (x,y,dx,dy) or None (whole frame), mode,
    effect_params, start, end, feather: optional soft-edge description}.
    duration: length (seconds) the output will run — used only to bound the
    synthetic mask source for feathered passes (see note below).
    Every pass, including whole-frame ones, is built via the same
    crop -> effect -> [feather blend] -> overlay skeleton, chained in one
    filter_complex so only a single encode pass is needed."""
    filters = []
    current = "[0:v]"
    for i, p in enumerate(passes):
        x, y, dx, dy = p["region"] if p["region"] is not None else (0, 0, width, height)
        enable_expr = f"between(t,{p['start']},{p['end']})"
        feather = p.get("feather")

        if not feather:
            effect_str = effect_filter_string(p["mode"], p["effect_params"], dx, dy)
            crop_expr = f"crop={dx}:{dy}:{x}:{y}"
            chain = crop_expr if effect_str is None else f"{crop_expr},{effect_str}"
            blur_label, out_label = f"b{i}", f"v{i}"
            filters.append(f"[0:v]{chain}[{blur_label}]")
            filters.append(f"{current}[{blur_label}]overlay={x}:{y}:enable='{enable_expr}'[{out_label}]")
            current = f"[{out_label}]"
            continue

        # Feathered pass: crop a region grown by the fade width, apply the
        # effect to it, build a soft rectangular mask (solid over the
        # original box, linearly ramping to 0 over the grown border), and
        # blend the effect back onto the original using that mask before
        # overlaying the result onto the main frame.
        #
        # Two things this filter chain must do or ffmpeg will misbehave:
        #  - the cropped region feeds both the effect and the mask-merge,
        #    so it must be explicitly `split` rather than reused directly —
        #    ffmpeg rejects a plain filter-pad label used as input twice.
        #  - the synthetic `color` mask source defaults to infinite
        #    duration, and `maskedmerge` waits for every input to finish —
        #    so the source must be given an explicit, bounded duration or
        #    the encode never terminates.
        f_ = feather
        crop_l = f"bc{i}"
        crop_a, crop_b = f"bc{i}a", f"bc{i}b"
        blur_l = f"bb{i}"
        mbase_l, mrect_l, mask_l, merge_l, out_l = f"mb{i}", f"mr{i}", f"mk{i}", f"mm{i}", f"v{i}"
        mask_duration = max(duration, p["end"], 0.1)

        filters.append(f"[0:v]crop={f_['big_w']}:{f_['big_h']}:{f_['cx0']}:{f_['cy0']}[{crop_l}]")
        filters.append(f"[{crop_l}]split=2[{crop_a}][{crop_b}]")
        effect_str = effect_filter_string(p["mode"], p["effect_params"], f_["big_w"], f_["big_h"])
        if effect_str is None:
            filters.append(f"[{crop_a}]copy[{blur_l}]")
        else:
            filters.append(f"[{crop_a}]{effect_str}[{blur_l}]")
        filters.append(f"color=c=black:s={f_['big_w']}x{f_['big_h']}:d={mask_duration:.3f}[{mbase_l}]")
        filters.append(
            f"[{mbase_l}]drawbox=x={f_['inner_x']}:y={f_['inner_y']}:"
            f"w={f_['inner_w']}:h={f_['inner_h']}:color=white:t=fill[{mrect_l}]"
        )
        filters.append(f"[{mrect_l}]{boxblur_expr(f_['ramp'], f_['big_w'], f_['big_h'])}[{mask_l}]")
        filters.append(f"[{crop_b}][{blur_l}][{mask_l}]maskedmerge[{merge_l}]")
        filters.append(f"{current}[{merge_l}]overlay={f_['cx0']}:{f_['cy0']}:enable='{enable_expr}'[{out_l}]")
        current = f"[{out_l}]"

    filter_complex = ";".join(filters)
    return ["-filter_complex", filter_complex, "-map", current, "-map", "0:a?", "-c:a", "copy"]


def _parse_out_time_seconds(line):
    try:
        h, m, s = line.split("=", 1)[1].strip().split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, IndexError):
        return None


_PROGRESS_KEYS = {
    "frame", "fps", "stream_0_0_q", "bitrate", "total_size",
    "out_time_us", "out_time_ms", "out_time", "dup_frames",
    "drop_frames", "speed", "progress", "stream_1_0_q",
}


def _is_progress_line(stripped):
    key = stripped.split("=", 1)[0]
    return key in _PROGRESS_KEYS


def run_ffmpeg(ffmpeg_exe, cmd_args, total_duration=None):
    """Run ffmpeg, printing a live % complete indicator on one line.
    On Ctrl-C, terminates the ffmpeg child process before re-raising.

    stderr is merged into the same pipe as stdout (rather than opened as a
    second, separate pipe) and that single pipe is continuously drained here.
    Two separate pipes where only one is read is a classic deadlock: once
    ffmpeg fills the unread pipe's OS buffer, it blocks trying to write to
    it, which stalls the whole process (encoding appears to "hang", the
    output file stops growing, and even Ctrl-C can't cleanly unstick it).
    """
    cmd = [ffmpeg_exe, "-y", "-progress", "pipe:1", "-nostats"] + cmd_args
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    last_pct = -1
    interrupted = False
    # Last few lines of ffmpeg's actual log output (banner, warnings, errors),
    # for error messages — the repetitive machine-readable -progress
    # key=value lines are excluded so a real error isn't pushed out by them.
    tail = deque(maxlen=8)
    try:
        for line in process.stdout:
            line = line.rstrip("\n")
            stripped = line.strip()
            if not _is_progress_line(stripped):
                tail.append(line)
            if total_duration and stripped.startswith("out_time="):
                seconds = _parse_out_time_seconds(stripped)
                if seconds is not None:
                    pct = max(0.0, min(100.0, (seconds / total_duration) * 100))
                    if int(pct) != last_pct:
                        last_pct = int(pct)
                        sys.stdout.write(f"\r  {pct:5.1f}% complete")
                        sys.stdout.flush()
            elif stripped == "progress=end":
                sys.stdout.write("\r  100.0% complete\n")
                sys.stdout.flush()
    except KeyboardInterrupt:
        interrupted = True
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        if not interrupted and process.poll() is None:
            process.wait()

    if process.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + "\n".join(tail))


def run_ffmpeg_for_output(ffmpeg_exe, cmd_args, output_path, total_duration=None):
    """Like run_ffmpeg, but silently deletes output_path if anything goes
    wrong (a failed encode, or a Ctrl-C interrupt) so no partial file is left."""
    try:
        run_ffmpeg(ffmpeg_exe, cmd_args, total_duration=total_duration)
    except BaseException:
        try:
            if output_path and os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            pass
        raise


# --------------------------------------------------------------------------
# Dry-run preview (no probing, no file I/O beyond a harmless existence check)
# --------------------------------------------------------------------------
def format_size(dx, dy):
    return f"{dx}x{dy}" if (dx is not None and dy is not None) else "to frame edge"


def format_window(start, end):
    s = f"{start:g}" if start is not None else "0"
    e = f"{end:g}" if end is not None else "end"
    return f"{s}s to {e}"


def format_mode(mode, strength):
    return f"{mode.upper()} {strength:g}"


def dry_run_preview(path, args, blur_passes_from_file, logger):
    logger.info(f"[DRY RUN] {path}")

    if args.test:
        orig_path = make_output_path(path, args.output_dir, suffix="_test-orig", create_dir=False)
        logger.info(f"  would extract a ~5s test clip -> {orig_path}")
        if blur_passes_from_file is not None:
            for i, p in enumerate(blur_passes_from_file, 1):
                suffix = f"_test_blurred_{i:02d}"
                blurred_path = make_output_path(path, args.output_dir, suffix=suffix, create_dir=False)
                fade_desc = f", fade {p['fade']}px" if p["fade"] else ""
                logger.info(
                    f"  would extract a ~5s blurred test clip for pass {i}: "
                    f"region ({p['x']},{p['y']}) size {format_size(p['dx'], p['dy'])}, "
                    f"mode {format_mode(p['mode'], p['strength'])}{fade_desc} -> {blurred_path}"
                )
        else:
            blurred_path = make_output_path(path, args.output_dir, suffix="_test_blurred", create_dir=False)
            logger.info(f"  would extract a ~5s blurred test clip -> {blurred_path}")
        return

    output_path = make_output_path(path, args.output_dir, create_dir=False)
    if blur_passes_from_file is not None:
        logger.info(f"  {len(blur_passes_from_file)} blur pass(es) from '{args.param_file}':")
        for i, p in enumerate(blur_passes_from_file, 1):
            fade_desc = f", fade {p['fade']}px" if p["fade"] else ""
            logger.info(
                f"    pass {i}: region ({p['x']},{p['y']}) size {format_size(p['dx'], p['dy'])}, "
                f"mode {format_mode(p['mode'], p['strength'])}{fade_desc}, "
                f"window {format_window(p['start'], p['end'])}"
            )
    else:
        fade_desc = f", fade {args.fade}px" if args.fade else ""
        if args.blur_all:
            region_desc = "entire frame"
        else:
            region_desc = f"region at ({args.x},{args.y}), size {format_size(args.dx, args.dy)}"
        logger.info(f"  blur {region_desc}, mode {format_mode(args.mode, args.strength)}{fade_desc}, "
                    f"window {format_window(args.start, args.end)}")
    logger.info(f"  -> {output_path}")
    logger.info("  (region/time bounds are not checked against the video in dry-run mode)")


# --------------------------------------------------------------------------
# Per-file processing
# --------------------------------------------------------------------------
def process_file(path, args, ffmpeg_exe, blur_passes_from_file, logger):
    check_input_file(path)

    if args.dry_run:
        dry_run_preview(path, args, blur_passes_from_file, logger)
        return

    width, height, _fps, duration = get_video_info(path)
    passes = resolve_passes(args, blur_passes_from_file, width, height, duration)
    attach_feather(passes, width, height)

    if args.test:
        process_test_clip(path, args, ffmpeg_exe, duration, passes, width, height,
                           blur_passes_from_file is not None, logger)
        return

    blur_args = build_blur_args(passes, duration, width, height)
    output_path = make_output_path(path, args.output_dir, create_dir=True)

    print(f"Loading {os.path.basename(path)} for blurring")
    logger.info(f"  -> {output_path}")
    run_ffmpeg_for_output(
        ffmpeg_exe,
        ["-i", path] + blur_args + ["-c:v", "libx264", "-preset", "medium", "-crf", "20", output_path],
        output_path,
        total_duration=duration,
    )


def process_test_clip(path, args, ffmpeg_exe, duration, passes, width, height, per_pass, logger):
    clip_len = min(5.0, duration)
    print(f"Loading {os.path.basename(path)} for blurring")

    if per_pass:
        # One original + one blurred 5s clip per line of the parameter file.
        # Each pair starts at that line's own S (start) time, since that's
        # when the pass is meant to take effect in the real video.
        for i, p in enumerate(passes, 1):
            test_start = p["start"] if p["start"] is not None else 0.0
            test_start = max(0.0, min(test_start, max(0.0, duration - clip_len)))

            orig_path = make_output_path(path, args.output_dir, suffix=f"_test-orig_{i:02d}", create_dir=True)
            logger.info(f"  -> {orig_path}  (pass {i} of {len(passes)}, original)")
            run_ffmpeg_for_output(ffmpeg_exe, [
                "-ss", f"{test_start:.3f}", "-i", path, "-t", f"{clip_len:.3f}",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-c:a", "aac", orig_path,
            ], orig_path, total_duration=clip_len)

            blurred_path = make_output_path(path, args.output_dir, suffix=f"_test_blurred_{i:02d}", create_dir=True)
            single_pass = dict(p)
            single_pass["start"], single_pass["end"] = 0.0, clip_len
            blur_args = build_blur_args([single_pass], clip_len, width, height)
            logger.info(f"  -> {blurred_path}  (pass {i} of {len(passes)}, blurred)")
            run_ffmpeg_for_output(ffmpeg_exe, [
                "-ss", f"{test_start:.3f}", "-i", path, "-t", f"{clip_len:.3f}",
            ] + blur_args + ["-c:v", "libx264", "-preset", "medium", "-crf", "18", blurred_path],
                blurred_path, total_duration=clip_len)
        return

    # Single combined blurred clip (whole-frame, or one CLI-specified region).
    test_start = args.start if args.start is not None else 0.0
    test_start = max(0.0, min(test_start, max(0.0, duration - clip_len)))

    orig_path = make_output_path(path, args.output_dir, suffix="_test-orig", create_dir=True)
    logger.info(f"  -> {orig_path}")
    run_ffmpeg_for_output(ffmpeg_exe, [
        "-ss", f"{test_start:.3f}", "-i", path, "-t", f"{clip_len:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", orig_path,
    ], orig_path, total_duration=clip_len)

    blurred_path = make_output_path(path, args.output_dir, suffix="_test_blurred", create_dir=True)
    test_passes = []
    for p in passes:
        sp = dict(p)
        sp["start"], sp["end"] = 0.0, clip_len
        test_passes.append(sp)
    blur_args = build_blur_args(test_passes, clip_len, width, height)
    logger.info(f"  -> {blurred_path}")
    run_ffmpeg_for_output(ffmpeg_exe, [
        "-ss", f"{test_start:.3f}", "-i", path, "-t", f"{clip_len:.3f}",
    ] + blur_args + ["-c:v", "libx264", "-preset", "medium", "-crf", "18", blurred_path],
        blurred_path, total_duration=clip_len)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Obscure a region (or the whole frame) of one or more MP4 videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,       # replaced below by -H (and -h as an alias)
        allow_abbrev=False,   # avoid e.g. "-D" ambiguously matching -DX/-DY
    )
    parser.add_argument(
        "-H", "-h", "--help", action="help", default=argparse.SUPPRESS,
        help="Show this help message and exit (all other switches are ignored)",
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("-F", dest="file_pattern", metavar="mm",
                      help='Input filename or wildcard pattern, e.g. "movie*.mp4"')
    src.add_argument("-R", dest="recursive", action="store_true",
                      help="Recursively process all .mp4 files in this folder and subfolders")
    src.add_argument("-C", dest="current_folder", action="store_true",
                      help="Process all .mp4 files in the current folder")

    parser.add_argument("-O", dest="output_dir", metavar="ff",
                         help="Output folder (default: same folder as source file)")

    parser.add_argument("-A", dest="blur_all", action="store_true",
                         help="Blur the entire frame")
    parser.add_argument("-X", dest="x", type=int, metavar="xx",
                         help="X of upper-left corner of blur region")
    parser.add_argument("-Y", dest="y", type=int, metavar="yy",
                         help="Y of upper-left corner of blur region")
    parser.add_argument("-DX", dest="dx", type=int, metavar="xx1",
                         help="Width of blur region")
    parser.add_argument("-DY", dest="dy", type=int, metavar="yy1",
                         help="Height of blur region")

    parser.add_argument("-S", dest="start", type=float, metavar="ss",
                         help="Start time (s) of blurring")
    parser.add_argument("-E", dest="end", type=float, metavar="ee",
                         help="End time (s) of blurring")

    parser.add_argument("-T", dest="test", action="store_true",
                         help="Test mode: extract 5s clip(s) (original + blurred)")
    parser.add_argument("-M", dest="mode_value", nargs=2, metavar=("mode", "gg"),
                         help="Blur mode + strength 0-100: G gaussian, P pixelate, "
                              "B box blur, M median, S solid fill. E.g. '-M P 60'. "
                              "Default: G 50")
    parser.add_argument("-I", dest="param_file", metavar="ii",
                         help="Read blur parameters from file ii (X;DX;Y;DY;M;D;S;E per line); "
                              "overrides -X/-DX/-Y/-DY/-M/-D/-S/-E")
    parser.add_argument("-D", dest="fade", type=int, default=0, metavar="dd",
                         help="Fade: grow the blur box by dd pixels on each side, linearly "
                              "ramping the effect strength from full at the box edge to 0 at "
                              "the outer edge (soft edge instead of a hard rectangle). "
                              "Not usable with -A. Ignored when -I is given.")
    parser.add_argument("-N", dest="dry_run", action="store_true",
                         help="Dry run: show what would happen, without opening/probing any "
                              "video or writing any files")

    args = parser.parse_args()

    if args.mode_value is not None:
        mode_char = args.mode_value[0].upper()
        if mode_char not in VALID_MODES:
            parser.error(f"-M mode must be one of G,P,B,M,S (got '{args.mode_value[0]}')")
        try:
            strength = float(args.mode_value[1])
        except ValueError:
            parser.error(f"-M strength must be a number 0-100 (got '{args.mode_value[1]}')")
        if not (0.0 <= strength <= 100.0):
            parser.error("-M strength must be between 0 and 100")
        args.mode, args.strength = mode_char, strength
    else:
        args.mode, args.strength = "G", 50.0

    if args.fade < 0:
        parser.error("-D must not be negative")
    if args.fade and args.blur_all:
        parser.error("-D cannot be combined with -A (there is no edge to feather on a "
                      "whole-frame blur)")

    if args.param_file:
        if args.blur_all:
            parser.error("-A cannot be combined with -I")
        # -X/-Y/-DX/-DY/-M/-D/-S/-E are simply ignored when -I is given.
    else:
        if args.blur_all and any(v is not None for v in (args.x, args.y, args.dx, args.dy)):
            parser.error("-A cannot be combined with -X/-Y/-DX/-DY")
        if not args.blur_all and (args.x is None or args.y is None):
            parser.error("-X and -Y are required unless -A or -I is given")

    return args


def run(args):
    blur_passes_from_file = None
    if args.param_file:
        try:
            blur_passes_from_file = load_param_file(args.param_file)
        except ValueError as exc:
            sys.exit(f"ERROR: invalid parameter file '{args.param_file}': {exc}")
        print(f"Loaded {len(blur_passes_from_file)} blur pass(es) from '{args.param_file}'.")
        cli_overrides_given = (
            any(v is not None for v in (args.x, args.y, args.dx, args.dy, args.start, args.end))
            or args.mode_value is not None or args.fade
        )
        if cli_overrides_given:
            print("Note: -I was given, so -X/-Y/-DX/-DY/-M/-D/-S/-E on the command line are ignored.")

    logger = setup_logger()

    global cv2, imageio_ffmpeg
    imageio_ffmpeg = ensure_module("imageio_ffmpeg")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not args.dry_run:
        cv2 = ensure_module("cv2", "opencv-python")

    files = discover_files(args)
    if not files:
        logger.error("No matching .mp4 files found.")
        sys.exit(1)

    if args.dry_run:
        logger.info(f"[DRY RUN] Found {len(files)} file(s); no files will be opened or changed.")
    else:
        logger.info(f"Found {len(files)} file(s) to process.")

    succeeded, failed = 0, 0
    for path in files:
        try:
            process_file(path, args, ffmpeg_exe, blur_passes_from_file, logger)
            succeeded += 1
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - intentional: skip & log, keep going
            logger.error(f"FAILED: {path}: {exc}")
            failed += 1
            continue

    logger.info(f"{succeeded} succeeded, {failed} failed."
                f"{' See blur_errors.log for details.' if failed else ''}")
    print("Done")


def main():
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nInterrupted. Any file that was still being written has been removed.")
        sys.exit(130)


if __name__ == "__main__":
    main()
