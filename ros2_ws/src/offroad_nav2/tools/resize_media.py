#!/usr/bin/env python3
"""Fast resize of images and videos to the network input size (default 640x320, aspect 2:1).

A source that is not 2:1 is first centre-cropped to 2:1 (never stretched), then scaled. Uses ffmpeg (one process per file,
files run in parallel), so videos and images share one code path.

  resize_media.py clip.mp4 frame.png                 # -> clip_640x320.mp4, frame_640x320.png next to the inputs
  resize_media.py data/ -o out/ -r                   # whole directory tree, mirrored into out/
  resize_media.py clip.mp4 --width 1280 --height 640 --crf 18
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMAGES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
VIDEOS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def vf(w, h):
    # centre crop to w:h (2:1 by default), then scale; floor to even numbers for yuv420
    return (f"crop='if(gte(iw*{h},ih*{w}),trunc(ih*{w}/{h}/2)*2,iw)':'if(gte(iw*{h},ih*{w}),ih,trunc(iw*{h}/{w}/2)*2)',"
            f"scale={w}:{h}:flags=lanczos")


def convert(src, dst, a):
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y' if a.overwrite else '-n', '-i', str(src), '-vf', vf(a.width, a.height)]
    if src.suffix.lower() in VIDEOS:
        cmd += ['-c:v', 'libx264', '-preset', a.preset, '-crf', str(a.crf), '-pix_fmt', 'yuv420p', '-c:a', 'copy']
    else:
        cmd += ['-frames:v', '1']
        if dst.suffix.lower() in ('.jpg', '.jpeg'):
            cmd += ['-q:v', '2']
    r = subprocess.run(cmd + [str(dst)], capture_output=True, text=True)
    return src, dst, r.returncode, r.stderr.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', type=Path, help='image/video files or directories')
    p.add_argument('-o', '--out', type=Path, help='output directory (default: next to the input, suffix _WxH)')
    p.add_argument('-r', '--recursive', action='store_true', help='descend into directories')
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=320)
    p.add_argument('--crf', type=int, default=20, help='x264 quality for videos (lower = better)')
    p.add_argument('--preset', default='veryfast', help='x264 preset')
    p.add_argument('-j', '--jobs', type=int, default=4, help='files converted in parallel')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()

    jobs = []
    for inp in a.inputs:
        if inp.is_dir():
            files = inp.rglob('*') if a.recursive else inp.glob('*')
            for f in sorted(files):
                if f.suffix.lower() in IMAGES | VIDEOS:
                    rel = f.relative_to(inp)
                    dst = (a.out / rel) if a.out else f.with_name(f'{f.stem}_{a.width}x{a.height}{f.suffix}')
                    jobs.append((f, dst))
        elif inp.suffix.lower() in IMAGES | VIDEOS:
            dst = (a.out / inp.name) if a.out else inp.with_name(f'{inp.stem}_{a.width}x{a.height}{inp.suffix}')
            jobs.append((inp, dst))
        else:
            print(f'skip (not an image/video): {inp}', file=sys.stderr)
    if not jobs:
        sys.exit('nothing to do')
    bad = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for src, dst, rc, err in ex.map(lambda j: convert(*j, a), jobs):
            print(('ok   ' if rc == 0 else 'FAIL ') + f'{src} -> {dst}' + ('' if rc == 0 else f'\n     {err}'))
            bad += rc != 0
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
