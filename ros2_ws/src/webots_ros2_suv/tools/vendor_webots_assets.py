#!/usr/bin/env python3
"""Download every Webots asset the world needs and point the world and the PROTOs at the local copies.

Webots fetches the official PROTOs, textures, meshes and sounds from raw.githubusercontent.com (EXTERNPROTO
"https://...", `webots://` paths the local install does not have, default sounds of ContactProperties).
Without internet the world loads with missing objects and textures. This script:

  1. crawls worlds/*.wbt and protos/*.proto for those URLs, then the downloaded PROTOs for theirs (absolute
     URLs, `webots://` paths and paths relative to the PROTO), recursively;
  2. mirrors each file under protos/webots/projects/... (the path below `projects/` in the Webots repository);
  3. rewrites the URLs in the world, the local PROTOs and the downloaded PROTOs into relative paths.

Some PROTOs build file names in their JavaScript templates (TexturedBackground: cubic/<texture>_<side>.jpg);
those are listed in EXTRA. To find what is still fetched from the net, run the sim without network
(`docker run --network none ...`) and look for "Cannot download" / "Unable to find resource" in the Webots
log; add the paths to EXTRA. Controllers of those PROTOs: see CONTROLLERS.

    python3 tools/vendor_webots_assets.py           # download what is missing and rewrite
    python3 tools/vendor_webots_assets.py --check   # only list URLs that still point to the net
"""
import argparse
import os
import re
import sys
import urllib.request

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PKG, 'protos', 'webots')          # mirror root: OUT/projects/...
RAW = 'https://raw.githubusercontent.com/cyberbotics/webots/%s/%s'
INSTALLED = 'R2025a'                                  # webots:// and default sounds come from this release

GITHUB = re.compile(r'https://raw\.githubusercontent\.com/cyberbotics/webots/([^/]+)/(projects/[^"\'\s)\]]+)')
WEBOTS = re.compile(r'webots://(projects/[^"\'\s)\]]*)')
FILE_EXT = r'(?:proto|jpg|jpeg|png|hdr|obj|dae|stl|wav|mp3|ogg|bvh|json)'
RELATIVE = re.compile(r'"([^"\s:]+\.' + FILE_EXT + r')"')

# file names built by JavaScript templates of the PROTOs in this world: (release, path below the repo root)
SIDES = ('back', 'bottom', 'front', 'left', 'right', 'top')
EXTRA = [(INSTALLED, f'projects/default/worlds/textures/cubic/mountains_{s}.{e}')
         for s in SIDES for e in ('jpg', 'hdr')]
EXTRA += [(INSTALLED, f'projects/appearances/protos/textures/leather/{n}.jpg')        # Leather of the steering wheel
          for n in ('dollaro_dark_leather_base_color', 'dollaro_leather_roughness', 'dollaro_leather_normal',
                    'dollaro_leather_occlusion')]
EXTRA += [(INSTALLED, f'projects/appearances/protos/textures/pavement/braun_square_stone_pavement_{n}.jpg')
          for n in ('base_color', 'roughness', 'normal', 'occlusion')]                # Pavement of Road
# Controllers and plugins of downloaded PROTOs. Webots looks for them next to the PROTO (../controllers/<name>,
# ../plugins/...). The compiled ones ship with the Webots install of the image, so the mirror links there
# (the links dangle outside the image).
CONTROLLERS = ['projects/objects/traffic/controllers',        # generic_traffic_light: switches the start light
               'projects/vehicles/plugins']                   # robot window "automobile" of Car (Webots GUI)
WEBOTS_HOME = '/usr/local/webots'
# ContactProperties sounds: Webots' defaults are URLs; the world sets them to these copies
SOUNDS = [(INSTALLED, f'projects/default/worlds/sounds/{n}.wav') for n in ('bump', 'roll', 'slide')]


MANIFEST = os.path.join(OUT, 'SOURCES.txt')     # "<release> <path>" of every downloaded file


def local(path):
    return os.path.join(OUT, path)


def manifest():
    if not os.path.exists(MANIFEST):
        return {}
    return {p: t for t, p in (line.split() for line in open(MANIFEST) if line.strip())}


def save_manifest(sources):
    with open(MANIFEST, 'w') as f:
        f.write(''.join(f'{t} {p}\n' for p, t in sorted(sources.items())))


def download(tag, path, sources):
    tag = sources.get(path, tag)        # keep the release a file was first taken from
    dst = local(path)
    if os.path.exists(dst):
        return dst, False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = RAW % (tag, path)
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
    except Exception as e:      # noqa: BLE001 - report and go on, the offline run shows what is missing
        print(f'  FAILED {url}: {e}')
        return None, False
    with open(dst, 'wb') as f:
        f.write(data)
    sources[path] = tag
    print(f'  {len(data) // 1024:6d} KB  {path}  ({tag})')
    return dst, True


def references(text, tag, path):
    """(release, repo path) of every asset a file refers to. `path` = repo path of the file itself (None for
    files of this package, whose relative paths are local)."""
    out = [(t, p) for t, p in GITHUB.findall(text)]
    out += [(INSTALLED, p) for p in WEBOTS.findall(text) if not p.endswith('/')]
    if path is not None:
        base = os.path.dirname(path)
        for rel in RELATIVE.findall(text):
            p = os.path.normpath(os.path.join(base, rel))
            if p.startswith('projects/'):
                out.append((tag, p))
    return out


def rewrite(text, file_dir):
    """Absolute and webots:// URLs -> paths relative to `file_dir` (in the mirror)."""
    def rel(p):
        r = os.path.relpath(local(p), file_dir)
        return r + '/' if p.endswith('/') and not r.endswith('/') else r
    text = GITHUB.sub(lambda m: rel(m.group(2)), text)
    return WEBOTS.sub(lambda m: rel(m.group(1)), text)


def own_files():
    for d in ('worlds', 'protos'):
        for name in sorted(os.listdir(os.path.join(PKG, d))):
            if name.endswith(('.wbt', '.proto')):
                yield os.path.join(PKG, d, name)


def mirrored_protos():
    for root, _, files in os.walk(OUT):
        for name in files:
            if name.endswith('.proto'):
                yield os.path.join(root, name)


def set_sounds(world_text, world_dir):
    """Give every ContactProperties of WorldInfo the local sounds instead of Webots' default URLs."""
    snd = {n: os.path.relpath(local(p), world_dir) for n, (_, p) in zip(('bump', 'roll', 'slide'), SOUNDS)}

    def fix(m):     # m: one ContactProperties node up to its closing brace at the same indent
        block = m.group(0)
        for n, path in snd.items():
            if f'{n}Sound' not in block:
                block = block.replace('ContactProperties {\n', f'ContactProperties {{\n      {n}Sound "{path}"\n', 1)
        return block
    return re.sub(r'^( *)ContactProperties \{\n.*?^\1\}', fix, world_text, flags=re.S | re.M)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--check', action='store_true', help='only list what still points to the net')
    args = ap.parse_args()

    if args.check:
        bad = 0
        for f in list(own_files()) + list(mirrored_protos()):
            text = open(f, encoding='utf-8').read()
            for m in list(GITHUB.finditer(text)) + list(WEBOTS.finditer(text)):
                print(f'{os.path.relpath(f, PKG)}: {m.group(0)}')
                bad += 1
        print(f'{bad} remote references')
        return 1 if bad else 0

    todo = [(INSTALLED, None, f) for f in own_files()] + [(t, p, None) for t, p in EXTRA + SOUNDS]
    # PROTOs mirrored earlier: their URLs are local by now, scan their relative references again
    sources = manifest()
    todo += [(sources.get(os.path.relpath(f, OUT), INSTALLED), os.path.relpath(f, OUT), f)
             for f in mirrored_protos()]
    seen = set()
    while todo:
        tag, path, f = todo.pop()
        if f is None:
            if path in seen:
                continue
            seen.add(path)
            f, _ = download(tag, path, sources)
            if f is None or not f.endswith('.proto'):
                continue
        text = open(f, encoding='utf-8', errors='replace').read()
        for t, p in references(text, tag, path):
            if p not in seen:
                todo.append((t, p, None))

    save_manifest(sources)
    for path in CONTROLLERS:
        link = local(path)
        if not os.path.lexists(link):
            os.makedirs(os.path.dirname(link), exist_ok=True)
            os.symlink(os.path.join(WEBOTS_HOME, path), link)
            print(f'linked {path} -> {os.path.join(WEBOTS_HOME, path)}')

    for f in list(own_files()) + list(mirrored_protos()):
        text = open(f, encoding='utf-8').read()
        new = rewrite(text, os.path.dirname(f))
        if f.endswith('.wbt'):
            new = set_sounds(new, os.path.dirname(f))
        if new != text:
            with open(f, 'w', encoding='utf-8') as fh:
                fh.write(new)
            print(f'rewrote {os.path.relpath(f, PKG)}')
    size = sum(os.lstat(os.path.join(r, n)).st_size for r, _, fs in os.walk(OUT) for n in fs)
    n = sum(len(fs) for _, _, fs in os.walk(OUT))
    print(f'{n} files, {size / 1e6:.1f} MB in {os.path.relpath(OUT, PKG)}')


if __name__ == '__main__':
    sys.exit(main())
