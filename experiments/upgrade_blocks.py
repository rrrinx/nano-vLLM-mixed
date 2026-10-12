"""Clone a verified experiment package and apply orchestration updates in a NEW directory."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_files(root, hashes):
    root = root.resolve()
    found = []
    for name, expected in hashes.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name:
            raise ValueError(f'Package entries must be safe relative POSIX paths: {name}')
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError(f'Missing, changed or unsafe package file: {name}')
        found.append((name, path, expected))
    return found


def upgrade(source, destination, update_root):
    source, destination, update_root = source.resolve(), destination.resolve(), update_root.resolve()
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Destination must be a new directory outside the source package')
    parent_path = source / 'PACKAGE_MANIFEST.json'
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    patch = json.loads((update_root / 'UPDATE_MANIFEST.json').read_text(encoding='utf-8'))
    originals = checked_files(source, parent['sha256'])
    changes = checked_files(update_root / 'payload', patch['sha256'])
    # Only immutable allowlisted files are copied: no running results, plans,
    # checkpoints or Python environment are traversed. Keep using the old plan.
    destination.mkdir(parents=True)
    hashes = dict(parent['sha256'])
    for name, path, expected in originals + changes:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if digest(target) != expected:
            raise ValueError(f'Copy verification failed: {name}')
        hashes[name] = expected
    manifest = dict(parent, created=datetime.now().isoformat(timespec='seconds'), sha256=hashes,
                    upgrade={'parent_package': str(source), 'parent_manifest_sha256': digest(parent_path),
                             'update_manifest_sha256': digest(update_root / 'UPDATE_MANIFEST.json'),
                             'feature': 'selected-block-ids-and-audited-result-reuse'})
    (destination / 'PACKAGE_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    checked_files(destination, hashes)
    print('UPDATED PACKAGE:', destination)
    print('Original package, plans, results and active processes were not modified.')
    print('Reuse the original absolute --plan path; --reuse-from must point to a stopped prior stage.')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    args = parser.parse_args()
    upgrade(args.source, args.destination, Path(__file__).resolve().parent)
