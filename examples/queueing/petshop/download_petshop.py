"""Download the pinned public PetShop dataset and verify Git blob SHA-1 hashes."""
import argparse, hashlib, json, urllib.request
from pathlib import Path

def blob_sha(data):
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=Path(__file__).with_name('source_manifest.json'))
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    m=json.loads(args.manifest.read_text())
    for f in m['files']:
        p=(args.data_dir/f['path']).resolve()
        if not p.is_relative_to(args.data_dir.resolve()) or p == args.data_dir.resolve():
            raise ValueError('Manifest path escapes data directory')
        if p.exists():
            data=p.read_bytes()
            if len(data)==f['size'] and blob_sha(data)==f['sha']:
                continue
            raise ValueError(f'Existing file differs from pinned source: {p}')
        if args.verify_only:
            raise FileNotFoundError(p)
        url=f"https://raw.githubusercontent.com/{m['repository']}/{m['revision']}/{f['path']}"
        with urllib.request.urlopen(url,timeout=60) as response:
            data=response.read(f['size'] + 1)
        if len(data)!=f['size'] or blob_sha(data)!=f['sha']:
            raise ValueError(f'Download failed integrity check: {url}')
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(data)
    print(f"Verified {len(m['files'])} files at {m['revision']}")
if __name__=='__main__':
    main()
