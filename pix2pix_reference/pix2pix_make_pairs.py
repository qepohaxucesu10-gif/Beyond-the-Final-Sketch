import argparse
import csv
from pathlib import Path
from PIL import Image

ap = argparse.ArgumentParser(description='Create aligned Face-to-B25 pix2pix pairs from released manifests.')
ap.add_argument('--reference-root', type=Path, required=True, help='Directory containing released manifests and paired_data/.')
args = ap.parse_args()
EXT = args.reference_root
DATA = EXT / 'paired_data'

for split, folder in [('train32', 'train'), ('valid43', 'val')]:
    with (EXT / f'PIX2PIX_{split.upper()}_MANIFEST.csv').open() as f:
        rows = list(csv.DictReader(f))
    out = DATA / folder
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        a = Image.open(r['face_path']).convert('RGB').resize((256,256), Image.Resampling.BICUBIC)
        b = Image.open(r['step_25_path']).convert('RGB').resize((256,256), Image.Resampling.BICUBIC)
        ab = Image.new('RGB',(512,256)); ab.paste(a,(0,0)); ab.paste(b,(256,0)); ab.save(out/(r['sample_uid']+'.png'))
print('PAIRS_READY', len(list((DATA/'train').glob('*.png'))), len(list((DATA/'val').glob('*.png'))))
