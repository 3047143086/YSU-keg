"""Extract a common frozen evidence bank for all three experimental controls."""

import argparse
import html
import json
import os
import sys
import time
from html.parser import HTMLParser
from pathlib import Path


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, value):
        self.parts.append(value)


def clean_text(value):
    if isinstance(value, list):
        value = ' '.join(str(part) for part in value)
    parser = TextOnly()
    parser.feed(html.unescape(str(value or '')))
    return ' '.join(' '.join(parser.parts).split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / 'vendor'))
    import clip
    import numpy as np
    import torch
    from PIL import Image
    from stage1_core import atomic_json, sha256

    started = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(20260906)
    config = json.loads((root / 'experiment.json').read_text())
    items = [json.loads(line) for line in (root / 'items.jsonl').read_text().splitlines()]
    images = json.loads((root / 'image_manifest.json').read_text())
    model, preprocess = clip.load(str(root / 'weights/ViT-B-16.pt'), device='cuda', jit=False)
    model.eval()
    model.requires_grad_(False)
    n_items, image_views, text_chunks = len(items), config['image_views'], config['text_chunks']
    bank = np.zeros((n_items, image_views + text_chunks, 512), dtype=np.float16)
    mask = np.zeros((n_items, image_views + text_chunks), dtype=np.bool_)
    batch, positions = [], []

    @torch.no_grad()
    def flush_images():
        if not batch:
            return
        features = model.encode_image(torch.stack(batch).cuda()).float()
        features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
        for (item_id, view), vector in zip(positions, features):
            bank[item_id, view] = vector
            mask[item_id, view] = True
        batch.clear()
        positions.clear()

    with torch.no_grad():
        for entry in images:
            if not entry['available']:
                continue
            item_id = entry['item_id']
            with Image.open(entry['path']) as source:
                picture = source.convert('RGB')
                width, height = picture.size
                crop_width, crop_height = max(1, round(width * 2 / 3)), max(1, round(height * 2 / 3))
                views = [picture]
                for x, y in ((0, 0), (width - crop_width, 0), (0, height - crop_height),
                             (width - crop_width, height - crop_height)):
                    views.append(picture.crop((x, y, x + crop_width, y + crop_height)))
                for view, image in enumerate(views):
                    batch.append(preprocess(image))
                    positions.append((item_id, view))
                    if len(batch) >= 64:
                        flush_images()
            if item_id % 500 == 0:
                record = {'stage': 'image_encoding', 'item_id': item_id, 'items': n_items,
                          'elapsed_seconds': time.monotonic() - started}
                atomic_json(root / 'extraction_progress.json', record)
                print(json.dumps(record), flush=True)
        flush_images()

        token_batch, positions = [], []
        tokenizer = clip.clip._tokenizer
        sot, eot = tokenizer.encoder['<|startoftext|>'], tokenizer.encoder['<|endoftext|>']

        def flush_text():
            if not token_batch:
                return
            features = model.encode_text(torch.stack(token_batch).cuda()).float()
            features = torch.nn.functional.normalize(features, dim=-1).cpu().numpy()
            for (item_id, slot), vector in zip(positions, features):
                bank[item_id, slot] = vector
                mask[item_id, slot] = True
            token_batch.clear()
            positions.clear()

        for row in items:
            text = ' '.join(clean_text(row.get(key)) for key in ('title', 'brand', 'description')).strip()
            encoded = tokenizer.encode(text)
            for index in range(text_chunks):
                chunk = encoded[index * 75:(index + 1) * 75]
                if not chunk:
                    break
                tokens = torch.zeros(77, dtype=torch.long)
                tokens[:len(chunk) + 2] = torch.tensor([sot] + chunk + [eot])
                token_batch.append(tokens)
                positions.append((row['item_id'], image_views + index))
                if len(token_batch) >= 128:
                    flush_text()
            if row['item_id'] % 1000 == 0:
                record = {'stage': 'text_encoding', 'item_id': row['item_id'], 'items': n_items,
                          'elapsed_seconds': time.monotonic() - started}
                atomic_json(root / 'extraction_progress.json', record)
                print(json.dumps(record), flush=True)
        flush_text()

    global_vectors = []
    global_masks = []
    for start, stop in ((0, image_views), (image_views, image_views + text_chunks)):
        count = mask[:, start:stop].sum(axis=1)
        pooled = bank[:, start:stop].astype(np.float32).sum(axis=1) / np.maximum(count[:, None], 1)
        pooled /= np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-8)
        global_vectors.append(pooled)
        global_masks.append(count > 0)
    output = root / 'features'
    output.mkdir(exist_ok=False)
    arrays = {'local_bank': bank, 'local_mask': mask, 'image_feat': global_vectors[0],
              'text_feat': global_vectors[1], 'global_bank': np.stack(global_vectors, axis=1),
              'global_mask': np.stack(global_masks, axis=1)}
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise RuntimeError('Nonfinite extracted features: ' + name)
        np.save(output / (name + '.npy'), array)
    manifest = {'encoder': json.loads((root / 'encoder_manifest.json').read_text()),
                'items': n_items, 'image_views': image_views, 'text_chunks': text_chunks,
                'global_pooling': 'normalized mean of exactly the same image views and text chunks used by local',
                'missing_image_items': int((~global_masks[0]).sum()),
                'missing_text_items': int((~global_masks[1]).sum()),
                'sha256': {path.name: sha256(path) for path in output.glob('*.npy')},
                'elapsed_seconds': time.monotonic() - started}
    atomic_json(root / 'feature_manifest.json', manifest)
    print('FEATURES_READY', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
