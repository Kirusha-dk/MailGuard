"""Verify downloaded experimental v86 state without training or inference."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import torch
from safetensors.torch import load

from v86_month import CAMPAIGN, PARENT_SHA, PARENT_STEP
from v86_quality_guard import experiment_tag


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(package, state, backbone, root, asset_digest=None, release=None):
    raw = package.read_bytes()
    current = json.loads((state/'report.json').read_text())
    if release is not None:
        if (release['draft'] or not release['prerelease'] or
                release['tag_name'] != experiment_tag(current)):
            raise ValueError('Expected published experimental prerelease missing')
        assets = [a for a in release['assets'] if a['name'] == 'v86-experiment.zip']
        if len(assets) != 1 or not assets[0].get('digest'):
            raise ValueError('Expected release asset digest missing')
        asset_digest = assets[0]['digest']
    if asset_digest is not None and asset_digest != 'sha256:' + digest(raw):
        raise ValueError('Downloaded release asset digest mismatch')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        state_files = {'checkpoint.pt', 'report.json', 'select-scores.npz',
                       'select-scores-metadata.json', 'migration.json',
                       'best-adapter.safetensors', 'best-development.json'}
        source_files = {str(p.relative_to(root)) for p in (root/'research/v86').glob('*.json')}
        docs = {'docs/v86-english-month.md', 'docs/v84-source-attribution.md'}
        required = {'v86-state/' + n for n in state_files} | source_files | docs | {
            '.cache/v82/backbone/manifest.json'}
        if len(names) != len(set(names)) or set(names) != required:
            raise ValueError('Missing, duplicate or unexpected package members')
        checkpoint = z.read('v86-state/checkpoint.pt')
        if digest(checkpoint) != digest((state/'checkpoint.pt').read_bytes()):
            raise ValueError('Release does not preserve the expected own checkpoint')
        c = torch.load(io.BytesIO(checkpoint), map_location='cpu', weights_only=True)
        p = c['progress']
        if not {'order', 'cursor', 'exposures', 'active_seen', 'hard_indices'} <= p.keys():
            raise ValueError('Incomplete order, cursor or exposure state')
        report = json.loads(z.read('v86-state/report.json'))
        frozen = json.loads((root/'research/v86/frozen.json').read_text())
        if (c['metadata']['campaign'] != CAMPAIGN or
                c['metadata']['parent_checkpoint_sha256'] != PARENT_SHA or
                c['metadata']['corpus'] != frozen or
                json.loads(z.read('research/v86/frozen.json')) != frozen):
            raise ValueError('Campaign, immutable parent or frozen source drift')
        if (not report['session_complete'] or report['session'] != current['session'] or
                report['session'] not in p['completed_sessions'] or p['active_session'] is not None or
                report['global_optimizer_steps'] != p['step'] or
                current['global_optimizer_steps'] != p['step']):
            raise ValueError('Incomplete or different archived session')
        if (len(c['optimizer']['state']) != 26 or
                any(int(v['step']) != p['step'] for v in c['optimizer']['state'].values()) or
                c['scheduler']['last_epoch'] != p['step'] - PARENT_STEP or
                c['rng'].numel() == 0 or len(p['exposures']) != frozen['train_rows'] or
                int(p['exposures'].max()) > 3 or 'hard_indices' not in p):
            raise ValueError('Incomplete training state or invalid exposure ledger')
        for name in ('best-adapter.safetensors', 'best-development.json', 'migration.json',
                     'select-scores.npz', 'select-scores-metadata.json'):
            if z.read('v86-state/' + name) != (state/name).read_bytes():
                raise ValueError('Archived best adapter, selection or migration differs: ' + name)
        best = load(z.read('v86-state/best-adapter.safetensors'))
        if sum(n.endswith(('.a', '.b')) for n in best) != 24:
            raise ValueError('Incomplete best LoRA adapter')
        bb = json.loads(z.read('.cache/v82/backbone/manifest.json'))
        if bb != c['metadata']['backbone']:
            raise ValueError('Archived pinned backbone manifest drift')
        for name, expected in bb['files'].items():
            if digest((backbone/name).read_bytes()) != expected:
                raise ValueError('Pinned backbone file drift: ' + name)
        for name in docs:
            if z.read(name) != (root/name).read_bytes():
                raise ValueError('Source attribution or conditions differ: ' + name)
        return dict(campaign=CAMPAIGN, session=report['session'],
                    release_url=release.get('html_url') if release else None,
                    published_experimental_prerelease_verified=release is not None,
                    global_optimizer_steps=p['step'], phase_optimizer_steps=p['step']-PARENT_STEP,
                    inherited_parent_steps=PARENT_STEP, completed_dates=len(p['completed_sessions']),
                    downloaded_asset_sha256=digest(raw), checkpoint_sha256=digest(checkpoint),
                    best_adapter_sha256=digest(z.read('v86-state/best-adapter.safetensors')),
                    downloaded_release_checked=True, own_checkpoint_matches=True,
                    optimizer_scheduler_rng_order_cursor_exposures_hard_indices_present=True,
                    best_adapter_and_metadata_preserved=True, source_attribution_and_conditions_preserved=True,
                    frozen_manifest_unchanged=True, pinned_backbone_files_verified=True,
                    raw_mail_included=False, verification_optimizer_steps=0,
                    verification_inference_rows=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--state', type=Path, default=Path('v86-state'))
    p.add_argument('--backbone', type=Path, default=Path('.cache/v82/backbone'))
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--asset-digest')
    p.add_argument('--release-metadata', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('v86-archive-verification.json'))
    args = p.parse_args()
    result = verify(args.package, args.state, args.backbone, args.root, args.asset_digest,
                    json.loads(args.release_metadata.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
