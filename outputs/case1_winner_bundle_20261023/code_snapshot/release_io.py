"""Streaming reader that keeps exploratory analysis on train by default."""
from __future__ import annotations
import gzip
import json
from pathlib import Path


def iter_release_metrics(directory, *, split='train', allow_heldout=False):
    """Yield versioned candidate records from a mechanically verified dataset.

    Reading held-out outcomes is a separate deliberate action for future final
    evaluation, not something the current verification/EDA does implicitly.
    """
    root=Path(directory)
    if split not in ('train','validation','test','challenge'):
        raise ValueError('Unknown split')
    if split!='train' and not allow_heldout:
        raise PermissionError('Held-out outcomes are closed for exploratory analysis')
    state=json.loads((root/'generation_state.json').read_text(encoding='utf-8'))
    if state['status']!='verified_complete':
        raise ValueError('Dataset has not completed mechanical verification')
    if state['dataset_version']!='phase4-mathematics-v3':
        raise ValueError('Unexpected dataset version')
    for marker in sorted((root/'partitions'/split).glob('*/done.json')):
        receipt=json.loads(marker.read_text(encoding='utf-8'))
        if receipt['data_split']!=split:raise ValueError('Partition split mismatch')
        path=marker.parent/receipt['attempt']/'trajectory_metrics.jsonl.gz'
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                record=json.loads(line)
                if record['data_split']!=split or record['dataset_version']!=state['dataset_version']:
                    raise ValueError('Record split/version mismatch')
                yield record
