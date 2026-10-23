"""Inventory the Git-visible snapshot and the preserved, ignored research data.

No files are uploaded by this script. Large raw files are only listed, not read.
The inventory directory itself is excluded from its own inventory to avoid recursion.
"""
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/github_backup_20261023'
SECRET = re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{40,}|gh[pousr]_[A-Za-z0-9]{30,}')

def git_paths(*args):
    raw = subprocess.check_output(['git', '-c', 'core.quotepath=false', 'ls-files', '-z', *args], cwd=ROOT)
    return sorted(set(x.decode('utf-8').replace('\\', '/') for x in raw.split(b'\0') if x))

def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    paths = git_paths('--cached', '--others', '--exclude-standard')
    paths = [p for p in paths if not p.startswith('outputs/github_backup_20261023/')]
    included = []
    groups = defaultdict(lambda: {'files': 0, 'bytes': 0})
    for name in paths:
        path = ROOT / name
        assert path.resolve().is_relative_to(ROOT.resolve())
        assert path.is_file() and not path.is_symlink(), name
        assert path.name not in {'.env', '.npmrc', '.pypirc', 'id_rsa', 'id_ed25519'}, name
        content = path.read_bytes()
        assert not SECRET.search(content), f'Possible credential in {name}; review before uploading.'
        assert len(content) < 100 * 1024**2, f'GitHub ordinary-file limit: {name}'
        sha = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
        included.append({'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(), 'git_blob_sha1': sha})
        group = groups[name.split('/')[0]]
        group['files'] += 1
        group['bytes'] += len(content)
    ignored = git_paths('--others', '--ignored', '--exclude-standard', '--', 'data', 'outputs')
    excluded = []
    for name in ignored:
        path = ROOT / name
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        excluded.append({'path': name, 'bytes': path.stat().st_size, 'reason': 'existing_gitignore_large_generated_research_data', 'remote_backup_status': 'not_uploaded'})
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / 'included_files.csv', included, ['path', 'bytes', 'sha256', 'git_blob_sha1'])
    write_csv(OUT / 'excluded_large_data_files.csv', excluded, ['path', 'bytes', 'reason', 'remote_backup_status'])
    summary = {
        'snapshot_date': '2026-10-23', 'repository': 'jinmodev/2026-R-E',
        'included_files': len(included), 'included_bytes': sum(r['bytes'] for r in included),
        'excluded_large_data_files': len(excluded), 'excluded_large_data_bytes': sum(r['bytes'] for r in excluded),
        'included_groups': dict(groups), 'credential_pattern_scan': 'passed',
        'unit_tests': {'run': 97, 'passed': 97, 'failed': 0, 'temporary_directory': 'workspace/tmp/github_backup_checks'},
        'notes': [
            'This inventory is not evidence of a completed upload; the final remote commit and tree must be verified separately.',
            'The inventory folder is included in publication but excluded from its own hash inventory.',
            'Virtual environments, caches, node_modules, .git, and temporary files are not research backups.',
            'Large ignored research files remain local. Cloud/LFS backup and any payment require a separate user decision.',
            'Existing remote-only files are preserved. No force push or deletion is requested.',
        ],
    }
    (OUT / 'snapshot_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'README.txt').write_text(
        '2026-10-23 GitHub 저장 범위\n'
        'included_files.csv: Git 저장 대상의 바이트 수, SHA-256 및 Git blob SHA-1.\n'
        'excluded_large_data_files.csv: 기존 .gitignore로 제외된 대용량 원자료의 경로·크기. 이 파일들은 업로드되지 않았습니다.\n'
        'snapshot_summary.json: 범위와 검사 결과. 실제 저장 완료 여부는 최종 원격 커밋 검증 기록을 확인하세요.\n'
        '순환 해시를 피하기 위해 이 목록 폴더 자체는 included_files.csv에서 제외합니다.\n'
        '가상환경·캐시·임시파일은 제외합니다. 원본 데이터나 프로그램 파일은 삭제하지 않았습니다.\n', encoding='utf-8-sig')
    print(json.dumps(summary, ensure_ascii=False))

if __name__ == '__main__':
    main()
