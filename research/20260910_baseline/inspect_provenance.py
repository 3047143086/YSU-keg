"""Read only: locate the two historical BRIDGE runs in bounded log folders.

Run this script on server 213. It does not evaluate a model, alter files, or
print test metric values. A log match is evidence to inspect, not proof that
the present checkpoint bytes were produced by that log.
"""

import hashlib
import json
import os
import re
from pathlib import Path


ROOT = Path('/data/sunzixuan/bridge_official')
RUNS = ('bridge_baby_seed2026_screen', 'bridge_sports_s2026_confirm')
RELATIVE_DIRS = (
    'experiments_cbcr', 'experiments_c2r', 'experiments_c2r_confirm',
    'src/log', 'src/logs', 'log', 'logs',
)
MAX_ENTRIES = 20000
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_MATCHES = 40
TEXT_SUFFIXES = {'.log', '.out', '.txt', '.json', '.jsonl', '.yaml', '.yml',
                 '.sh', '.py', '.csv', ''}
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
CONFIG = re.compile(
    r'\b(?:model|dataset|seed|log_file_name|valid_metric|epochs|stopping_step|'
    r'learning_rate|hyper_parameters|embedding_size|feat_embed_dim|'
    r'num_layers|freq_decomp_method|behavior_topk|behavior_weight|'
    r'behavior_eval_topk|train_candidate_topk|use_activity_simplex|'
    r'msr_weight|vision_feature_file|text_feature_file|data_path)\s*[:=]',
    re.IGNORECASE,
)
CONFIG_RUN = re.compile(r'={3,}\s*(\d+)\s*/\s*(\d+)\s*:\s*Parameters', re.IGNORECASE)
BEST = re.compile(r'best valid(?:ation)? (?:result|score)|best eval result in epoch', re.IGNORECASE)
END = re.compile(r'Finished training|All Over|Traceback \(most recent|RuntimeError:|'
                 r'out of memory|KeyboardInterrupt|Process finished|exit(?:_code)?\s*[:=]',
                 re.IGNORECASE)


def snippet(index, line):
    return {'line': index + 1, 'text': line[:900]}


def inspect_file(path, data):
    raw_text = data.decode('utf-8', errors='replace')
    folded = (str(path) + '\n' + raw_text).lower()
    matches = [run for run in RUNS if run in folded]
    if not matches:
        return None
    lines = [ANSI.sub('', line) for line in raw_text.splitlines()]
    run_mentions = []
    configs = []
    parameter_runs = []
    best = []
    endings = []
    validation_blocks = []
    test_markers = 0
    for index, line in enumerate(lines):
        lower = line.lower()
        if 'test result' in lower or 'best test' in lower:
            test_markers += 1
            continue
        if any(run in lower for run in RUNS):
            run_mentions.append(snippet(index, line))
        if CONFIG.search(line):
            configs.append(snippet(index, line))
        run_match = CONFIG_RUN.search(line)
        if run_match:
            parameter_runs.append({
                'line': index + 1, 'run_index': int(run_match.group(1)),
                'declared_total': int(run_match.group(2)), 'text': line[:1200],
            })
        if BEST.search(line) and 'updated!!!' not in lower:
            best.append(snippet(index, line))
        if END.search(line):
            endings.append(snippet(index, line))
        if re.search(r'\bvalid(?:ation)? result\s*:', line, re.IGNORECASE):
            block = [snippet(index, line)]
            if index + 1 < len(lines):
                following = lines[index + 1]
                if ('test' not in following.lower()
                        and re.search(r'recall|ndcg', following, re.IGNORECASE)):
                    block.append(snippet(index + 1, following))
            validation_blocks.append(block)

    # Preserve the initial configuration and any final overrides without
    # printing hundreds of per-epoch validation updates.
    config_selection = configs if len(configs) <= 70 else configs[:50] + configs[-20:]
    return {
        'path': str(path), 'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(), 'matched_runs': matches,
        'line_count': len(lines), 'run_mentions': run_mentions[:3] + run_mentions[-3:]
        if len(run_mentions) > 6 else run_mentions,
        'config_snippets': config_selection,
        'parameter_run_header_count': len(parameter_runs),
        'parameter_run_headers': parameter_runs[:8] + parameter_runs[-2:]
        if len(parameter_runs) > 10 else parameter_runs,
        'best_validation_snippets': best[-8:],
        'last_validation_blocks': validation_blocks[-2:],
        'end_status_snippets': endings[-8:],
        'test_result_markers_present': test_markers > 0,
        'test_result_marker_count': test_markers,
        'test_metric_values_omitted': True,
        'checkpoint_bytes_linkage_proven_by_log_match': False,
    }


def main():
    result = {
        'purpose': 'read_only_historical_baseline_training_provenance',
        'root': str(ROOT), 'requested_runs': list(RUNS),
        'searched_directories': [], 'missing_directories': [],
        'skipped': [], 'errors': [], 'matches': [],
        'files_examined': 0, 'bytes_read': 0, 'entries_examined': 0,
        'search_complete_within_declared_scope': True,
        'bounds': {'entries': MAX_ENTRIES, 'file_bytes': MAX_FILE_BYTES,
                   'total_bytes': MAX_TOTAL_BYTES, 'matches': MAX_MATCHES},
    }
    seen = set()
    stop = False
    root_resolved = ROOT.resolve()
    for relative in RELATIVE_DIRS:
        directory = ROOT / relative
        if not directory.is_dir():
            result['missing_directories'].append(str(directory))
            continue
        if directory.is_symlink() or root_resolved not in directory.resolve().parents:
            result['skipped'].append({'path': str(directory), 'reason': 'symlink_or_outside_root'})
            continue
        result['searched_directories'].append(str(directory))
        for current, child_dirs, files in os.walk(str(directory), followlinks=False):
            child_dirs[:] = sorted(name for name in child_dirs
                                   if not (Path(current) / name).is_symlink())
            for name in sorted(files):
                path = Path(current) / name
                result['entries_examined'] += 1
                if (result['entries_examined'] > MAX_ENTRIES
                        or result['bytes_read'] >= MAX_TOTAL_BYTES
                        or len(result['matches']) >= MAX_MATCHES):
                    result['search_complete_within_declared_scope'] = False
                    stop = True
                    break
                if path.is_symlink() or path.suffix.lower() not in TEXT_SUFFIXES:
                    continue
                if str(path) in seen:
                    continue
                seen.add(str(path))
                try:
                    size = path.stat().st_size
                    if size > MAX_FILE_BYTES or result['bytes_read'] + size > MAX_TOTAL_BYTES:
                        result['skipped'].append({'path': str(path), 'bytes': size,
                                                  'reason': 'read_budget'})
                        result['search_complete_within_declared_scope'] = False
                        continue
                    data = path.read_bytes()
                    result['files_examined'] += 1
                    result['bytes_read'] += len(data)
                    item = inspect_file(path, data)
                    if item is not None:
                        result['matches'].append(item)
                except (OSError, UnicodeError) as error:
                    result['errors'].append({'path': str(path), 'error': str(error)})
                    result['search_complete_within_declared_scope'] = False
            if stop:
                break
        if stop:
            break
    result['runs_found'] = sorted({run for item in result['matches']
                                   for run in item['matched_runs']})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
