"""Exercise real Bash orchestration using an offline model-command stand-in."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / 'scripts', self.root / 'scripts')
        # The stand-in records command boundaries; table builders run real Python.
        self.python = self.root / 'python'
        self.python.write_text('#!' + sys.executable + '\n' + '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['COMMAND_LOG'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
if args[0] != '-m':
    os.execv(sys.executable, [sys.executable, *args])
record = {'doc_id': 'd', 'generated_summary': 'Costs increased.',
          'meta': {'target_words': int(args[args.index('--target-words') + 1]) if '--target-words' in args else 0}}
for option in ['--output', '--output-records', '--output-summary', '--output-jsonl', '--output-markdown']:
    if option in args:
        path = Path(args[args.index(option) + 1])
        path.parent.mkdir(parents=True, exist_ok=True)
        value = {'by_version': {'before_correction': {'count': 1}}} if option == '--output-summary' else record
        path.write_text(json.dumps(value) + '\\n')
''')
        self.python.chmod(0o755)
        self.log = self.root / 'commands'
        for name in ['documents', 'summaries']:
            (self.root / name).write_text('{"doc_id":"d"}\n')
        self.env = {**os.environ, 'CONFIG': '', 'PYTHON': str(self.python),
                    'COMMAND_LOG': str(self.log), 'PYTHONPATH': str(ROOT), 'DOCUMENTS': str(self.root / 'documents'),
                    'SUMMARIES': str(self.root / 'summaries'), 'RUN_DIR': str(self.root / 'run'),
                    'OUTPUT_DIR': str(self.root / 'tables')}

    def run_script(self, name, **env):
        return subprocess.run(['bash', 'scripts/' + name], cwd=self.root,
                              env={**self.env, **env}, capture_output=True, text=True)

    def calls(self, module):
        return [row for line in self.log.read_text().splitlines()
                if (row := json.loads(line))[:2] == ['-m', 'src.' + module]]

    def test_disabled_correction_does_not_export_old_results(self):
        run = self.root / 'run'
        run.mkdir()
        for name in ['corrections.bm25.jsonl', 'verification.after.bm25.jsonl']:
            (run / name).write_text('{"old":true}\n')
        result = self.run_script('run_pipeline.sh', RETRIEVAL_METHOD='bm25', RUN_CORRECTION='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        call = self.calls('export_cases')[0]
        self.assertNotIn('--corrections', call)
        self.assertNotIn('--verification-after', call)

    def test_disabled_after_verification_does_not_export_old_verification(self):
        run = self.root / 'run'
        run.mkdir()
        (run / 'verification.after.bm25.jsonl').write_text('{"old":true}\n')
        result = self.run_script('run_pipeline.sh', RETRIEVAL_METHOD='bm25', RUN_CORRECTION='1', RUN_AFTER_VERIFY='0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn('--verification-after', self.calls('export_cases')[0])

    def test_bm25_comparison_does_not_load_embedding(self):
        result = self.run_script('run_compare_retrieval.sh', METHODS='bm25')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls('retrieval_embedding'), [])
        self.assertEqual(self.calls('retrieval_ariel'), [])

    def test_comparison_dispatches_selected_ablation_options(self):
        methods = 'embedding hybrid ariel ariel_no_coverage ariel_no_expansion ariel_no_risk_weighting'
        result = self.run_script('run_compare_retrieval.sh', METHODS=methods)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls('retrieval_bm25')), 1)
        self.assertEqual(len(self.calls('retrieval_embedding')), 1)
        self.assertEqual(len(self.calls('retrieval_hybrid')), 1)
        calls = self.calls('retrieval_ariel')
        self.assertEqual(len(calls), 4)
        self.assertNotIn('--disable-coverage-rerank', calls[0])
        self.assertIn('--disable-coverage-rerank', calls[1])
        self.assertIn('--disable-adaptive-expansion', calls[2])
        self.assertEqual(calls[3][calls[3].index('--number-date-alpha') + 1], '0.5')
        self.assertEqual(len(self.calls('verify_nli')), 6)

    def test_controlled_length_forwards_labels_and_isolates_config(self):
        config = self.root / 'controlled.env'
        config.write_text('LENGTH_SPECS="tiny:80:128:160 wide:520:640:900"\n'
                          'RUN_CORRECTION=1\nSUMMARY_TARGET_WORDS=999\n'
                          'CHUNKS=shared_chunks.jsonl\nNLI_LIMIT=2\n')
        result = self.run_script('run_controlled_length_experiment.sh', CONFIG=str(config),
                                 SOURCE_DOCUMENTS=str(self.root / 'documents'),
                                 RUN_ROOT=str(self.root / 'lengths'))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls('correct_local'), [])
        calls = self.calls('summarize')
        self.assertEqual([c[c.index('--target-words') + 1] for c in calls], ['80', '520'])
        self.assertEqual([c[c.index('--chunks') + 1] for c in self.calls('retrieval_bm25')],
                         ['shared_chunks.jsonl', 'shared_chunks.jsonl'])
        self.assertEqual([c[c.index('--limit') + 1] for c in self.calls('verify_nli')], ['2', '2'])
        table = json.loads((self.root / 'tables/controlled_length.json').read_text())
        self.assertEqual([r['label'] for r in table], ['tiny', 'wide'])

    def test_missing_requested_length_is_not_silently_omitted(self):
        run = self.root / 'lengths/short'
        run.mkdir(parents=True)
        (run / 'evaluation.ariel.summary.json').write_text(json.dumps(
            {'by_version': {'before_correction': {'count': 1}}}))
        result = subprocess.run([sys.executable, 'scripts/summarize_controlled_length_results.py',
                                 '--run-root', str(self.root / 'lengths'), '--labels', 'short', 'long'],
                                cwd=self.root, env=self.env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('long', result.stderr)

    def table_data(self, method='ariel'):
        run = self.root / 'run'
        run.mkdir(exist_ok=True)
        summary = {'by_version': {'before_correction': {'count': 1}},
                   'by_length_bucket': {'short': {'before_correction': {'count': 0}},
                                        'long': {'before_correction': {'count': 1}}}}
        (run / f'evaluation.{method}.summary.json').write_text(json.dumps(summary))
        (run / f'evaluation.{method}.jsonl').write_text(json.dumps(
            {'summary_version': 'before_correction', 'summary_word_count': 10, 'length_bucket': 'long'}) + '\n')

    def test_tables_use_evaluator_buckets(self):
        self.table_data()
        result = self.run_script('make_tables.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('| Long | 1 |', (self.root / 'tables/table_2_length_buckets.md').read_text())

    def test_tables_fail_when_requested_method_is_missing(self):
        self.table_data('bm25')
        result = self.run_script('make_tables.sh', PRIMARY_METHOD='ariel')
        self.assertNotEqual(result.returncode, 0)

    def test_tables_fail_when_requested_correction_run_is_missing(self):
        self.table_data()
        result = self.run_script('make_tables.sh', CORRECTION_RUN_DIR=str(self.root / 'absent'))
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
