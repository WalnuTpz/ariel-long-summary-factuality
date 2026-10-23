"""Offline regressions for evidence fusion and ablation semantics."""
import json
import os
import subprocess
import sys
import unittest
from src import retrieval_ariel as ariel, retrieval_hybrid as hybrid


def retrieval(ids, query='Costs rose in 2021.'):
    return {'query_id': 'q', 'doc_id': 'd', 'query': query,
            'top_k': [{'chunk_id': i, 'chunk_index': n, 'score': 1., 'text': 'Costs rose.'}
                      for n, i in enumerate(ids)]}


class RetrievalTests(unittest.TestCase):
    def test_coverage_ablation_keeps_expansion(self):
        recall = retrieval(['c0'])
        chunks = [{'doc_id': 'd', 'chunk_id': 'c0', 'chunk_index': 0, 'text': 'Costs rose.'},
                  {'doc_id': 'd', 'chunk_id': 'c1', 'chunk_index': 1, 'text': 'The year was 2021.'}]
        rows, _ = ariel.retrieve_ariel([recall], [recall], chunks=chunks, enable_coverage=False)
        evidence = rows[0]['top_k'][0]
        self.assertTrue(evidence['expanded_context'])
        self.assertIn('2021', evidence['text'])
        self.assertEqual(evidence['score'], evidence['base_score'])

    def test_disabled_ariel_matches_fixed_hybrid(self):
        bm25, dense = retrieval(['a']), retrieval(['b', 'c'])
        dense['top_k'][1]['score'] = 0.
        expected = hybrid.fuse_one_result(bm25, dense)
        rows, _ = ariel.retrieve_ariel([bm25], [dense], enable_coverage=False,
                                     enable_expansion=False,
                                     alpha_by_risk={risk: .5 for risk in ariel.RISK_TYPES})
        self.assertEqual([(r['chunk_id'], r['score']) for r in rows[0]['top_k']],
                         [(r['chunk_id'], r['score']) for r in expected['top_k']])

    def test_percent_terms_are_not_bare_numbers(self):
        self.assertEqual(ariel.number_date_coverage('Costs rose 5%.', 'Costs rose by 5.')[0], 0.)
        self.assertEqual(ariel.number_date_coverage('Costs rose 5%.', 'Costs rose 5%.')[0], 1.)

    def test_equal_scores_have_stable_order(self):
        code = '''
import json
from src.retrieval_hybrid import fuse_one_result
from src.retrieval_ariel import retrieve_ariel
r = json.loads(%r)
h = fuse_one_result(r, r)
a = retrieve_ariel([r], [r], enable_coverage=False, enable_expansion=False)[0][0]
print(json.dumps([[x['chunk_id'] for x in row['top_k']] for row in [h,a]]))
''' % json.dumps(retrieval(['c', 'a', 'b']))
        orders = [subprocess.check_output([sys.executable, '-c', code],
                                         env={**os.environ, 'PYTHONHASHSEED': str(seed)}, text=True)
                  for seed in [1, 2, 3]]
        self.assertEqual(len(set(orders)), 1)


if __name__ == '__main__':
    unittest.main()
