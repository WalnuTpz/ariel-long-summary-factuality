"""Offline regressions for sentence identity, correction, and evaluation."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import Mock, patch

from src import correct_local, evaluate_summary, export_cases, load_data, retrieval_bm25, summarize, verify_nli
from src.split_summary_sentences import split_sentences, split_summary_records
from src.utils import read_jsonl, write_jsonl
from src.verification_schema import SUPPORTED, UNSUPPORTED, NOT_ENOUGH_INFO, build_verification_inputs


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def save(self, name, records):
        path = self.root / name
        write_jsonl(records, path)
        return path

    def correct(self, original, response):
        summary = {'doc_id': 'd', 'summary_id': 'sum', 'generated_summary': original}
        sentences, _ = split_summary_records([summary])
        verification = {s['query_id']: {'label': UNSUPPORTED, 'scores': {'contradiction': .9},
                                      'evidence': ['Costs fell.'], 'evidence_ids': ['c']}
                        for s in sentences}
        corrector = Mock()
        corrector.generate.return_value = json.dumps({'corrected_sentence': response})
        record = correct_local.build_corrected_record(
            {**summary, 'original_summary': original}, sentences, verification,
            corrector, 'test', 3, 1200, 5, 45, 128, 'keep', .7,
            correct_local.CorrectionStats(), False, False)
        return record, sentences

    def test_source_index_survives_filtering(self):
        record, _ = load_data.normalize_record(
            {'doc_id': 'd', 'document': 'Long enough.', 'meta': {'source_index': 42}},
            index=1, min_document_chars=1)
        self.assertEqual(record.meta['source_index'], 42)

    def test_long_abbreviation_is_protected_before_its_prefix(self):
        self.assertEqual(split_sentences('The U.S.C. 123 applies. Costs fell.'),
                         ['The U.S.C. 123 applies.', 'Costs fell.'])

    def test_rewrite_retains_a_boundary_in_reconstructed_text(self):
        record, _ = self.correct('Costs increased.', 'Costs fell')
        self.assertEqual(record['corrected_summary'], 'Costs fell.')

    def test_unchanged_unpunctuated_sentence_is_kept(self):
        record, _ = self.correct('Costs increased', 'Costs increased')
        self.assertEqual(record['corrected_summary'], 'Costs increased')
        self.assertEqual(record['sentence_edits'][0]['action'], 'keep')

    def test_rewrite_with_quoted_terminal_punctuation_is_preserved(self):
        for replacement in ['The report said “costs fell.”', 'The report said ‘costs fell.’']:
            with self.subTest(replacement=replacement):
                record, _ = self.correct('Costs increased.', replacement)
                self.assertEqual(record['corrected_summary'], replacement)

    def test_query_ids_include_summary_identity(self):
        rows, _ = split_summary_records([
            {'doc_id': 'd', 'summary_id': 'a', 'summary': 'Costs increased.'},
            {'doc_id': 'd', 'summary_id': 'b', 'summary': 'Costs decreased.'}])
        self.assertEqual(len({r['query_id'] for r in rows}), 2)

    def test_correction_preserves_filtered_text(self):
        record, _ = self.correct('It is. Costs increased.', 'Costs fell.')
        self.assertEqual(record['corrected_summary'], 'It is. Costs fell.')

    def test_short_rewrite_keeps_verification_unit(self):
        record, before = self.correct('Costs increased.', 'It fell.')
        # Use a replacement below the default eight-character filter.
        record['sentence_edits'][0]['corrected_sentence'] = 'It is.'
        record['corrected_summary'] = 'It is.'
        after, _ = split_summary_records([record], summary_field='corrected_summary')
        self.assertEqual([(r['query_id'], r['text']) for r in after],
                         [(before[0]['query_id'], 'It is.')])

    def test_rewrite_without_terminal_punctuation_keeps_sentence_ids(self):
        summary = {'doc_id': 'd', 'summary_id': 'sum',
                   'generated_summary': 'Costs increased. Programs expanded.'}
        before, _ = split_summary_records([summary])
        correction = {**summary, 'original_summary': summary['generated_summary'],
                      'corrected_summary': 'Costs fell Programs expanded.',
                      'sentence_edits': [{'sentence_id': 1, 'corrected_sentence': 'Costs fell'}]}
        after, _ = split_summary_records([correction], summary_field='corrected_summary')
        self.assertEqual([(r['query_id'], r['text']) for r in after],
                         [(before[0]['query_id'], 'Costs fell'),
                          (before[1]['query_id'], 'Programs expanded.')])

    def test_multi_sentence_rewrite_is_rejected(self):
        record, _ = self.correct('Costs increased.', 'Costs fell. Programs expanded.')
        self.assertEqual(record['corrected_summary'], 'Costs increased.')
        self.assertEqual(record['sentence_edits'][0]['action'], 'keep')

    def test_correction_has_its_own_system_prompt(self):
        summary = {'doc_id': 'd', 'summary_id': 'sum', 'generated_summary': 'Costs increased.'}
        sentences, _ = split_summary_records([summary])
        with patch.object(correct_local, 'HuggingFaceSummarizer') as model:
            model.return_value.generate.return_value = '{"corrected_sentence": "Costs fell."}'
            correct_local.correct_local_jsonl(
                self.save('summaries', [summary]), self.save('sentences', sentences),
                self.save('verification', [{**sentences[0], 'label': UNSUPPORTED}]),
                self.root / 'corrected')
        self.assertIn('JSON', model.call_args.kwargs.get('system_prompt', ''))

    def test_correction_does_not_substitute_another_summary(self):
        summary = {'doc_id': 'd', 'summary_id': 'a', 'summary': 'Costs increased.'}
        sentences, _ = split_summary_records([{**summary, 'summary_id': 'b'}])
        with patch.object(correct_local, 'HuggingFaceSummarizer'):
            with self.assertRaises(KeyError):
                correct_local.correct_local_jsonl(self.save('summaries', [summary]),
                    self.save('sentences', sentences), self.save('verification', []), self.root / 'out')

    def test_reference_is_not_the_candidate_summary(self):
        path = self.save('summaries', [{'doc_id': 'd', 'summary': 'Costs increased.'}])
        rows, _ = evaluate_summary.load_summaries(path, {'d': {'reference_summary': 'Costs fell.'}}, None)
        self.assertEqual(rows[('d', 'd_summary')]['reference_summary'], 'Costs fell.')
        rows, _ = evaluate_summary.load_summaries(path, {}, None)
        self.assertEqual(rows[('d', 'd_summary')]['reference_summary'], '')

    def test_verification_is_not_borrowed_from_another_summary(self):
        summaries = self.save('summaries', [
            {'doc_id': 'd', 'summary_id': 'a', 'summary': 'Costs increased.'},
            {'doc_id': 'd', 'summary_id': 'b', 'summary': 'Costs decreased.'}])
        verification = self.save('verification', [{'doc_id': 'd', 'summary_id': 'a', 'label': SUPPORTED}])
        output = self.root / 'eval'
        evaluate_summary.evaluate_summary_jsonl(summaries, output, verification_before_path=verification)
        rows = read_jsonl(output)
        self.assertEqual(rows[0]['supported_ratio'], 1)
        self.assertIsNone(rows[1]['supported_ratio'])

    def test_missing_retrieval_is_nei_not_dropped(self):
        sentences, _ = split_summary_records([{'doc_id': 'd', 'summary': 'Costs increased.'}])
        output = self.root / 'verification'
        with patch.object(verify_nli, 'NliVerifier') as model:
            model.return_value.model_name = 'test'
            verify_nli.verify_nli_jsonl(output, sentences_path=self.save('sentences', sentences),
                                       retrieval_path=self.save('retrieval', []))
            model.return_value.predict.assert_not_called()
        rows = read_jsonl(output)
        self.assertEqual([r['label'] for r in rows], [NOT_ENOUGH_INFO])

    def test_label_order_must_be_a_permutation(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            verify_nli.parse_label_order('neutral,neutral,entailment')

    def test_all_short_summary_is_preserved(self):
        summary = {'doc_id': 'd', 'summary_id': 'sum', 'summary': 'It is.'}
        output = self.root / 'out'
        with patch.object(correct_local, 'HuggingFaceSummarizer') as model:
            correct_local.correct_local_jsonl(self.save('summaries', [summary]),
                self.save('sentences', []), self.save('verification', []), output)
            model.return_value.generate.assert_not_called()
        self.assertEqual([r['corrected_summary'] for r in read_jsonl(output)], ['It is.'])

    def test_unchanged_sentence_is_not_a_successful_correction(self):
        case, _ = export_cases.classify_case(
            {'label': UNSUPPORTED}, {'action': 'keep'}, {'label': SUPPORTED}, False)
        self.assertEqual(case, export_cases.CASE_RISK_UNCORRECTED)

    def test_chat_template_special_tokens_are_not_added_twice(self):
        model = summarize.HuggingFaceSummarizer.__new__(summarize.HuggingFaceSummarizer)
        model.system_prompt = 'Return JSON.'
        model.tokenizer = Mock(chat_template='template', pad_token_id=0, eos_token_id=1)
        tensor = Mock(shape=(1, 3))
        tensor.to.return_value = tensor
        model.tokenizer.return_value = {'input_ids': tensor}
        model.tokenizer.decode.return_value = '{"corrected_sentence": "Costs fell."}'
        model.model = Mock()
        model.model.generate.return_value = [[1, 2, 3, 4]]
        model.torch = Mock()
        model.torch.inference_mode.return_value = nullcontext()
        model.device, model.max_prompt_tokens, model.do_sample = 'cpu', 100, False
        model.generate('Correct this sentence.', 20)
        self.assertFalse(model.tokenizer.call_args.kwargs['add_special_tokens'])
        messages = model.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(messages[0]['content'], 'Return JSON.')

    def test_correction_to_case_export_uses_matching_sentence_evidence(self):
        original = 'It is. Costs increased. Programs expanded.'
        summaries = [{'doc_id': 'd', 'summary_id': 'sum', 'generated_summary': original}]
        before, _ = split_summary_records(summaries)
        chunks = [{'doc_id': 'd', 'chunk_id': 'c', 'chunk_index': 0,
                   'text': 'Costs fell. Programs expanded.'}]
        verifier = Mock(model_name='test')
        predictions = {
            'Costs increased.': ('contradiction', UNSUPPORTED),
            'Costs fell.': ('entailment', SUPPORTED),
            'Programs expanded.': ('entailment', SUPPORTED),
        }

        def predict(premise, hypothesis):
            raw, label = predictions[hypothesis]
            return {'raw_label': raw, 'label': label, 'scores': {raw: .99}}

        verifier.predict.side_effect = predict

        def verify(sentences):
            retrieval, _ = retrieval_bm25.retrieve_bm25(chunks, sentences)
            inputs, _ = build_verification_inputs(sentences, retrieval, 'BM25 + NLI', 'nli')
            return verify_nli.verify_inputs(inputs, verifier)[0]

        before_labels = verify(before)
        summary_path = self.save('summaries', summaries)
        correction_path = self.root / 'corrections'
        with patch.object(correct_local, 'HuggingFaceSummarizer') as model:
            model.return_value.generate.return_value = '{"corrected_sentence": "Costs fell"}'
            correct_local.correct_local_jsonl(summary_path, self.save('sentences', before),
                                              self.save('before', before_labels), correction_path)
        corrections = read_jsonl(correction_path)
        after, _ = split_summary_records(corrections, summary_field='corrected_summary')
        after_labels = verify(after)
        self.assertEqual(len(after_labels), 2)
        evaluation_path = self.root / 'evaluation'
        evaluate_summary.evaluate_summary_jsonl(summary_path, evaluation_path,
            corrections_path=correction_path, verification_before_path=self.root / 'before',
            verification_after_path=self.save('after', after_labels))
        evaluation = read_jsonl(evaluation_path)
        self.assertEqual([r['supported_ratio'] for r in evaluation], [.5, 1.])
        cases_path = self.root / 'cases'
        export_cases.export_cases(self.root / 'before', cases_path,
            corrections_path=correction_path, verification_after_path=self.root / 'after',
            evaluation_path=evaluation_path)
        cases = read_jsonl(cases_path)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]['case_type'], export_cases.CASE_SUCCESSFUL_CORRECTION)
        self.assertEqual(cases[0]['corrected_sentence'], 'Costs fell.')
        self.assertTrue(corrections[0]['corrected_summary'].startswith('It is. '))

    def test_chat_template_error_is_not_hidden(self):
        model = summarize.HuggingFaceSummarizer.__new__(summarize.HuggingFaceSummarizer)
        model.system_prompt = 'Return JSON.'
        model.tokenizer = Mock(chat_template='template')
        model.tokenizer.apply_chat_template.side_effect = RuntimeError('invalid template')
        with self.assertRaisesRegex(RuntimeError, 'invalid template'):
            model.format_chat_prompt('Correct this sentence.')


if __name__ == '__main__':
    unittest.main()
