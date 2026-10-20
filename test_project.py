"""Deterministic regression tests. Fake inference checks wiring, not Qwen accuracy."""
import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch
from data import Message, Concern, Interpretation, Topic, HK, ContextIndex, load_messages, select_messages
from model import discover_model, server_command, LocalLLM, interpret_batch, discover_topics
from analysis import compute_signals, classify_status, evidence_ids, review_coordination, run_pipeline
import run

def message(mid, day=20, text='钓鱼太累了', parent=None, platform='forum'):
    return Message(mid, parent, platform, 'general', 'user' + mid, None, text, datetime(2025, 8, day, 12, tzinfo=HK))

def concern(m, position='supports', kind='firsthand', style='literal'):
    return Concern(m.id + ':0', m.id, 'fishing', '钓鱼太累', 'Fishing is exhausting', 'Gameplay', position, 'critical', style, kind, m.content)

def response(obj):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(obj, ensure_ascii=False)}}]}

def fake_model(payload):
    data = json.loads(payload['messages'][1]['content'])
    if 'messages' in data:
        items = []
        for m in data['messages']:
            items.append({'id': m['id'], 'literal_en': m['text'], 'meaning_en': 'Fishing feels like work', 'feeling': 'frustrated', 'concerns': [{'target': 'fishing', 'issue_zh': '钓鱼太累', 'issue_en': 'Fishing is exhausting', 'category': 'Gameplay', 'position': 'supports', 'stance': 'critical', 'style': 'humorous', 'evidence_type': 'firsthand', 'quote': m['text']}]})
        return response({'items': items})
    return response({'groups': [{'ids': [t['id'] for t in data['topics']], 'title': '钓鱼太累', 'title_en': 'Fishing is exhausting', 'category': 'Gameplay'}]})

class ProjectTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def client(self, transport=fake_model):
        return LocalLLM(cache_dir=self.root / 'cache', transport=transport)

    def gguf(self, name='Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf'):
        p = self.root / 'models' / 'nested folder' / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'GGUFtest fixture')
        return p

    def source(self, messages):
        from dataclasses import asdict
        p = self.root / 'messages.jsonl'
        p.write_text('\n'.join((json.dumps({**asdict(m), 'created_at': m.created_at.isoformat()}, ensure_ascii=False) for m in messages)), encoding='utf8')
        return p

    def test_downloaded_filename_and_nested_folder(self):
        self.assertEqual(self.gguf(), discover_model(self.root))

    def test_explicit_path_with_spaces(self):
        p = self.gguf('a model.gguf')
        self.assertEqual(discover_model(self.root, str(p)), p)

    def test_missing_download(self):
        with self.assertRaises(FileNotFoundError):
            discover_model(self.root)

    def test_multiple_models_require_choice(self):
        self.gguf('one.gguf')
        self.gguf('two.gguf')
        with self.assertRaises(ValueError):
            discover_model(self.root)

    def test_html_is_not_model(self):
        p = self.gguf()
        p.write_bytes(b'<html>')
        with self.assertRaises(ValueError):
            discover_model(self.root)

    def test_incomplete_extension(self):
        p = self.gguf('unfinished.gguf.part')
        with self.assertRaises(ValueError):
            discover_model(self.root, str(p))

    def test_server_arguments_cpu_and_quoted_paths(self):
        m = self.gguf()
        exe = self.root / 'tools' / 'cpu folder' / 'llama-server.exe'
        exe.parent.mkdir(parents=True)
        exe.touch()
        cmd = server_command(self.root)
        self.assertEqual(cmd[cmd.index('-m') + 1], str(m))
        self.assertEqual(cmd[cmd.index('-ngl') + 1], '0')

    def test_future_rows_excluded(self):
        p = self.source([message('a', 20), message('b', 21)])
        self.assertEqual([m.id for m in load_messages(p, '2025-08-20')], ['a'])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            load_messages(self.source([message('a'), message('a')]), '2025-08-20')

    def test_parent_context_changes_signature(self):
        a = message('a', text='Fishing is exhausting')
        b = message('b', text='Fishing is great')
        x = message('x', text='+1', parent='a')
        y = message('y', text='+1', parent='b')
        ctx = ContextIndex([a, b, x, y])
        self.assertNotEqual(ctx.signature(x), ctx.signature(y))
        self.assertEqual(ctx.bundle(x)['parents'][0]['id'], 'a')

    def test_future_parent_not_used(self):
        a = message('a', 21)
        b = message('b', 20, parent='a')
        self.assertEqual(ContextIndex([a, b]).ancestors(b), [])

    def test_unique_wording_enters_selection(self):
        ms = [message(str(i), text=t) for i, t in enumerate(['釣魚上班', '手腕要斷了', '又要做水邊日課', '這不是休閒', '+1'])]
        self.assertEqual(len(select_messages(ms, ContextIndex(ms), 96)), 5)

    def test_remote_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            LocalLLM('https://example.com/v1')

    def test_cache_avoids_second_request(self):
        ms = [message('a')]
        c = self.client()
        interpret_batch(ms, ContextIndex(ms), c)
        interpret_batch(ms, ContextIndex(ms), c)
        self.assertEqual(c.calls, 1)
        self.assertEqual(c.hits, 1)

    def test_invalid_output_has_repair_feedback(self):
        requests = []

        def transport(p):
            requests.append(p)
            return response({'items': []}) if len(requests) == 1 else fake_model(p)
        ms = [message('a')]
        interpret_batch(ms, ContextIndex(ms), self.client(transport))
        self.assertIn('failed validation', requests[1]['messages'][-1]['content'])

    def test_invented_quote_rejected(self):

        def bad(p):
            r = fake_model(p)
            obj = json.loads(r['choices'][0]['message']['content'])
            obj['items'][0]['concerns'][0]['quote'] = 'nonexistent evidence'
            return response(obj)
        ms = [message('a')]
        with self.assertRaises(ValueError):
            interpret_batch(ms, ContextIndex(ms), self.client(bad))

    def test_semantic_grouping_precedes_minimum_count(self):
        ms = [message(str(i), text=str(i) + '不同的表达') for i in range(5)]
        its = [Interpretation(m.id, 'literal', 'meaning', 'frustrated', [concern(m)]) for m in ms]
        topics = discover_topics(its, self.client())
        self.assertEqual(len(topics), 1)
        self.assertEqual(len(topics[0].members), 5)

    def test_disagreement_and_quotes_not_support(self):
        ms = [message(str(i)) for i in range(4)]
        topic = Topic('t', '题', 'issue', 'Gameplay', [concern(ms[0]), concern(ms[1], 'opposes', 'disagreement'), concern(ms[2], 'supports', 'quotation'), concern(ms[3], style='sarcastic')])
        by = {m.id: m for m in ms}
        s = compute_signals(topic, by, ms, date(2025, 8, 20))
        self.assertEqual(s['support_messages'], 2)
        self.assertEqual(s['opposition_messages'], 1)
        self.assertEqual(s['quotation_messages'], 1)
        self.assertEqual(s['humorous_support_messages'], 1)
        self.assertEqual(set(evidence_ids(topic, by)), {'0', '3'})

    def test_renewed_growth_wins_over_old_peak(self):
        s = dict(support_messages=10, common_platforms=1, baseline_reviewed=20, recent_reviewed=20, recent_support=4, recent_actors=4, recent_active_days=2, share_ratio=4, days_since_peak=8, peak_count=20, recent_review_days=3, episode_days=1, age_days=60)
        self.assertEqual(classify_status(s), 'growing')

    def test_short_episode_can_have_old_background(self):
        s = dict(support_messages=10, common_platforms=0, days_since_peak=3, peak_count=8, recent_review_days=3, recent_support=0, episode_days=2, age_days=60)
        self.assertEqual(classify_status(s), 'one-off')

    def test_organic_coordination_not_called_fake(self):
        ms = [message(str(i)) for i in range(20)]
        c = self.client(lambda p: response(dict(assessment='uncertain', title='活动', title_en='Campaign', reason='Could be fans')))
        alert, _ = review_coordination(dict(rows=ms, exclusive_name_fraction=1, clock_fraction=1), c)
        self.assertIsNone(alert)

    def test_pipeline_semantic_end_to_end(self):
        ms = [message(str(i), 18 + i % 3, text=t) for i, t in enumerate(['钓鱼像上班', '手腕累死了', '水边日课受不了', '休闲怎么这么累', '又在钓鱼打卡'])]
        report = run_pipeline(self.source(ms), '2025-08-20', self.root / 'out.json', client=self.client(), coordination=False)
        self.assertEqual(len(report['trends']), 1)
        self.assertEqual(report['trends'][0]['status'], 'new')

    def test_offline_never_claims_semantics(self):
        with patch('socket.socket', side_effect=AssertionError('network')):
            r = run_pipeline(self.source([message('a')]), '2025-08-20', self.root / 'offline.json', mode='offline')
        self.assertEqual(r['trends'], [])

    def test_cli_defaults_to_local(self):
        with patch.object(run, 'ensure_server', return_value=(None, None)), patch.object(run, 'run_pipeline', return_value={'trends': []}) as pipeline:
            self.assertEqual(run.main(['--input', 'x', '--as-of', '2025-08-20', '--out', 'y']), 0)
            self.assertEqual(pipeline.call_args.args[3], 'local')

    def test_root_server_discovery(self):
        self.gguf()
        exe = self.root / 'llama-server.exe'
        exe.touch()
        self.assertEqual(server_command(self.root)[0], str(exe))

    def test_autostart_server_is_stopped(self):
        from unittest.mock import Mock
        proc = Mock()
        handle = Mock()
        with patch.object(run, 'ensure_server', return_value=(proc, handle)), patch.object(run, 'run_pipeline', return_value={'trends': []}):
            self.assertEqual(run.main(['--input', 'x', '--as-of', '2025-08-20', '--out', 'y']), 0)
        proc.terminate.assert_called_once()
        handle.close.assert_called_once()

    def test_future_data_cannot_change_result(self):
        ms = [message(str(i), 18 + i % 3) for i in range(5)]
        a = run_pipeline(self.source(ms), '2025-08-20', self.root / 'a.json', client=self.client(), coordination=False)
        b = run_pipeline(self.source(ms + [message('future', 21)]), '2025-08-20', self.root / 'b.json', client=self.client(), coordination=False)
        self.assertEqual(a, b)

    def test_sampling_includes_comparison_baseline(self):
        ms = [message(str(day) + '-' + str(i), day, text='unique ' + str(day) + ' ' + str(i)) for day in range(1, 29) for i in range(6)]
        selected = select_messages(ms, ContextIndex(ms), 30)
        self.assertTrue(any((m.created_at.day >= 26 for m in selected)))
        self.assertTrue(any((12 <= m.created_at.day < 26 for m in selected)))

    def test_serve_does_not_require_dataset(self):
        with patch.object(run, 'server_command', return_value=['test']), patch.object(run.subprocess, 'call', return_value=0):
            self.assertEqual(run.main(['--serve']), 0)
if __name__ == '__main__':
    unittest.main()
