"""Validate the published snapshot and the matched-route comparisons."""
import csv
import json
import statistics
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class LatestResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / 'app/latest-results.json').read_text())

    def aggregate(self, rows):
        return dict(count=len(rows), goals=sum(r['goal'] for r in rows), safe=sum(r['safe'] for r in rows),
                    latency=sum(r['apiSeconds'] for r in rows)/sum(r['responses'] for r in rows),
                    decisions=statistics.mean(r['decisions'] for r in rows),
                    safetyEvents=statistics.mean(r['safetyEvents'] for r in rows))

    def test_public_data_matches_rendered_data(self):
        self.assertEqual((ROOT/'app/latest-results.json').read_bytes(), (ROOT/'public/data/latest-results.json').read_bytes())
        text = (ROOT/'public/data/latest-results.json').read_text()
        for private in ['/data/rose', '/home/rose', 'xoxb-', 'xapp-', 'api_key']:
            self.assertNotIn(private, text)
        with (ROOT/'public/data/latest-results.csv').open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), len(self.data['models']))
        for csv_row, model in zip(rows, self.data['models']):
            for key in ['latency', 'decisions', 'safetyEvents', 'count']:
                self.assertAlmostEqual(float(csv_row[key]), model[key])

    def test_coverage_and_episode_aggregates(self):
        episodes = self.data['episodes']
        self.assertEqual(len(episodes), self.data['validatedEpisodes'])
        self.assertEqual(len({(r['slug'],r['task']) for r in episodes}), len(episodes))
        for model in self.data['models']:
            rows = [r for r in episodes if r['slug'] == model['slug']]
            self.assertEqual(sorted(r['task'] for r in rows), model['tasks'])
            self.assertEqual(sorted(set(self.data['tasks'])-set(model['tasks'])), model['missingTasks'])
            self.assertEqual(model['count'] + len(model['missingTasks']), model['expected'])
            for key,value in self.aggregate(rows).items():
                self.assertAlmostEqual(model[key], value)
            for row in rows:
                self.assertEqual(row['safe'], row['goal'] and row['safetyEvents'] == 0)
                self.assertGreater(row['responses'], 0)

    def test_pairs_use_only_shared_routes(self):
        for pair in self.data['pairs']:
            old = [r for r in self.data['episodes'] if r['slug'] == pair['older']]
            new = [r for r in self.data['episodes'] if r['slug'] == pair['newer']]
            shared = {r['task'] for r in old} & {r['task'] for r in new}
            self.assertEqual(pair['tasks'], sorted(shared))
            for label, rows in [('old',old), ('new',new)]:
                for key,value in self.aggregate([r for r in rows if r['task'] in shared]).items():
                    self.assertAlmostEqual(pair[label][key], value)
            for key,change in pair['changes'].items():
                self.assertAlmostEqual(change, (pair['new'][key]/pair['old'][key]-1)*100)
        partial = {p['key']:len(p['tasks']) for p in self.data['pairs']}
        self.assertEqual(partial['astra'],8)
        self.assertEqual(partial['deepseek'],8)
