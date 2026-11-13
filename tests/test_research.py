"""Offline checks of methodological boundaries in the actual published notebook."""
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'itml-test-matplotlib'))
import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/investment_track_ml.ipynb'
FEATURES = ['ret_1','ret_5','ret_20','vol_20','vol_60','ma_20_gap','ma_50_gap',
            'volume_z_20','range_pct','rsi_14','spy_ret_5','spy_vol_20']


def helpers():
    namespace = {'HORIZONS': (4, 5, 20), 'FEATURES': FEATURES}
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if 'definitions' in cell.metadata.get('tags', []):
            exec(compile(cell.source, str(NOTEBOOK) + '#' + cell.id, 'exec'), namespace)
    return namespace


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ns = helpers()

    def test_notebook_structure_syntax_and_clean_outputs(self):
        notebook = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
                if 'install' not in cell.metadata.get('tags', []):
                    compile(cell.source, cell.id, 'exec')

    def test_training_labels_must_end_before_evaluation(self):
        train = pd.DataFrame({'date': pd.to_datetime(['2020-01-01','2020-01-02']),
                              'label_end_4': pd.to_datetime(['2020-01-05','2020-01-06']),
                              'target': [0,1]})
        evaluation = pd.DataFrame({'date': pd.to_datetime(['2020-01-07','2020-01-08'])})
        self.ns['check_split']('TEST',4,'valid',train,evaluation)
        for first_date in ['2020-01-06', '2020-01-05']:
            with self.subTest(first_date=first_date), self.assertRaises(AssertionError):
                self.ns['check_split']('TEST',4,'leaky',train,
                    evaluation.assign(date=pd.to_datetime([first_date,'2020-01-08'])))

    def test_baseline_comes_from_training_not_holdout(self):
        metrics, baseline = self.ns['metrics_with_baseline'](
            np.array([0,0,0,1]), np.full(4,.7), np.array([1,1,1,0]))
        np.testing.assert_array_equal(baseline, np.full(4,.75))
        self.assertEqual(metrics['baseline_accuracy'], .25)
        self.assertEqual(metrics['reference_base_rate'], .75)

    def test_future_prices_do_not_change_past_features(self):
        dates = pd.bdate_range('2024-01-01', periods=180)
        rng = np.random.default_rng(42)
        close = 100*np.exp(np.cumsum(rng.normal(0,.01,len(dates))))
        frame = pd.DataFrame({'date':dates, 'close':close, 'high':close*1.01,
            'low':close*.99, 'open':close, 'volume':rng.integers(10000,20000,len(dates))})
        def context(data):
            data = data.copy()
            data['ret_5'] = data.close.pct_change(5)
            data['vol_20'] = data.close.pct_change().rolling(20).std()*np.sqrt(252)
            return data
        original = self.ns['make_features'](frame, context(frame))
        changed = frame.copy()
        changed.loc[121:, ['open','high','low','close']] *= 1.2
        changed.loc[121:, 'volume'] *= 2
        result = self.ns['make_features'](changed, context(changed))
        pd.testing.assert_frame_equal(original.loc[:120,FEATURES], result.loc[:120,FEATURES])
        for horizon in (4,5,20):
            self.assertEqual(original[f'label_end_{horizon}'].iloc[0], dates[horizon])
            self.assertAlmostEqual(original[f'forward_{horizon}'].iloc[0], close[horizon]/close[0]-1)
            self.assertTrue(original[f'forward_{horizon}'].tail(horizon).isna().all())

    def test_gate_rejects_missing_or_weak_evidence(self):
        passing = dict(validation_brier_improvement=.01, roc_auc=.60,
                       brier=.20, baseline_brier=.22, log_loss=.60,
                       baseline_log_loss=.65, reference_base_rate=.5)
        gate = self.ns['research_signal']
        self.assertEqual(gate(.60,passing), 'Research candidate only')
        self.assertEqual(gate(.40,passing), 'Review downside; not a sell instruction')
        self.assertEqual(gate(.53,passing), 'Neutral')
        for name,value in [('validation_brier_improvement',0),('roc_auc',np.nan),
                           ('roc_auc',.54),('brier',.22),('log_loss',.65)]:
            with self.subTest(metric=name,value=value):
                self.assertEqual(gate(.99,dict(passing,**{name:value})), 'Insufficient model evidence')

    def test_full_model_pipeline_on_synthetic_history(self):
        # Real model selection/refit on synthetic data; no market download or investment conclusion.
        rng = np.random.default_rng(123)
        dates = pd.bdate_range('2018-01-01', periods=1100)
        close = 100*np.exp(np.cumsum(rng.normal(.0002,.01,len(dates))))
        frame = pd.DataFrame({'date':dates,'close':close,'open':close,
            'high':close*1.01,'low':close*.99,'volume':rng.integers(10000,50000,len(dates))})
        context = frame[['date']].copy()
        context['ret_5'] = frame.close.pct_change(5)
        context['vol_20'] = frame.close.pct_change().rolling(20).std()*np.sqrt(252)
        featured = self.ns['make_features'](frame,context)
        self.ns['EXPECTED_LAST'] = dates[-1]
        result = self.ns['analyze_symbol_horizon']('SYNTHETIC',featured,4)
        self.assertTrue(0 <= result['current_probability'] <= 1)
        self.assertEqual(result['latest_date'], str(dates[-1].date()))
        self.assertEqual(len(result['validation_rows']),9)
        self.assertEqual(result['test_row']['period'], 'historical_holdout')
        self.assertEqual(len(result['test_truth']),len(result['test_probability']))

    def test_archived_summary_is_internally_consistent(self):
        summary = json.loads((ROOT/'examples/historical_summary.json').read_text())
        self.assertEqual(summary['symbols']*len(summary['horizons_sessions']),summary['asset_horizon_pairs'])
        self.assertEqual(summary['pairs_passing_research_gate']+summary['pairs_with_insufficient_evidence'],
                         summary['asset_horizon_pairs'])
        self.assertEqual(summary['split_boundary_checks'],summary['asset_horizon_pairs']*4)


if __name__ == '__main__':
    unittest.main()
