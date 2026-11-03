import importlib.util,pathlib,tempfile,unittest
p=pathlib.Path(__file__).with_name('stream_goodput.py')
spec=importlib.util.spec_from_file_location('metric',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cfg={'text_tps':4.8,'audio_equivalent_tps':3.3,'audio_unit_s':.02,'lateness_decay_s':1.,'buffer_full_credit_s':10.,'buffer_decay_s':20.,'mu':3.3,'beta':3.3,'metric_version':'soft_ttft_v2'}
row={'request_path':'audio','source_request_id':'r','ttfp_slo_ms':1000.,'slo_ms':10000.,'slo_tier':'standard'}
def result(first=1.,status='success',done=True):
 return {'status':status,'done_received':done,'audio_chunks':[{'arrival_monotonic_s':first,'duration_s':2.}], 'ttfp_s':first,'e2e_s':3.}
class MetricTests(unittest.TestCase):
 def test_on_time(self):self.assertAlmostEqual(m.request_score(row,result(),cfg)['contribution'],6.6)
 def test_soft_late(self):self.assertAlmostEqual(m.request_score(row,result(1.5),cfg)['contribution'],4.95)
 def test_no_complete_gate(self):self.assertAlmostEqual(m.request_score(row,result(status='error',done=False),cfg)['contribution'],6.6)
 def test_floor(self):self.assertEqual(m.request_score(row,result(4.),cfg)['contribution'],0.)
 def test_old_unchanged(self):
  old=dict(cfg);old.pop('metric_version');self.assertEqual(m.request_score(row,result(1.5),old)['contribution'],0.)
 def test_beta_monotonic(self):
  vals=[m.request_score(row,result(1.5),dict(cfg,beta=b))['contribution'] for b in [1.65,3.3,6.6,13.2]]
  self.assertEqual(vals,sorted(vals,reverse=True))
 def test_rebuffer(self):
  s=m.score_units([(0.,.5,1.65),(1.,.5,1.65)],cfg)
  self.assertAlmostEqual(s['rebuffer_s'],.5);self.assertAlmostEqual(s['utility'],2.475);self.assertAlmostEqual(s['net'],.825)
 def test_missing(self):self.assertEqual(m.request_score(row,None,cfg)['contribution'],0.)
unittest.main()
