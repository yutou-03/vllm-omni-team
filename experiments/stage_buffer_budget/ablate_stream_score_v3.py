"""Post-run score decomposition; no counterfactual scheduling claims."""
import argparse, json, math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root
    arms = {}
    for name in ('nohold16', 'buffer16', 'dynamic16', 'fixed16'):
        path = root / name / 'stream_score.json'
        if not path.exists():
            continue
        score = json.loads(path.read_text())
        cfg, rows = score['metric_config'], score['requests']
        T = score['arrival_window_s']
        if not all(r['complete'] and r['rebuffer_s'] == 0 for r in rows):
            raise ValueError('w-only ablation requires complete streams without playback stalls')
        # With no stalls, all generated/read units are on the ideal playback
        # schedule. Removing w(B) gives h=1 for each unit, hence raw_units.
        tiers = {}
        for tier in sorted({r['tier'] for r in rows}):
            subset = [r for r in rows if r['tier'] == tier]
            tiers[tier] = {
                'requests': len(subset),
                'visible_units': sum(r['raw_units'] for r in subset),
                'weighted_utility': sum(r['utility'] for r in subset),
                'buffer_weight_loss': sum(r['raw_units'] - r['utility'] for r in subset),
                'first_output_penalty': sum(r['first_penalty'] for r in subset),
                'score_contribution': sum(r['contribution'] for r in subset),
            }
        w_disabled = sum(max(0, r['raw_units'] - r['first_penalty']) for r in rows) / T
        arms[name] = {
            'T_s': T,
            'G_frozen': score['G_stream'],
            'G_without_unread_buffer_weight': w_disabled,
            'G_without_first_output_penalty': sum(r['utility'] for r in rows) / T,
            'weighted_buffer_loss_per_T': sum(r['raw_units'] - r['utility'] for r in rows) / T,
            'first_output_penalty_per_T': sum(r['first_penalty'] for r in rows) / T,
            'mean_peak_client_unread_s': sum(r['peak_client_buffer_s'] for r in rows) / len(rows),
            'tiers': tiers,
            'all_complete_no_stall': True,
        }
    comparisons = {}
    for arm, base in [('buffer16', 'nohold16'), ('dynamic16', 'nohold16'),
                      ('dynamic16', 'buffer16')]:
        if arm in arms and base in arms:
            comparisons[f'{arm}_vs_{base}'] = {
                label: 100 * (arms[arm][key] / arms[base][key] - 1)
                for label, key in [('frozen_gain_pct', 'G_frozen'),
                                   ('w_disabled_gain_pct', 'G_without_unread_buffer_weight')]
            }
    result = {'scope': 'offline score ablation of identical saved client traces',
              'warning': 'Removing w(B) changes the scoring definition; it does not replay a different scheduling/load trajectory.',
              'arms': arms, 'comparisons': comparisons}
    (root / 'score_ablation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'comparisons': comparisons,
                      'mean_peak_client_unread_s': {k:v['mean_peak_client_unread_s'] for k,v in arms.items()}}, indent=2))

if __name__ == '__main__':
    main()
