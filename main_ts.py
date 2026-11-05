"""
main_ts.py — CLI entry point for DiNof time-series training & evaluation.

Usage:
    # Train on synthetic SWaT data
    python main_ts.py --mode train --workdir runs/swat_vpsde

    # Evaluate anomaly detection
    python main_ts.py --mode eval --workdir runs/swat_vpsde

    # Full demo (generate data → train → eval → plot)
    python demo_run.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'demo'],
                  'What to run: train | eval | demo')
flags.DEFINE_string('workdir', 'runs/swat_vpsde',
                    'Working directory for checkpoints and logs.')
flags.DEFINE_string('data_dir', 'data',
                    'Root data directory (should contain swat_synthetic/).')
flags.DEFINE_enum('source', 'synthetic', ['synthetic', 'csv'],
                  'Data source: synthetic (generated) or csv (real SWaT).')
flags.DEFINE_string('config', 'swat_vpsde',
                    'Config name under configs/ts/ (default: swat_vpsde).')


def main(argv):
    del argv  # unused

    # Dynamic config import
    cfg_module = __import__(
        f'configs.ts.{FLAGS.config}',
        fromlist=['get_config']
    )
    config = cfg_module.get_config()

    if FLAGS.mode == 'train':
        from runner_ts import train_ts
        train_ts(config,
                 workdir=FLAGS.workdir,
                 data_dir=FLAGS.data_dir,
                 source=FLAGS.source)

    elif FLAGS.mode == 'eval':
        from runner_ts import evaluate_anomaly_ts
        evaluate_anomaly_ts(config,
                            workdir=FLAGS.workdir,
                            data_dir=FLAGS.data_dir,
                            source=FLAGS.source)

    elif FLAGS.mode == 'demo':
        # Delegate to demo_run.py logic
        import demo_run
        demo_run.run_demo(config,
                          workdir=FLAGS.workdir,
                          data_dir=FLAGS.data_dir)


if __name__ == '__main__':
    app.run(main)
