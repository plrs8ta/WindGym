**First-version implementation and initial training results**

The environment, PPO trainer, checkpoint selection, and paired evaluator run successfully. The initial trained policy does not outperform the configured conventional yaw controller. These results describe a synthetic simulation experiment, not turbine field performance.

The local study used the supplied normal configuration: one V80 turbine, steady 8 m/s wind within each episode, random absolute wind bearing, independently sampled initial error and sensor bias, 12 historical windows, and 30 minutes of scored simulated operation. Training seed was 42. PPO completed 100,352 decisions (whole-rollout rounding of the requested 100,000), using four environment copies. Training and validation took approximately 203 seconds on this machine; this is not a portable runtime guarantee.

Checkpoint selection used five fixed validation cases with seeds 1,000,000 through 1,000,004. The best reward checkpoint was selected at 92,160 decisions after 450 PPO optimization epochs. The final model was also saved separately. No test metrics were used to select the checkpoint.

The final comparison used 20 fresh test cases with seeds 3,000,000 through 3,000,019. A quarter were unbiased cases. Each controller received matching wind, initialization, sensor-noise seeds and actuator limits. Earlier execution pilots used a different test-seed range.

| Mean over 20 test episodes | PPO | Filtered yaw controller | Hold |
|---|---:|---:|---:|
| Absolute physical yaw error | 4.20 degrees | 3.37 degrees | 6.33 degrees |
| Energy per 30-minute episode | 343.97 kWh | 345.23 kWh | 339.60 kWh |
| Actual yaw travel per episode | 56.20 degrees | 6.55 degrees | 0 degrees |
| Motor starts per episode | 56.20 | 6.55 | 0 |
| Time within 2-degree tolerance | 28.78% | 39.83% | 15.00% |

PPO's paired mean energy change relative to the filtered controller was -0.366%, with a scenario-bootstrap 95% interval of approximately [-0.827%, +0.022%]. Its paired mean yaw-error change was +0.838 degrees, with interval [-0.207, +1.940]. These intervals describe scenario variation conditional on this one trained policy; they do not measure variation across training seeds.

The practical limitation is excessive yaw movement and poorer held-out alignment than the conventional baseline. Further work should use validation scenarios to study motion penalties, observation history and training stability, and repeat independent training seeds. If these final test outcomes inform later tuning, use a new final test set. No engineering acceptance threshold or improvement claim is established by this run.

Verified implementation checks:

1. 25 new tests passed, covering physical yaw sign, 0/360 wrapping, fixed heading under changing wind, rate/rest/travel limits, full-interval reward and energy, persistent sensor bias, paired noise, history integrity, hidden-truth exclusion, numerical failures, checkpoints and frozen evaluation.
2. The related reward-calculator and baseline-controller suites passed: 26 additional tests.
3. Gymnasium and Stable-Baselines3 environment checks passed with the 120-value observation and three-action interface. A noisy, changing-wind episode also completed successfully.
4. The wheel built successfully; importing from its extracted contents, resetting, and completing an episode worked independently of the source checkout.
5. Ruff formatting and lint passed for the new Python files, with N999 excluded for the repository's established `WindGym` package spelling. The working changes passed whitespace checks.

Run artifacts are local and intentionally ignored by Git:

- `runs/yaw_alignment/first-study/training.json`
- `runs/yaw_alignment/first-study/validation.csv`
- `runs/yaw_alignment/first-study/best_model.zip`
- `runs/yaw_alignment/first-study/evaluation/summary.json`
- `runs/yaw_alignment/first-study/evaluation/episodes.csv`

Use the [run guide](README.md) to reproduce the workflow with a new output directory.
