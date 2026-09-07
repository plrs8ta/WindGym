**Run the first yaw-alignment experiment**

This version provides a single V80 turbine using WindGym's PyWake backend, a physical yaw actuator, biased/noisy sensors, 120 observation features, PPO training, and paired baseline evaluation. It is a simulation learning framework; a successful smoke run does not demonstrate improved turbine control.

The [initial results and verification record](RESULTS.md) include a completed 100,352-decision run. Its held-out performance remains behind the conventional baseline.

Run the commands below from the repository root with the project's Python environment. Each training/evaluation output directory must be new, so existing results cannot be overwritten accidentally.

1. **Check the environment.**

   ```bash
   .venv/bin/python -m WindGym.yaw_alignment check --config examples/yaw_alignment/config.yaml
   ```

   Expected result: Gymnasium and SB3 checks pass; observation shape `(120,)`, three actions.

2. **Run a small training smoke test.**

   ```bash
   .venv/bin/python -m WindGym.yaw_alignment train \
     --config examples/yaw_alignment/smoke.yaml \
     --output runs/yaw_alignment/my-smoke \
     --steps 256 --n-envs 1 --rollout-steps 64 \
     --eval-every 128 --validation-episodes 2
   ```

   This checks learning, validation and checkpoint saving. The smoke configuration has 60-second simulated episodes, so its scores are not an alignment benchmark. PPO finishes whole rollouts: `training.json` records requested and actual decision counts. Validation occurs after optimizer updates at rollout boundaries, including the final update; `validation.csv` records the completed PPO epochs. The run manifest also records library versions and source hashes.

3. **Evaluate the saved policy against baselines.**

   ```bash
   .venv/bin/python -m WindGym.yaw_alignment evaluate \
     --run runs/yaw_alignment/my-smoke \
     --output runs/yaw_alignment/my-smoke/evaluation --episodes 4
   ```

   Evaluation loads the best validation checkpoint and uses a separate fixed test-seed set. PPO, filtered conventional yaw control, and hold receive identical scenario/noise seeds and actuator limits. One quarter of test cases have zero direction and encoder bias. A fixed calibration baseline can be added with `--calibration-deg`; this must come from calibration/validation work, not hidden test biases. Conventional filter and deadband parameters are exposed as `--filter-seconds` and `--deadband-deg` and should be tuned before final testing.

4. **Start a learning experiment with full episodes.**

   ```bash
   .venv/bin/python -m WindGym.yaw_alignment train \
     --config examples/yaw_alignment/config.yaml \
     --output runs/yaw_alignment/first-study --steps 100000 --seed 42
   ```

   The normal configuration uses 30 minutes of simulated operation per episode. Training runtime depends on measured simulator throughput; simulated minutes are not wall-clock minutes. Keep the first study at steady 8 m/s with randomized wind bearing, sensor bias and initial error. Repeat with independently chosen training seeds before making performance claims. The supplied `robustness.yaml` adds synthetic noise, varying wind speed and changing wind direction for a subsequent experiment.

5. **Inspect physical metrics.**

   | Output | What it contains |
   |---|---|
   | `config.json`, `training.json` | Environment settings, seed sets and run status |
   | `model.zip`, `best_model.zip` | Final and validation-selected PPO checkpoints |
   | `validation.csv` | Validation reward and mean physical error over training |
   | `evaluation/episodes.csv` | Per-controller, per-scenario error, energy, travel and motor starts |
   | `evaluation/summary.json`, `trajectory.csv` | Paired comparison, scenario uncertainty, and substep traces from one common case |

   Energy is integrated over the complete scored episode. Lower alignment error alone does not establish an acceptable energy/movement tradeoff. The reported confidence intervals describe variation over scenarios for one trained policy, not uncertainty across training seeds. The evaluator does not label a controller successful without engineering acceptance thresholds.

**Use the environment from Python**

```python
from WindGym.yaw_alignment import AlignmentConfig, Scenario, YawAlignmentEnv

env = YawAlignmentEnv(AlignmentConfig())
try:
    obs, info = env.reset(
        seed=42,
        options={"scenario": Scenario(
            wind_direction_deg=270, wind_speed_mps=8,
            initial_error_deg=10, direction_bias_deg=5,
        )},
    )
    obs, reward, terminated, truncated, info = env.step(2)  # clockwise
    print(info["true_error_deg"], info["yaw_travel_deg"])
finally:
    env.close()
```

The policy receives only `obs`. `info` contains simulator truth for reward auditing and evaluation and must not be passed to the policy. Action IDs are `0=counterclockwise`, `1=hold`, `2=clockwise`. Positive true error means a clockwise motion improves alignment in fixed wind.

The observation consists of 12 causal windows with ten features in the declared `env.feature_names` order. Direction and heading use sine/cosine; wind speed is divided by 25 m/s, power by turbine rated power, and encoder displacement by the requested step size. Input saturation bounds are explicit in `sensors.py`, and saturation counts are logged. Historical sensor values are retained, not regenerated with new noise.

The reward is negative squared true error outside a 2-degree tolerance, minus actual yaw travel and motor-start penalties. Ground truth is permitted for this simulation-training reward and remains excluded from policy inputs. The deployed inference computation would require only the trained actor, measured history and the actuator governor; live reward-driven adaptation is outside this version.

**Code responsibilities**

| Module | Responsibility |
|---|---|
| `config.py` | Validated configuration, independent scenario sampling and wind traces |
| `plant.py` | WindGym PyWake adapter, heading conversion, actuator mechanics |
| `sensors.py` | Persistent sensor bias, causal noise, feature assembly |
| `env.py` | Gymnasium reset/step, history, interval reward and diagnostics |
| `controllers.py`, `experiment.py` | Conventional controls, PPO training and paired evaluation |

The new environment owns physical heading directly instead of calling the existing WindFarmEnv yaw controller. Both reuse WindGym's PyWake flow adapter. This avoids a wind-relative target or coordinate update automatically supplying the correct wind alignment.

Current scope is one turbine with a steady-state aerodynamic solution at each simulation substep. Changing wind traces are supported, but transient wake transport, multiple turbines, HAWC2/OpenFAST loads, hardware deployment and structural-fatigue validation remain later stages. The actuator is a simplified rate/rest/travel model, and numerical integration samples the flow at each substep endpoint.

To run the focused tests on this machine:

```bash
.venv/bin/python -m pytest -p no:capture tests/test_yaw_alignment.py --no-cov -q
```

`-p no:capture` avoids this local Python runtime's `readline` import crash in pytest's capture initialization. It does not disable any tests. On an unaffected Python installation, ordinary pytest capture can be used.
