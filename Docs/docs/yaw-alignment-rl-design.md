**RL design for correcting yaw misalignment**

Status: the first single-turbine PyWake version is implemented in `WindGym/yaw_alignment`, with observation history, actuator limits, PPO training and paired evaluation. See [the runnable example](../../examples/yaw_alignment/README.md). Numerical settings below are initial simulation settings, not turbine operating recommendations. Dynamic-wake, multiple-turbine and higher-fidelity validation stages remain future work; no control-performance claim follows from a smoke test.

Start with one turbine operating below rated power. The objective is to reduce persistent physical yaw misalignment while avoiding unnecessary nacelle motion. Test energy production separately. Intentional farm wake steering is a different objective.

1. **Define the physical problem and the hidden state.**

   Use degrees for logging and explicitly convert to radians for trigonometric functions. Let theta be the true local incoming wind direction and psi the physical nacelle heading, both expressed as clockwise bearings in the same reference frame. Define:

   ```text
   wrap360(x) = x mod 360
   wrap180(x) = (x + 180) mod 360 - 180
   true_error = wrap180(theta - psi)
   measured_theta = wrap360(theta + direction_bias + direction_noise)
   measured_psi = wrap360(psi + encoder_bias + encoder_noise)
   measured_error = wrap180(measured_theta - measured_psi)
   ```

   Positive true_error means a clockwise nacelle movement reduces error when wind is fixed. A clockwise increase of 1 degree changes an error of +10 degrees to +9 degrees. At the 0/360 boundary, theta=1 degree and psi=359 degrees must give +2 degrees.

   If the turbine provides a relative nacelle-vane signal directly, model that sensor directly instead of inventing two independent sensors. Later replace constant direction_bias with a calibrated function of wind speed, rotor operation, and misalignment if data support it.

   The simulator's hidden state includes actual wind, sensor bias, physical heading, actuator state and, for multiple turbines, evolving wakes. The policy observes imperfect measurements. This is a partially observed control problem: a single reading cannot generally distinguish real misalignment from sensor bias.

   Sample initial physical error and sensor bias independently. Randomize the absolute wind bearing across the full circle between single-turbine episodes, even when wind is steady within an episode. Otherwise the policy could memorize a fixed compass heading from the encoder and bypass the sensor-bias problem. Never create the training shortcut that every +10-degree reading corresponds to the same correction. Power and movement history can help identify the bias, but do not guarantee observability in all operating conditions. Begin below rated power with an isolated turbine; include additional reference measurements if the operational problem is unobservable from the available signals.

2. **Build the observation and action interfaces.**

   Each observation contains 12 consecutive ten-second windows. One window has the following 10 features; flatten the history to 120 float32 values for the first policy.

   | Feature group | Values per window | Purpose |
   |---|---|---|
   | Measured relative direction | sin(measured_error), cos(measured_error) | Represent circular alignment measurements without a wrap discontinuity |
   | Measured nacelle heading | sin(measured_psi), cos(measured_psi) | Describe physical orientation using the encoder |
   | Operating measurements | normalized measured wind speed, normalized measured power | Help distinguish changing inflow from response to yaw movement |
   | Previous movement and request | signed encoder displacement in the previous interval / 1 degree, previous requested direction in {-1,0,+1} | Show what was requested and what physically happened |
   | Actuator readiness | normalized remaining motion-inhibit time, yaw-enabled flag | Let the policy observe when movement is unavailable |

   Use circular averaging for direction signals and interval averages for power and wind speed. Calculate signed encoder displacement with angle unwrapping. Accumulate encoder path length separately for the movement penalty; net displacement would miss back-and-forth travel.

   Normalize using declared engineering scales and training-only statistics; freeze any fitted normalization during validation and testing. Use feature bounds that account for measurement noise, and log out-of-range values. The policy must not receive true_error, sampled bias, clean_obs, future wind, training reward, or a simulator-only reference controller output. Its critic uses the same observation history as its actor.

   Generate sensor errors on the causal sensor time series before forming histories. A persistent sensor bias is shared across that sensor's samples; do not draw a different persistent bias for each history slot. Handle missing/invalid sensor data explicitly; the first lab assumes valid data throughout.

   Use a discrete action space for the first policy:

   | Gym action ID | Requested physical movement |
   |---|---|
   | 0 | Counterclockwise, up to 1 degree |
   | 1 | Hold heading |
   | 2 | Clockwise, up to 1 degree |

   Decide every 10 simulated seconds. For an illustrative rate limit of 0.3 degrees/second, a 1-degree request takes about 3.33 seconds, followed by a hold for the remainder of the interval. Integrate partial final movements correctly. Replace this example rate with the turbine's documented rate and actuator behavior.

   The actuator governor enforces rate limits, physical travel limits, motor/brake state and any required dwell time before applying a request. A prohibited request produces an enforced hold or limited movement and is logged alongside actual movement. Rate and travel limits are enforced physically, not only through reward penalties. No step may rotate the nacelle automatically to the true wind direction.

3. **Specify the reward and the policy.**

   For the first simulation experiment, use an alignment reward with privileged simulator truth. This directly teaches the stated alignment objective:

   ```text
   alignment_loss = mean_over_interval(
       [max(abs(true_error) - 2 degrees, 0) / 10 degrees]^2
   )
   travel = total_actual_encoder_path_length_in_interval / 1 degree
   starts = number_of_actual_stopped_to_moving_transitions_in_interval
   reward = -alignment_loss - 0.01 * travel - 0.02 * starts
   ```

   The 2-degree tolerance and penalty weights are lab choices. They create a region where holding earns zero alignment loss and unnecessary movement costs reward. An unchanged 10-degree error gives -0.64 alignment reward per interval; at 1 degree, holding gives zero. Sweep penalty weights on validation scenarios to determine how much alignment is traded for less movement. These costs do not measure structural fatigue.

   Average error loss and integrate energy across the complete action interval, including movement and settling. Use substep data so the agent cannot earn a favorable score only at the interval endpoint. Starts persist across interval boundaries: continuous movement across a boundary does not count as a new start.

   True error is available to the reward calculator during simulation training, but never to the policy or its observation history. This is permissible training supervision, not a deployable sensor. At deployment, the frozen policy needs observations and the actuator governor; reward calculation and the critic are not required. Online training on a real turbine would require a validated alignment reference or a different, measurable reward.

   Evaluate energy as a separate outcome using the time integral of power. If a later objective is energy optimization, a separate experiment can replace the alignment term with normalized interval energy, E_interval / (P_rated * interval_duration), while retaining movement penalties. Do not silently substitute farm energy for alignment: that can reward intentional misalignment for wake steering. Do not reward zero biased measured_error.

   Use PPO from Stable-Baselines3 with an MLP actor and critic:

   ```text
   120 history features -> actor: 64 tanh -> 64 tanh -> probabilities of 3 actions
                       -> critic: 64 tanh -> 64 tanh -> predicted future return
   ```

   The actor chooses movements. The critic estimates future discounted reward to guide training. PPO collects trajectories, estimates whether actions worked better or worse than expected, and updates the networks with a clipped policy objective. The objective is expected discounted return, E[sum_t gamma^t * reward_t].

   Initial training settings: learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, n_steps=512 per environment, batch_size=64, n_epochs=10, clip_range=0.2, ent_coef=0.01, four environments and CPU execution. These are a starting configuration, not validated tuning. With ten-second decisions, gamma=0.99 discounts a reward 60 seconds later by approximately 0.94. Retune the physical horizon if decision timing changes.

   During simulation training, sample from the actor's action distribution. Evaluate the frozen policy with deterministic prediction. Check that evaluation does not update model weights or normalization. Plain SB3 PPO is not recurrent; history stacking provides temporal input. Consider a recurrent policy only if the history comparison shows a material limitation.

4. **Implement the environment contract in WindGym.**

   Recommended separation of responsibilities:

   ```text
   sensor history -> PPO actor -> physical yaw request -> actuator governor
          ^                                                |
          |                                                v
      sensor model <- WindGym physics and coordinate adapter
                              |
                              v
                       training reward -> PPO training update
   ```

   Keep the physics, sensor model, controller, reward calculation and evaluator independently replaceable. Start with PyWake for controlled single-turbine exercises. Use DYNAMIKS for wind transients and wake propagation. PyWake's steady flow solution does not by itself validate dynamic wake control or structural loads.

   The outer Gymnasium environment exposes Discrete(3) and a 120-value Box observation space. In the first implementation, a physical-heading plant calls WindGym's shared PyWake flow adapter directly, rather than passing through the existing WindFarmEnv controller. It converts physical heading to PyWake's yaw coordinate as yaw=wind_direction-heading. The governor and coordinate conversion operate before measurements are returned. This keeps one owner of physical yaw movement and avoids the existing controller's wind-relative updates.

   In the existing WindFarmEnv, ActionMethod='yaw' is incremental, while ActionMethod='wind' commands an offset expressed in the simulator's wind-relative coordinates. ActionMethod='absolute' raises NotImplementedError. The first alignment implementation bypasses these controllers. Tests verify heading invariance under a held command when wind crosses north, rather than assuming a mode name proves physical behavior.

   A normalized zero target in 'wind' mode must not become an oracle that aligns a biased-sensor controller to the true wind. Use true wind inside the physics/coordinate conversion only, never to decide a physical heading command. Verify the adapter's sign convention against the equations above and the actual backend; do not assume both use the same yaw sign.

   The sensor design follows the persistent-bias concepts in core/measurement_manager.py and implements an explicit per-sensor time series in yaw_alignment/sensors.py. This ensures that every historical sample of one sensor shares its persistent bias and that comparisons use identical noise innovations. The existing 'noise' configuration field alone is not a complete sensor-error experiment.

   Implement the proposed alignment reward in the outer environment; it is not a built-in RewardCalculator mode. Derive true local rotor misalignment explicitly, because the backend's yaw relative to global wind may differ from local inflow under wakes. Use actual mechanical rotation for travel costs, excluding changes caused solely by a wind-relative coordinate transformation.

   The first implementation uses simulation_seconds=1 and decision_seconds=10, independently of WindFarmEnv's dt_env/delay aggregation. All action substeps enter its reward and energy calculation. If integrating the existing WindFarmEnv path later, inspect aggregation carefully when delay exceeds dt_env: it retains only the final observation window for agent mean power, while baseline averaging can span the full action interval.

   reset(seed) should sample the scenario, sensor biases and initial orientation; reset actuator/sensor/history state; complete flow settling and two minutes of causal sensor prehistory under a specified hold; then return (observation, info). Reset every controller on the same scenario and identical initial history. Exclude the common prehistory from scored episode metrics.

   step(action) should apply governed physical movement; integrate the simulator and sensors for ten seconds; compute interval reward and diagnostic metrics; update history; and return (observation, reward, terminated, truncated, info).

   Use 30 minutes of scored simulated operation per initial episode: 180 policy decisions. Set truncated=True at the time limit. Do not end an episode merely because alignment has been achieved; the controller must maintain it. The first lab excludes equipment failures and other operational modes. Numerical simulator failures invalidate the run and should raise an error rather than yield a favorable terminal transition. If shutdown scenarios are added later, model the ensuing downtime so early termination cannot avoid future costs.

   Log true_error, measured_error, sampled_bias, requested_action, applied_movement, yaw_travel, yaw_starts, energy and each reward component. True error, sampled bias and reward components remain diagnostic/training data only; measured signals and previous requests enter the policy only as declared in its observation interface. Keep a separate evaluation script that never feeds diagnostic truth back to the policy.

5. **Train and evaluate in stages.**

   | Stage | Scenario | Evidence to collect |
   |---|---|---|
   | Interface check | Fixed wind; known positive/negative errors; zero sensor bias | Correct turn sign, circular angles, rate limits and physical heading under hold |
   | First learning run | One turbine at 8 m/s; steady wind bearing randomized between episodes; independent initial error within +/-15 degrees and bias within +/-10 degrees | Whether the policy learns to correct alignment with imperfect observations |
   | Robustness | Speeds 6-10 m/s; changing direction; illustrative 1-degree direction noise; later calibrated sensor and actuator variations | Held-out performance and the effect of history length |
   | Two turbines | Dynamic wake backend; initially one controller controlling both turbines | Alignment, energy and yaw activity under coupled inflow |
   | Turbine-specific validation | Calibrated sensor/actuator models and an appropriate higher-fidelity simulator | Whether the result persists beyond the training model |

   The ranges above are synthetic starting distributions. Replace them with operational distributions and exclude above-rated, curtailed and faulted conditions until their observation/reward behavior is deliberately modeled.

   Compare against a filtered measured-error yaw controller with deadband and actuator limits, and against the same controller with an available fixed calibration correction. Tune their filters, deadbands and correction on training/validation data only. A zero-bias case checks that RL does not damage an already functioning controller. WindGym's built-in local baseline reads simulated flow directly; label it an ideal-information reference, not the sole realistic comparator.

   Use the same exogenous wind traces, persistent sensor biases, initial states and actuator limits for every paired comparison. For state-dependent sensor errors, share the exogenous noise innovations and error model, not a prerecorded sensor reading that would ignore the controller's different physical state. Use five independent training seeds; choose checkpoints using validation traces and lock separate test traces/bias cases before testing. Do not randomly split neighboring samples from the same trajectory across these sets.

   Report mean absolute true yaw error and its high-percentile tail, fraction of time within the declared tolerance, total energy, yaw travel and motor starts. Show paired differences and uncertainty across independent scenarios/seeds; successive time samples are correlated and are not independent replicates. A training reward curve alone does not establish success. Treat structural load or fatigue claims as untested until a suitable load model has been evaluated.

   First success requires lower held-out physical misalignment than the equally informed conventional baseline while meeting predeclared energy and yaw-activity tolerances. Include steady unbiased cases in the acceptance set. Define tolerances from the study's engineering objectives before training, and report failure to meet them rather than redefining success after seeing results. No numerical performance target is established by this design document.

   For N turbines, start with one centralized policy: observation contains each turbine's sensor history plus measured inflow/layout context; action is MultiDiscrete([3] * N). Average the alignment reward and movement costs across turbines so their scale is stable as N changes. Keep the target at zero local misalignment for correction studies. If a farm supervisor later requests intentional wake steering, explicitly introduce its target offset into observation and the tracking objective; evaluate that as a separate control task.

Sources and verified code entry points:

- [Gymnasium Env API](https://gymnasium.farama.org/api/env/) specifies reset, step, diagnostic info and time-limit semantics.
- [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) documents discrete actions, actor/critic training parameters and observation stacking.
- [Stable-Baselines3 experiment guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) supports separate evaluation, normalization and repeated training runs.
- [WindGym environment](../../WindGym/wind_farm_env.py), [measurement manager](../../WindGym/core/measurement_manager.py), [reward calculator](../../WindGym/core/reward_calculator.py), and [basic controllers](../../WindGym/BasicControllers/BasicControllers.py) were inspected for this design.

Next study task: run full-duration training and compare physical alignment, energy and yaw activity across independent training seeds and held-out scenarios. The implemented smoke configuration verifies execution only.
