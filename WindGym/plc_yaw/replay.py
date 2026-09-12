"""Read measured PLC inputs, execute scans, and export auditable timelines."""

import csv
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from .config import SIGNAL_NAMES, PLCInputs, PLCYawConfig
from .controller import PLCYawController
from .env import PLCYawEnv, PLCYawScenario
from .reference import provenance


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_trace(output, rows, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "trace.csv", rows)
    transitions = []
    keys = [
        "gbStartYawMotorCW",
        "gbStartYawMotorCCW",
        "DQ_NacelleStartYawCW",
        "DQ_NacelleStartYawCCW",
        "DQ_NacelleOpenElectricalYawBrake",
        "physical_phase",
        "logical_phase",
        "logical_inhibit_5s",
        "physical_inhibit_22s",
        "restart_release",
    ]
    previous = None
    for row in rows:
        state = tuple(row[k] for k in keys)
        if state != previous or row["events"]:
            transitions.append({k: row[k] for k in ["time_s", *keys, "events"]})
        previous = state
    write_csv(output / "events.csv", transitions)
    metadata = {**metadata, **provenance()}
    (output / "summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    plot_trace(output / "execution.svg", rows)
    return output


def plot_trace(path, rows):
    """Standalone scientific timeline; no web dependencies or server required."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [row["time_s"] for row in rows]
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, layout="constrained")
    for key, label in [
        ("grVaneDirection_1sec", "vane 1 s"),
        ("grVaneDirection_30sec", "vane 30 s"),
        ("grVaneDirection_5min", "vane 5 min"),
    ]:
        axes[0].plot(t, [row[key] for row in rows], label=label)
    if "true_error_deg" in rows[0]:
        axes[0].plot(
            t,
            [row["true_error_deg"] for row in rows],
            "--",
            label="true error (simulation)",
        )
    axes[0].axhline(1, color="gray", lw=0.8)
    axes[0].axhline(-1, color="gray", lw=0.8)
    axes[0].set_ylabel("Yaw error [deg]")
    for key, label, offset in [
        ("gbStartYawMotorCW", "logical CW", 3),
        ("gbStartYawMotorCCW", "logical CCW", 2),
        ("DQ_NacelleStartYawCW", "motor CW", 1),
        ("DQ_NacelleStartYawCCW", "motor CCW", 0),
    ]:
        axes[1].step(
            t, [int(row[key]) * 0.7 + offset for row in rows], where="post", label=label
        )
    axes[1].set_yticks(
        [0, 1, 2, 3], ["motor CCW", "motor CW", "logical CCW", "logical CW"]
    )
    for key, label, offset in [
        ("DQ_NacelleOpenElectricalYawBrake", "electrical brake OPEN", 3),
        ("DQ_NacelleValveOpenYawBrake1", "valve 1", 2),
        ("DQ_NacelleValveOpenYawBrake2", "valve 2", 1),
        ("DQ_NacelleValveOpenYawBrake3", "valve 3", 0),
    ]:
        axes[2].step(
            t, [int(row[key]) * 0.7 + offset for row in rows], where="post", label=label
        )
    axes[2].set_yticks([0, 1, 2, 3], ["valve 3", "valve 2", "valve 1", "brake OPEN"])
    for key, label, offset in [
        ("logical_inhibit_5s", "5 s inhibit", 2),
        ("physical_inhibit_22s", "22 s inhibit", 1),
        ("restart_release", "restart released", 0),
    ]:
        axes[3].step(
            t, [int(row[key]) * 0.7 + offset for row in rows], where="post", label=label
        )
    axes[3].set_yticks([0, 1, 2], ["restart release", "22 s inhibit", "5 s inhibit"])
    axes[3].set_xlabel("Time [s] — PLC outputs apply from each scan timestamp")
    axes[0].legend(loc="upper right", ncols=2, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "PLC automatic yaw execution | illustrative parameters, not a site-validated trace"
    )
    fig.savefig(path)
    plt.close(fig)


def load_input_csv(path, config):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        unknown = set(headers) - {"time_s", *SIGNAL_NAMES}
        required = {
            "time_s",
            "grVaneDirection_1sec",
            "grVaneDirection_30sec",
            "grVaneDirection_5min",
            "grWindSpeed_30sec",
            "grWindSpeed_5sec",
            "grNacelleYawLoopPressure_1sec",
            "grCableTwistTotal",
        }
        if unknown or required - set(headers) or len(set(headers)) != len(headers):
            raise ValueError(
                f"Invalid CSV columns: unknown={unknown}, missing={required - set(headers)}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError("Replay CSV is empty")
    state = asdict(PLCInputs())
    events = []
    previous_ms = -1
    for index, row in enumerate(rows):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row {index + 2} has a different column count")
        time = float(row["time_s"])
        if (
            not math.isfinite(time)
            or time < 0
            or not math.isclose(
                time * 1000 / config.scan_ms, round(time * 1000 / config.scan_ms)
            )
        ):
            raise ValueError("CSV time_s must be finite nonnegative scan boundaries")
        now_ms = round(time * 1000)
        if now_ms <= previous_ms or (index == 0 and now_ms != 0):
            raise ValueError("CSV must start at 0 with strictly increasing timestamps")
        previous_ms = now_ms
        for plc, name in SIGNAL_NAMES.items():
            value = row.get(plc, "")
            if value is None:
                raise ValueError(f"Missing CSV value for {plc} at row {index + 2}")
            if not value.strip():
                if index == 0 and plc in required:
                    raise ValueError(f"First row requires {plc}")
                continue  # Documented zero-order hold, flags default to PLCInputs.
            value = value.strip()
            if isinstance(state[name], bool):
                if value.lower() not in ("0", "1", "true", "false"):
                    raise ValueError(f"{plc} must be 0/1/true/false")
                state[name] = value.lower() in ("1", "true")
            else:
                state[name] = float(value)
        events.append((now_ms, PLCInputs(**state)))
    return events


def replay(path, output, config=None):
    config = config or PLCYawConfig()
    events = load_input_csv(path, config)
    controller = PLCYawController(config)
    rows = []
    event_index = 0
    for now in range(0, events[-1][0] + config.scan_ms, config.scan_ms):
        while event_index + 1 < len(events) and events[event_index + 1][0] <= now:
            event_index += 1
        inputs = events[event_index][1]
        row = controller.scan(inputs)
        row.update({plc: getattr(inputs, name) for plc, name in SIGNAL_NAMES.items()})
        rows.append(row)
    return write_trace(
        output,
        rows,
        {
            "mode": "open_loop_input_replay",
            "config": config.to_dict(),
            "source_csv": Path(path).name,
            "scans": len(rows),
            "initial_inputs": asdict(events[0][1]),
            "note": "Predicted outputs; no measured output comparison or plant feedback",
        },
    )


def demo(output, config=None):
    config = config or PLCYawConfig(episode_seconds=180)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("normal_cw", config, PLCYawScenario(), []),
        ("normal_ccw", config, PLCYawScenario(initial_error_deg=-18), []),
        ("wind_time_stop", replace(config, stop_mode=2), PLCYawScenario(), []),
        ("pressure_stop", replace(config, stop_mode=3), PLCYawScenario(), []),
        (
            "pressure_timeout",
            replace(config, stop_mode=3, pressure_rise_per_second=0),
            PLCYawScenario(initial_pressure=0),
            [],
        ),
        (
            "wind_loss",
            config,
            PLCYawScenario(),
            [{"time_s": 45, "signals": {"active_wind": False}}],
        ),
        (
            "electrical_open",
            config,
            PLCYawScenario(),
            [{"time_s": 45, "signals": {"nacelle_open_electrical": True}}],
        ),
        (
            "mode_exit",
            config,
            PLCYawScenario(),
            [{"time_s": 45, "signals": {"auto_mode": False}}],
        ),
    ]
    summary = []
    for name, settings, scenario, events in cases:
        env = PLCYawEnv(settings)
        try:
            env.reset(seed=42, options={"scenario": scenario, "events": events})
            rows = []
            done = False
            while not done:
                _, _, _, done, info = env.step([0.0])
                rows.extend(info["trace"])
            write_trace(
                output / name,
                rows,
                {
                    "config": settings.to_dict(),
                    "scenario": asdict(scenario),
                    "events": events,
                    "seed": 42,
                    "action": "zero vane correction (reference controller)",
                    "metrics": info["episode_metrics"],
                    "plant_model": "synthetic yaw-rate and pressure",
                },
            )
            summary.append(dict(case=name, **info["episode_metrics"]))
        finally:
            env.close()
    write_csv(output / "summary.csv", summary)
    return output
