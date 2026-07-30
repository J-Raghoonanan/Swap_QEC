"""
Compare end-only and interleaved rho2 purification at matched copy budget.

The two schedules are

    end-only:    P^T E^T(rho_0)
    interleaved: (P E)^T(rho_0),

where E is one round of Z dephasing and
P(rho) = rho^2 / Tr(rho^2).  Both purified schedules therefore use T
binary-tree purification levels, N = 2^T leaf copies, and N - 1 pairwise
merges.  No Clifford twirling is applied in this focused comparison.

Run from the repository root with

    python -m src.simulation.rho2_sims.interleaving_study
    python -m src.simulation.rho2_sims.interleaving_study --p 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit.quantum_info import DensityMatrix, Statevector

from .configs import AASpec, NoiseMode, NoiseSpec, NoiseType, StateKind, TargetSpec
from .noise_engine import apply_noise_to_density_matrix
from .rho2_purification import apply_rho2_purification
from .state_factory import build_target


# Focused study parameters.
M = 1
DEFAULT_DEPHASING_P = 0.3
T = 5

NO_PURIFICATION = "no_purification"
END_ONLY = "end_only_matched"
INTERLEAVED = "interleaved_matched"


def _output_paths(dephasing_p: float) -> tuple[Path, Path, Path]:
    """Return parameter-specific CSV, PDF, and PNG paths."""
    probability_tag = f"{round(100 * dephasing_p):03d}"
    stem = f"matched_budget_M1_plus_p{probability_tag}_T{T}"
    data_path = Path("data/rho2_interleaving") / f"{stem}.csv"
    figure_pdf_path = Path("figures/rho2_interleaving") / f"{stem}.pdf"
    return data_path, figure_pdf_path, figure_pdf_path.with_suffix(".png")


def _fidelity_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    """Return <psi|rho|psi> for a pure target state."""
    value = np.vdot(psi.data, rho.data @ psi.data)
    return float(np.real(value))


def _trace_distance_to_pure(rho: DensityMatrix, psi: Statevector) -> float:
    """Return (1/2) ||rho - |psi><psi|||_1."""
    target_density = np.outer(psi.data, np.conj(psi.data))
    difference = rho.data - target_density
    hermitian_difference = (difference + difference.conj().T) / 2.0
    eigenvalues = np.linalg.eigvalsh(hermitian_difference)
    return 0.5 * float(np.sum(np.abs(eigenvalues)))


def _purity(rho: DensityMatrix) -> float:
    """Return Tr(rho^2)."""
    return float(np.real(np.trace(rho.data @ rho.data)))


def _record(
    rows: List[Dict[str, object]],
    *,
    schedule: str,
    event_index: int,
    noise_rounds_completed: int,
    operation: str,
    purification_round_in_block: int,
    purification_rounds_completed: int,
    rho: DensityMatrix,
    psi: Statevector,
    dephasing_p: float,
) -> None:
    """Append one machine-readable trajectory checkpoint."""
    uses_matched_budget = schedule != NO_PURIFICATION
    input_copies = 2**T if uses_matched_budget else 1
    pairwise_merges = input_copies - 1 if uses_matched_budget else 0
    fidelity = _fidelity_to_pure(rho, psi)

    rows.append(
        {
            "schedule": schedule,
            "event_index": event_index,
            "noise_rounds_completed": noise_rounds_completed,
            "operation": operation,
            "purification_round_in_block": purification_round_in_block,
            "purification_rounds_completed": purification_rounds_completed,
            "M": M,
            "initial_state": "plus",
            "noise_type": NoiseType.dephase_z.value,
            "dephasing_probability_p": dephasing_p,
            "total_noise_rounds_T": T,
            "clifford_twirling": False,
            "purification_depth_budget": T if uses_matched_budget else 0,
            "input_copies": input_copies,
            "pairwise_merges": pairwise_merges,
            "fidelity": fidelity,
            "infidelity": 1.0 - fidelity,
            "purity": _purity(rho),
            "trace_distance": _trace_distance_to_pure(rho, psi),
        }
    )


def run_study(dephasing_p: float = DEFAULT_DEPHASING_P) -> pd.DataFrame:
    """Run the three trajectories and return all event-level data."""
    if not 0.0 <= dephasing_p <= 1.0:
        raise ValueError(f"dephasing probability must be in [0, 1], got {dephasing_p}")

    _, psi = build_target(TargetSpec(M=M, kind=StateKind.hadamard))
    rho_initial = DensityMatrix(psi)
    noise = NoiseSpec(
        noise_type=NoiseType.dephase_z,
        mode=NoiseMode.iid_p,
        p=dephasing_p,
    )
    aa = AASpec()
    rows: List[Dict[str, object]] = []

    # Reference: dephasing only.
    rho = rho_initial.copy()
    _record(
        rows,
        schedule=NO_PURIFICATION,
        event_index=0,
        noise_rounds_completed=0,
        operation="initial",
        purification_round_in_block=0,
        purification_rounds_completed=0,
        rho=rho,
        psi=psi,
        dephasing_p=dephasing_p,
    )
    for noise_round in range(1, T + 1):
        rho = apply_noise_to_density_matrix(rho, noise, twirling=None)
        _record(
            rows,
            schedule=NO_PURIFICATION,
            event_index=noise_round,
            noise_rounds_completed=noise_round,
            operation="noise",
            purification_round_in_block=0,
            purification_rounds_completed=0,
            rho=rho,
            psi=psi,
            dephasing_p=dephasing_p,
        )

    # Matched end-only schedule: all T noise rounds, then all T P rounds.
    rho = rho_initial.copy()
    event_index = 0
    _record(
        rows,
        schedule=END_ONLY,
        event_index=event_index,
        noise_rounds_completed=0,
        operation="initial",
        purification_round_in_block=0,
        purification_rounds_completed=0,
        rho=rho,
        psi=psi,
        dephasing_p=dephasing_p,
    )
    for noise_round in range(1, T + 1):
        event_index += 1
        rho = apply_noise_to_density_matrix(rho, noise, twirling=None)
        _record(
            rows,
            schedule=END_ONLY,
            event_index=event_index,
            noise_rounds_completed=noise_round,
            operation="noise",
            purification_round_in_block=0,
            purification_rounds_completed=0,
            rho=rho,
            psi=psi,
            dephasing_p=dephasing_p,
        )
    for purification_round in range(1, T + 1):
        event_index += 1
        rho, _ = apply_rho2_purification(rho, aa)
        _record(
            rows,
            schedule=END_ONLY,
            event_index=event_index,
            noise_rounds_completed=T,
            operation="purification",
            purification_round_in_block=purification_round,
            purification_rounds_completed=purification_round,
            rho=rho,
            psi=psi,
            dephasing_p=dephasing_p,
        )

    # Matched interleaved schedule: one P round after every noise round.
    rho = rho_initial.copy()
    event_index = 0
    _record(
        rows,
        schedule=INTERLEAVED,
        event_index=event_index,
        noise_rounds_completed=0,
        operation="initial",
        purification_round_in_block=0,
        purification_rounds_completed=0,
        rho=rho,
        psi=psi,
        dephasing_p=dephasing_p,
    )
    for noise_round in range(1, T + 1):
        event_index += 1
        rho = apply_noise_to_density_matrix(rho, noise, twirling=None)
        _record(
            rows,
            schedule=INTERLEAVED,
            event_index=event_index,
            noise_rounds_completed=noise_round,
            operation="noise",
            purification_round_in_block=0,
            purification_rounds_completed=noise_round - 1,
            rho=rho,
            psi=psi,
            dephasing_p=dephasing_p,
        )

        event_index += 1
        rho, _ = apply_rho2_purification(rho, aa)
        _record(
            rows,
            schedule=INTERLEAVED,
            event_index=event_index,
            noise_rounds_completed=noise_round,
            operation="purification",
            purification_round_in_block=1,
            purification_rounds_completed=noise_round,
            rho=rho,
            psi=psi,
            dephasing_p=dephasing_p,
        )

    return pd.DataFrame(rows)


def _initial_fidelity(data: pd.DataFrame) -> float:
    initial = data[
        (data["schedule"] == NO_PURIFICATION)
        & (data["operation"] == "initial")
    ]
    return float(initial.iloc[0]["fidelity"])


def _final_fidelity(data: pd.DataFrame, schedule: str) -> float:
    schedule_data = data[data["schedule"] == schedule]
    final_event = int(schedule_data["event_index"].max())
    final_row = schedule_data[schedule_data["event_index"] == final_event]
    return float(final_row.iloc[0]["fidelity"])


def make_figure(
    data: pd.DataFrame,
    dephasing_p: float,
    figure_pdf_path: Path,
    figure_png_path: Path,
) -> None:
    """Create a trajectory panel and a direct final-fidelity comparison."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
        }
    )

    colors = {
        NO_PURIFICATION: "#707070",
        END_ONLY: "#D55E00",
        INTERLEAVED: "#0072B2",
    }
    initial_fidelity = _initial_fidelity(data)

    baseline = data[
        (data["schedule"] == NO_PURIFICATION)
        & (data["operation"] == "noise")
    ].sort_values("noise_rounds_completed")
    end_noise = data[
        (data["schedule"] == END_ONLY) & (data["operation"] == "noise")
    ].sort_values("noise_rounds_completed")
    interleaved_post = data[
        (data["schedule"] == INTERLEAVED)
        & (data["operation"] == "purification")
    ].sort_values("noise_rounds_completed")

    final_values = {
        NO_PURIFICATION: _final_fidelity(data, NO_PURIFICATION),
        INTERLEAVED: _final_fidelity(data, INTERLEAVED),
        END_ONLY: _final_fidelity(data, END_ONLY),
    }

    figure, (trajectory_ax, final_ax) = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.5),
        gridspec_kw={"width_ratios": [2.15, 1.0]},
    )

    initial_x = np.array([0])
    noise_steps = baseline["noise_rounds_completed"].to_numpy(dtype=int)

    trajectory_ax.plot(
        np.concatenate((initial_x, noise_steps)),
        np.concatenate(([initial_fidelity], baseline["fidelity"].to_numpy())),
        color=colors[NO_PURIFICATION],
        linestyle=":",
        marker="o",
        markersize=4,
        label="No purification",
    )
    trajectory_ax.plot(
        np.concatenate(
            (
                initial_x,
                interleaved_post["noise_rounds_completed"].to_numpy(dtype=int),
            )
        ),
        np.concatenate(
            ([initial_fidelity], interleaved_post["fidelity"].to_numpy())
        ),
        color=colors[INTERLEAVED],
        linewidth=2.1,
        marker="o",
        markersize=4.5,
        label="Interleaved: P after each E",
    )

    # Repeated x=T creates the visible end-only purification jump.
    end_x = np.concatenate(
        (
            initial_x,
            end_noise["noise_rounds_completed"].to_numpy(dtype=int),
            np.array([T]),
        )
    )
    end_y = np.concatenate(
        (
            np.array([initial_fidelity]),
            end_noise["fidelity"].to_numpy(),
            np.array([final_values[END_ONLY]]),
        )
    )
    trajectory_ax.plot(
        end_x,
        end_y,
        color=colors[END_ONLY],
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markersize=4,
        label=f"End-only: {T} P rounds at t={T}",
    )
    trajectory_ax.scatter(
        [T],
        [float(end_noise.iloc[-1]["fidelity"])],
        s=50,
        facecolors="white",
        edgecolors=colors[END_ONLY],
        linewidths=1.5,
        zorder=5,
    )
    trajectory_ax.scatter(
        [T],
        [final_values[END_ONLY]],
        s=55,
        marker="D",
        color=colors[END_ONLY],
        zorder=5,
    )

    trajectory_ax.set_title("Fidelity trajectory")
    trajectory_ax.set_xlabel("Completed dephasing rounds, t")
    trajectory_ax.set_ylabel("State fidelity to |+>")
    trajectory_ax.set_xticks(range(T + 1))
    trajectory_ax.set_xlim(-0.1, T + 0.35)
    trajectory_minimum = min(
        baseline["fidelity"].min(),
        interleaved_post["fidelity"].min(),
        final_values[END_ONLY],
    )
    trajectory_ax.set_ylim(max(0.0, trajectory_minimum - 0.05), 1.015)
    trajectory_ax.axhline(
        0.5,
        color="#999999",
        linestyle="-.",
        linewidth=0.8,
        zorder=0,
    )
    trajectory_ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    trajectory_ax.spines[["top", "right"]].set_visible(False)
    trajectory_ax.legend(loc="upper right", frameon=False)

    schedule_order = [NO_PURIFICATION, INTERLEAVED, END_ONLY]
    bar_labels = ["No\npurification", "Interleaved", "End-only"]
    bar_values = [final_values[schedule] for schedule in schedule_order]
    marker_shapes = ["o", "o", "D"]
    for index, (schedule, value, marker) in enumerate(
        zip(schedule_order, bar_values, marker_shapes)
    ):
        final_ax.scatter(
            index,
            value,
            s=95,
            marker=marker,
            color=colors[schedule],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
    final_ax.set_title(f"Final fidelity after T={T}", pad=25)
    final_ax.set_xticks(np.arange(len(schedule_order)), bar_labels)
    final_ax.set_xlim(-0.35, 2.15)
    final_span = max(bar_values) - min(bar_values)
    final_padding = max(0.004, 0.25 * final_span)
    final_lower = max(0.0, min(bar_values) - final_padding)
    final_upper = min(1.0, max(bar_values) + final_padding)
    final_ax.set_ylim(final_lower, final_upper)
    final_ax.set_ylabel("Final fidelity")
    final_ax.axhline(
        0.5,
        color="#999999",
        linestyle="-.",
        linewidth=0.8,
        zorder=0,
    )
    final_ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    final_ax.set_axisbelow(True)
    final_ax.spines[["top", "right"]].set_visible(False)
    label_offset = 0.025 * (final_upper - final_lower)
    for index, value in enumerate(bar_values):
        final_ax.text(
            index,
            value + label_offset,
            f"{value:.6f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    end_minus_interleaved = final_values[END_ONLY] - final_values[INTERLEAVED]
    final_ax.text(
        0.5,
        1.01,
        f"End-only - interleaved = {end_minus_interleaved:+.6f}",
        transform=final_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        clip_on=False,
    )

    input_copies = 2**T
    figure.suptitle(
        r"Matched-budget $\rho^2$ purification: end-only vs interleaved",
        fontsize=13,
        y=0.995,
    )
    figure.text(
        0.5,
        0.94,
        (
            f"M={M}, |+> initial state, Z dephasing p={dephasing_p:.2f}; "
            f"N={input_copies} leaf copies and {input_copies - 1} pairwise merges"
        ),
        ha="center",
        va="top",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.02, 0.99, 0.90))

    figure_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_pdf_path, bbox_inches="tight")
    figure.savefig(figure_png_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched-budget end-only vs interleaved rho2 study"
    )
    parser.add_argument(
        "--p",
        type=float,
        default=DEFAULT_DEPHASING_P,
        help=f"Z-dephasing probability (default: {DEFAULT_DEPHASING_P})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_path, figure_pdf_path, figure_png_path = _output_paths(args.p)
    data = run_study(args.p)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)
    make_figure(data, args.p, figure_pdf_path, figure_png_path)

    print(f"Saved trajectory data: {data_path}")
    print(f"Saved figure: {figure_pdf_path}")
    print(f"Saved preview: {figure_png_path}")
    print(
        "Final fidelities: "
        f"no purification={_final_fidelity(data, NO_PURIFICATION):.6f}, "
        f"interleaved={_final_fidelity(data, INTERLEAVED):.6f}, "
        f"end-only={_final_fidelity(data, END_ONLY):.6f}"
    )


if __name__ == "__main__":
    main()
