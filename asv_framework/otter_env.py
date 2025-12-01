"""
Lightweight Otter USV simulation wrapper.

This wraps `python_vehicle_simulator.vehicles.otter` and exposes a simple
reset/step interface for data collection and controller testing.
"""
from __future__ import annotations

import numpy as np

from .paths import add_vendor_paths

# Make sure vendor modules are importable
add_vendor_paths()

from python_vehicle_simulator.vehicles.otter import otter  # type: ignore
from python_vehicle_simulator.lib.gnc import attitudeEuler  # type: ignore


class OtterEnv:
    """
    Minimal simulation environment for the Otter surface vehicle.

    Args:
        sample_time: integration timestep [s]
        use_autopilot: if True, defer control to the built-in heading autopilot
        heading_deg: desired heading when autopilot is used [deg]
        current_speed: ocean current speed [m/s]
        current_beta_deg: ocean current direction [deg]
        tau_X: surge force used by the heading autopilot [N]
    """

    def __init__(
        self,
        sample_time: float = 0.05,
        use_autopilot: bool = False,
        heading_deg: float = 0.0,
        current_speed: float = 0.0,
        current_beta_deg: float = 0.0,
        tau_X: float = 120.0,
    ) -> None:
        control_mode = (
            "headingAutopilot" if use_autopilot else "stepInput"
        )
        self.sample_time = float(sample_time)
        self.vehicle = otter(
            control_mode,
            heading_deg,
            current_speed,
            current_beta_deg,
            tau_X,
        )
        self.t = 0.0
        self.eta = np.zeros(6, dtype=float)
        self.nu = self.vehicle.nu.copy()
        self.u_actual = self.vehicle.u_actual.copy()
        self.use_autopilot = use_autopilot

    def reset(self, eta: np.ndarray | None = None, nu: np.ndarray | None = None) -> None:
        """
        Reset simulation state.

        Args:
            eta: optional pose [x,y,z,phi,theta,psi]
            nu: optional body velocities [u,v,w,p,q,r]
        """
        self.t = 0.0
        self.eta = np.zeros(6, dtype=float) if eta is None else np.array(eta, float)
        self.nu = self.vehicle.nu.copy() if nu is None else np.array(nu, float)
        self.u_actual = self.vehicle.u_actual.copy()

    def step(self, u_control: np.ndarray | None = None) -> dict:
        """
        Advance one timestep.

        Args:
            u_control: propeller rev commands [n1, n2] (rad/s).
                       Ignored when autopilot is on.
        Returns:
            info dictionary with time, pose, velocities, control, and actual propeller states.
        """
        if self.use_autopilot:
            u_cmd = self.vehicle.headingAutopilot(self.eta, self.nu, self.sample_time)
        else:
            if u_control is None:
                u_cmd = np.zeros(2, dtype=float)
            else:
                u_cmd = np.array(u_control, float)

        self.nu, self.u_actual = self.vehicle.dynamics(
            self.eta, self.nu, self.u_actual, u_cmd, self.sample_time
        )
        self.eta = attitudeEuler(self.eta, self.nu, self.sample_time)
        self.t += self.sample_time

        return {
            "t": self.t,
            "eta": self.eta.copy(),
            "nu": self.nu.copy(),
            "u_control": u_cmd.copy(),
            "u_actual": self.u_actual.copy(),
        }

    @staticmethod
    def to_planar(sample: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract planar state-action tuples suitable for SE(2) learning.

        Args:
            sample: dict from `step`
        Returns:
            q = [x, y, psi], q_dot = [u, v, r], u_cmd = [n1, n2], u_actual = [n1, n2]
        """
        eta = sample["eta"]
        nu = sample["nu"]
        u_cmd = sample["u_control"]
        u_actual = sample["u_actual"]
        q = np.array([eta[0], eta[1], eta[5]], dtype=float)
        q_dot = np.array([nu[0], nu[1], nu[5]], dtype=float)
        u_cmd = np.array(u_cmd, dtype=float)
        u_actual = np.array(u_actual, dtype=float)
        return q, q_dot, u_cmd, u_actual
