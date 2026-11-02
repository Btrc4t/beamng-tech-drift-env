"""Reusable building blocks for the BeamNG.tech drift environments.

These modules are independent of gymnasium and of any specific environment:

- :mod:`~beamng_tech_env.core.constants` - shared tuning values and limits
- :mod:`~beamng_tech_env.core.obs` - pure observation math (drift, BEV, IMU)
- :class:`~beamng_tech_env.core.game.GameDriver` - game launch/scenario/recovery
- :class:`~beamng_tech_env.core.rig.VehicleRig` - sensors + observation for one vehicle
- :class:`~beamng_tech_env.core.replay.ReplayController` - .rpl playback control

The drive and replay environments in :mod:`beamng_tech_env.envs` compose these
blocks; the drive env spawns a vehicle, the replay env attaches to the vehicle
a replay spawns.
"""
from beamng_tech_env.core.game import GameDriver
from beamng_tech_env.core.replay import ReplayController
from beamng_tech_env.core.rig import VehicleRig

__all__ = ["GameDriver", "ReplayController", "VehicleRig"]
