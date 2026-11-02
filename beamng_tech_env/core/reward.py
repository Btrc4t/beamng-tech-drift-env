"""The per-step drift reward as a pure function of the step's data.

Shared by the live env (``BeamNGBaseEnv._compute_reward``) and the offline
regeneration (``data.regenerate`` without ``--lidar``), so the reward
recomputed from a ``.rpl`` uses exactly the formula the env uses live.
"""
import math

from beamng_tech_env.core.constants import (
    CENTERLINE_DIST_MAX,
    OFFROAD_REWARD_MULT,
    PROGRESS_BREAK_DIST,
    PROGRESS_RATE,
    REVERSING_REWARD_MULT,
)


def compute_reward(distance, drift_score, dmg_delta,
                   centerline_dist=float("nan"), offroad=False,
                   reversing=False):
    """The reward ``(total, components)`` for one step. All values are floats.

    ``distance``: metres travelled this step. ``drift_score``: the drift score
    added this step (``drift_from_state``'s ``scoreAdded``). ``dmg_delta``:
    the damage increase this step, penalized in the total (capped at -1.0).
    ``centerline_dist``: the distance (metres) from the car to the road
    centerline of the surface it is closest to (NaN when there is no road
    nearby); the progress reward is scaled by how close the car is to it.
    ``offroad``: the car is off the road (the env's on_road determination is
    False): the total reward is scaled to OFFROAD_REWARD_MULT. ``reversing``:
    the car moves backwards (the drift data's ``reversing`` flag): the total
    reward is scaled to REVERSING_REWARD_MULT. When both apply, the lowest
    multiplier wins; the two never stack.
    """
    # drift score (self-computed from State) - a positive component
    # 1 percent of the score will be a reward
    drift_reward = min(0.8, max(0.0, drift_score) * 0.01)

    # progress: a single reward from the per-step distance - positive when
    # the car travels more than the break-even distance, held at 0 when the
    # car just jiggles (travels less)
    progress_reward = max(0.0, PROGRESS_RATE * distance - PROGRESS_BREAK_DIST)

    # centerline: the progress reward is scaled by how close the car is to the
    # road centerline (the closer, the more of it is kept). The distance is
    # clamped to [0, CENTERLINE_DIST_MAX] and inverted so on-centerline = 1.0
    # and far = 0.0; a missing centerline (off-road / no navigraph, NaN)
    # counts as far (0.0).
    if math.isnan(centerline_dist):
        centerline_dist = CENTERLINE_DIST_MAX
    centerline_dist = min(max(centerline_dist, 0.0), CENTERLINE_DIST_MAX)
    centerline_norm = 1.0 - centerline_dist / CENTERLINE_DIST_MAX
    progress_reward *= centerline_norm

    dmg_reward = max(-1.0, dmg_delta * -0.005)

    # off-road / reversing: the total reward is scaled down; the lowest
    # multiplier wins when both apply (the two never stack)
    reward_mult = 1.0
    if offroad:
        reward_mult = min(reward_mult, OFFROAD_REWARD_MULT)
    if reversing:
        reward_mult = min(reward_mult, REVERSING_REWARD_MULT)

    comps = {
        "drift_delta": max(0.0, drift_score),
        "drift_reward": drift_reward,
        "dmg_delta": max(0.0, dmg_delta),
        "dmg_reward": dmg_reward,
        "centerline_dist": centerline_dist,
        "centerline_norm": centerline_norm,
        "progress_reward": progress_reward,
        "offroad": bool(offroad),
        "reversing": bool(reversing),
        "reward_mult": reward_mult,
    }
    total = (drift_reward + progress_reward + dmg_reward) * reward_mult
    return total, comps
