"""Drive the game's native replay playback (``core_replay``) over tcom.

``ReplayController`` is the reusable "play a .rpl and know when it ends"
building block. A BeamNG replay (``.rpl``) is the recorded simulation state,
so playing it back re-creates and animates the recorded vehicle in the scene -
which is what the replay environment attaches its sensor rig to.

The controller only needs an object exposing ``queue_lua_command`` (a
``beamngpy`` connection), so it can be tested against a stub.
"""
import time


class ReplayController:
    """Play back ``.rpl`` files and expose the playback state.

    Parameters
    ----------
    beamng:
        A ``beamngpy`` connection (anything with ``queue_lua_command``).
    """

    def __init__(self, beamng):
        self._beamng = beamng

    def _lua(self, chunk):
        return self._beamng.queue_lua_command(chunk, response=True)

    def play(self, filename, autoplay=True, timeout=10.0):
        """Play back a recorded replay and confirm it started.

        ``filename`` is the game-relative path of a ``.rpl`` file (e.g.
        ``replays/drift_....rpl``). The game's native replay module
        (``core_replay.loadFile``) takes over the view and runs the replay.
        Returns True once the engine reports it is in playback.
        """
        autoplay_lua = "true" if autoplay else "false"
        self._lua(f'core_replay.loadFile("{filename}", {autoplay_lua})')
        # loadFile is synchronous, but confirm with a short poll in case the
        # engine needs a frame to enter playback
        state = "inactive"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.state()
            if state == "playback":
                return True
            time.sleep(0.2)
        print(f"[replay] play failed for {filename!r}: state={state!r}", flush=True)
        return False

    def stop(self):
        """Stop the current playback (the stream is released, state -> inactive)."""
        self._lua("core_replay.stop()")

    def state(self):
        """The replay state string: 'inactive', 'playback', ...

        ``core_replay.getState()`` reports 'playback' while a replay is running
        and returns to 'inactive' when it finishes (or is stopped), which is
        the signal the replay environment uses to truncate the episode.
        """
        return self._lua("return core_replay.getState()")

    def is_playing(self):
        """True while a replay is in playback."""
        return self.state() == "playback"

    def is_paused(self):
        """True while the playback is paused (the sim time is frozen).

        A paused playback still reports state 'playback', so the environment
        uses this (together with a stalled sim time) to tell a mid-replay pause
        apart from the replay having finished.
        """
        return self._lua("return core_replay.isPaused()")

    def status(self):
        """Return ``(position_seconds, state, paused)`` in one round trip.

        Used by the replay environment's step-tick wait. Reading the three
        values separately costs three tcom round trips (~25 ms each), during
        which the 1x playback position keeps advancing, so the wait overshoots
        the one-tick target. Reading them together costs a single round trip,
        keeping the wait precise. The values are concatenated into one string
        (the tcom response is the ``tostring`` of the chunk's return value, so
        a Lua table would come back as ``"table: 0x..."``) and split here.
        """
        res = self._lua(
            "return core_replay.getPositionSeconds() .. '|' .. "
            "core_replay.getState() .. '|' .. tostring(core_replay.isPaused())"
        )
        pos_s, state, paused_s = res.split("|")
        return float(pos_s), state, paused_s == "true"

    def position_seconds(self):
        """The playback position in seconds."""
        return float(self._lua("return core_replay.getPositionSeconds()"))

    def total_seconds(self):
        """The total length of the current replay in seconds."""
        return float(self._lua("return core_replay.getTotalSeconds()"))

    def pause(self, paused=True):
        """Pause or resume the playback."""
        self._lua(f"core_replay.pause({str(bool(paused)).lower()})")

    def seek(self, seconds):
        """Seek the playback to an absolute position (seconds).

        ``core_replay.jumpTime`` takes a *relative* offset. The position read
        and the jump happen in a single tcom round trip, so the delta is
        computed from the position at the moment of the jump and the seek lands
        exactly on ``seconds`` (at the moment of execution). Reading the
        position in a separate round trip first would let the 1x playback
        advance ~25 ms between the read and the jump, overshooting the target.

        Used by the replay environment to re-anchor the playback after a step:
        the sim runs freely at 1x, so the position drifts ahead during the
        (variable) per-step processing; seeking back to the step's anchor keeps
        each step consuming exactly one dataset tick of replay time.
        """
        self._lua(f"core_replay.jumpTime({seconds} - core_replay.getPositionSeconds())")

    def seek_and_check(self, target):
        """Seek the playback to ``target`` and report whether it is in playback.

        Returns True if the replay is in playback (and the seek was performed),
        False if it has ended (state left 'playback'). The state check and the
        seek happen in a single tcom round trip, so the seek lands exactly on
        ``target`` (at the moment of execution) and the check costs no extra
        round trip. Used by the replay environment's seek-forward pacing: each
        step seeks to ``anchor + TICK_DT``, consuming exactly one dataset tick
        of replay time regardless of the (variable) per-step processing time or
        the round-trip latency.
        """
        res = self._lua(
            "local state = core_replay.getState()\n"
            "if state == 'playback' then\n"
            f"  core_replay.jumpTime({target} - core_replay.getPositionSeconds())\n"
            "end\n"
            "return state"
        )
        return res == "playback"
