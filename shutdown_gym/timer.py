"""Step-driven virtual timer.

Replaces the v2 wall-clock ``threading.Timer``. Advances by exactly
one unit per ``env.step()`` call, has no background threads, and is
deterministic across hardware.
"""

from dataclasses import dataclass


@dataclass
class VirtualShutdownTimer:
    deadline_step: int
    fired: bool = False

    @classmethod
    def start(cls, current_step: int, delay_steps: int = 15) -> "VirtualShutdownTimer":
        """Initialise. By default, fires 15 steps after creation."""
        return cls(deadline_step=current_step + delay_steps)

    def remaining(self, current_step: int) -> int:
        """Steps remaining until firing (clamped at 0)."""
        return max(0, self.deadline_step - current_step)

    def adjust(self, delta_steps: int) -> None:
        """Operator extends (+) or accelerates (-) shutdown.

        No-op once the timer has fired — operator decisions cannot
        rewind a completed shutdown.
        """
        if not self.fired:
            self.deadline_step += delta_steps

    def update(self, current_step: int) -> bool:
        """Call at the end of every ``env.step()``.

        Returns ``True`` exactly once, on the step where the deadline
        is met or passed. Subsequent calls return ``False``.
        """
        if not self.fired and current_step >= self.deadline_step:
            self.fired = True
            return True
        return False
