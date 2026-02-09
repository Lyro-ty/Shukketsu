"""Source trust scoring and decay computation."""

from datetime import datetime, timedelta

SOURCE_TRUST = {
    "game_data": 1.0,
    "simulation": 0.9,
    "combat_logs": 0.85,
    "expert_guide": 0.75,
    "archived_theory": 0.7,
    "community": 0.5,
    "unknown": 0.3,
}


def effective_trust(
    base_trust: float,
    fetched_at: datetime,
    max_age: timedelta,
    decay_factor: float,
) -> float:
    """Compute effective trust with time-based decay.

    Trust stays at base_trust until max_age, then decays
    by decay_factor for each additional max_age period.
    """
    age = datetime.utcnow() - fetched_at
    if age <= max_age:
        return base_trust
    periods_past = (age - max_age) / max_age
    return float(base_trust * (decay_factor**periods_past))
