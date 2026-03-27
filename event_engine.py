"""
event_engine.py
Transforms continuous metrics into discrete, actionable events.
Three patterns:
  1. ThresholdEventEngine: Static thresholds with debounce cooldowns
  2. RollingAnomalyDetector: Bollinger Band-style dynamic anomaly detection
  3. SceneStateMachine: State machine with persistence-based transitions
"""

import numpy as np
from enum import Enum
from datetime import datetime, timedelta, timezone
from collections import deque

import config


# ============================================================
# PATTERN 1: Static thresholds with debouncing
# ============================================================

class ThresholdEventEngine:
    """
    Fires events when metrics cross defined thresholds.
    
    Includes a cooldown mechanism per event type to prevent
    rapid-fire spam when a metric oscillates around a boundary.
    This is analogous to how AWS IoT Events and industrial
    SCADA systems handle alarm debouncing.
    """

    def __init__(self, cooldown_seconds=None):
        self.cooldown_period = timedelta(
            seconds=cooldown_seconds or config.EVENT_COOLDOWN_SECONDS
        )
        self.cooldowns = {}  # event_type -> last_fired datetime
        self.events = []
        self.prev_umbrella_count = 0
        self.prev_is_daytime = None

    def _can_fire(self, event_type, now):
        """Check if enough time has passed since last event of this type."""
        last = self.cooldowns.get(event_type)
        return last is None or (now - last) >= self.cooldown_period

    def _emit(self, timestamp, event_type, severity, value, threshold,
              description, scene_state=""):
        """Record and print an event."""
        self.cooldowns[event_type] = timestamp
        event = {
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "severity": severity,
            "metric_value": round(value, 2) if isinstance(value, float) else value,
            "threshold": round(threshold, 2) if isinstance(threshold, float) else threshold,
            "description": description,
            "scene_state": scene_state,
        }
        self.events.append(event)
        print(f"  EVENT: [{severity}] {description}")
        return event

    def check(self, timestamp, metrics, scene_state=""):
        """
        Run all threshold checks against current metrics.
        
        Args:
            timestamp: datetime object (UTC)
            metrics: dict of current metric values
            scene_state: current state machine state string
        
        Returns:
            List of event dicts generated this check
        """
        fired = []

        # --- Crowd density checks ---
        person_count = metrics.get("person_count", 0)

        if person_count >= config.CROWD_CRITICAL_THRESHOLD:
            if self._can_fire("crowd_critical", timestamp):
                fired.append(self._emit(
                    timestamp, "crowd_critical", "CRITICAL",
                    person_count, config.CROWD_CRITICAL_THRESHOLD,
                    f"Critical crowd density: {person_count} people detected",
                    scene_state,
                ))
        elif person_count >= config.CROWD_WARNING_THRESHOLD:
            if self._can_fire("crowd_warning", timestamp):
                fired.append(self._emit(
                    timestamp, "crowd_warning", "WARNING",
                    person_count, config.CROWD_WARNING_THRESHOLD,
                    f"Elevated crowd: {person_count} people detected",
                    scene_state,
                ))

        # --- Rain detection (umbrella proxy) ---
        umbrella_count = metrics.get("umbrella_count", 0)

        if umbrella_count > 0 and self.prev_umbrella_count == 0:
            if self._can_fire("rain_started", timestamp):
                fired.append(self._emit(
                    timestamp, "rain_started", "INFO",
                    umbrella_count, 1,
                    f"Umbrellas detected ({umbrella_count}) - possible rain",
                    scene_state,
                ))
        elif umbrella_count == 0 and self.prev_umbrella_count > 0:
            if self._can_fire("rain_stopped", timestamp):
                fired.append(self._emit(
                    timestamp, "rain_stopped", "INFO",
                    0, 0,
                    "No more umbrellas detected - rain may have stopped",
                    scene_state,
                ))
        self.prev_umbrella_count = umbrella_count

        # --- Day/night transition ---
        is_daytime = metrics.get("is_daytime")
        if is_daytime is not None and self.prev_is_daytime is not None:
            if is_daytime and not self.prev_is_daytime:
                fired.append(self._emit(
                    timestamp, "sunrise", "INFO",
                    metrics.get("brightness", 0), 110,
                    "Daylight detected - sunrise",
                    scene_state,
                ))
            elif not is_daytime and self.prev_is_daytime:
                fired.append(self._emit(
                    timestamp, "sunset", "INFO",
                    metrics.get("brightness", 0), 110,
                    "Darkness detected - sunset",
                    scene_state,
                ))
        if is_daytime is not None:
            self.prev_is_daytime = is_daytime

        # --- Sudden emptiness (everyone left) ---
        if person_count == 0 and metrics.get("motion_pct", 0) < 0.5:
            if self._can_fire("scene_empty", timestamp):
                fired.append(self._emit(
                    timestamp, "scene_empty", "INFO",
                    person_count, 0,
                    "Scene is completely empty with no motion",
                    scene_state,
                ))

        return fired

    def get_all_events(self):
        """Return all events fired since start."""
        return self.events


# ============================================================
# PATTERN 2: Rolling window anomaly detection
# ============================================================

class RollingAnomalyDetector:
    """
    Bollinger Band-style anomaly detection for a single metric.
    
    Maintains a rolling window of recent values and flags new values
    that exceed mean + (num_std * standard_deviation). This adapts
    to gradual changes (like rush hour building up) while still
    catching sudden spikes.
    """

    def __init__(self, window_size=None, num_std=None):
        """
        Args:
            window_size: Number of samples in the rolling window
            num_std: Number of standard deviations for anomaly boundary
        """
        self.window = deque(
            maxlen=window_size or config.ANOMALY_WINDOW_SIZE
        )
        self.num_std = num_std or config.ANOMALY_NUM_STD

    def evaluate(self, value):
        """
        Add a value and check if it's anomalous.
        
        Args:
            value: Current metric value (float)
        
        Returns:
            Tuple of (is_anomaly: bool or None, mean: float, upper_bound: float)
            Returns (None, 0, 0) if the window isn't full yet.
        """
        self.window.append(value)

        if len(self.window) < self.window.maxlen:
            return None, 0, 0

        mean = float(np.mean(self.window))
        std = float(np.std(self.window))
        upper = mean + (self.num_std * std)
        is_anomaly = value > upper

        return is_anomaly, round(mean, 2), round(upper, 2)

    def get_rolling_avg(self):
        """Get the current rolling average (or 0 if window is empty)."""
        if len(self.window) == 0:
            return 0.0
        return float(np.mean(self.window))


# ============================================================
# PATTERN 3: Scene state machine
# ============================================================

class SceneState(Enum):
    """Possible states for the overall scene."""
    QUIET = "quiet"
    NORMAL = "normal"
    BUSY = "busy"
    CRITICAL = "critical"


class SceneStateMachine:
    """
    Models the scene as a finite state machine with hysteresis.
    
    State transitions require the condition to persist for a
    configurable number of consecutive readings. This prevents
    flickering between states when person counts bounce around
    a boundary.
    
    Analogous to AWS IoT Events detector models, which require
    conditions to be true for a specified duration before
    triggering state changes.
    
    States and transitions:
        QUIET --(>=5 people)--> NORMAL --(>=20)--> BUSY --(>=40)--> CRITICAL
        QUIET <--(<3 people)-- NORMAL <--(<15)-- BUSY <--(<30)-- CRITICAL
    """

    def __init__(self, persistence_count=None):
        """
        Args:
            persistence_count: Consecutive readings needed to confirm transition
        """
        self.state = SceneState.QUIET
        self.persistence = persistence_count or config.STATE_PERSISTENCE_COUNT
        self.pending_state = None
        self.pending_counter = 0

        # Define transitions: for each state, when to go up/down
        self.transitions = {
            SceneState.QUIET: {
                "up_thresh": config.STATE_QUIET_TO_NORMAL,
                "up_target": SceneState.NORMAL,
            },
            SceneState.NORMAL: {
                "up_thresh": config.STATE_NORMAL_TO_BUSY,
                "up_target": SceneState.BUSY,
                "down_thresh": config.STATE_NORMAL_TO_QUIET,
                "down_target": SceneState.QUIET,
            },
            SceneState.BUSY: {
                "up_thresh": config.STATE_BUSY_TO_CRITICAL,
                "up_target": SceneState.CRITICAL,
                "down_thresh": config.STATE_BUSY_TO_NORMAL,
                "down_target": SceneState.NORMAL,
            },
            SceneState.CRITICAL: {
                "down_thresh": config.STATE_CRITICAL_TO_BUSY,
                "down_target": SceneState.BUSY,
            },
        }

    def update(self, person_count):
        """
        Evaluate the current person count and potentially transition state.
        
        Args:
            person_count: Number of people detected
        
        Returns:
            Transition string "old_state -> new_state" if a transition
            occurred, or None if the state didn't change.
        """
        rules = self.transitions[self.state]
        target = None

        # Check upward transition
        if "up_thresh" in rules and person_count >= rules["up_thresh"]:
            target = rules["up_target"]
        # Check downward transition
        elif "down_thresh" in rules and person_count < rules["down_thresh"]:
            target = rules["down_target"]

        if target and target != self.state:
            # Same pending target - increment counter
            if self.pending_state == target:
                self.pending_counter += 1
                if self.pending_counter >= self.persistence:
                    old = self.state
                    self.state = target
                    self.pending_state = None
                    self.pending_counter = 0
                    return f"{old.value} -> {target.value}"
            else:
                # New pending target - reset counter
                self.pending_state = target
                self.pending_counter = 1
        else:
            # No transition condition met - reset pending
            self.pending_state = None
            self.pending_counter = 0

        return None

    def get_state(self):
        """Get current state as a string."""
        return self.state.value
