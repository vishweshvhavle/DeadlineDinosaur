#!/usr/bin/env python3
"""Simple test of the paper's resolution progression logic"""

import time

class SimpleResolutionScheduler:
    def __init__(self, num_stages=5, stage_duration=1.0):
        self.num_stages = num_stages
        self.stage_duration = stage_duration
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def get_current_stage(self):
        if self.start_time is None:
            return 1
        elapsed = time.time() - self.start_time
        stage = int(elapsed / self.stage_duration) + 1
        return min(stage, self.num_stages)

    def get_resolution_scale(self):
        stage = self.get_current_stage()
        # Paper's progression: 1/5, 1/4, 1/3, 1/2, 1/1
        progression = [1/5, 1/4, 1/3, 1/2, 1.0]
        if stage <= len(progression):
            return progression[stage - 1]
        else:
            return 1.0

def test_progression():
    print("Testing paper's progression: 1/5, 1/4, 1/3, 1/2, 1/1")
    print("=" * 70)

    scheduler = SimpleResolutionScheduler(num_stages=5, stage_duration=1.0)
    scheduler.start()

    expected_scales = [
        (1, 1/5, "1/5 (20%)"),
        (2, 1/4, "1/4 (25%)"),
        (3, 1/3, "1/3 (33%)"),
        (4, 1/2, "1/2 (50%)"),
        (5, 1.0, "1/1 (100%)"),
        (5, 1.0, "1/1 (100%) - stays at full"),
    ]

    for expected_stage, expected_scale, desc in expected_scales:
        stage = scheduler.get_current_stage()
        scale = scheduler.get_resolution_scale()

        # Calculate example resolution
        full_h, full_w = 1618, 1214
        ds_h, ds_w = int(full_h * scale), int(full_w * scale)

        status = "✓" if abs(scale - expected_scale) < 0.001 else "✗"
        print(f"{status} Stage {stage}: scale={scale:.4f} ({desc}), resolution={ds_h}x{ds_w}")

        assert abs(scale - expected_scale) < 0.001, f"Expected {expected_scale}, got {scale}"
        assert stage == expected_stage, f"Expected stage {expected_stage}, got {stage}"

        time.sleep(1.1)

    print("=" * 70)
    print("✓ All tests passed! Progression is correct: 1/5, 1/4, 1/3, 1/2, 1/1")

if __name__ == "__main__":
    test_progression()
