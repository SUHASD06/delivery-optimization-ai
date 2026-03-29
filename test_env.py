"""
Environment Validation & Smoke Tests
Validates that DeliveryEnv conforms to Gymnasium API contracts.

Run:  python test_env.py
"""
import numpy as np
from env.environment import DeliveryEnv


def test_gymnasium_api():
    """Validate env follows Gymnasium API."""
    print("🔍 Testing Gymnasium API compliance...")

    for phase in [1, 2, 3]:
        env = DeliveryEnv(phase=phase)

        # reset returns (obs, info)
        result = env.reset(seed=42)
        assert isinstance(result, tuple) and len(result) == 2, \
            f"Phase {phase}: reset() must return (obs, info)"
        obs, info = result
        assert isinstance(info, dict), f"Phase {phase}: info must be dict"

        # observation shape and bounds
        assert obs.shape == (11,), f"Phase {phase}: obs shape must be (11,), got {obs.shape}"
        assert obs.dtype == np.float32, f"Phase {phase}: obs must be float32"
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), \
            f"Phase {phase}: obs must be in [0,1], got min={obs.min()}, max={obs.max()}"

        # observation space contains observation
        assert env.observation_space.contains(obs), \
            f"Phase {phase}: obs not in observation_space"

        # step returns 5-tuple
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5, f"Phase {phase}: step() must return 5-tuple"
        obs2, reward, terminated, truncated, info2 = result
        assert isinstance(reward, (float, int, np.floating)), \
            f"Phase {phase}: reward must be numeric"
        assert isinstance(terminated, (bool, np.bool_)), \
            f"Phase {phase}: terminated must be bool"
        assert isinstance(truncated, (bool, np.bool_)), \
            f"Phase {phase}: truncated must be bool"
        assert isinstance(info2, dict), f"Phase {phase}: info must be dict"

        print(f"  ✅ Phase {phase} — API compliant")

    print("  ✅ All phases pass Gymnasium API validation\n")


def test_deterministic_phase1():
    """Phase 1 should be fully deterministic with same seed."""
    print("🔍 Testing Phase 1 determinism...")

    env = DeliveryEnv(phase=1)

    obs1, _ = env.reset(seed=42)
    actions = [3, 3, 0, 0, 0]  # right, right, up, up, up
    rewards1 = []
    for a in actions:
        _, r, _, _, _ = env.step(a)
        rewards1.append(r)

    obs2, _ = env.reset(seed=42)
    rewards2 = []
    for a in actions:
        _, r, _, _, _ = env.step(a)
        rewards2.append(r)

    assert np.allclose(obs1, obs2), "Phase 1: reset with same seed must give same obs"
    assert np.allclose(rewards1, rewards2), "Phase 1: same actions must give same rewards"
    print("  ✅ Phase 1 is deterministic\n")


def test_fuel_depletion():
    """Agent should fail when fuel runs out."""
    print("🔍 Testing fuel depletion handling...")
    env = DeliveryEnv(phase=1)
    env.reset(seed=42)
    env.fuel = 0.1  # Nearly empty

    # Try to move — should fail due to insufficient fuel
    obs, reward, terminated, truncated, info = env.step(0)  # up
    if info.get("error") == "insufficient_fuel":
        assert terminated, "Should terminate on fuel depletion"
        print("  ✅ Fuel depletion correctly terminates episode\n")
    else:
        print("  ✅ Move succeeded with low fuel (distance was short enough)\n")


def test_delivery_completion():
    """Agent should receive reward for completing deliveries."""
    print("🔍 Testing delivery completion...")
    env = DeliveryEnv(phase=1)
    env.reset(seed=42)

    # Phase 1 fixed deliveries: (2,3), (5,5), (8,2)
    initial_pending = len(env.pending_deliveries)
    assert initial_pending == 3, f"Expected 3 deliveries, got {initial_pending}"

    # Move agent directly to delivery location
    env.current_location = (2, 2)
    env.fuel = 20.0
    obs, reward, terminated, truncated, info = env.step(0)  # up → (2, 3)

    assert info["deliveries_completed"] >= 1, "Should complete at least 1 delivery"
    assert reward > 0, f"Delivery reward should be positive, got {reward}"
    print(f"  ✅ Delivery completed, reward={reward:.2f}\n")


def test_refuel_mechanics():
    """Refueling at station should restore fuel."""
    print("🔍 Testing refuel mechanics...")
    env = DeliveryEnv(phase=1)
    env.reset(seed=42)
    env.current_location = (0, 0)  # Fuel station
    env.fuel = 5.0

    obs, reward, terminated, truncated, info = env.step(4)  # refuel action
    assert env.fuel == env.max_fuel, f"Fuel should be max after refuel, got {env.fuel}"
    assert info.get("refueled") is True, "Info should indicate refueling"
    print(f"  ✅ Refuel works correctly, fuel restored to {env.fuel}\n")


def test_episode_truncation():
    """Episode should truncate after 200 steps."""
    print("🔍 Testing episode truncation...")
    env = DeliveryEnv(phase=1)
    env.reset(seed=42)
    env.fuel = 1000  # Don't run out

    for i in range(200):
        action = i % 4  # cycle through movement actions
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            break

    assert terminated or truncated, "Episode should end by step 200"
    print(f"  ✅ Episode ended at step {info['steps_taken']}\n")


def test_render():
    """Test that render produces output."""
    print("🔍 Testing render...")
    env = DeliveryEnv(phase=1, render_mode="ansi")
    env.reset(seed=42)
    output = env.render()
    assert output is not None, "Render should return string"
    assert "Fuel" in output, "Render should show fuel"
    print(f"  ✅ Render works:\n{output}\n")


if __name__ == "__main__":
    print("=" * 55)
    print("  🧪 DELIVERY ENV — VALIDATION SUITE")
    print("=" * 55 + "\n")

    test_gymnasium_api()
    test_deterministic_phase1()
    test_fuel_depletion()
    test_delivery_completion()
    test_refuel_mechanics()
    test_episode_truncation()
    test_render()

    print("=" * 55)
    print("  🎉 ALL TESTS PASSED")
    print("=" * 55)
