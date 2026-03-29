"""
PPO Agent Smoke Test
Quick training + inference test to validate the full pipeline works.

Run:  python test_ppo.py
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.environment import DeliveryEnv

print("=" * 50)
print("  🧪 PPO SMOKE TEST")
print("=" * 50)

# 1. Setup Environment
os.makedirs("test_logs", exist_ok=True)
env = Monitor(DeliveryEnv(phase=1), "test_logs")
print("\n✅ Environment created successfully")
print(f"   Observation space: {env.observation_space}")
print(f"   Action space: {env.action_space}")

# 2. Create Model with same architecture as training
policy_kwargs = dict(net_arch=[256, 256])
model = PPO("MlpPolicy", env, verbose=0, n_steps=64, batch_size=16,
            policy_kwargs=policy_kwargs)
print("✅ PPO model created")

# 3. Quick Train
print("\n🔄 Training for 500 steps ...")
model.learn(total_timesteps=500)
print("✅ Training completed")

# 4. Test Inference
obs, _ = env.reset()
total_reward = 0
done = False
steps = 0

print("\n🎮 Running test episode...")
while not done and steps < 50:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    steps += 1

print(f"   Steps: {steps}")
print(f"   Total reward: {total_reward:.2f}")
print(f"   Deliveries: {info.get('deliveries_completed', 0)}")
print(f"   Fuel used: {info.get('fuel_used', 0):.2f}")

# 5. Test save/load
model.save("test_logs/test_model")
loaded = PPO.load("test_logs/test_model", env=env)
obs, _ = env.reset()
action, _ = loaded.predict(obs, deterministic=True)
assert env.action_space.contains(action), "Loaded model produced invalid action"
print("\n✅ Save/load cycle works")

print("\n" + "=" * 50)
print("  🎉 ALL SMOKE TESTS PASSED")
print("=" * 50)
