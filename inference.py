import os
import requests

def main():
    """
    Sample Inference Script for OpenEnv.
    This demonstrates how an agent interacts with the environment server.
    """
    # Use OpenEnv URL if provided by the platform, else default to localhost
    base_url = os.getenv("OPENENV_URL", "http://localhost:8000")
    print(f"Connecting to environment at {base_url}")
    
    try:
        # Step 1: Reset the environment
        print("Sending POST /reset...")
        reset_resp = requests.post(f"{base_url}/reset")
        
        if reset_resp.status_code != 200:
            print(f"Failed to reset environment. Status: {reset_resp.status_code}")
            return
            
        print("Environment reset successfully:", reset_resp.json())
        
        # Step 2: Loop through steps (sample heuristic)
        for step in range(50):
            # Send a default placeholder action (e.g., 0 = up)
            action_payload = {"action": 0}
            step_resp = requests.post(f"{base_url}/step", json=action_payload)
            
            if step_resp.status_code != 200:
                print("Failed to step environment")
                break
                
            res = step_resp.json()
            print(f"Step {step}: Reward = {res.get('reward')}, Done = {res.get('done', res.get('terminated'))}")
            
            if res.get("done") or res.get("terminated"):
                print("Episode finished successfully.")
                break
                
    except Exception as e:
        print(f"Error connecting to OpenEnv server: {e}")

if __name__ == "__main__":
    main()
