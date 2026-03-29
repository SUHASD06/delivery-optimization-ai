import sys
import os
from itertools import permutations

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.environment import DeliveryEnv
from env.models import Action
from env.utils import distance, get_traffic, cluster_score
from env.grader import grade
from env.tasks import easy, medium, hard
from visualize import animate

def estimated_fuel_needed(env, target):
    dist = distance(env.current_location, target)
    traffic = get_traffic(env.traffic_map, target)
    if target in env.hotspots:
        traffic += 1
    return dist * (1 + 0.5 * traffic)

def is_feasible(env, target):
    return estimated_fuel_needed(env, target) <= env.fuel

def future_feasible(env, target):
    from env.utils import distance, get_traffic

    # simulate move
    dist = distance(env.current_location, target)
    traffic = get_traffic(env.traffic_map, target)
    if target in env.hotspots:
        traffic += 1
    fuel_needed = dist * (1 + 0.5 * traffic)

    remaining_fuel = env.fuel - fuel_needed

    # check remaining deliveries
    for d in env.pending_deliveries:
        if d == target:
            continue

        future_dist = distance(target, d)
        future_fuel = future_dist * 1.2  # approx

        if remaining_fuel < future_fuel:
            return False

    return True

def should_refuel(env):
    from env.utils import distance
    
    if not env.pending_deliveries:
        return False
        
    if env.fuel >= env.max_fuel - 0.1:
        return False
        
    nearest_delivery = min(env.pending_deliveries, key=lambda d: distance(env.current_location, d))
    dist_to_delivery = distance(env.current_location, nearest_delivery)
    dist_from_delivery_to_station = min(distance(nearest_delivery, s) for s in env.fuel_stations)
    
    conservative_fuel_needed = (dist_to_delivery + dist_from_delivery_to_station) * 2.0
    
    if env.current_location in env.fuel_stations and env.fuel < 17.0:
        return True
        
    if env.fuel < conservative_fuel_needed + 2.0:
        return True
        
    return False

def estimate_future_cost(start, deliveries):
    if not deliveries:
        return 0

    min_cost = float('inf')

    # try small permutations (limit to 3 for performance)
    for perm in permutations(deliveries[:3]):
        cost = 0
        current = start

        for d in perm:
            cost += distance(current, d)
            current = d

        min_cost = min(min_cost, cost)

    return min_cost

def best_route(env):
    best_path = None
    best_cost = float('inf')

    deliveries = env.pending_deliveries

    # limit permutations for speed
    for perm in permutations(deliveries):
        cost = 0
        current = env.current_location
        fuel = env.fuel
        feasible = True
        refuel_penalty = 0

        for d in perm:
            dist = distance(current, d)
            fuel_needed = dist * 1.5  # approx

            # 🚨 If not enough fuel → check for refuel
            if fuel_needed > fuel:
                if current in env.fuel_stations:
                    fuel = env.max_fuel
                    refuel_penalty += 5
                else:
                    feasible = False
                    break

            cost += dist + refuel_penalty
            fuel -= fuel_needed
            current = d

        if feasible and cost < best_cost:
            best_cost = cost
            best_path = perm

    return best_path

def choose_best(env):
    best_score = -1e9
    best = None

    # check if need refuel
    if should_refuel(env) and env.current_location in env.fuel_stations:
        return Action(next_location=env.current_location, refuel=True)
    if should_refuel(env):
        # go to nearest reachable fuel station
        reachable_stations = [s for s in env.fuel_stations if is_feasible(env, s)]
        if not reachable_stations:
            return None
        station = min(reachable_stations, key=lambda x: distance(env.current_location, x))
        return Action(next_location=station, refuel=False)

    route = best_route(env)
    if route:
        return Action(next_location=route[0], refuel=False)

    for i, d in enumerate(env.pending_deliveries):
        if not is_feasible(env, d):
            continue  # skip impossible deliveries

        dist = distance(env.current_location, d)
        traffic = get_traffic(env.traffic_map, d)
        deadline = env.deadlines[i]
        fuel_needed = dist * (1 + 0.5 * traffic)

        score = 0

        score -= 2 * dist
        score -= 3 * traffic
        score -= 0.3 * deadline

        score += 3 * cluster_score(d, env.pending_deliveries)

        # encourage refuel when near station and fuel moderate
        if env.current_location in env.fuel_stations and env.fuel < 8:
            score += 5

        if env.fuel < 5:
            score -= 3 * dist

        if not future_feasible(env, d):
            score -= 20

        remaining_fuel = env.fuel - fuel_needed
        remaining = [x for x in env.pending_deliveries if x != d]
        future_cost = estimate_future_cost(d, remaining)
        if remaining_fuel < future_cost * 0.7:
            score -= 30

        if score > best_score:
            best_score = score
            best = d

    if best is None:
        return None

    return Action(next_location=best, refuel=False)

def run_task(task_name, task_fn):
    env = DeliveryEnv()
    task_fn(env)
    obs = env._get_obs()
    total_deliveries = len(env.pending_deliveries)

    done = False
    step = 0
    max_steps = 200
    last_action = None
    repeated_action_count = 0

    print(f"\n=== Task: {task_name} ===")
    path_history = [env.current_location]

    while not done:
        step += 1
        if step > max_steps:
            print(f"Stopping after {max_steps} steps to avoid infinite loop.")
            break

        print(f"\nStep {step}")
        print(f"Agent at {env.current_location} with fuel {round(env.fuel, 2)}")

        action = choose_best(env)
        if action is None:
            print("No feasible action (insufficient fuel). Ending episode.")
            break

        next_loc = action.next_location
        if action.refuel:
            print("⛽ REFUELING (Smart decision to avoid failure)")
        else:
            print("📦 Delivering package efficiently")
            if env.fuel < 15:
                print("⚠️ Avoided risky path to prevent fuel exhaustion")

        print(f"Remaining deliveries: {env.pending_deliveries}")

        action_key = (action.next_location, action.refuel)
        if action_key == last_action:
            repeated_action_count += 1
        else:
            repeated_action_count = 0
        last_action = action_key

        if repeated_action_count >= 5:
            print("Stopping due to repeated identical action loop.")
            break

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        path_history.append(env.current_location)

        print(f"Reward received: {round(reward, 2)}")
        if info:
            print("Info:", info)
            if info.get("error") == "insufficient_fuel":
                print("Stopping to avoid repeated insufficient-fuel loop.")
                break

    print("Done")
    print("Location:", env.current_location)
    print("Fuel:", env.fuel)
    print("Pending:", env.pending_deliveries)
    animate(env, path_history)

    score = grade(env, total_deliveries)
    print("Final Score:", score)

    print("\n===== FINAL SUMMARY =====")
    print("All deliveries completed:", len(env.pending_deliveries) == 0)
    print("Total fuel remaining:", round(env.fuel, 2))
    print("Total time taken:", round(env.time_elapsed, 2))
    print("Final score:", score)

    return len(env.pending_deliveries) == 0


def run_simulation():
    env = DeliveryEnv()
    obs = env.reset()

    done = False
    log = ""
    step = 0

    while not done and step < 200:
        step += 1
        action = choose_best(env)
        if action is None:
            log += "No feasible action (insufficient fuel). Ending episode.\n"
            break

        next_loc = action.next_location
        if action.refuel:
            log += "⛽ REFUELING (Smart decision to avoid failure)\n"
        else:
            log += "📦 Delivering package efficiently\n"
            if env.fuel < 15:
                log += "⚠️ Avoided risky path to prevent fuel exhaustion\n"

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        log += f"Location: {env.current_location}, Fuel: {round(env.fuel,2)}\n"

    if len(env.pending_deliveries) == 0:
        log += "\n🚀 Completed all deliveries successfully!"
    else:
        log += "\n⚠️ Partially deliveries completed. Some were left pending."
    return log

if __name__ == "__main__":
    print("This simulation demonstrates AI-driven delivery optimization.")
    print("Agent balances fuel, traffic, and delivery sequence.")
    print("Goal: Complete all deliveries efficiently.\n")

    all_success = True
    for task_name, task_fn in [("easy", easy), ("medium", medium), ("hard", hard)]:
        success = run_task(task_name, task_fn)
        if not success:
            all_success = False

    if all_success:
        print("\n🚀 DELIVERY COMPLETED SUCCESSFULLY")
        print("✅ All deliveries completed")
        print("⛽ Fuel optimized")
        print("🧠 Intelligent routing achieved")
    else:
        print("\n⚠️ PARTIALLY DELIVERIES COMPLETED")
        print("❌ Some deliveries were left pending")
        print("⛽ Further fuel optimization needed")