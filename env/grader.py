def grade(env, total_deliveries):

    completed = total_deliveries - len(env.pending_deliveries)

    completion_score = completed / total_deliveries

    fuel_penalty = 0
    if env.fuel <= 0:
        fuel_penalty = 0.2

    time_penalty = min(env.time_elapsed / 100, 0.3)

    score = completion_score - fuel_penalty - time_penalty

    return max(0.0, min(1.0, score))