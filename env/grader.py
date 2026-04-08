def grade(env, total_deliveries):
    """
    Score an episode. Returns a float strictly in (0.01, 0.99).
    The evaluator rejects exact boundary values 0.0 and 1.0.
    """
    completed = total_deliveries - len(env.pending_deliveries)

    completion_score = completed / max(total_deliveries, 1)

    fuel_penalty = 0.0
    if env.fuel <= 0:
        fuel_penalty = 0.2

    time_penalty = min(env.time_elapsed / 100, 0.3)

    score = completion_score - fuel_penalty - time_penalty

    # Clamp strictly inside (0, 1) — boundary values are rejected by evaluator
    return round(max(0.01, min(0.99, score)), 6)