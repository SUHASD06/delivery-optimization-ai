def easy(env):
    env.reset()
    env.pending_deliveries = [(1,1), (2,2)]
    env.deadlines = [50, 60]
    env.fuel = 20

def medium(env):
    env.reset()
    env.pending_deliveries = [(2,3), (5,5), (8,2)]
    env.deadlines = [30, 50, 70]
    env.fuel = 16

def hard(env):
    env.reset()
    env.pending_deliveries = [(2,3), (5,5), (8,2), (9,9), (3,7)]
    env.deadlines = [20, 40, 60, 80, 100]
    env.fuel = 20