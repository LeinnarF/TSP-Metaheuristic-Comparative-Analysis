import random

def hybrid_pso_tsp(G,
                   num_particles=30,
                   iterations=500,
                   w=0.7, c1=1.4, c2=1.4,
                   vmax=80,
                   local_search_iters=40):

    nodes = list(G.keys())
    n = len(nodes)

    # ---------------- COST ----------------
    def tour_cost(perm):
        return sum(G[perm[i]][perm[(i+1)%n]] for i in range(n))

    # ---------------- 2-OPT ----------------
    def two_opt(route):
        best = route[:]
        best_cost = tour_cost(best)

        for _ in range(local_search_iters):
            i, j = sorted(random.sample(range(n), 2))
            if j - i < 2:
                continue
            new = best[:i] + best[i:j+1][::-1] + best[j+1:]
            c = tour_cost(new)
            if c < best_cost:
                best, best_cost = new, c
        return best

    # ---------------- SWAP OPERATORS ----------------
    def get_swaps(a, b):
        a = a[:]
        pos = {v:i for i,v in enumerate(a)}
        swaps = []

        for i in range(n):
            if a[i] != b[i]:
                j = pos[b[i]]
                swaps.append((i,j))
                pos[a[i]] = j
                pos[a[j]] = i
                a[i], a[j] = a[j], a[i]
        return swaps

    def apply_swaps(p, swaps):
        p = p[:]
        for i,j in swaps:
            p[i], p[j] = p[j], p[i]
        return p

    # ---------------- INIT SWARM ----------------
    swarm = []
    for _ in range(num_particles):
        p = nodes[:]
        random.shuffle(p)
        p = two_opt(p)

        swarm.append({
            "pos": p,
            "vel": [],
            "pbest": p[:],
            "pbest_cost": tour_cost(p)
        })

    gbest = min(swarm, key=lambda x: x["pbest_cost"])
    gbest_pos = gbest["pbest"][:]
    gbest_cost = gbest["pbest_cost"]

    # ---------------- MAIN LOOP ----------------
    for _ in range(iterations):
        for p in swarm:

            X = p["pos"]
            V = p["vel"]

            S_p = get_swaps(X, p["pbest"])
            S_g = get_swaps(X, gbest_pos)

            newV = []

            for s in V:
                if random.random() < w:
                    newV.append(s)

            for s in S_p:
                if random.random() < c1:
                    newV.append(s)

            for s in S_g:
                if random.random() < c2:
                    newV.append(s)

            newV = newV[:vmax]

            X_new = apply_swaps(X, newV)
            X_new = two_opt(X_new)
            cost = tour_cost(X_new)

            p["pos"] = X_new
            p["vel"] = newV

            if cost < p["pbest_cost"]:
                p["pbest"] = X_new[:]
                p["pbest_cost"] = cost

                if cost < gbest_cost:
                    gbest_cost = cost
                    gbest_pos = X_new[:]

    return gbest_pos, gbest_cost
