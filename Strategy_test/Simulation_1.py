import random
import csv
from collections import defaultdict, Counter


# CONFIGURATION


NUM_MANAGERS = 5
NUM_GUIDES = 120
MANAGER_CAPACITY = 10
EXTRA_DEFERRAL_ROUNDS = 1
MANAGERS = [f"M{i+1}" for i in range(NUM_MANAGERS)]


# TOURS AND MANAGER MAPPING


TOUR_TO_MANAGER = {
    # Cooking class (M1)
    "cooking_class_chef": "M1",
    "cooking_class_host": "M1",

    # Tipsy tours (M2)
    "tipsy_tour_guide": "M2",
    "tipsy_tour_host": "M2",

    # Food tour (M3)
    "food_tour": "M3",

    # Walking + ghost (M4)
    "walking_english_day": "M4",
    "walking_english_night": "M4",
    "walking_spanish": "M4",
    "walking_vatican": "M4",
    "walking_colosseum": "M4",
    "ghost_tour": "M4",

    # Colosseum walking tour (M5)
    "colosseum_walking_tour": "M5",
}

ALL_TOURS = list(TOUR_TO_MANAGER.keys())

# Tour weights for guide selection. Tour weights control how likely 
# a guide is to select a given tour when choosing their set of tours.
# Give M4 and M1 heavier weights (they’re most popular)

TOUR_WEIGHTS = {
    "cooking_class_chef":        0.12,  # M1
    "cooking_class_host":        0.10,  # M1
    "tipsy_tour_guide":          0.07,  # M2
    "tipsy_tour_host":           0.06,  # M2
    "food_tour":                 0.07,  # M3
    "walking_english_day":       0.13,  # M4
    "walking_english_night":     0.12,  # M4
    "walking_spanish":           0.08,  # M4
    "walking_vatican":           0.07,  # M4
    "walking_colosseum":         0.07,  # M4
    "ghost_tour":                0.07,  # M4
    "colosseum_walking_tour":    0.04,  # M5
}


# GUIDE GENERATION


def build_guides(num_guides=NUM_GUIDES):
    return [f"G{i+1}" for i in range(num_guides)]

# Randomly sample how many tours a guide can do
def sample_num_tours_for_guide():
    """How many different tours a guide can do."""
    r = random.random()
    if r < 0.4:
        return 1
    elif r < 0.8:
        return 2
    else:
        return 3


def build_guide_tour_choices(guides):
    """Each guide selects 1–3 tours (weighted)."""
    tour_ids = ALL_TOURS
    weights = [TOUR_WEIGHTS[t] for t in tour_ids]

    guide_tours = {}
    for g in guides:
        k = sample_num_tours_for_guide()
        chosen = random.choices(tour_ids, weights=weights, k=k)
        # de-duplicate while preserving order
        seen, cleaned = set(), []
        for t in chosen:
            if t not in seen:
                seen.add(t)
                cleaned.append(t)
        guide_tours[g] = cleaned
    return guide_tours


def build_guide_manager_prefs_from_tours(guide_tours, tour_to_manager, bias_m4_prob=0.7):
    """
    Build manager priority list from tour list.
    Bias: if M4 appears, move it to the front for most guides.
    """
    guide_mgr_prefs = {}
    for g, tours in guide_tours.items():
        mgr_list, seen = [], set()
        for t in tours:
            m = tour_to_manager[t]
            if m not in seen:
                seen.add(m)
                mgr_list.append(m)

        # Bias toward M4
        if "M4" in mgr_list and random.random() < bias_m4_prob:
            mgr_list = ["M4"] + [m for m in mgr_list if m != "M4"]

        guide_mgr_prefs[g] = mgr_list
    return guide_mgr_prefs

# ================================================================
# MANAGER PREFERENCES
# ================================================================

def build_manager_preferences(guides, managers):
    manager_prefs = {}
    for m in managers:
        arr = guides[:]
        random.shuffle(arr)
        manager_prefs[m] = arr
    return manager_prefs

# ================================================================
# SIMULATION
# ================================================================

def simulate_scheduling(
    guides,
    guide_mgr_prefs,
    manager_prefs,
    guide_tours,
    num_managers=NUM_MANAGERS,
    extra_deferral_rounds=EXTRA_DEFERRAL_ROUNDS,
    manager_capacity=MANAGER_CAPACITY,
):
    managers = MANAGERS
    total_rounds = num_managers + extra_deferral_rounds
    big = len(guides) + 1000

    # Manager rank maps
    manager_rank = {m: {g: idx for idx, g in enumerate(manager_prefs[m])} for m in managers}

    guide_state = {
        g: {"status": "pending", "current_pref_idx": None, "assigned_manager": None,
            "assigned_round": None, "deferral_count": 0}
        for g in guides
    }

    round_rows = []

    print("\n=================== GUIDE TOURS & MANAGER PREFS ===================")
    for g in sorted(guides, key=lambda x: int(x[1:])):
        print(f"{g}: tours={guide_tours[g]} -> managers={guide_mgr_prefs[g]}")
    print("===================================================================\n")

    print("=================== MANAGER PREFERENCES (top 10) ==================")
    for m in managers:
        print(f"{m}: {manager_prefs[m][:10]}")
    print("===================================================================\n")

    print("\n=================== START SIMULATION ===================\n")

    for round_idx in range(1, total_rounds + 1):
        is_extra = round_idx > num_managers
        print(f"\n--- ROUND {round_idx} {'(Extra Deferral Round)' if is_extra else ''} ---")

        # Applications
        applications = defaultdict(list)
        for g in guides:
            st = guide_state[g]
            if st["status"] == "assigned":
                continue
            prefs = guide_mgr_prefs.get(g, [])
            if not prefs:
                continue
            if is_extra:
                if st["status"] != "deferred":
                    continue
                pref_idx = st["current_pref_idx"]
            else:
                if st["status"] == "deferred":
                    pref_idx = st["current_pref_idx"]
                else:
                    pref_idx = round_idx - 1
                    if pref_idx >= len(prefs):
                        continue
                    st["current_pref_idx"] = pref_idx
            if pref_idx is None or pref_idx >= len(prefs):
                continue
            applications[prefs[pref_idx]].append(g)

        # Manager selections
        assignments_this_round = defaultdict(list)
        deferred_this_round = []
        for m in managers:
            cands = applications.get(m, [])
            if not cands:
                continue
            rank = manager_rank[m]
            sorted_cands = sorted(cands, key=lambda g: rank.get(g, big))
            accepted = sorted_cands[:manager_capacity]
            rejected = sorted_cands[manager_capacity:]
            for g in accepted:
                st = guide_state[g]
                st.update({"status": "assigned", "assigned_manager": m, "assigned_round": round_idx})
                assignments_this_round[m].append(g)
            for g in rejected:
                guide_state[g]["status"] = "deferred"
                guide_state[g]["deferral_count"] += 1
                deferred_this_round.append(g)

        # Print summary
        total_assigned = sum(len(v) for v in assignments_this_round.values())
        print(f"Assigned: {total_assigned}, Deferred: {len(deferred_this_round)}")
        for m in managers:
            print(f"  {m}: {len(assignments_this_round[m])} guides -> {assignments_this_round[m]}")

        row = {"Round": round_idx, "GuidesAssigned": total_assigned,
               "GuidesDeferred": len(deferred_this_round)}
        for m in managers:
            row[f"{m}_count"] = len(assignments_this_round[m])
            row[f"{m}_guides"] = " ".join(assignments_this_round[m])
        row["Deferred_guides"] = " ".join(deferred_this_round)
        round_rows.append(row)

    # Mark overflows
    for g, st in guide_state.items():
        if st["status"] != "assigned":
            st["status"] = "overflow"

    print("\n=================== END SIMULATION ===================\n")
    return guide_state, round_rows

# ================================================================
# OUTPUT
# ================================================================

def save_round_csv(round_rows):
    fn = "round_summary.csv"
    fields = ["Round", "GuidesAssigned", "GuidesDeferred"]
    for m in MANAGERS:
        fields += [f"{m}_count", f"{m}_guides"]
    fields.append("Deferred_guides")
    with open(fn, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(round_rows)
    print(f"Saved '{fn}'")


def summarize_results(guide_state):
    total = len(guide_state)
    assigned = [g for g, st in guide_state.items() if st["status"] == "assigned"]
    overflow = [g for g, st in guide_state.items() if st["status"] == "overflow"]
    assigned_rounds = Counter(st["assigned_round"] for st in guide_state.values() if st["assigned_round"])
    total_by_manager = Counter(st["assigned_manager"] for st in guide_state.values() if st["assigned_manager"])

    print("=== FINAL SUMMARY ===")
    print(f"Total guides: {total}")
    print(f"Assigned: {len(assigned)}")
    print(f"Overflow: {len(overflow)}\n")
    print("Assignments per round:")
    for r in sorted(assigned_rounds):
        print(f"  Round {r}: {assigned_rounds[r]}")
    print("\nFinal manager distribution:")
    for m in sorted(total_by_manager):
        print(f"  {m}: {total_by_manager[m]} guides")

# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    random.seed(0)
    guides = build_guides(NUM_GUIDES)
    guide_tours = build_guide_tour_choices(guides)
    guide_mgr_prefs = build_guide_manager_prefs_from_tours(guide_tours, TOUR_TO_MANAGER, bias_m4_prob=0.7)
    manager_prefs = build_manager_preferences(guides, MANAGERS)
    guide_state, round_rows = simulate_scheduling(guides, guide_mgr_prefs, manager_prefs, guide_tours)
    summarize_results(guide_state)
    save_round_csv(round_rows)




