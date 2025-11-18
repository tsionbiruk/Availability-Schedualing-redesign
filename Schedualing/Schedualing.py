import random
import csv
from collections import defaultdict, Counter
import datetime as dt  


# CONFIGURATION


NUM_MANAGERS = 5
NUM_GUIDES = 120
MANAGER_CAPACITY = 10
EXTRA_DEFERRAL_ROUNDS = 1
MANAGERS = [f"M{i+1}" for i in range(NUM_MANAGERS)]

# Slots in a day for availability + tours
SLOTS = ["morning", "afternoon", "evening"]  


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

TOUR_WEIGHTS = {
    # M1
    "cooking_class_chef":        0.12,
    "cooking_class_host":        0.10,

    # M2
    "tipsy_tour_guide":          0.10,
    "tipsy_tour_host":           0.08,

    # M3
    "food_tour":                 0.10,

    # M4 
    "walking_english_day":       0.10,
    "walking_english_night":     0.09,
    "walking_spanish":           0.07,
    "walking_vatican":           0.06,
    "walking_colosseum":         0.06,
    "ghost_tour":                0.06,

    # M5
    "colosseum_walking_tour":    0.06,
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


# MANAGER PREFERENCES

def build_manager_preferences(guides, managers):
    manager_prefs = {}
    for m in managers:
        arr = guides[:]
        random.shuffle(arr)
        manager_prefs[m] = arr
    return manager_prefs


#  DECEMBER DATES + DUMMY AVAILABILITY + DUMMY TOUR CALENDAR

def build_december_dates(year=2025, month=12):
    """
    Build ISO date strings for all days of the given December.
    Example: ["2025-12-01", ..., "2025-12-31"]
    """
    start = dt.date(year, month, 1)
    if month == 12:
        next_month = dt.date(year + 1, 1, 1)
    else:
        next_month = dt.date(year, month + 1, 1)

    dates = []
    cur = start
    while cur < next_month:
        dates.append(cur.isoformat())
        cur += dt.timedelta(days=1)
    return dates


def build_dummy_guide_availability(guides, dates, p_day_available=0.7):
    """
    For each guide and each day in December, randomly generate a set
    of slots they are available in (morning/afternoon/evening).

    - A day is 'blank' if the guide is not available at all.
    - If available, they pick 1–3 slots.
    """
    availability = {}
    for g in guides:
        day_map = {}
        for d in dates:
            # Decide if this guide is available AT ALL this day
            if random.random() > p_day_available:
                # not available -> left blank (no key for this date)
                continue

            # Available: choose 1, 2, or 3 slots
            num_slots = random.choice([1, 2, 3])
            chosen_slots = random.sample(SLOTS, k=num_slots)
            day_map[d] = chosen_slots

        availability[g] = day_map
    return availability


def _sample_booking_status(is_weekend):
    """
    Simple heuristic:
    - Weekends: more likely to be 'booked'
    - Weekdays: more 'unbooked' / 'likely'
    """
    r = random.random()
    if is_weekend:
        if r < 0.65:       # 65% booked
            return "booked"
        elif r < 0.90:     # 25% likely
            return "likely"
        else:
            return "unbooked"  # 10%
    else:
        if r < 0.45:       # 45% booked
            return "booked"
        elif r < 0.80:     # 35% likely
            return "likely"
        else:
            return "unbooked"  # 20%


def build_dummy_tour_calendar(dates, tours=ALL_TOURS):
    """
    For every date, slot, and tour type, create a 'tour instance'
    with a booking status and its manager.

    This gives you the full "tour days list" for December:
    - booked tours
    - likely-to-be-booked tours
    - unbooked tours
    """
    tour_instances = []

    for d_str in dates:
        d_obj = dt.date.fromisoformat(d_str)
        is_weekend = d_obj.weekday() >= 5  # 5=Sat, 6=Sun

        for slot in SLOTS:
            for tour in tours:
                status = _sample_booking_status(is_weekend)
                tour_instances.append({
                    "date": d_str,
                    "slot": slot,
                    "tour": tour,
                    "manager": TOUR_TO_MANAGER[tour],
                    "status": status,  # 'booked', 'likely', 'unbooked'
                })

    return tour_instances

def assign_guides_to_tours(
    tour_calendar,
    guide_availability,
    guide_tours,
    guide_state,
    guides,
):
    """
    Assign guides to each tour instance in December.
    Priority of filling:
      1) booked  -> must be covered if at all possible
      2) likely  -> pre-assigned if enough guides
      3) unbooked -> filled with remaining capacity

    Rules for selecting a guide:
      - must be available on that date & slot
      - must be able to do that tour
      - prefer guides whose assigned_manager == tour.manager for BOOKED tours
      - otherwise, use anyone (including overflow) who fits
      - break ties by picking the guide with the fewest assigned tours so far
    """
    from collections import defaultdict

    guide_assign_counts = defaultdict(int)
    assigned_instances = []

    # We’ll go status by status (booked first)
    status_order = ["booked", "likely", "unbooked"]

    for target_status in status_order:
        for entry in tour_calendar:
            if entry["status"] != target_status:
                continue

            date = entry["date"]
            slot = entry["slot"]
            tour = entry["tour"]
            manager = entry["manager"]

            preferred = []  # guides whose assigned_manager == this manager
            fallback = []   # any other compatible guide

            for g in guides:
                # Must be able to do this tour
                if tour not in guide_tours.get(g, []):
                    continue

                # Must be available on this date & slot
                slots_for_day = guide_availability.get(g, {}).get(date, [])
                if slot not in slots_for_day:
                    continue

                st = guide_state.get(g, {"assigned_manager": None, "status": "overflow"})
                if st.get("assigned_manager") == manager:
                    preferred.append(g)
                else:
                    fallback.append(g)

            # Choose candidate pool
            if target_status == "booked":
                # For booked tours we try really hard to match the manager,
                # but if that fails, we still fill from fallback.
                pool = preferred if preferred else fallback
            else:
                # For likely / unbooked we can use anyone
                pool = preferred + fallback

            if not pool:
                # No compatible guide for this specific tour instance
                assigned_instances.append({**entry, "assigned_guide": None})
                continue

            # Pick the least-loaded guide so far
            chosen = min(pool, key=lambda g: guide_assign_counts[g])
            guide_assign_counts[chosen] += 1

            assigned_instances.append({**entry, "assigned_guide": chosen})

    return assigned_instances

def compute_guide_assignment_stats(tour_assignments, guides):
    """
    Compute, for each guide:
      - total number of tours assigned
      - number of booked / likely / unbooked tours

    Also returns global counts:
      - total tour instances in the calendar
      - assigned vs unassigned instances
    """
    per_guide = {
        g: {"total": 0, "booked": 0, "likely": 0, "unbooked": 0}
        for g in guides
    }

    total_instances = len(tour_assignments)
    assigned_instances = 0
    unassigned_instances = 0

    for entry in tour_assignments:
        g = entry.get("assigned_guide")
        status = entry.get("status", "unbooked")

        if g is None:
            unassigned_instances += 1
            continue

        assigned_instances += 1

        if g not in per_guide:
            per_guide[g] = {"total": 0, "booked": 0, "likely": 0, "unbooked": 0}

        per_guide[g]["total"] += 1
        if status in ("booked", "likely", "unbooked"):
            per_guide[g][status] += 1

    return per_guide, total_instances, assigned_instances, unassigned_instances


def print_assignment_summary(per_guide, total_instances, assigned_instances, unassigned_instances):
    """
    Pretty-print global stats + a small per-guide summary.
    """
    num_guides = len(per_guide)
    tours_per_guide = [stats["total"] for stats in per_guide.values()]
    zero_tour_guides = [g for g, s in per_guide.items() if s["total"] == 0]

    avg_tours = sum(tours_per_guide) / num_guides if num_guides else 0
    min_tours = min(tours_per_guide) if tours_per_guide else 0
    max_tours = max(tours_per_guide) if tours_per_guide else 0

    print("\n=== GLOBAL TOUR ASSIGNMENT STATS ===")
    print(f"Total tour instances in calendar: {total_instances}")
    print(f"Assigned tour instances:          {assigned_instances}")
    print(f"Unassigned tour instances:        {unassigned_instances}")
    if total_instances > 0:
        print(f"Assignment rate:                  {assigned_instances / total_instances:.1%}")

    print("\n=== PER-GUIDE WORKLOAD STATS ===")
    print(f"Number of guides:                 {num_guides}")
    print(f"Average tours per guide:          {avg_tours:.2f}")
    print(f"Min tours per guide:              {min_tours}")
    print(f"Max tours per guide:              {max_tours}")
    print(f"Guides with 0 tours:              {len(zero_tour_guides)}")

    # Optionally print a few sample guides
    print("\nSample of guide workloads:")
    sample_guides = sorted(per_guide.keys(), key=lambda g: int(g[1:]))[:10]
    for g in sample_guides:
        s = per_guide[g]
        print(
            f"  {g}: total={s['total']} "
            f"(booked={s['booked']}, likely={s['likely']}, unbooked={s['unbooked']})"
        )


def save_guide_assignment_stats_csv(per_guide, filename="guide_assignment_stats.csv"):
    """
    Save per-guide stats to CSV:
      guide, total_tours, booked_tours, likely_tours, unbooked_tours
    """
    fields = ["guide", "total_tours", "booked_tours", "likely_tours", "unbooked_tours"]
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for g, stats in sorted(per_guide.items(), key=lambda x: int(x[0][1:])):
            w.writerow({
                "guide": g,
                "total_tours": stats["total"],
                "booked_tours": stats["booked"],
                "likely_tours": stats["likely"],
                "unbooked_tours": stats["unbooked"],
            })
    print(f"Saved per-guide assignment stats to '{filename}'")


def save_guide_availability_csv(guide_availability, dates, filename="guide_availability_december.csv"):
    """
    Save availability in a "spreadsheet-friendly" format:
    one row per (guide, date) with flags for each slot.
    """
    fields = ["guide", "date"] + SLOTS
    rows = []

    for g, day_map in guide_availability.items():
        for d in dates:
            slots = day_map.get(d, [])
            row = {
                "guide": g,
                "date": d,
            }
            for s in SLOTS:
                row[s] = 1 if s in slots else 0
            rows.append(row)

    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f" Saved guide availability to '{filename}'")


def save_tour_calendar_csv(tour_calendar, filename="tour_calendar_december.csv"):
    """
    Save the tour-days list: one row per tour instance.
    """
    fields = ["date", "slot", "tour", "manager", "status"]
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(tour_calendar)
    print(f" Saved tour calendar to '{filename}'")

def save_tour_assignments_csv(tour_assignments, filename="tour_assignments_december.csv"):
    """
    Save final tour assignments:
    one row per (date, slot, tour) with the assigned guide.
    """
    fields = ["date", "slot", "tour", "manager", "status", "assigned_guide"]
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(tour_assignments)
    print(f" Saved tour assignments to '{filename}'")

def print_tour_load_histogram(per_guide):
    """
    Print a simple histogram of how many tours guides got:
      - 0 tours
      - 1–5
      - 6–10
      - 11–20
      - 21+
    """
    tours_per_guide = [stats["total"] for stats in per_guide.values()]

    # Define bins: (low, high, label)
    bins = [
        (0, 0, "0"),
        (1, 5, "1–5"),
        (6, 10, "6–10"),
        (11, 20, "11–20"),
        (21, 9999, "21+"),
    ]

    # Count how many guides fall into each bin
    bin_counts = []
    for lo, hi, label in bins:
        count = sum(1 for t in tours_per_guide if lo <= t <= hi)
        bin_counts.append((label, count))

    print("\n=== HISTOGRAM: TOURS PER GUIDE ===")
    total_guides = len(per_guide)
    for label, count in bin_counts:
        if total_guides > 0:
            pct = 100.0 * count / total_guides
        else:
            pct = 0.0
        # ASCII bar: one block per guide per 2% (adjust as you like)
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"{label:6}: {count:3d} guides ({pct:5.1f}%) {bar}")


# SIMULATION (GUIDE -> MANAGER MATCHING)

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


# OUTPUT


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
    print(f" Saved '{fn}'")

def save_global_assignment_summary_csv(
    total_instances,
    assigned_instances,
    unassigned_instances,
    num_guides,
    avg_tours,
    min_tours,
    max_tours,
    zero_tour_count,
    filename="global_assignment_summary.csv",
):
    """
    Save the global summary statistics of the simulation to a CSV file.
    """
    fields = [
        "total_tour_instances",
        "assigned_tour_instances",
        "unassigned_tour_instances",
        "assignment_rate_percent",
        "number_of_guides",
        "average_tours_per_guide",
        "min_tours_per_guide",
        "max_tours_per_guide",
        "guides_with_zero_tours",
    ]

    assignment_rate = 100 * assigned_instances / total_instances if total_instances else 0

    row = {
        "total_tour_instances": total_instances,
        "assigned_tour_instances": assigned_instances,
        "unassigned_tour_instances": unassigned_instances,
        "assignment_rate_percent": round(assignment_rate, 2),
        "number_of_guides": num_guides,
        "average_tours_per_guide": round(avg_tours, 2),
        "min_tours_per_guide": min_tours,
        "max_tours_per_guide": max_tours,
        "guides_with_zero_tours": zero_tour_count,
    }

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)

    print(f" Saved global summary statistics to '{filename}'")


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


# MAIN


if __name__ == "__main__":
    random.seed(0)

    # 1) Build guides, their tours, and manager preferences 
    guides = build_guides(NUM_GUIDES)
    guide_tours = build_guide_tour_choices(guides)
    guide_mgr_prefs = build_guide_manager_prefs_from_tours(guide_tours, TOUR_TO_MANAGER, bias_m4_prob=0.7)
    manager_prefs = build_manager_preferences(guides, MANAGERS)

    # 2) Run the manager–guide scheduling simulation 
    guide_state, round_rows = simulate_scheduling(guides, guide_mgr_prefs, manager_prefs, guide_tours)
    summarize_results(guide_state)
    save_round_csv(round_rows)

    # 3) Build December availability + tour-days list
    december_dates = build_december_dates(year=2025, month=12)
    guide_availability = build_dummy_guide_availability(guides, december_dates, p_day_available=0.7)
    tour_calendar = build_dummy_tour_calendar(december_dates, tours=ALL_TOURS)

    # 4) Save them to CSV so you can inspect them in Excel / Sheets
    save_guide_availability_csv(guide_availability, december_dates,
                                filename="guide_availability_december.csv")
    save_tour_calendar_csv(tour_calendar, filename="tour_calendar_december.csv")

    # 5) Print a tiny preview so the console isn't crazy
    print("\n=== SAMPLE AVAILABILITY (first 3 guides, first few days) ===")
    for g in guides[:3]:
        days = guide_availability[g]
        print(f"{g}:")
        for d in december_dates[:5]:
            slots = days.get(d, [])
            print(f"  {d}: {slots}")
        print()

    print("=== SAMPLE TOUR CALENDAR (first 10 entries) ===")
    
    for row in tour_calendar[:10]:
        print(row)
    
    # 6) Assign guides to specific tour instances (using booked/likely/unbooked)
    tour_assignments = assign_guides_to_tours(
        tour_calendar=tour_calendar,
        guide_availability=guide_availability,
        guide_tours=guide_tours,
        guide_state=guide_state,
        guides=guides,
    )

    save_tour_assignments_csv(tour_assignments)

    print("\n=== SAMPLE TOUR ASSIGNMENTS (first 15) ===")
    for row in tour_assignments[:15]:
        print(row)

    # 7) Compute and display stats on assignments
    per_guide_stats, total_instances, assigned_instances, unassigned_instances = \
        compute_guide_assignment_stats(tour_assignments, guides)

    print_assignment_summary(per_guide_stats, total_instances, assigned_instances, unassigned_instances)
    save_guide_assignment_stats_csv(per_guide_stats)
    print_tour_load_histogram(per_guide_stats)

    # 8) Save global summary as CSV
    num_guides = len(per_guide_stats)
    tours_per_guide = [s["total"] for s in per_guide_stats.values()]
    avg_tours = sum(tours_per_guide) / num_guides if num_guides else 0
    min_tours = min(tours_per_guide) if tours_per_guide else 0
    max_tours = max(tours_per_guide) if tours_per_guide else 0
    zero_tour_count = sum(1 for s in per_guide_stats.values() if s["total"] == 0)

    save_global_assignment_summary_csv(
        total_instances,
        assigned_instances,
        unassigned_instances,
        num_guides,
        avg_tours,
        min_tours,
        max_tours,
        zero_tour_count,
    )





