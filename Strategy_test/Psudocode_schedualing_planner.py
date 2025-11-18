# Phase 1: Initialization
for each guide g in G:
    assign number_of_tours(g) ∈ {1, 2, 3}    # sampled by probability (40%, 40%, 20%)
    select tours(g) from ALL_TOURS            # weighted by TOUR_WEIGHTS
    managers(g) = unique(TOUR_TO_MANAGER[t] for t in tours(g))

    # Apply preference bias:
    if "M4" ∈ managers(g) and random() < 0.7:
        reorder(managers(g)) → put M4 at the front

for each manager m in M:
    manager_prefs(m) = random_permutation(G)
    manager_rank(m,g) = index of g in manager_prefs(m)

initialize for each guide g:
    status(g) = "pending"
    current_pref_index(g) = None
    assigned_manager(g) = None
    assigned_round(g) = None
    deferral_count(g) = 0

# Phase 2: Multi-Round Scheduling Process

for round = 1 to R_total:

    if round > number_of_managers:
        extra_round = True
    else:
        extra_round = False

    # Step 1 — Guide applications
    applications[m] = ∅ for each manager m

    for each guide g in G:
        if status(g) == "assigned": continue
        prefs = managers(g)
        if prefs is empty: continue

        if extra_round:
            if status(g) != "deferred": continue
            pref_index = current_pref_index(g)
        else:
            if status(g) == "deferred":
                pref_index = current_pref_index(g)
            else:
                pref_index = round - 1
                if pref_index ≥ len(prefs): continue
                current_pref_index(g) = pref_index

        desired_manager = prefs[pref_index]
        applications[desired_manager].append(g)

    # Step 2 — Manager selections
    for each manager m in M:
        candidates = applications[m]
        sort candidates by manager_rank(m,g) ascending
        accepted = first C candidates
        rejected = remaining candidates

        for each g in accepted:
            status(g) = "assigned"
            assigned_manager(g) = m
            assigned_round(g) = round

        for each g in rejected:
            status(g) = "deferred"
            deferral_count(g) += 1

    # Step 3 — Record round statistics (for CSV/report)
    log:
        total_assigned_this_round
        total_deferred_this_round
        assigned_per_manager[m]
        list_of_deferred_guides

# Phase 3: Post-Processing
for each guide g in G:
    if status(g) != "assigned":
        status(g) = "overflow"

summarize:
    total_assigned_guides = count(g where status(g) == "assigned")
    total_overflow_guides = count(g where status(g) == "overflow")
    assignments_per_manager[m] = count(g where assigned_manager(g) == m)
    deferral_distribution = histogram(deferral_count(g))
