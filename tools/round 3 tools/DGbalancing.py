def optimize_delta_gamma(
    total_delta,
    total_gamma,
    deltas,
    gammas,
    weights,
    threshold,
    max_coeffs,
    min_coeff=8,
    delta_tolerance=1
):
    keys = list(deltas.keys())
    sign = -1 if total_delta > 0 else 1

    # Filter usable keys based on weights
    usable_keys = [
        k for k in keys
        if (total_delta < 0 and weights[k] <= threshold)
        or (total_delta > 0 and weights[k] >= -threshold)
    ]
    
    best = None
    best_within_tolerance = None
    best_gamma_diff = float("inf")
    best_delta_error = float("inf")  # fallback metric

    for i in range(len(usable_keys)):
        for j in range(i, len(usable_keys)):
            k1, k2 = usable_keys[i], usable_keys[j]
            d1, d2 = deltas[k1], deltas[k2]
            g1, g2 = gammas[k1], gammas[k2]
            m1, m2 = max_coeffs[k1], max_coeffs[k2]

            for c1 in range(min(min_coeff, m1), m1 + 1):
                for c2 in range(min(min_coeff, m2), m2 + 1):
                    c1_signed = -sign * c1
                    c2_signed = -sign * c2

                    delta_sum = c1_signed * d1 + c2_signed * d2
                    delta_error = abs(delta_sum - total_delta)

                    gamma_sum = c1_signed * g1 + c2_signed * g2
                    gamma_diff = abs(gamma_sum - total_gamma)

                    candidate = {
                        'keys': (k1, k2),
                        'coeffs': (-c1_signed, -c2_signed),
                        'delta_sum': -delta_sum,
                        'gamma_sum': -gamma_sum,
                        'delta_diff': delta_error,
                        'gamma_diff': gamma_diff
                    }

                    # Try to satisfy tolerance first
                    if delta_error <= delta_tolerance:
                        if gamma_diff < best_gamma_diff:
                            best_gamma_diff = gamma_diff
                            best_within_tolerance = candidate
                    else:
                        # Otherwise track best fallback
                        if (delta_error < best_delta_error) or (
                            delta_error == best_delta_error and gamma_diff < best_gamma_diff):
                            best_delta_error = delta_error
                            best_gamma_diff = gamma_diff
                            best = candidate

    return best_within_tolerance if best_within_tolerance else best



if __name__ == "__main__":
    total_delta = -100
    total_gamma = -6
    threshold = 0.002

    deltas = {
        'k1': 1,
        'k2': 10,
        'k3': 10,
        'k4': 0.3,
        'k5': 0.5,
        'k6': 0.8
    }

    gammas = {
        'k1': 0,
        'k2': 5,
        'k3': 2,
        'k4': 0.6,
        'k5': 0.4,
        'k6': 0.5
    }

    weights = {
        'k1': 0,
        'k2': 0,   # excluded if total_delta < 0 and threshold = 0.5
        'k3': 0,
        'k4': 0,
        'k5': 0,   # excluded
        'k6': 0
    }

    max_coeffs = {
        'k1': 1,
        'k2': 1,
        'k3': 1,
        'k4': 1,
        'k5': 1,
        'k6': 1
    }

    result = optimize_delta_gamma(total_delta, total_gamma, deltas, gammas, weights, threshold, max_coeffs)

    if result:
        print("✅ Optimal solution found:")
        print(f"  Keys:      {result['keys']}")
        print(f"  Coeffs:    {result['coeffs']}")
        print(f"  Δ sum:     {result['delta_sum']} (target {total_delta})")
        print(f"  Γ sum:     {result['gamma_sum']} (target {total_gamma})")
        print(f"  Γ diff:    {result['gamma_diff']}")
    else:
        print("❌ No valid combination found.")
