def measure_overlap(a, b, c, d):
    """Returns the length of the intersection of [a, b] with (c, d]."""
    return max(0, min(b, d) - max(a, c))

def expected_payout(B1, B2):
    """
    Computes the total expected payout when B1 is fixed at 200
    and B2 ranges from 285 to 320.
    """
    # Distribution is uniform over 110 units: [160–200] and [250–320]
    total_width = 110

    # [160–200]: Only B1 can be used here
    overlap1_b1 = measure_overlap(160, 200, float('-inf'), B1)
    payout1 = overlap1_b1 * (320 - B1)

    # [250–320]: Split between B1 and B2
    overlap2_b1 = measure_overlap(250, 320, float('-inf'), B1)
    overlap2_b2 = measure_overlap(250, 320, B1, B2)
    payout2 = overlap2_b1 * (320 - B1) + overlap2_b2 * (320 - B2)

    total_expected = (payout1 + payout2) / total_width
    return total_expected

def evaluate_fixed_lower_bid():
    B1 = 200
    print(f"{'B2':>5} | {'Expected Payout':>18}")
    print("-" * 26)
    for B2 in range(285, 321):
        payout = expected_payout(B1, B2)
        print(f"{B2:>5} | {payout:>18.4f}")

if __name__ == "__main__":
    evaluate_fixed_lower_bid()
