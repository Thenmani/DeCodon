import math
from collections import Counter

# Kyte-Doolittle Hydrophobicity (Water-Repellency)
HYDROPATHY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Intermolecular "Stickiness" / Cross-Linking Propensity inside Droplets
AGGREGATION_PROPENSITY = {
    'G': 0.05, 'P': 0.05, 'A': 0.25, 'S': 0.55, 'T': 0.50,
    'N': 0.60, 'Q': 0.65, 'D': 0.40, 'E': 0.55, 'K': 0.30,
    'R': 0.70, 'H': 0.45, 'C': 0.75, 'M': 0.65, 'V': 0.80,
    'I': 0.85, 'L': 0.85, 'F': 0.95, 'Y': 0.95, 'W': 1.00
}

ULTRA_FLEX_SPACERS = set("GP")  # Glycine & Proline prevent beta-sheet freezing
POS_CHARGED = set("KRH")
NEG_CHARGED = set("DE")


def extract_physics_features(seq: str) -> dict:
    """Decodes an amino acid sequence window into 6 biophysical droplet features."""
    seq = seq.upper().strip()
    n = max(len(seq), 1)
    counts = Counter(seq)

    # 1. Water-repellency (GRAVY score)
    gravy = sum(HYDROPATHY.get(aa, 0.0) for aa in seq) / n

    # 2. Velcro Sticker & Cross-Linking Score
    sticker_frac = sum(AGGREGATION_PROPENSITY.get(aa, 0.3) for aa in seq) / n

    # 3. Backbone Fluidity (G/P Breaker Fraction)
    spacer_frac = sum(counts[aa] for aa in ULTRA_FLEX_SPACERS) / n

    # 4. Net Charge Density
    net_charge = abs(sum(counts[aa] for aa in POS_CHARGED) - sum(counts[aa] for aa in NEG_CHARGED)) / n

    # 5. Shannon Sequence Entropy
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)

    # 6. Freezing Index (Stickiness-to-Fluidity Ratio)
    sticker_spacer_ratio = sticker_frac / max(spacer_frac, 0.08)

    return {
        "gravy_water_repellency": round(gravy, 4),
        "sticker_velcro_frac": round(sticker_frac, 4),
        "spacer_flexibility_frac": round(spacer_frac, 4),
        "net_charge_density": round(net_charge, 4),
        "sequence_entropy": round(entropy, 4),
        "sticker_to_spacer_ratio": round(sticker_spacer_ratio, 4),
    }


def apply_mutation(seq: str, mutation_code: str) -> tuple[str, dict]:
    """Applies a point mutation like 'G298S' (1-indexed) and returns mutated sequence + codon info."""
    mutation_code = mutation_code.strip().upper()
    orig_aa = mutation_code[0]
    new_aa = mutation_code[-1]
    pos = int(mutation_code[1:-1])

    idx = pos - 1
    if idx < 0 or idx >= len(seq):
        raise ValueError(f"Position {pos} is out of range for sequence of length {len(seq)}.")

    mutated_seq = seq[:idx] + new_aa + seq[idx + 1:]

    CODON_MAP = {
        'G': 'GGT', 'S': 'AGT', 'W': 'TGG', 'Y': 'TAT', 'F': 'TTT',
        'R': 'CGT', 'P': 'CCT', 'L': 'CTT', 'A': 'GCT', 'T': 'ACT',
        'V': 'GTT', 'M': 'ATG', 'E': 'GAA', 'K': 'AAA', 'C': 'TGT'
    }
    codon_info = {
        "position": pos,
        "wild_aa": orig_aa,
        "mutated_aa": new_aa,
        "wild_codon": CODON_MAP.get(orig_aa, "NNN"),
        "mutated_codon": CODON_MAP.get(new_aa, "NNN"),
    }
    return mutated_seq, codon_info