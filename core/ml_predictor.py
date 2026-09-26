import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from core.physics_engine import extract_physics_features, apply_mutation

FEATURE_KEYS = [
    "gravy_water_repellency",
    "sticker_velcro_frac",
    "spacer_flexibility_frac",
    "net_charge_density",
    "sequence_entropy",
    "sticker_to_spacer_ratio",
]

PRESET_PROTEINS = {
    "TDP43": {
        "name": "TDP-43 (TARDBP)",
        "disease": "ALS (Lou Gehrig's Disease)",
        "default_mutation": "G298S",
        "sequence": (
            "MSEYIRVTEDENDEPIEIPSEDDGTVLLSTVTAQFPGACGLRYRNPVSQCMRGVRLVEGILHAPDAGWGNLVYVVNYPKDNKRKMDETDASSAVKVKRAVQKTSDLIVLGLPWKTTEQDLKEYFSTFGEVLMVQVKKDLKTGHSKGFGFVRFTEYETQVKVMSQRHMIDGRWCDCKLPNSKQSQDEPLRSRKVFVGRCTEDMTEDELREFFSQYGDVMDVFIPKPFRAFAFVTFADDQIAQSLCGEDLIIKGISVHISNAEPKHNSNRQLERSGRFGGNPGGFGNQGGFGNSRGGGAGLGNNQGSNMGGGMNFGAFSINPAMMAAAQAALQSSWGMMGMLASQQNQSGPSGNNQNQGNMQREPNQAFGSGNNSYSGSNSGAAIGWGSASNAGSGSGFNGGFGSSMDSKSSGWGM"
        ),
    },
    "FUS": {
        "name": "FUS (RNA-Binding Protein)",
        "disease": "Severe ALS & Frontotemporal Dementia",
        "default_mutation": "G156E",
        "sequence": (
            "MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGNYGQDQSSMSSGGGSGGGYGNQDQSGGGGSGGYGQQDRG"
        ),
    },
    "TAU": {
        "name": "Tau (MAPT)",
        "disease": "Alzheimer's Disease",
        "default_mutation": "P301L",
        "sequence": (
            "MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTPTEDGSEEPGSETSDAKSTPTAEDVTAPLVDEGAPGKQAAAQPHTEIPEGTTAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDKKAKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKDNIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL"
        ),
    },
}


class DropletFreezingPredictor:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=42)
        self._train_physics_informed_baseline()

    def _train_physics_informed_baseline(self):
        rng = np.random.default_rng(42)
        n_samples = 800

        healthy = np.column_stack([
            rng.normal(-1.10, 0.35, n_samples),  # gravy
            rng.normal(0.32, 0.05, n_samples),   # sticker_velcro_frac
            rng.normal(0.45, 0.07, n_samples),   # spacer_flexibility_frac
            rng.normal(0.02, 0.03, n_samples),   # net_charge_density
            rng.normal(2.10, 0.30, n_samples),   # sequence_entropy
            rng.normal(0.72, 0.12, n_samples),   # sticker_to_spacer_ratio
        ])

        frozen = np.column_stack([
            rng.normal(-0.75, 0.35, n_samples),  # gravy
            rng.normal(0.40, 0.05, n_samples),   # sticker_velcro_frac
            rng.normal(0.30, 0.07, n_samples),   # spacer_flexibility_frac
            rng.normal(0.08, 0.04, n_samples),   # net_charge_density
            rng.normal(2.25, 0.30, n_samples),   # sequence_entropy
            rng.normal(1.15, 0.18, n_samples),   # sticker_to_spacer_ratio
        ])

        X = np.vstack([healthy, frozen])
        y = np.array([0] * n_samples + [1] * n_samples)
        self.model.fit(X, y)

    def analyze_mutation(self, protein_key: str, mutation_code: str) -> dict:
        protein = PRESET_PROTEINS[protein_key]
        wild_seq = protein["sequence"]
        mut_seq, codon_info = apply_mutation(wild_seq, mutation_code)

        # Focused 9-residue local droplet motif window around the mutation site
        pos = codon_info["position"] - 1
        start, end = max(0, pos - 4), min(len(wild_seq), pos + 5)

        wild_feats = extract_physics_features(wild_seq[start:end])
        mut_feats = extract_physics_features(mut_seq[start:end])

        X_wild = np.array([[wild_feats[k] for k in FEATURE_KEYS]])
        X_mut = np.array([[mut_feats[k] for k in FEATURE_KEYS]])

        wild_risk = float(self.model.predict_proba(X_wild)[0][1])
        mut_risk = float(self.model.predict_proba(X_mut)[0][1])

        deltas = {k: round(mut_feats[k] - wild_feats[k], 4) for k in FEATURE_KEYS}

        return {
            "protein_name": protein["name"],
            "disease": protein["disease"],
            "mutation": mutation_code,
            "codon_transition": f"{codon_info['wild_codon']} ({codon_info['wild_aa']}) → {codon_info['mutated_codon']} ({codon_info['mutated_aa']})",
            "wild_features": wild_feats,
            "mutated_features": mut_feats,
            "physics_deltas": deltas,
            "wild_freezing_risk_pct": round(wild_risk * 100, 1),
            "mutated_freezing_risk_pct": round(mut_risk * 100, 1),
            "risk_shift_pct": round((mut_risk - wild_risk) * 100, 1),
        }