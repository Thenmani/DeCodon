<div align="center">

# 🧬 DeCodon
### **Decoding 'Codons' in ML Way** *(To Keep Brain Cells Alive & Fluid)*
**When Cellular "Liquid Droplets" Freeze into ALS & Alzheimer's Disease**

### 🌐 **Live Application: [https://decodon.onrender.com/](https://decodon.onrender.com/)**



</div>

---

## 💡 The Problem: Why Do Brain Cells Die in ALS & Alzheimer's?

Inside healthy neurons, proteins organize into dynamic **"Liquid Droplets"** (via *Liquid-Liquid Phase Separation*)—acting like miniature lava lamps where cellular workers move freely to repair RNA and maintain the cell.

However, a **single 1-letter DNA/mRNA codon mutation**—such as swapping **Glycine (`GGT` → `G`)**, a tiny flexible hinge, for **Serine (`AGT` → `S`)**, a sticky residue—alters the biophysical forces of the protein strand. The fluid droplet **freezes into a solid aggregate brick**, trapping cellular workers inside and causing motor and brain neurons to die.

---

## 🎯 Why DeCodon Matters (Real-World Impact)

* ⚡ **Instant Mutation Screening (Months in a Wet Lab → Seconds in ML):** There are millions of possible 1-letter genetic mutations, and testing them physically in a lab takes months and thousands of dollars each. DeCodon uses physics-informed ML to instantly predict which specific mutations will cause healthy liquid protein droplets in neurons to freeze into solid bricks (triggering ALS and Alzheimer's)—enabling **early-stage detection and recovery** before irreversible neuron damage occurs.
* 💊 **From Cryptic Math to Drug Hypotheses:** Traditional bioinformatics tools only output raw biophysical numbers that are hard to interpret. DeCodon's AI Agent translates those exact physical shifts into plain-English disease mechanisms and actionable therapeutic hypotheses to help researchers design targeted drugs faster and restore healthy droplet fluidity in the earliest stages of disease.

---

## 🚀 What DeCodon Does

**DeCodon** bridges molecular biophysics, machine learning, and agentic AI in a single-screen interactive platform:

```text
[1. Neuron] ➔ [2. DNA Blueprint] ➔ [3. mRNA] ➔ [4. 3-Letter Codon] ➔ [5. 20-Letter Protein Coil]
                                                                               │
                                                                               ▼
[Mechanistic & Therapeutic Report] ◄── [Gemini AI Agent] ◄── [6-Feature Physics Engine + ML Classifier]
```

1. **Interactive Protein Pipeline & Cell Simulator:** Visualizes the end-to-end biological flow from neuron to 3-letter codon to coiled protein strand, paired with a real-time HTML5 Canvas simulator comparing **Healthy Liquid Droplets (`Glycine G`)** vs. **Frozen Solid Aggregates (`Serine S`)**.
2. **6-Feature Biophysical Engine:** Computes exact sequence-level biophysical shifts between wild-type and mutated proteins (Hydropathy/GRAVY water-repellency, sticky aromatic/polar Velcro rings, flexible glycine/proline spacers, net electrical charge, and disorder propensity).
3. **Physics-Informed ML Classifier:** Uses a Scikit-Learn Gradient Boosting model trained on biophysical phase-separation signatures to quantify the exact **Liquid-to-Solid Freezing Risk (%)** and mutation risk shift ($\Delta\%$).
4. **Autonomous AI Agent:** Synthesizes the biophysical feature deltas and ML risk scores into an actionable **Mechanistic & Therapeutic Report**, proposing targeted small-molecule and antisense oligonucleotide (ASO) rescue strategies.

---

## 🧪 Featured Clinical Disease Targets

| Target Protein | Gene | Associated Disease | Benchmark Point Mutation | Codon Transition | Biophysical Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TDP-43** | `TARDBP` | **ALS** (Amyotrophic Lateral Sclerosis) | **`G298S`** | `GGT (G)` → `AGT (S)` | Replaces flexible Glycine hinge with H-bonding Serine, accelerating solid fibril freezing |
| **FUS** | `FUS` | **Severe ALS / FTD** | **`G156E`** | `GGT (G)` → `GAA (E)` | Introduces bulky negatively charged Glutamate into the prion-like low-complexity domain |
| **Tau** | `MAPT` | **Alzheimer's Disease** | **`P301L`** | `CCG (P)` → `CTG (L)` | Removes helix-breaking Proline kink and adds hydrophobic Leucine, driving neurofibrillary tangles |

---

## 🛠️ Tech Stack

* **Backend & Packaging:** Python, FastAPI, Uvicorn, `uv` (`pyproject.toml` & `uv.lock`)
* **Physics & ML Engine:** Scikit-Learn (`GradientBoostingClassifier`), NumPy, Custom Biophysical Feature Extractor
* **AI Reasoning Agent:** Google GenAI SDK (`gemini-2.5-flash`)
* **Frontend:** HTML5, Tailwind CSS, HTML5 Canvas Physics Animation, Marked.js

---

## ⚡ Quick Start (Run Locally with `uv`)

### 1. Clone the Repository
```bash
git clone [https://github.com/Thenmani/DeCodon.git](https://github.com/Thenmani/DeCodon.git)
cd DeCodon
```

### 2. Sync Dependencies via `uv`
```bash
uv sync
```

### 3. Add Your Gemini API Key
Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 4. Launch the Server
```bash
uv run uvicorn main:app --reload --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---
### Run complete test suite
uv run python -m unittest discover -s tests -v

---

<div align="center">
  <strong>Gemini AI powered Live ML and Agentic solution</strong> • Built to decode neurodegenerative phase transitions.
</div>
