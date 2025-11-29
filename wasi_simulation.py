import pandas as pd
import numpy as np

# --- Simulation Parameters ---
n_participants = 1000
np.random.seed(42) # for reproducibility

# --- Generate Participant Demographics ---
participant_id = np.arange(1, n_participants + 1)
age = np.random.normal(24.5, 5.2, n_participants)
age = np.clip(age, 16, 40)

gender = np.random.binomial(1, 0.5, n_participants)
gender_label = ["Female" if g == 0 else "Male" for g in gender]

sport = np.random.choice(["Running", "Jumping", "Cutting"], n_participants, p=[0.4, 0.3, 0.3])

# --- Generate Biomechanical Variables ---
fv_data = np.random.normal(0.15, 0.04, n_participants)
fv_data = np.clip(fv_data, 0.05, 0.35)

ta_data = np.random.normal(0.08, 0.03, n_participants)
ta_data = np.clip(ta_data, 0.02, 0.20)

ld_data = np.random.normal(0.12, 0.05, n_participants)
ld_data = np.clip(ld_data, 0.03, 0.30)

ba_data = np.random.normal(0.10, 0.04, n_participants)
ba_data = np.clip(ba_data, 0.02, 0.25)

# --- Add Correlation Structure ---
correlation_factor = 0.4
fv_data = fv_data + correlation_factor * (ta_data - 0.08)
fv_data = np.clip(fv_data, 0.05, 0.35)

# --- Calculate WASe Scores ---
weights = {"FV": 0.35, "TA": 0.28, "LD": 0.22, "BA": 0.15}
wase_scores = (weights["FV"] * fv_data + 
               weights["TA"] * ta_data + 
               weights["LD"] * ld_data + 
               weights["BA"] * ba_data)

# --- Generate Injury Outcomes ---
beta_0 = -2.5
beta_1 = 3.0
injury_probability = 1 / (1 + np.exp(-(beta_0 + beta_1 * wase_scores)))

threshold = np.percentile(injury_probability, 80)
injury_status = (injury_probability >= threshold).astype(int)

# --- Create Full Dataset ---
full_dataset = pd.DataFrame({
    "Participant_ID": participant_id,
    "Age": np.round(age, 1),
    "Gender": gender_label,
    "Sport": sport,
    "Force_Variability_FV": np.round(fv_data, 4),
    "Temporal_Asymmetry_TA": np.round(ta_data, 4),
    "Load_Distribution_LD": np.round(ld_data, 4),
    "Bilateral_Asymmetry_BA": np.round(ba_data, 4),
    "WASe_Score": np.round(wase_scores, 4),
    "Injury_Probability": np.round(injury_probability, 4),
    "Injury_Status": injury_status
})

# --- Save to Excel ---
output_file = "/home/ubuntu/WASI_Full_Dataset_1000_Calibrated.xlsx"
full_dataset.to_excel(output_file, index=False)

print(f"Simulation complete. Full dataset saved to {output_file}")
