import numpy as np
import pandas as pd

# --- Simulate GNSS Signal Data for Spoofing/Jamming Detection ---
n_samples_per_class = 20000        # Number of samples for each main scenario
n_samples_partial_spoof = 1600     # Challenging: Partial spoofing
n_samples_lowp_spoof = 1600        # Challenging: Low-power spoofing
n_samples_lowp_jam = 1600          # Challenging: Low-power/narrowband jamming

n_sats = 8           # Number of satellites per sample
n_features = 3       # CNR, Doppler, SignalPower
n_timesteps = 30     # Simulated time windows per sample for vision-based method

# --- Simulate Functions ---
def simulate_normal(n):
    X = []
    for _ in range(n):
        CNR = np.clip(np.random.normal(45, 4, (n_sats, n_timesteps)), 32, 55)
        Doppler = np.random.normal(0, 80, (n_sats, n_timesteps))
        Power = CNR - 174 + np.random.normal(0, 1, (n_sats, n_timesteps))
        feat = np.stack([CNR, Doppler, Power], axis=0)
        X.append(feat)
    return np.array(X), np.zeros(n, dtype=int)


def simulate_jam(n):
    X = []
    for _ in range(n):
        CNR = np.clip(np.random.normal(17, 5, (n_sats, n_timesteps)), 0, 27)
        drop = np.random.rand(n_sats, n_timesteps) < 0.18
        CNR[drop] = 0
        Doppler = np.random.normal(0, 300, (n_sats, n_timesteps))
        Power = CNR - 174 + np.random.normal(0, 1, (n_sats, n_timesteps))
        feat = np.stack([CNR, Doppler, Power], axis=0)
        X.append(feat)
    return np.array(X), np.full(n, 2)

def simulate_spoof(n):
    X = []
    for _ in range(n):
        base_cnr = np.random.normal(50, 2)
        CNR = np.clip(np.random.normal(base_cnr, 0.6, (n_sats, n_timesteps)), 35, 54)
        base_dopp = np.random.uniform(-35, 35)
        Doppler = np.random.normal(base_dopp, 5, (n_sats, n_timesteps))
        Power = CNR - 174 + np.random.normal(0, 1, (n_sats, n_timesteps))
        feat = np.stack([CNR, Doppler, Power], axis=0)
        X.append(feat)
    return np.array(X), np.ones(n, dtype=int)

def simulate_partial_spoof(n):
    X = []
    for _ in range(n):
        # randomly choose spoofed satellites
        spoof_mask = np.random.choice([0, 1], size=(n_sats, 1), p=[0.6, 0.4])
        spoof_mask = np.repeat(spoof_mask, n_timesteps, axis=1)
        base_cnr = np.random.normal(46, 3)
        CNR = np.clip(np.random.normal(base_cnr, 4, (n_sats, n_timesteps)), 30, 55)
        # Add spoofing boost to some satellites/timesteps
        CNR += spoof_mask * np.random.uniform(4, 10, (n_sats, n_timesteps))
        base_dopp = np.random.normal(0, 80)
        Doppler = np.random.normal(base_dopp, 80, (n_sats, n_timesteps))
        Doppler += spoof_mask * np.random.uniform(-10, 10, (n_sats, n_timesteps))
        Power = CNR - 174 + np.random.normal(0, 1, (n_sats, n_timesteps))
        feat = np.stack([CNR, Doppler, Power], axis=0)
        X.append(feat)
    return np.array(X), np.ones(n, dtype=int) # Label 1 (spoof, challenging)

def simulate_lowp_spoof(n):
    X = []
    for _ in range(n):
        base_cnr = np.random.normal(43, 3)
        CNR = np.clip(np.random.normal(base_cnr, 2, (n_sats, n_timesteps)), 33, 47)
        base_dopp = np.random.normal(0, 60)
        Doppler = np.random.normal(base_dopp, 12, (n_sats, n_timesteps))
        Power = CNR - 174 + np.random.normal(0, 1, (n_sats, n_timesteps))
        feat = np.stack([CNR, Doppler, Power], axis=0)
        X.append(feat)
    return np.array(X), np.ones(n, dtype=int) # Label 1

def simulate_lowp_jam(n):
    X = []
    for _ in range(n):
        base_cnr = np.random.normal(38, 3)
        CNR = np.clip(np.random.normal(base_cnr, 2, (n_sats, n_timesteps)), 25, 44)
        # select a few satellites/times to degrade further
        mask = np.random.rand(n_sats, n_timesteps) < 0.1
        CNR[mask] -= np.random.uniform(7, 13, np.sum(mask))
        CNR = np.clip(CNR, 0, None)
        Doppler = np.random.normal(0, 90, (n_sats, n_timesteps))
        Power = CNR - 174 + np.random.normal(0, 1, (n_sats, n_timesteps))
        feat = np.stack([CNR, Doppler, Power], axis=0)
        X.append(feat)
    return np.array(X), np.full(n, 2) # Label 2

# --- Simulate All Classes ---
X_normal, y_normal = simulate_normal(n_samples_per_class)
X_jam, y_jam = simulate_jam(n_samples_per_class)
X_spoof, y_spoof = simulate_spoof(n_samples_per_class)
X_partial_spoof, y_partial_spoof = simulate_partial_spoof(n_samples_partial_spoof)
X_lowp_spoof, y_lowp_spoof = simulate_lowp_spoof(n_samples_lowp_spoof)
X_lowp_jam, y_lowp_jam = simulate_lowp_jam(n_samples_lowp_jam)

# --- Combine and Shuffle ---
X_all = np.concatenate([
    X_normal, X_jam, X_spoof,
    X_partial_spoof, X_lowp_spoof, X_lowp_jam
], axis=0)
y_all = np.concatenate([
    y_normal, y_jam, y_spoof,
    y_partial_spoof, y_lowp_spoof, y_lowp_jam
], axis=0)

# Shuffle
rng = np.random.default_rng(42)
indices = rng.permutation(len(X_all))
X_all = X_all[indices]
y_all = y_all[indices]

# --- Reshape for ML Pipelines ---
n_samples = X_all.shape[0]
X_flat = X_all.reshape(n_samples, n_features * n_sats * n_timesteps)

# --- Create DataFrame for ML Pipelines ---
colnames = []
for feat_idx, feat_name in enumerate(['CNR', 'Doppler', 'SignalPower']):
    for sat in range(n_sats):
        for t in range(n_timesteps):
            colnames.append(f'{feat_name}_sat{sat}_t{t}')
df = pd.DataFrame(X_flat, columns=colnames)
df['label'] = y_all

# --- Save to CSV ---
csv_path = '/home/nader/Desktop/GNSS/gnss_dataset.csv'
df.to_csv(csv_path, index=False)
print(f"Dataset saved to {csv_path}")
print(f"Shape: {df.shape}")
print("Label counts:", df['label'].value_counts().to_dict())

# --- Save as Numpy Array ---
np.save('gnss_dataset.npy', {'features': X_flat, 'labels': y_all})

