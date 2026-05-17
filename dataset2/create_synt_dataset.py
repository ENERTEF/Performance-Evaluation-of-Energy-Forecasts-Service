import os
import pandas as pd
import numpy as np

np.random.seed(42)

# 1. Periodo de tempo (Freq: 30 minutos)
timestamps = pd.date_range(start="2025-01-01 00:00", end="2025-05-31 23:30", freq="30min")
n_records = len(timestamps)

# 2. Gerar condicoes ambientais base (macroscopicas) baseadas na API
base_hs = np.random.normal(loc=2.5, scale=0.8, size=n_records).clip(0.5, 6.0)
base_te = np.random.normal(loc=8.5, scale=1.2, size=n_records).clip(4.0, 14.0)

# Outras variaveis derivadas estatisticamente das reais para manter realismo
base_hmax = base_hs * np.random.uniform(1.5, 1.9, n_records)
base_h1_3 = base_hs * 1.05  # Approximate
base_h1_10 = base_hs * 1.27 # Approximate


buoys = [f"Boia_{i}" for i in range(1, 13)]
failing_buoys = ['Boia_9', 'Boia_10', 'Boia_11', 'Boia_12']

mock_data = []

# 3. Gerar os dados especificos para cada WEC
for i, buoy_id in enumerate(buoys):
    # Simular o desfasamento temporal espacial (fase shift) entre WECs
    # Shift de i elementos (simula um atraso ligeiro na propagacao da onda)
    hs = np.roll(base_hs, shift=i) + np.random.normal(0, 0.05, n_records)
    te = np.roll(base_te, shift=i) + np.random.normal(0, 0.2, n_records)
    
    # Calcular o fluxo real de potencia em kW/m: 0.49 * Hs^2 * Te
    wave_power_flux = 0.49 * (hs**2) * te
    
    # Transformar o Fluxo na Geracao da WEC C5 (estimativa de absorcao)
    # A C5 e um point absorber, assumimos uma capture width ratio e um max de 350kW
    capture_width_ratio = 2.5 # metros hipoteticos de captura
    energy = (wave_power_flux * capture_width_ratio)
    
    performance_multiplier = np.ones(n_records)
    
    mask_epoch2 = (timestamps >= "2025-05-01") & (timestamps < "2025-05-15")
    mask_epoch3 = (timestamps >= "2025-05-15")
    
    performance_multiplier[mask_epoch2] = 0.85
    
    if buoy_id in failing_buoys:
        performance_multiplier[mask_epoch3] = 0.45 
    else:
        performance_multiplier[mask_epoch3] = 0.85 
        
    energy = energy * performance_multiplier
    energy = energy.clip(0, 350) + np.random.normal(0, 2.0, n_records)
    energy = energy.clip(0, 350)
    
    epoch_labels = np.ones(n_records, dtype=int)
    epoch_labels[mask_epoch2] = 2
    epoch_labels[mask_epoch3] = 3
    
    df_buoy = pd.DataFrame({
        'PCTimeStamp': timestamps,
        'Buoy_ID': buoy_id,
        'Hs__m': hs,
        'Te__s': te,
        'H1/3__m': np.roll(base_h1_3, shift=i) + np.random.normal(0, 0.05, n_records),
        'H1/10__m': np.roll(base_h1_10, shift=i) + np.random.normal(0, 0.05, n_records),
        'Hmax__m': np.roll(base_hmax, shift=i) + np.random.normal(0, 0.1, n_records),
        'HTmax__m': np.roll(base_hmax, shift=i) * 0.9 + np.random.normal(0, 0.1, n_records),
        'Havg__m': hs * 0.6 + np.random.normal(0, 0.05, n_records),
        'Hsms__m': hs * 1.1 + np.random.normal(0, 0.05, n_records),
        "NumberOfWaves":       (1800 / (te * 0.8) + np.random.normal(0, 15, n_records)).astype(int).clip(10, None),
        'THmax__s': te * 1.2 + np.random.normal(0, 0.5, n_records),
        'Tavg__s': te * 0.8 + np.random.normal(0, 0.2, n_records),
        'Tmax__s': te * 1.5 + np.random.normal(0, 0.5, n_records),
        # Variavel alvo sintetica necessaria para a avaliacao DEA/XGBoost funcionar
        'Energy_Generation_kW': energy,
        'Epoch_Marker': epoch_labels
    })
    
    mock_data.append(df_buoy)

final_dataset = pd.concat(mock_data, ignore_index=True)

csv_path = 'wec_c5_mock_data_epochs.csv'
final_dataset.to_csv(csv_path, index=False)

print(f"Dataset sintetico (API Waverider | 30min) gerado. Total: {len(final_dataset)} rows")