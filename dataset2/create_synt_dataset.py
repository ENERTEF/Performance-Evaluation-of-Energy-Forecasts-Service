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

    spatial_factor = 0.75 + (i * 0.04)

    hs_local = np.roll(base_hs, shift=i) * spatial_factor + np.random.normal(0, 0.05, n_records)
    te_local = np.roll(base_te, shift=i) + np.random.normal(0, 0.2, n_records)
    
    # Calcular o fluxo real de potencia em kW/m: 0.49 * Hs^2 * Te
    wave_power_flux = 0.49 * (hs_local**2) * te_local
    
    # 2. CURVA REALISTA DO WEC (Saturacao Assintotica)
    # Substitui a reta linear por uma curva que achata nos 350kW
    # Formula: P_max * (1 - exp(-k * WPF))
    ideal_energy = 350.0 * (1.0 - np.exp(-0.015 * wave_power_flux))
    
    # 1. Definir as mascaras temporais
    mask_epoch1 = timestamps < "2025-05-01"
    mask_epoch2 = (timestamps >= "2025-05-01") & (timestamps < "2025-05-15")
    mask_epoch3 = timestamps >= "2025-05-15"
    
    # 2. Inicializar vetores de DGP (Data Generating Process) para SFA
    base_eff = np.ones(n_records)
    u = np.zeros(n_records)
    
    # 3. Ruido simetrico v ~ N(0, sigma_v) (comum a todos os periodos)
    v = np.random.normal(0, 4.0, n_records)
    
    # 4. Ineficiencia tecnica u ~ |N(0, sigma_u)|
    # Epoch 1: Treino (tem de existir uma ineficiencia natural residual para o SFA aprender)
    u[mask_epoch1] = np.abs(np.random.normal(0, 2.0, np.sum(mask_epoch1)))
    
    # Epoch 2: Fator ambiental sub-otimo para toda a frota
    base_eff[mask_epoch2] = 0.85
    u[mask_epoch2] = np.abs(np.random.normal(0, 5.0, np.sum(mask_epoch2)))
    
    # Epoch 3: Avaria grave mecanica vs Operacao Saudavel
    if buoy_id in failing_buoys:
        base_eff[mask_epoch3] = 0.45 
        # Aumento massivo da variancia de ineficiencia u (PTO avariado)
        u[mask_epoch3] = np.abs(np.random.normal(0, 35.0, np.sum(mask_epoch3))) 
    else:
        # Boias saudaveis regressam ao baseline perfeito
        base_eff[mask_epoch3] = 1.0 
        u[mask_epoch3] = np.abs(np.random.normal(0, 2.0, np.sum(mask_epoch3)))
        
    # 5. Aplicar a decomposicao SFA: Fronteira + Ruido - Ineficiencia
    energy = (ideal_energy * base_eff) + v - u
    energy = energy.clip(0, 350)
    
    epoch_labels = np.ones(n_records, dtype=int)
    epoch_labels[mask_epoch2] = 2
    epoch_labels[mask_epoch3] = 3
    
    df_buoy = pd.DataFrame({
        'PCTimeStamp': timestamps,
        'Buoy_ID': buoy_id,
        'Hs__m': hs_local,
        'Te__s': te_local,
        'Wave_Power_Flux': wave_power_flux,
        'H1/3__m': np.roll(base_h1_3, shift=i) + np.random.normal(0, 0.05, n_records),
        'H1/10__m': np.roll(base_h1_10, shift=i) + np.random.normal(0, 0.05, n_records),
        'Hmax__m': np.roll(base_hmax, shift=i) + np.random.normal(0, 0.1, n_records),
        'HTmax__m': np.roll(base_hmax, shift=i) * 0.9 + np.random.normal(0, 0.1, n_records),
        'Havg__m': hs_local * 0.6 + np.random.normal(0, 0.05, n_records),
        'Hsms__m': hs_local * 1.1 + np.random.normal(0, 0.05, n_records),
        "NumberOfWaves":       (1800 / (te_local * 0.8) + np.random.normal(0, 15, n_records)).astype(int).clip(10, None),
        'THmax__s': te_local * 1.2 + np.random.normal(0, 0.5, n_records),
        'Tavg__s': te_local * 0.8 + np.random.normal(0, 0.2, n_records),
        'Tmax__s': te_local * 1.5 + np.random.normal(0, 0.5, n_records),
        # Variavel alvo sintetica necessaria para a avaliacao DEA/XGBoost funcionar
        'Energy_Generation_kW': energy,
        'Epoch_Marker': epoch_labels
    })
    
    mock_data.append(df_buoy)

final_dataset = pd.concat(mock_data, ignore_index=True)

csv_path = 'wec_c5_mock_data_epochs.csv'
final_dataset.to_csv(csv_path, index=False)

print(f"Dataset sintetico (API Waverider | 30min) gerado. Total: {len(final_dataset)} rows")