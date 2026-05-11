import os
import pandas as pd
import numpy as np

# Garantir que a pasta datasets existe
os.makedirs("datasets", exist_ok=True)

# Definir semente para garantir reprodutibilidade dos dados gerados
np.random.seed(42)

# 1. Definir o periodo de tempo (Estendido para 5 meses)
# Janeiro a Abril representarao os 80% de treino (Golden Period).
# Maio representara os 20% de teste (onde injetamos as ineficiencias).
timestamps = pd.date_range(start="2025-01-01 00:00", end="2025-05-31 23:00", freq="h")
n_records = len(timestamps)

# 2. Gerar condicoes ambientais base (comuns as 3 boias)
base_wave_hs = np.random.normal(loc=2.5, scale=0.8, size=n_records).clip(0.5, 6.0)
base_wave_tp = np.random.normal(loc=9.0, scale=1.5, size=n_records).clip(5.0, 15.0)
base_wind_speed = np.random.normal(loc=15.0, scale=5.0, size=n_records).clip(0.0, 30.0)
base_sst = np.random.normal(loc=14.5, scale=0.5, size=n_records)

# 3. Definir as coordenadas das 3 boias (distanciadas em poucos km)
buoys = {
    'Boia_1': {'lat': 41.140, 'lon': -8.700},
    'Boia_2': {'lat': 41.155, 'lon': -8.710},
    'Boia_3': {'lat': 41.130, 'lon': -8.725}
}

mock_data = []

# 4. Gerar os dados especificos para cada boia
for buoy_id, coords in buoys.items():
    # Adicionar ruido espacial
    hs = base_wave_hs + np.random.normal(0, 0.05, n_records)
    tp = base_wave_tp + np.random.normal(0, 0.2, n_records)
    
    # Calcular geracao de energia teorica (proporcional a Hs^2 * Tp)
    energy = (hs**2 * tp) * 4.2 
    
    # Criar um multiplicador de performance inicializado a 1.0 (Golden Period total)
    performance_multiplier = np.ones(n_records)
    
    # Criar mascaras temporais exatas em vez de divisoes inteiras
    mask_epoch2 = (timestamps >= "2025-05-01") & (timestamps < "2025-05-15")
    mask_epoch3 = (timestamps >= "2025-05-15")
    
    # Epoca 2: Degradacao Comum (1 a 15 de Maio). Todas as boias perdem 15%
    performance_multiplier[mask_epoch2] = 0.85
    
    # Epoca 3: Degradacao Especifica (15 a 31 de Maio).
    if buoy_id == 'Boia_3':
        performance_multiplier[mask_epoch3] = 0.45 # Boia 3 sofre anomalia grave
    else:
        performance_multiplier[mask_epoch3] = 0.85 # Restantes mantem degradacao comum
        
    # Aplicar o multiplicador a producao de energia
    energy = energy * performance_multiplier
        
    # Limitar a potencia nominal do equipamento (350kW) e adicionar ruido mecanico
    energy = energy.clip(0, 350) + np.random.normal(0, 2.0, n_records)
    energy = energy.clip(0, 350)
    
    # Gerar a coluna de indicacao da Epoca para ajudar na visualizacao
    epoch_labels = np.ones(n_records, dtype=int)
    epoch_labels[mask_epoch2] = 2
    epoch_labels[mask_epoch3] = 3
    
    # Criar o DataFrame para esta boia
    df_buoy = pd.DataFrame({
        'PCTimeStamp': timestamps,
        'Buoy_ID': buoy_id,
        'Buoy_Latitude': coords['lat'],
        'Buoy_Longitude': coords['lon'],
        'SST': base_sst + np.random.normal(0, 0.1, n_records),
        'Salinity': np.random.normal(35.2, 0.1, n_records),
        'Conductivity': np.random.normal(52.4, 0.5, n_records),
        'Dissolved_Oxygen': np.random.normal(8.5, 0.3, n_records),
        'Turbidity': np.random.normal(3.0, 0.5, n_records),
        'Chlorophyll_a': np.random.normal(1.5, 0.2, n_records),
        'Wave_Hs': hs,
        'Wave_Tp': tp,
        'Wave_Dir': np.random.normal(270, 10, n_records) % 360,
        'Current_Speed': np.random.normal(0.4, 0.1, n_records).clip(0, None),
        'Current_Dir': np.random.normal(180, 20, n_records) % 360,
        'Air_Temperature': base_sst - 1.0 + np.random.normal(0, 0.5, n_records),
        'Atmospheric_Pressure': np.random.normal(1015, 5, n_records),
        'Relative_Humidity': np.random.normal(80, 5, n_records).clip(0, 100),
        'Solar_Radiation': np.random.normal(400, 200, n_records).clip(0, 1000),
        'Wind_Speed': base_wind_speed + np.random.normal(0, 0.5, n_records),
        'Wind_Direction': np.random.normal(300, 15, n_records) % 360,
        'Rainfall': np.random.exponential(0.5, n_records).clip(0, 10),
        'Battery_Voltage': np.random.normal(12.5, 0.1, n_records),
        'Energy_Generation_kW': energy,
        'Epoch_Marker': epoch_labels
    })
    
    mock_data.append(df_buoy)

# 5. Juntar tudo num unico dataset
final_dataset = pd.concat(mock_data, ignore_index=True)

# Exportar para CSV (Garantir que guarda dentro da pasta datasets para evitar erros de caminho)
csv_path = 'wec_c5_mock_data_epochs.csv'
final_dataset.to_csv(csv_path, index=False)

print(f"Dataset sintetico do Golden Period criado com sucesso.")
print(f"Guardado em '{csv_path}'. Total de registos: {len(final_dataset)}")