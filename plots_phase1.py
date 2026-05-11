import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error

# Configurar estilo dos graficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def main():
    print("A carregar dados e modelo...")
    df = pd.read_csv("datasets/wec_c5_mock_data_epochs.csv")
    model = joblib.load("wec_phase1_model.joblib")
    
    # Forçar a conversao para datetime
    df["PCTimeStamp"] = pd.to_datetime(df["PCTimeStamp"])
    
    # 1. Replicar a Feature Engineering necessaria para o modelo
    df["Wave_Power_Flux"] = df["Wave_Hs"] ** 2 * df["Wave_Tp"]
    df["hour"] = df["PCTimeStamp"].dt.hour
    df["month"] = df["PCTimeStamp"].dt.month
    
    # Adicionar as novas variaveis fisicas para corresponder ao treino
    df["Wave_Steepness"] = df["Wave_Hs"] / (df["Wave_Tp"] ** 2)
    diff = np.abs(df["Wave_Dir"] - df["Wind_Direction"])
    df["Misalignment"] = np.minimum(diff, 360 - diff)
    df["Wind_Power_Density"] = df["Wind_Speed"] ** 3
    
    # A lista TEM de ser exatamente igual a FEATURE_COLS do phase1.py
    features = [
        "Wave_Hs",
        "Wave_Tp",
        "Wave_Power_Flux",       # engineered feature
        "Wave_Steepness",        # engineered feature
        "Misalignment",          # engineered feature
        "Wind_Power_Density",    # engineered feature
        "Wind_Speed",
        "Current_Speed",
        "Wave_Dir",
        "Air_Temperature",
        "Atmospheric_Pressure",
        # Temporal features
        "hour",
        "month",
    ]
    
    # Fazer o forecast (Analise Absoluta)
    print("A gerar previsoes...")
    df["Predicted_Energy_kW"] = model.predict(df[features])
    df["Predicted_Energy_kW"] = df["Predicted_Energy_kW"].clip(lower=0, upper=350)
    
    # Calcular Incerteza Estatistica (Z-score de 1.28 para ~P10 a ~P90)
    # Assumindo um RMSE historico na ordem dos ~47 kW (do teu log)
    ESTIMATED_RMSE = 47.0 
    Z_SCORE_90 = 1.28 
    
    df["Predicted_Energy_P10"] = (df["Predicted_Energy_kW"] - (ESTIMATED_RMSE * Z_SCORE_90)).clip(lower=0)
    df["Predicted_Energy_P90"] = (df["Predicted_Energy_kW"] + (ESTIMATED_RMSE * Z_SCORE_90)).clip(upper=350)
    
    # Criar a figura com 3 subplots
    fig = plt.figure(figsize=(18, 12))
    
    # =========================================================================
    # Grafico 1: Importancia das Variaveis (O Filtro para o DEA)
    # =========================================================================
    ax1 = plt.subplot(2, 2, 1)
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    importances.tail(5).plot(kind='barh', color='#2c3e50', ax=ax1)
    ax1.set_title("1. Importância das Variáveis (Output da Fase 1 para Input do DEA)", fontweight='bold')
    ax1.set_xlabel("Score de Importância (XGBoost)")
    
    # =========================================================================
    # Grafico 2: Real vs Forecast com Intervalo de Confiança Estatístico
    # =========================================================================
    ax2 = plt.subplot(2, 2, 2)
    df_b3 = df[df["Buoy_ID"] == "Boia_3"].set_index("PCTimeStamp")
    
    # Fazer resample diario para o grafico nao ficar uma mancha tremida
    df_b3_daily = df_b3[["Energy_Generation_kW", "Predicted_Energy_kW", "Predicted_Energy_P10", "Predicted_Energy_P90"]].resample("D").mean()
    
    # Zona sombreada de Incerteza baseada no RMSE
    ax2.fill_between(
        df_b3_daily.index, 
        df_b3_daily["Predicted_Energy_P10"], 
        df_b3_daily["Predicted_Energy_P90"], 
        color='#2ecc71', alpha=0.2, label='Intervalo Confiança Estimado (±1.28 RMSE)'
    )
    
    ax2.plot(df_b3_daily.index, df_b3_daily["Predicted_Energy_kW"], label='Forecast (Gémeo Digital)', color='#27ae60', linestyle='--')
    ax2.plot(df_b3_daily.index, df_b3_daily["Energy_Generation_kW"], label='Produção Real', color='#e74c3c', linewidth=2)
    
    # Marcar as novas epocas baseadas no "Golden Period"
    ax2.axvline(pd.to_datetime("2025-05-01"), color='gray', linestyle=':', alpha=0.7)
    ax2.axvline(pd.to_datetime("2025-05-15"), color='gray', linestyle=':', alpha=0.7)
    
    ax2.text(pd.to_datetime("2025-04-15"), 300, 'Época 1\n(Golden Period)', ha='center')
    ax2.text(pd.to_datetime("2025-05-07"), 300, 'Época 2\n(-15%)', ha='center', fontsize=9)
    ax2.text(pd.to_datetime("2025-05-23"), 300, 'Época 3\n(Anomalia)', ha='center', fontsize=9)
    
    ax2.set_title("2. Previsão Probabilística Estimada: Boia 3 (Real vs Esperado)", fontweight='bold')
    ax2.set_ylabel("Potência (kW)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.legend()
    
    # =========================================================================
    # Grafico 3: Pair Plot / Precursor da PPF (A Ponte para a Fase 2)
    # =========================================================================
    ax3 = plt.subplot(2, 1, 2)
    
    # Focar apenas na Epoca 3 para ver a falha da Boia 3 em relacao as outras
    df_epoca3 = df[df["Epoch_Marker"] == 3]
    
    sns.scatterplot(
        data=df_epoca3, 
        x="Wave_Power_Flux", 
        y="Energy_Generation_kW", 
        hue="Buoy_ID", 
        alpha=0.6, 
        ax=ax3,
        palette=["#3498db", "#f1c40f", "#e74c3c"]
    )
    
    # Desenhar uma linha de tendencia superior (Simulacao visual da Fronteira DEA)
    x_vals = np.linspace(df_epoca3["Wave_Power_Flux"].min(), df_epoca3["Wave_Power_Flux"].max(), 100)
    # Linha teorica superior ajustada ao novo dataset
    y_vals = np.clip(x_vals * 3.5, 0, 350) 
    ax3.plot(x_vals, y_vals, color='black', linestyle='--', label='Fronteira Teórica de Produção (PPF)')
    
    ax3.set_title("3. Comparação de Eficiência na Época 3 (A base do DEA)", fontweight='bold')
    ax3.set_xlabel("Input Principal: Fluxo de Potência da Onda (Wave_Power_Flux)")
    ax3.set_ylabel("Output: Geração de Energia Real (kW)")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig("wec_phase1_analysis.png", dpi=300)
    print("Graficos guardados com sucesso em 'wec_phase1_analysis.png'!")

if __name__ == "__main__":
    main()