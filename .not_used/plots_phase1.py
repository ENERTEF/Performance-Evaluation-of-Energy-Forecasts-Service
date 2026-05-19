import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def main():
    print("A carregar dados e modelo...")
    df = pd.read_csv("dataset2/wec_c5_mock_data_epochs.csv")
    model = joblib.load("wec_phase1_model.joblib")

    df["PCTimeStamp"] = pd.to_datetime(df["PCTimeStamp"])

    # Feature engineering: replicar exactamente o que foi feito no treino
    df["Wave_Power_Flux"] = 0.49 * (df["Hs__m"] ** 2) * df["Te__s"]
    df["hour"]  = df["PCTimeStamp"].dt.hour
    df["month"] = df["PCTimeStamp"].dt.month

    features = [
        "Hs__m",
        "Te__s",
        "Wave_Power_Flux",
        "H1/3__m",
        "H1/10__m",
        "Hmax__m",
        "HTmax__m",
        "Havg__m",
        "Hsms__m",
        "NumberOfWaves",
        "THmax__s",
        "Tavg__s",
        "Tmax__s",
        "hour",
        "month",
    ]

    # Forecast (analise absoluta)
    print("A gerar previsoes...")
    df["Predicted_Energy_kW"] = model.predict(df[features])
    df["Predicted_Energy_kW"] = df["Predicted_Energy_kW"].clip(lower=0, upper=350)

    # Intervalo de confianca estatistico: +/- 1.28 * RMSE  (~P10-P90)
    ESTIMATED_RMSE = 16.5
    Z_SCORE_90     = 1.28

    df["Predicted_Energy_P10"] = (df["Predicted_Energy_kW"] - ESTIMATED_RMSE * Z_SCORE_90).clip(lower=0)
    df["Predicted_Energy_P90"] = (df["Predicted_Energy_kW"] + ESTIMATED_RMSE * Z_SCORE_90).clip(upper=350)

    # -------------------------------------------------------------------------
    # Layout: GridSpec 2x2 com o grafico inferior a ocupar as duas colunas.
    # Usar GridSpec em vez de misturar subplot(2,2,...) com subplot(2,1,...)
    # evita o conflito de sistemas de grelha que comprimia tudo num canto.
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    gs  = fig.add_gridspec(
        nrows=2, ncols=2,
        hspace=0.20,   # espaco vertical entre linhas
        wspace=0.20,   # espaco horizontal entre colunas
    )

    # =========================================================================
    # Grafico 1 (linha 0, coluna 0): Importancia das Variaveis
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    importances = (
        pd.Series(model.feature_importances_, index=features)
        .sort_values(ascending=True)
    )
    importances.tail(8).plot(kind="barh", color="#2c3e50", ax=ax1)
    ax1.set_title(
        "1. Importancia das Variaveis\n(Output da Fase 1 para Input do DEA)",
        fontweight="bold",
        fontsize=10,
    )
    ax1.set_xlabel("Score de Importancia (XGBoost)", fontsize=9)
    ax1.tick_params(axis="both", labelsize=8)

    # =========================================================================
    # Grafico 2 (linha 0, coluna 1): Real vs Forecast com intervalo P10-P90
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    df_b9 = df[df["Buoy_ID"] == "Boia_9"].set_index("PCTimeStamp")
    df_b9_daily = df_b9[
        ["Energy_Generation_kW", "Predicted_Energy_kW",
         "Predicted_Energy_P10", "Predicted_Energy_P90"]
    ].resample("D").mean()

    # 1. Plotar a linha real (vermelha) para TODO o periodo
    ax2.plot(
        df_b9_daily.index, df_b9_daily["Energy_Generation_kW"],
        label="Producao Real", color="#e74c3c", linewidth=2,
    )

    # 2. Encontrar o ponto de corte estatistico (80% / 20%)
    # Nos dados que vao ate 31 de Maio, os 80% recaem exatamante no dia 1 de Maio
    split_date = pd.to_datetime("2025-05-01") 

    # 3. Separar os dados In-Sample e Out-of-Sample
    train_mask = df_b9_daily.index < split_date
    test_mask  = df_b9_daily.index >= split_date

    df_train = df_b9_daily[train_mask]
    df_test  = df_b9_daily[test_mask]

    # 4. Plotar a linha do modelo no Treino (Fina e cinzenta - In-Sample)
    ax2.plot(
        df_train.index, df_train["Predicted_Energy_kW"],
        label="Modelo (Treino / In-Sample)", color="gray", linestyle="--", linewidth=1.5
    )

    # 5. Plotar o Forecast Verdadeiro (Forte e verde - Out-of-Sample)
    ax2.plot(
        df_test.index, df_test["Predicted_Energy_kW"],
        label="Forecast (Teste / Out-of-Sample)", color="#27ae60", linestyle="--", linewidth=2.5
    )

    # 6. Adicionar a Banda de Incerteza APENAS na zona de Forecast
    ax2.fill_between(
        df_test.index,
        df_test["Predicted_Energy_P10"],
        df_test["Predicted_Energy_P90"],
        color="#2ecc71", alpha=0.3,
        label="Intervalo Confianca (+/-1.28 RMSE)",
    )

    # Linhas Divisorias de Epoca/Treino
    ax2.axvline(pd.to_datetime("2025-05-01"),  color="black", linestyle="-", lw=1.5, label="Corte de Treino (80%)")
    ax2.axvline(pd.to_datetime("2025-05-15"),  color="gray", linestyle=":", alpha=0.7)

    # Anotacoes posicionadas dentro do eixo Y
    y_annot = ax2.get_ylim()[1] * 0.88 if ax2.get_ylim()[1] > 0 else 280
    ax2.text(pd.to_datetime("2025-03-01"), y_annot, "ZONA DE TREINO\n(Epoca 1)", ha="center", fontsize=8, color="gray", fontweight="bold")
    ax2.text(pd.to_datetime("2025-05-07"), y_annot, "Epoca 2\n(-15%)",          ha="center", fontsize=8)
    ax2.text(pd.to_datetime("2025-05-23"), y_annot, "Epoca 3\n(Anomalia)",      ha="center", fontsize=8, color="#27ae60", fontweight="bold")

    ax2.set_title(
        "2. Previsao Probabilistica: Boia 9 (Real vs Esperado)",
        fontweight="bold", fontsize=10,
    )
    ax2.set_ylabel("Potencia (kW)", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.tick_params(axis="both", labelsize=8)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=20, ha="right")
    ax2.legend(fontsize=8, loc="lower left")

    # =========================================================================
    # Grafico 3 (linha 1, colunas 0+1): Scatter Epoca 3 -- base do DEA
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, :])   # ocupa as duas colunas da segunda linha

    df_epoca3     = df[df["Epoch_Marker"] == 3]
    expected_buoys = [f"Boia_{i}" for i in range(1, 13)]
    palette        = sns.color_palette("husl", len(expected_buoys))

    sns.scatterplot(
        data=df_epoca3,
        x="Wave_Power_Flux",
        y="Energy_Generation_kW",
        hue="Buoy_ID",
        hue_order=expected_buoys,
        alpha=0.6,
        ax=ax3,
        palette=palette,
    )

    # Fronteira teorica de producao (PPF visual)
    x_vals = np.linspace(df_epoca3["Wave_Power_Flux"].min(), df_epoca3["Wave_Power_Flux"].max(), 100)
    y_vals = np.clip(x_vals * 2.5, 0, 350)
    ax3.plot(x_vals, y_vals, color="black", linestyle="--", label="Fronteira Teorica (PPF)")

    ax3.set_title(
        "3. Comparacao de Eficiencia na Epoca 3 (Base do DEA)",
        fontweight="bold", fontsize=10,
    )
    ax3.set_xlabel("Input Principal: Fluxo de Potencia da Onda (Wave_Power_Flux)", fontsize=9)
    ax3.set_ylabel("Output: Geracao de Energia Real (kW)", fontsize=9)
    ax3.tick_params(axis="both", labelsize=8)
    ax3.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, ncol=1)

    #fig.suptitle(
    #    "WEC Phase 1 -- Analise de Desempenho dos Conversores de Energia das Ondas",
    #    fontsize=13, fontweight="bold", y=1.0,
    #)

    plt.savefig("plots/phase1/wec_phase1_analysis.png", dpi=600, bbox_inches="tight")
    print("Graficos guardados com sucesso em 'plots/phase1/wec_phase1_analysis.png'!")

if __name__ == "__main__":
    main()