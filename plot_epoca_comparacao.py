import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar o estilo do gráfico
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

def main():
    print("A carregar os dados e a calcular as médias por época...")
    df = pd.read_csv("datasets/wec_c5_mock_data_epochs.csv")

    # 1. Recriar a feature de Input
    df["Wave_Power_Flux"] = df["Wave_Hs"] ** 2 * df["Wave_Tp"]

    # 2. Calcular o ponto médio de operação de cada boia por época
    df_agg = df.groupby(["Epoch_Marker", "Buoy_ID"])[["Wave_Power_Flux", "Energy_Generation_kW"]].mean().reset_index()

    # 3. Preparar a Figura com 3 subgráficos (lado a lado)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, sharex=True)
    colors = {"Boia_1": "#2196F3", "Boia_2": "#4CAF50", "Boia_3": "#F44336"}
    markers = {"Boia_1": "o", "Boia_2": "s", "Boia_3": "^"}

    for i, epoch in enumerate([1, 2, 3]):
        ax = axes[i]
        df_ep = df_agg[df_agg["Epoch_Marker"] == epoch].copy()

        # Plotar as 3 boias como pontos grandes
        for _, row in df_ep.iterrows():
            buoy = row["Buoy_ID"]
            ax.scatter(
                row["Wave_Power_Flux"], 
                row["Energy_Generation_kW"], 
                color=colors[buoy], 
                marker=markers[buoy],
                s=250, # Tamanho do ponto
                label=buoy,
                edgecolor='black',
                linewidth=1.5,
                alpha=0.7, # Transparência para ver as boias sobrepostas
                zorder=5
            )

        # Desenhar a "Fronteira de Eficiência Média" (Linha a partir da origem)
        df_ep["Efficiency_Ratio"] = df_ep["Energy_Generation_kW"] / df_ep["Wave_Power_Flux"]
        best_buoy = df_ep.loc[df_ep["Efficiency_Ratio"].idxmax()]
        
        # Projetar a linha da fronteira (esticada para cobrir o gráfico)
        max_x = 100
        slope = best_buoy["Efficiency_Ratio"]
        ax.plot([0, max_x], [0, slope * max_x], color='gray', linestyle='--', zorder=1, label="Fronteira")

        # Formatação do eixo
        ax.set_title(f"Época {epoch}", fontweight='bold', fontsize=14)
        ax.set_xlabel("Input Médio: Wave Power Flux")
        if i == 0:
            ax.set_ylabel("Output Médio: Geração (kW)")
            ax.legend(title="Ativos", loc="upper left")
        
        # ---> ZOOM APLICADO AQUI <---
        ax.set_xlim(50, 80)
        ax.set_ylim(100, 250)

    # Título Geral
    plt.suptitle("Análise DEA Simplificada: Ponto de Operação Médio por Boia e Época", fontsize=16, y=1.05, fontweight='bold')
    plt.tight_layout()
    
    # Guardar a figura
    plt.savefig("wec_epoca_comparacao.png", dpi=300, bbox_inches='tight')
    print("Gráfico gerado com sucesso e guardado como 'wec_epoca_comparacao.png'!")

if __name__ == "__main__":
    main()