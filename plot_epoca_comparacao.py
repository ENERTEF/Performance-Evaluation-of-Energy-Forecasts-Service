import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar o estilo do grafico
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.0)

def main():
    print("A carregar os dados e a calcular as medias por epoca...")
    df = pd.read_csv("datasets/wec_c5_mock_data_epochs.csv")

    # 1. Recriar a feature de Input Principal
    if "Wave_Power_Flux" not in df.columns:
        df["Wave_Power_Flux"] = df["Wave_Hs"] ** 2 * df["Wave_Tp"]

    # 2. Calcular o ponto medio de operacao de cada boia por epoca
    df_agg = df.groupby(["Epoch_Marker", "Buoy_ID"])[["Wave_Power_Flux", "Wind_Speed", "Energy_Generation_kW"]].mean().reset_index()

    # 3. Preparar a Figura 3D com 3 subgraficos (lado a lado)
    fig = plt.figure(figsize=(18, 6))
    
    # Gerar paleta de 12 cores
    expected_buoys = [f"Boia_{i}" for i in range(1, 13)]
    palette = sns.color_palette("husl", len(expected_buoys))
    colors = {buoy: color for buoy, color in zip(expected_buoys, palette)}

    for i, epoch in enumerate([1, 2, 3], start=1):
        ax = fig.add_subplot(1, 3, i, projection='3d')
        df_ep = df_agg[df_agg["Epoch_Marker"] == epoch].copy()

        # Plotar as 12 boias no espaco 3D
        for _, row in df_ep.iterrows():
            buoy = row["Buoy_ID"]
            x_val = row["Wave_Power_Flux"]
            y_val = row["Wind_Speed"]
            z_val = row["Energy_Generation_kW"]
            
            # Ponto principal
            ax.scatter(
                x_val, y_val, z_val,
                color=colors[buoy],
                s=120,
                edgecolor='black',
                linewidth=1.0,
                alpha=0.9,
                label=buoy if epoch == 1 else ""
            )
            
            # Linha vertical (Stem) ligando o ponto ao chao (z=0)
            # Ajuda a perceber a quebra de producao visualmente
            ax.plot([x_val, x_val], [y_val, y_val], [0, z_val], color=colors[buoy], linestyle=':', alpha=0.5)

        # Formatacao dos eixos
        ax.set_title(f"Epoca {epoch}", fontweight='bold', fontsize=14)
        ax.set_xlabel("Input 1: Wave Power Flux")
        ax.set_ylabel("Input 2: Wind Speed")
        ax.set_zlabel("Output: Geracao (kW)")
        
        # Limites dinâmicos: O gráfico faz "zoom in" exato aos dados disponíveis
        # Calculamos os limites globais usando df_agg para garantir que as 3 épocas partilham a mesma escala
        x_min, x_max = df_agg["Wave_Power_Flux"].min(), df_agg["Wave_Power_Flux"].max()
        y_min, y_max = df_agg["Wind_Speed"].min(), df_agg["Wind_Speed"].max()
        z_max = df_agg["Energy_Generation_kW"].max()

        # Aplicar margens de 5% para os pontos não colarem às paredes do gráfico
        ax.set_xlim(x_min * 0.95, x_max * 1.05)
        ax.set_ylim(y_min * 0.95, y_max * 1.05)
        
        # O eixo Z (Energia) convém começar sempre no 0 para que a percepção visual da quebra de 45% seja real
        ax.set_zlim(0, z_max * 1.10)
        
        # Angulo de visualizacao (Elevacao e Azimute)
        ax.view_init(elev=20, azim=135)

    # Legenda apenas no primeiro grafico para nao poluir
    fig.axes[0].legend(title="Ativos", loc='upper left', bbox_to_anchor=(0.0, 1.15), ncol=4, fontsize=8)

    # Titulo Geral
    plt.suptitle("Analise DEA 3D: Ponto de Operacao Medio por Boia e Epoca", fontsize=16, y=1.05, fontweight='bold')
    plt.tight_layout()
    
    # Guardar a figura
    plt.savefig("wec_epoca_comparacao_3d.png", dpi=300, bbox_inches='tight')
    print("Grafico 3D gerado com sucesso e guardado como 'wec_epoca_comparacao_3d.png'!")

if __name__ == "__main__":
    main()