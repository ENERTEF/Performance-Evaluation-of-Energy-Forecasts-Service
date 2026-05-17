import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar o estilo do grafico
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.0)

def main():
    print("A carregar os dados e a calcular as medias por epoca...")
    # Usar o novo caminho garantido para o dataset
    df = pd.read_csv("dataset2/wec_c5_mock_data_epochs.csv")

    # 1. Recriar a feature de Input Principal com a nova formula fisica
    if "Wave_Power_Flux" not in df.columns:
        df["Wave_Power_Flux"] = 0.49 * (df["Hs__m"] ** 2) * df["Te__s"]

    # 2. Calcular o ponto medio de operacao de cada boia por epoca
    # Substituir Wind_Speed por NumberOfWaves para refletir o novo modelo 2D do DEA
    df_agg = df.groupby(["Epoch_Marker", "Buoy_ID"])[["Wave_Power_Flux", "NumberOfWaves", "Energy_Generation_kW"]].mean().reset_index()

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
            y_val = row["NumberOfWaves"] # Eixo Y e agora o Numero de Ondas
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
            ax.plot([x_val, x_val], [y_val, y_val], [0, z_val], color=colors[buoy], linestyle=':', alpha=0.5)

        # Formatacao dos eixos
        ax.set_title(f"Epoca {epoch}", fontweight='bold', fontsize=14)
        ax.set_xlabel("Input 1: Wave Power Flux")
        ax.set_ylabel("Input 2: Number Of Waves")
        ax.set_zlabel("Output: Geracao (kW)")
        
        # Limites dinamicos: O grafico faz "zoom in" exato aos dados disponiveis
        # Calculamos os limites globais usando df_agg para garantir que as 3 epocas partilham a mesma escala
        x_min, x_max = df_agg["Wave_Power_Flux"].min(), df_agg["Wave_Power_Flux"].max()
        y_min, y_max = df_agg["NumberOfWaves"].min(), df_agg["NumberOfWaves"].max()
        z_max = df_agg["Energy_Generation_kW"].max()

        # Aplicar margens de 5% para os pontos nao colarem as paredes do grafico
        ax.set_xlim(x_min * 0.95, x_max * 1.05)
        ax.set_ylim(y_min * 0.95, y_max * 1.05)
        
        # O eixo Z (Energia) convem comecar sempre no 0 para que a percepcao visual da quebra seja real
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