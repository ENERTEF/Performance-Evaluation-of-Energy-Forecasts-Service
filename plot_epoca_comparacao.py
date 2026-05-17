import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar o estilo do grafico
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.0)

def main():
    print("A carregar os dados e a calcular as medias por epoca...")
    df = pd.read_csv("dataset2/wec_c5_mock_data_epochs.csv")

    # 1. Recriar a feature de Input Principal
    if "Wave_Power_Flux" not in df.columns:
        df["Wave_Power_Flux"] = 0.49 * (df["Hs__m"] ** 2) * df["Te__s"]

    # Definir os inputs que queremos testar (as 4 dimensoes)
    inputs_to_test = ["Wave_Power_Flux", "Hs__m", "Te__s", "NumberOfWaves"]
    target = "Energy_Generation_kW"

    # 2. Calcular o ponto medio de operacao de cada boia por epoca
    cols_to_agg = inputs_to_test + [target]
    df_agg = df.groupby(["Epoch_Marker", "Buoy_ID"])[cols_to_agg].mean().reset_index()

    # Gerar paleta de 12 cores
    expected_buoys = [f"Boia_{i}" for i in range(1, 13)]
    palette = sns.color_palette("husl", len(expected_buoys))
    colors = {buoy: color for buoy, color in zip(expected_buoys, palette)}

    # Limite global do Eixo Y (Energia) para garantir escala justa em todos os plots
    y_max_global = df_agg[target].max() * 1.10

    # =========================================================================
    # PARTE A: O GRAFICO 3D
    # =========================================================================
    print("A gerar o grafico 3D...")
    fig3d = plt.figure(figsize=(18, 6))
    for i, epoch in enumerate([1, 2, 3], start=1):
        ax = fig3d.add_subplot(1, 3, i, projection='3d')
        df_ep = df_agg[df_agg["Epoch_Marker"] == epoch].copy()

        for _, row in df_ep.iterrows():
            buoy = row["Buoy_ID"]
            x_val = row["Wave_Power_Flux"]
            y_val = row["NumberOfWaves"]
            z_val = row[target]
            
            ax.scatter(x_val, y_val, z_val, color=colors[buoy], s=120, edgecolor='black', alpha=0.9, label=buoy if epoch == 1 else "")
            ax.plot([x_val, x_val], [y_val, y_val], [0, z_val], color=colors[buoy], linestyle=':', alpha=0.5)

        ax.set_title(f"Epoca {epoch}", fontweight='bold', fontsize=14)
        ax.set_xlabel("Input 1: Wave Power Flux")
        ax.set_ylabel("Input 2: Number Of Waves")
        ax.set_zlabel("Output: Geracao (kW)")
        
        ax.set_xlim(df_agg["Wave_Power_Flux"].min() * 0.95, df_agg["Wave_Power_Flux"].max() * 1.05)
        ax.set_ylim(df_agg["NumberOfWaves"].min() * 0.95, df_agg["NumberOfWaves"].max() * 1.05)
        ax.set_zlim(0, y_max_global)
        ax.view_init(elev=20, azim=135)

    fig3d.axes[0].legend(title="Ativos", loc='upper left', bbox_to_anchor=(0.0, 1.15), ncol=4, fontsize=8)
    plt.suptitle("Analise DEA 3D: Ponto de Operacao Medio por Boia e Epoca", fontsize=16, y=1.05, fontweight='bold')
    plt.tight_layout()
    plt.savefig("wec_epoca_comparacao_3d.png", dpi=300, bbox_inches='tight')

    # =========================================================================
    # PARTE B: OS N PLOTS (1 PARA CADA INPUT) Lado a Lado (1x3)
    # =========================================================================
    print("A gerar os N graficos 2D individuais em formato 1x3...")
    
    for i, input_col in enumerate(inputs_to_test):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        # Manter o mesmo limite do eixo X para as 3 epocas para ver a deslocacao horizontal
        x_min_global = df_agg[input_col].min() * 0.95
        x_max_global = df_agg[input_col].max() * 1.05

        for j, epoch in enumerate([1, 2, 3]):
            ax = axes[j]
            df_ep = df_agg[df_agg["Epoch_Marker"] == epoch]

            sns.scatterplot(
                data=df_ep, 
                x=input_col, 
                y=target, 
                hue="Buoy_ID", 
                hue_order=expected_buoys,
                palette=colors,
                alpha=0.8,
                s=150, 
                ax=ax,
                legend=(j == 0) # Apenas colocar legenda no primeiro plot
            )

            ax.set_title(f"Epoca {epoch}", fontweight='bold')
            ax.set_xlabel(f"INPUT {i+1}: {input_col}")
            
            if j == 0:
                ax.set_ylabel("OUTPUT: Geracao Media (kW)")
            else:
                ax.set_ylabel("")
                
            ax.set_xlim(x_min_global, x_max_global)
            ax.set_ylim(0, y_max_global)
            ax.grid(True, linestyle='--', alpha=0.6)
            
        # Ajustar a legenda
        axes[0].legend(title="Boias", bbox_to_anchor=(0, 1.2), loc='upper left', ncol=4)
        
        plt.suptitle(f"Ponto de Operacao Medio: Output vs {input_col}", fontweight='bold', y=1.05, fontsize=14)
        plt.tight_layout()
        filename = f"wec_media_epoca_2D_{input_col}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # =========================================================================
    # PARTE C: O PLOT N+1 (INPUT AGREGADO) Lado a Lado (1x3)
    # =========================================================================
    print("A gerar o grafico do Indice Agregado em formato 1x3...")
    
    df_agg["Aggregated_Input_Index"] = 0
    for col in inputs_to_test:
        col_min = df_agg[col].min()
        col_max = df_agg[col].max()
        if col_max != col_min:
            df_agg["Aggregated_Input_Index"] += (df_agg[col] - col_min) / (col_max - col_min)

    fig_agg, axes_agg = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    x_min_agg = df_agg["Aggregated_Input_Index"].min() - 0.1
    x_max_agg = df_agg["Aggregated_Input_Index"].max() + 0.1

    for j, epoch in enumerate([1, 2, 3]):
        ax = axes_agg[j]
        df_ep = df_agg[df_agg["Epoch_Marker"] == epoch]

        sns.scatterplot(
            data=df_ep, 
            x="Aggregated_Input_Index", 
            y=target, 
            hue="Buoy_ID", 
            hue_order=expected_buoys,
            palette=colors,
            alpha=0.8,
            s=150,
            ax=ax,
            legend=(j == 0)
        )
        
        ax.set_title(f"Epoca {epoch}", fontweight='bold')
        ax.set_xlabel("Indice Agregado")
        
        if j == 0:
            ax.set_ylabel("OUTPUT: Geracao Media (kW)")
        else:
            ax.set_ylabel("")
            
        ax.set_xlim(x_min_agg, x_max_agg)
        ax.set_ylim(0, y_max_global)
        ax.grid(True, linestyle='--', alpha=0.6)
        
    axes_agg[0].legend(title="Boias", bbox_to_anchor=(0, 1.2), loc='upper left', ncol=4)
    
    plt.suptitle("Multidimensionalidade: Output vs Indice Agregado (Normalizacao Conjunta)", fontweight='bold', y=1.05, fontsize=14)
    plt.tight_layout()
    plt.savefig("wec_media_epoca_2D_Aggregated.png", dpi=150, bbox_inches='tight')
    plt.close(fig_agg)
    
    print("Todos os graficos foram gerados com sucesso!")

if __name__ == "__main__":
    main()