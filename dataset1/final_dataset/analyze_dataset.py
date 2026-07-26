import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_synthetic_dataset(csv_path="final_wec_fleet_2026.csv"):
    print(f"A carregar dataset sintetico da frota... {csv_path}")
    
    try:
        df = pd.read_csv(csv_path, parse_dates=['PCTimeStamp'])
    except Exception as e:
        print(f"Erro ao carregar o CSV: {e}")
        return None

    # Obter lista de boias e definir as que avariam na Epoca 3
    buoys = df['Buoy_ID'].unique()
    failing_buoys = ['Boia_9', 'Boia_10', 'Boia_11', 'Boia_12']
    
    print("\n" + "="*60)
    print("1. INTEGRIDADE DA FROTA E DIMENSAO")
    print("="*60)
    print(f"Total de registos: {len(df)}")
    print(f"Numero de Boias: {len(buoys)} -> {sorted(buoys)}")
    print(f"Periodo: {df['PCTimeStamp'].min()} a {df['PCTimeStamp'].max()}")

    print("\n" + "="*60)
    print("2. TESTES DE LIMITES FISICOS (Clipping a 350kW)")
    print("="*60)
    out_of_bounds_high = (df['Energy_Generation_kW'] > 350.1).sum()
    out_of_bounds_low = (df['Energy_Generation_kW'] < 0).sum()
    
    print(f"  Registos > 350 kW (Saturacao violada): {out_of_bounds_high}")
    print(f"  Registos < 0 kW (Sem clipping no limite inferior): {out_of_bounds_low}")
    print(f"  Potencia Maxima Absoluta: {df['Energy_Generation_kW'].max():.2f} kW")

    print("\n" + "="*60)
    print("3. VALIDACAO DAS EPOCAS (Comportamento Medio de Energia)")
    print("="*60)
    
    # Criar uma coluna binaria de status para facilitar
    df['Status'] = np.where(df['Buoy_ID'].isin(failing_buoys), 'Degraded', 'Healthy')
    
    # Racio de captura aproximado (CWR proxy) = Energia / WPF
    # Adicionamos +0.1 ao WPF para evitar divisao por zero
    df['CWR_Proxy'] = df['Energy_Generation_kW'] / (df['Wave_Power_Flux'] + 0.1)
    
    # Analise pivot para ver como a media de energia (kW) reage por Epoca e Status
    pivot_energy = pd.pivot_table(
        df, 
        values='Energy_Generation_kW', 
        index='Epoch_Marker', 
        columns='Status', 
        aggfunc='mean'
    ).round(2)
    
    print("Media de Energia Gerada (kW) por Epoca e Estado da Boia:")
    print("(Epoca 1: Treino | Epoca 2: False Pos | Epoca 3: Falha Mecanica)")
    print(pivot_energy)
    
    print("\nVerificacao Logica:")
    print("-> Epoca 1: Valores saudaveis e degradados devem ser semelhantes.")
    print("-> Epoca 2: Ambos devem cair substancialmente (Spreading).")
    print("-> Epoca 3: Healthy recupera, Degraded colapsa.")

    return df

def plot_synthetic_golden_window(df_full):
    # Validar se recebemos dados
    if df_full is None:
        print("Nenhum dado fornecido para plotar.")
        return
        
    print("\nA processar dados sinteticos para visualizacao...")
    
    # Vamos focar a analise visual numa boia saudavel vs uma degradada para ser evidente
    df_gw = df_full[df_full['Buoy_ID'] == 'Boia_11'].copy() # Boia_11 e uma das que avaria
    df_gw.set_index('PCTimeStamp', inplace=True)
    df_gw.sort_index(inplace=True)
    
    power_col = 'Energy_Generation_kW'
    hs_col = 'Hs__m'
    wpf_col = 'Wave_Power_Flux'
    epoch_col = 'Epoch_Marker'
    
    # Definir estilo visual
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 12))
    
    epoch_colors = {1: '#2ca02c', 2: '#ff7f0e', 3: '#d62728'}
    
    # --- GRAFICO 1: Serie Temporal (Sazonalidade e Epocas) ---
    ax1 = plt.subplot(2, 1, 1)
    
    # Extrair dinamicamente as datas das epocas para o sombreamento
    t_start = df_gw.index.min()
    t_ep2 = df_gw[df_gw[epoch_col] == 2].index.min()
    t_ep3 = df_gw[df_gw[epoch_col] == 3].index.min()
    t_end = df_gw.index.max()
    
    ax1.axvspan(t_start, t_ep2, color=epoch_colors[1], alpha=0.1, label='Epoca 1 (Treino/Nominal)')
    ax1.axvspan(t_ep2, t_ep3, color=epoch_colors[2], alpha=0.15, label='Epoca 2 (Falso Positivo)')
    ax1.axvspan(t_ep3, t_end, color=epoch_colors[3], alpha=0.1, label='Epoca 3 (Mecanica/Degradada)')
    
    # Plot da Potencia (Eixo Y principal)
    color_pwr = 'tab:blue'
    ax1.set_ylabel('Potencia Gerada (kW)', color=color_pwr, fontsize=12)
    ax1.plot(df_gw.index, df_gw[power_col], color=color_pwr, alpha=0.8, label='Power (Boia_11)')
    ax1.tick_params(axis='y', labelcolor=color_pwr)
    
    # Plot do Hs (Eixo Y secundario)
    ax2 = ax1.twinx()
    color_hs = 'tab:gray' 
    ax2.set_ylabel('Significant Wave Height - Hs (m)', color=color_hs, fontsize=12)
    ax2.plot(df_gw.index, df_gw[hs_col], color=color_hs, alpha=0.5, label='Hs (m)')
    ax2.tick_params(axis='y', labelcolor=color_hs)
    
    # Juntar legendas
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    ax1.set_title('Dinamica Temporal Sintetica (Boia_11): Observar a quebra geral na Epoca 2 e o colapso na Epoca 3', fontsize=14, pad=15)
    
    # --- GRAFICO 2: Dispersao WPF vs Power (Nao-Linearidade e Epocas) ---
    ax3 = plt.subplot(2, 2, 3)
    df_scatter = df_gw.dropna(subset=[wpf_col, power_col])
    
    sns.scatterplot(data=df_scatter, x=wpf_col, y=power_col, hue=epoch_col, 
                    palette=epoch_colors, alpha=0.7, s=20, ax=ax3, edgecolor=None)
    
    ax3.set_xlabel('Wave Power Flux (kW/m)', fontsize=12)
    ax3.set_ylabel('Potencia Gerada (kW)', fontsize=12)
    ax3.set_title('Curva de Captura: Achatamento da eficiencia SFA nas Epocas 2 e 3', fontsize=14)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    # --- GRAFICO 3: Distribuicao de Potencia (Empilhada por Epoca) ---
    ax4 = plt.subplot(2, 2, 4)
    
    sns.histplot(data=df_gw, x=power_col, hue=epoch_col, palette=epoch_colors, 
                 bins=100, multiple="stack", ax=ax4, kde=False)
    ax4.axvline(0, color='black', linestyle='--', linewidth=2, label='Zero kW')
    
    ax4.set_xlabel('Potencia Gerada (kW)', fontsize=12)
    ax4.set_ylabel('Frequencia (n de periodos de 30min)', fontsize=12)
    ax4.set_title('Distribuicao da Potencia Sintetica (Boia_11)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('validacao_frota_final_2026.png', dpi=300, bbox_inches='tight')
    print("Graficos gerados com sucesso! Verifica o ficheiro 'validacao_frota_final_2026.png'")
    plt.show()

if __name__ == "__main__":
    csv_file = "final_wec_fleet_2026.csv"
    df_synth = analyze_synthetic_dataset(csv_file)
    plot_synthetic_golden_window(df_synth)