import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_raw_dataset(csv_path="clean_baseline_buoy.csv"):
    print(f"A carregar dataset...{csv_path}")
    
    # Carregar os dados. O index_col=0 e parse_dates=True assumem que 
    # a primeira coluna e o Timestamp em formato ISO
    try:
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    except Exception as e:
        print(f"Erro ao carregar o CSV: {e}")
        return

    print("\n" + "="*60)
    print("1. INTEGRIDADE TEMPORAL E DIMENSAO")
    print("="*60)
    print(f"Total de registos: {len(df)}")
    print(f"Data inicial: {df.index.min()}")
    print(f"Data final: {df.index.max()}")

    # Verificar os passos de tempo (esperado: 30 minutos)
    time_diffs = df.index.to_series().diff().value_counts()
    print("\nFrequencias de amostragem encontradas no index:")
    print(time_diffs.head(5))

    print("\n" + "="*60)
    print("2. ESTATISTICAS GERAIS (Min, Max, Media, Quartis)")
    print("="*60)
    # Focamos nas variaveis chave para nao poluir o terminal
    core_vars = ['waverider1.Hs__m', 'waverider1.Te__s', 'wec_power__W']
    
    # Verifica se as colunas existem antes de fazer describe
    available_core_vars = [var for var in core_vars if var in df.columns]
    print(df[available_core_vars].describe().round(2))

    print("\n" + "="*60)
    print("3. ANALISE PROFUNDA DE GAPS (BURACOS CONSECUTIVOS)")
    print("="*60)
    for var in available_core_vars:
        is_nan = df[var].isna()
        # Logica para agrupar NaNs consecutivos e contar o tamanho de cada bloco
        consecutive_nans = is_nan.groupby((~is_nan).cumsum()).sum()
        
        max_gap = consecutive_nans.max()
        max_gap_hours = max_gap * 0.5  # Assumindo step de 30 min
        total_nans = is_nan.sum()
        
        print(f"\nVariavel: {var}")
        print(f"  Total de NaNs: {total_nans} ({total_nans/len(df)*100:.2f}%)")
        print(f"  Maior falha consecutiva: {max_gap} passos ({max_gap_hours} horas seguidas)")
        
        # Quantas falhas sao maiores que 2 horas (4 passos)?
        # Estas sao as falhas que nao podemos interpolar em seguranca
        large_gaps = (consecutive_nans > 4).sum()
        print(f"  Buracos maiores que 2h (nao interpolaveis): {large_gaps} ocorrencias")

    print("\n" + "="*60)
    print("4. TESTES DE COERENCIA FISICA")
    print("="*60)
    print("Verificacao de valores fisicamente impossiveis:")
    if 'waverider1.Hs__m' in df.columns:
        print(f"  Hs < 0 m (Erro): {(df['waverider1.Hs__m'] < 0).sum()} registos")
        print(f"  Hs > 15 m (Anormal): {(df['waverider1.Hs__m'] > 15).sum()} registos")
    
    if 'waverider1.Te__s' in df.columns:
        print(f"  Te < 0 s (Erro): {(df['waverider1.Te__s'] < 0).sum()} registos")
        print(f"  Te > 30 s (Anormal): {(df['waverider1.Te__s'] > 30).sum()} registos")
        
    if 'wec_power__W' in df.columns:
        # Pelo teu artigo, o limite nominal e 350 kW
        print(f"  Potencia < 0 W (Consumo/Erro): {(df['wec_power__W'] < 0).sum()} registos")
        print(f"  Potencia > 350 kW (Saturacao/Clipping): {(df['wec_power__W'] > 350000).sum()} registos")
        print(f"  Potencia Maxima Absoluta Registada: {df['wec_power__W'].max():.2f} W")


def analyze_golden_window(csv_path="clean_baseline_buoy.csv"):
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    
    # 1. Isolar a Golden Window
    start_golden = "2026-03-01 00:00:00"
    end_golden = "2026-06-30 23:30:00"
    df_gw = df.loc[start_golden:end_golden].copy()
    
    print("\n" + "="*60)
    print("5. ANALISE ESPECIFICA DA GOLDEN WINDOW (Mar a Jun 2026)")
    print("="*60)
    
    # 2. Gaps apenas na Golden Window
    print("Gaps na Golden Window (Wavereider Hs):")
    is_nan = df_gw['waverider1.Hs__m'].isna()
    consecutive_nans = is_nan.groupby((~is_nan).cumsum()).sum()
    print(f"  Total NaNs: {is_nan.sum()} ({is_nan.sum()/len(df_gw)*100:.2f}%)")
    print(f"  Maior buraco: {consecutive_nans.max()} passos ({consecutive_nans.max()*0.5} horas)")
    
    # 3. Remover NaNs para analise estatistica de correlacoes
    df_clean = df_gw.dropna(subset=['waverider1.Hs__m', 'waverider1.Te__s', 'wec_power__W']).copy()
    
    # 4. Calcular o Wave Power Flux (WPF) Real
    df_clean['WPF_kW_m'] = 0.49 * (df_clean['waverider1.Hs__m']**2) * df_clean['waverider1.Te__s']
    df_clean['wec_power_kW'] = df_clean['wec_power__W'] / 1000.0
    
    # Filtrar potencia negativa para as correlacoes
    df_positive = df_clean[df_clean['wec_power_kW'] > 0]
    
    print("\nCorrelacoes Lineares (Como a potencia responde as ondas):")
    correlations = df_positive[['waverider1.Hs__m', 'waverider1.Te__s', 'WPF_kW_m', 'wec_power_kW']].corr()
    print(correlations['wec_power_kW'].round(3))
    
    print("\nRacio de Captura Medio (CWR aprox.):")
    # CWR = Energia Gerada / Energia Disponivel na frente de onda (assumindo largura caracteristica da boia)
    # Aqui apenas vemos o rácio de kW gerados por kW/m de onda
    ratio = (df_positive['wec_power_kW'] / df_positive['WPF_kW_m']).mean()
    print(f"  A boia extrai em media {ratio:.2f} kW por cada kW/m de Wave Power Flux")

def plot_golden_window(csv_path="clean_baseline_buoy.csv"):
    print("A carregar e a processar dados para visualizacao...")
    
    # 1. Carregar e isolar a Golden Window
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    df_gw = df.loc["2026-03-01 00:00:00":"2026-06-30 23:30:00"].copy()
    
    # 2. Calcular variaveis derivadas
    df_gw['wec_power_kW'] = df_gw['wec_power__W'] / 1000.0
    df_gw['WPF_kW_m'] = 0.49 * (df_gw['waverider1.Hs__m']**2) * df_gw['waverider1.Te__s']
    
    # Definir estilo visual
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 12))
    
    # --- GRAFICO 1: Serie Temporal (Sazonalidade e Gaps) ---
    ax1 = plt.subplot(2, 1, 1)
    
    # Plot da Potencia (Eixo Y principal)
    color_pwr = 'tab:blue'
    ax1.set_ylabel('Potencia Gerada (kW)', color=color_pwr, fontsize=12)
    ax1.plot(df_gw.index, df_gw['wec_power_kW'], color=color_pwr, alpha=0.7, label='Power (kW)')
    ax1.tick_params(axis='y', labelcolor=color_pwr)
    
    # Plot do Hs (Eixo Y secundario)
    ax2 = ax1.twinx()
    color_hs = 'tab:orange'
    ax2.set_ylabel('Significant Wave Height - Hs (m)', color=color_hs, fontsize=12)
    ax2.plot(df_gw.index, df_gw['waverider1.Hs__m'], color=color_hs, alpha=0.5, label='Hs (m)')
    ax2.tick_params(axis='y', labelcolor=color_hs)
    
    ax1.set_title('Dinâmica Temporal: Potencia vs Estado do Mar (Março - Junho 2026)', fontsize=14, pad=15)
    
    # --- GRAFICO 2: Dispersao WPF vs Power (Nao-Linearidade) ---
    ax3 = plt.subplot(2, 2, 3)
    # Filtrar NaNs para o scatter plot
    df_scatter = df_gw.dropna(subset=['WPF_kW_m', 'wec_power_kW'])
    
    scatter = ax3.scatter(df_scatter['WPF_kW_m'], df_scatter['wec_power_kW'], 
                          c=df_scatter['waverider1.Te__s'], cmap='viridis', 
                          alpha=0.6, s=15)
    
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Periodo de Onda - Te (s)')
    
    ax3.set_xlabel('Wave Power Flux (kW/m)', fontsize=12)
    ax3.set_ylabel('Potencia Gerada (kW)', fontsize=12)
    ax3.set_title('Curva de Captura: WPF vs Potencia', fontsize=14)
    # Adicionar linha do zero para destacar consumos
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
    
    # --- GRAFICO 3: Distribuicao de Potencia (O Paradoxo Negativo) ---
    ax4 = plt.subplot(2, 2, 4)
    
    sns.histplot(df_gw['wec_power_kW'].dropna(), bins=100, ax=ax4, color='slategray', kde=True)
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero kW')
    
    ax4.set_xlabel('Potencia Gerada (kW)', fontsize=12)
    ax4.set_ylabel('Frequencia (n de periodos de 30min)', fontsize=12)
    ax4.set_title('Distribuicao da Potencia (Nota: Cauda Negativa)', fontsize=14)
    ax4.legend()
    
    # Ajustar layout e guardar
    plt.tight_layout()
    plt.savefig('analise_golden_window.png', dpi=300, bbox_inches='tight')
    print("Graficos gerados com sucesso! Verifica o ficheiro 'analise_golden_window.png'")
    plt.show()


if __name__ == "__main__":
    analyze_raw_dataset()
    analyze_golden_window()
    plot_golden_window()