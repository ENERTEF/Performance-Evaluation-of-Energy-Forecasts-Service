"""
Baseline buoy (Boia 1) cleaning and reconstruction pipeline -- MODULAR VERSION.

Estrutura:
  1) Limpeza (clean_and_interpolate_baseline) -- so preenche falhas curtas
     (<= MAX_INTERP_STEPS); falhas longas ficam NaN de proposito.
  2) Analises de diagnostico (analyze_raw_dataset, analyze_golden_window).
  3) Visualizacao inicial (plot_golden_window).
  4) Reconstrucao modular em duas passagens (engineer_features ->
     detect_outliers_and_train_model -> reconstruct_dataset ->
     plot_reconstruction -> run_pipeline_reconstruction):
       - Passagem 1: treina um XGBoost "bruto" so com operacao ativa (>10 kW).
       - Filtro: remove os OUTLIER_QUANTILE piores residuos (saltos/anomalias).
       - Passagem 2: retreina so com os dados limpos.
       - Reconstrucao: substitui idle/curtailment E outliers pontuais pela
         previsao do modelo final, com um pouco de ruido mecanico injetado
         para manter a textura organica da serie.

Duas correcoes feitas ao juntar o bloco modular ao pipeline principal:

  FIX A (idle_mask sem filtro de mar): a versao que me mandaste substituia
  QUALQUER instante com potencia <= ACTIVE_POWER_KW, incluindo mar em
  espelho (WPF ~ 0), pela previsao do modelo. Isso reintroduz o problema
  original que ja tinhamos corrigido (Mini-Problema 2.1): potencia zero com
  mar sem energia e fisica real, nao curtailment, e nao deve ser reescrita.
  Voltei a exigir WPF > ACTIVE_WPF_KW_M para o idle_mask contar como
  curtailment.

  FIX B (limiar de outlier fixo em 40.0 kW): troquei o valor hardcoded por
  um multiplo do desvio-padrao dos residuos do modelo limpo
  (OUTLIER_ERROR_SIGMA * base_noise_std). Assim o limiar adapta-se ao nivel
  de ruido real da boia, em vez de assumir que 40 kW e sempre a fasquia
  certa (o mesmo espirito da banda de confianca dinamica que o artigo
  publicado usa em vez de um limiar fixo).

NOTA (para leres antes de correr isto em serio): a injecao de ruido
sintetico em reconstruct_dataset (proportional_noise) torna a serie
reconstruida visualmente mais "organica", mas esse ruido e fabricado, nao
medido. Se a Fase 2 da tua dissertacao (SFA) vai decompor o residuo em
ruido simetrico (v) e ineficiencia mecanica (u), injetar ruido artificial
aqui pode enviesar essa decomposicao mais tarde -- vale a pena documentar
esta decisao explicitamente e, se possivel, correr o teu SFA com e sem esta
injecao para veres se muda o gamma/lambda estimado.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# --- Constants (tune here) --------------------------------------------------
GOLDEN_WINDOW_START = "2026-03-01 00:00:00"
GOLDEN_WINDOW_END = "2026-06-30 23:30:00"

# Only bridge gaps up to 2h (4 steps of 30 min). Longer gaps stay NaN.
MAX_INTERP_STEPS = 4

# Training / curtailment mask: only genuinely healthy, non-commissioning
# operation counts as "active"; below this AND with sea energy = curtailed.
ACTIVE_POWER_KW = 10.0
ACTIVE_WPF_KW_M = 0.5

# Two-pass residual filter (reconstruction module).
OUTLIER_QUANTILE = 0.92      # pass-1: rejeita os (1-0.92)=8% piores residuos
OUTLIER_ERROR_SIGMA = 3.5    # pass-2: substitui pontos com erro > N x sigma
NOISE_MULTIPLIER = 0.10      # ruido mecanico injetado (fracao do sigma limpo)

# Colunas espectrais extra da API CorPower (usadas se existirem no CSV).
EXTRA_SPECTRAL_COLS = [
    "waverider1.H1/3__m",
    "waverider1.H1/10__m",
    "waverider1.HTmax__m",
    "waverider1.Havg__m",
    "waverider1.Hsms__m",
    "waverider1.NumberOfWaves",
    "waverider1.THmax__s",
    "waverider1.Tavg__s",
    "waverider1.Tmax__s",
    "waverider1.Spectral_directional_spread",
]
# -----------------------------------------------------------------------------


def clean_and_interpolate_baseline(input_csv="../raw/baseline_buoy_raw.csv", output_csv="clean_baseline_buoy.csv"):
    print("\n" + "="*60)
    print("A iniciar o processo de limpeza e interpolacao...")
    print("="*60)

    df_raw = pd.read_csv(input_csv, parse_dates=True, index_col=0)
    df_gw = df_raw.loc[GOLDEN_WINDOW_START:GOLDEN_WINDOW_END].copy()
    df_gw = df_gw.asfreq("30min")

    if 'wec_power__W' in df_gw.columns:
        linhas_negativas = (df_gw['wec_power__W'] < 0).sum()
        df_gw['wec_power__W'] = df_gw['wec_power__W'].clip(lower=0.0)
        print(f"-> {linhas_negativas} registos de potencia negativa passados a 0 W.")

    critical_cols = ['waverider1.Hs__m', 'waverider1.Te__s', 'wec_power__W']
    was_nan = df_gw[critical_cols].isna()

    cols_metocean = [c for c in df_gw.columns if c != 'wec_power__W']
    df_gw[cols_metocean] = df_gw[cols_metocean].interpolate(
        method='pchip', limit=MAX_INTERP_STEPS, limit_direction='both', limit_area='inside'
    )

    if 'wec_power__W' in df_gw.columns:
        df_gw['wec_power__W'] = df_gw['wec_power__W'].interpolate(
            method='linear', limit=MAX_INTERP_STEPS, limit_direction='both', limit_area='inside'
        )

    is_now_valid = df_gw[critical_cols].notna()
    interpolated_mask = (was_nan & is_now_valid).any(axis=1)
    df_gw['is_interpolated'] = interpolated_mask.astype(int)

    total_interpolados = df_gw['is_interpolated'].sum()
    print(f"-> Foram interpolados com sucesso {total_interpolados} registos (falhas curtas).")

    remaining_nans = df_gw['waverider1.Hs__m'].isna().sum()
    print(f"-> Restam {remaining_nans} registos com NaNs (falhas longas mantidas intactas).")

    df_gw.to_csv(output_csv)
    print(f"\nDataset intermediario guardado como: {output_csv}")
    print(f"Total de linhas na Golden Window: {len(df_gw)}")


def analyze_raw_dataset(csv_path="clean_baseline_buoy.csv"):
    print(f"\nA carregar dataset...{csv_path}")

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

    time_diffs = df.index.to_series().diff().value_counts()
    print("\nFrequencias de amostragem encontradas no index:")
    print(time_diffs.head(5))

    print("\n" + "="*60)
    print("2. ESTATISTICAS GERAIS (Min, Max, Media, Quartis)")
    print("="*60)
    core_vars = ['waverider1.Hs__m', 'waverider1.Te__s', 'wec_power__W']
    available_core_vars = [var for var in core_vars if var in df.columns]
    print(df[available_core_vars].describe().round(2))

    print("\n" + "="*60)
    print("3. ANALISE PROFUNDA DE GAPS (BURACOS CONSECUTIVOS)")
    print("="*60)
    for var in available_core_vars:
        is_nan = df[var].isna()
        consecutive_nans = is_nan.groupby((~is_nan).cumsum()).sum()

        max_gap = consecutive_nans.max()
        max_gap_hours = max_gap * 0.5
        total_nans = is_nan.sum()

        print(f"\nVariavel: {var}")
        print(f"  Total de NaNs: {total_nans} ({total_nans/len(df)*100:.2f}%)")
        print(f"  Maior falha consecutiva: {max_gap} passos ({max_gap_hours} horas seguidas)")

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
        print(f"  Potencia < 0 W (Consumo/Erro): {(df['wec_power__W'] < 0).sum()} registos")
        print(f"  Potencia > 350 kW (Saturacao/Clipping): {(df['wec_power__W'] > 350000).sum()} registos")
        print(f"  Potencia Maxima Absoluta Registada: {df['wec_power__W'].max():.2f} W")


def analyze_golden_window(csv_path="clean_baseline_buoy.csv"):
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)

    df_gw = df.loc[GOLDEN_WINDOW_START:GOLDEN_WINDOW_END].copy()

    print("\n" + "="*60)
    print("5. ANALISE ESPECIFICA DA GOLDEN WINDOW (Mar a Jun 2026)")
    print("="*60)

    print("Gaps na Golden Window (Wavereider Hs):")
    is_nan = df_gw['waverider1.Hs__m'].isna()
    consecutive_nans = is_nan.groupby((~is_nan).cumsum()).sum()
    print(f"  Total NaNs: {is_nan.sum()} ({is_nan.sum()/len(df_gw)*100:.2f}%)")
    print(f"  Maior buraco: {consecutive_nans.max()} passos ({consecutive_nans.max()*0.5} horas)")

    df_clean = df_gw.dropna(subset=['waverider1.Hs__m', 'waverider1.Te__s', 'wec_power__W']).copy()

    df_clean['WPF_kW_m'] = 0.49 * (df_clean['waverider1.Hs__m']**2) * df_clean['waverider1.Te__s']
    df_clean['wec_power_kW'] = df_clean['wec_power__W'] / 1000.0

    df_positive = df_clean[df_clean['wec_power_kW'] > 0]

    print("\nCorrelacoes Lineares (Como a potencia responde as ondas):")
    correlations = df_positive[['waverider1.Hs__m', 'waverider1.Te__s', 'WPF_kW_m', 'wec_power_kW']].corr()
    print(correlations['wec_power_kW'].round(3))

    print("\nRacio de Captura Medio (CWR aprox.):")
    ratio = (df_positive['wec_power_kW'] / df_positive['WPF_kW_m']).mean()
    print(f"  A boia extrai em media {ratio:.2f} kW por cada kW/m de Wave Power Flux")


def plot_golden_window(csv_path="clean_baseline_buoy.csv"):
    print("\nA carregar e a processar dados para visualizacao inicial...")

    df_gw = pd.read_csv(csv_path, parse_dates=True, index_col=0)

    df_gw['wec_power_kW'] = df_gw['wec_power__W'] / 1000.0
    df_gw['WPF_kW_m'] = 0.49 * (df_gw['waverider1.Hs__m']**2) * df_gw['waverider1.Te__s']

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 1, 1)
    mask_interp = df_gw['is_interpolated'] == 1

    color_pwr = 'tab:blue'
    ax1.set_ylabel('Potencia Gerada (kW)', color=color_pwr, fontsize=12)
    ax1.plot(df_gw.index, df_gw['wec_power_kW'], color=color_pwr, alpha=0.6, label='Power Real (kW)')
    ax1.scatter(df_gw.index[mask_interp], df_gw.loc[mask_interp, 'wec_power_kW'],
                color='red', s=12, zorder=5, label='Power Interpolada')
    ax1.tick_params(axis='y', labelcolor=color_pwr)

    ax2 = ax1.twinx()
    color_hs = 'tab:orange'
    ax2.set_ylabel('Significant Wave Height - Hs (m)', color=color_hs, fontsize=12)
    ax2.plot(df_gw.index, df_gw['waverider1.Hs__m'], color=color_hs, alpha=0.4, label='Hs Real (m)')
    ax2.scatter(df_gw.index[mask_interp], df_gw.loc[mask_interp, 'waverider1.Hs__m'],
                color='darkred', s=12, zorder=5, label='Hs Interpolado')
    ax2.tick_params(axis='y', labelcolor=color_hs)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    ax1.set_title('Dinamica Temporal: Pontos Vermelhos = falhas curtas reconstruidas; '
                   'falhas longas ficam como buracos reais no grafico', fontsize=13, pad=15)

    ax3 = plt.subplot(2, 2, 3)
    df_scatter = df_gw.dropna(subset=['WPF_kW_m', 'wec_power_kW'])

    scatter = ax3.scatter(df_scatter['WPF_kW_m'], df_scatter['wec_power_kW'],
                           c=df_scatter['waverider1.Te__s'], cmap='viridis',
                           alpha=0.6, s=15)

    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Periodo de Onda - Te (s)')

    ax3.set_xlabel('Wave Power Flux (kW/m)', fontsize=12)
    ax3.set_ylabel('Potencia Gerada (kW)', fontsize=12)
    ax3.set_title('Curva de Captura: WPF vs Potencia', fontsize=14)
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5)

    ax4 = plt.subplot(2, 2, 4)
    sns.histplot(df_gw['wec_power_kW'].dropna(), bins=100, ax=ax4, color='slategray', kde=True)
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero kW (Clip)')

    ax4.set_xlabel('Potencia Gerada (kW)', fontsize=12)
    ax4.set_ylabel('Frequencia (n de periodos de 30min)', fontsize=12)
    ax4.set_title('Distribuicao da Potencia (Apos Clip de Negativos)', fontsize=14)
    ax4.legend()

    plt.tight_layout()
    plt.savefig('analise_golden_window_com_flags.png', dpi=300, bbox_inches='tight')
    print("Graficos iniciais gerados com sucesso! Verifica 'analise_golden_window_com_flags.png'")


# ============================================================================
# MODULO DE RECONSTRUCAO SEMI-EMPIRICA (Two-Pass Residual Filtering)
# ============================================================================

def engineer_features(df):
    """
    Passo 1: calcula o WPF, a potencia em kW, e adiciona inercia temporal
    (lags de t-30min e t-1h) para Hs, Te e WPF.
    """
    df = df.copy()
    df['WPF_kW_m'] = 0.49 * (df['waverider1.Hs__m']**2) * df['waverider1.Te__s']
    df['wec_power_kW'] = df['wec_power__W'] / 1000.0

    base_cols = ['waverider1.Hs__m', 'waverider1.Te__s', 'WPF_kW_m']
    lag_cols = []
    for col in base_cols:
        for lag_steps, label in [(1, "30m"), (2, "1h")]:
            new_col = f"{col}_lag_{label}"
            df[new_col] = df[col].shift(lag_steps)
            lag_cols.append(new_col)

    available_spectral = [c for c in EXTRA_SPECTRAL_COLS if c in df.columns]
    missing_spectral = [c for c in EXTRA_SPECTRAL_COLS if c not in df.columns]
    if missing_spectral:
        print(f"-> Aviso: colunas espectrais extra nao encontradas e ignoradas: {missing_spectral}")

    features = base_cols + lag_cols + available_spectral
    return df, features


def detect_outliers_and_train_model(df, features):
    """
    Passo 2: treina um modelo bruto (pass 1), usa os residuos absolutos
    para identificar os OUTLIER_QUANTILE piores pontos (saltos/anomalias
    de sensor), remove-os, e treina o modelo final (pass 2) apenas com
    dados limpos.
    """
    # So consideramos "operacao ativa" pontos genuinamente saudaveis --
    # exclui ruido de comissionamento/idle (< ACTIVE_POWER_KW).
    active_mask = (df['wec_power_kW'] > ACTIVE_POWER_KW) & (df['WPF_kW_m'] > ACTIVE_WPF_KW_M)
    df_active = df[active_mask].dropna(subset=features + ['wec_power_kW']).copy()

    xgb_params = {
        'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05,
        'subsample': 0.85, 'colsample_bytree': 0.85,
        'reg_alpha': 0.5, 'reg_lambda': 2.0, 'random_state': 42,
    }

    # --- PASSAGEM 1: Treino Bruto ---
    model_rough = xgb.XGBRegressor(**xgb_params)
    model_rough.fit(df_active[features], df_active['wec_power_kW'])

    rough_preds = model_rough.predict(df_active[features])
    residuals = np.abs(df_active['wec_power_kW'] - rough_preds)

    error_threshold = residuals.quantile(OUTLIER_QUANTILE)
    clean_mask = residuals <= error_threshold

    df_clean = df_active[clean_mask].copy()
    df_outliers = df_active[~clean_mask].copy()

    print(f"-> Detecao de outliers concluida (limiar no quantil {OUTLIER_QUANTILE:.2f}):")
    print(f"   - Mantidos {len(df_clean)} registos saudaveis.")
    print(f"   - Removidos {len(df_outliers)} registos anomalos/saltos bruscos.")

    # --- PASSAGEM 2: Treino Final ---
    model_final = xgb.XGBRegressor(**xgb_params)
    model_final.fit(df_clean[features], df_clean['wec_power_kW'])

    final_preds = model_final.predict(df_clean[features])
    base_noise_std = (df_clean['wec_power_kW'] - final_preds).std()

    return model_final, base_noise_std, df_clean, df_outliers


def reconstruct_dataset(df, model, features, base_noise_std):
    """
    Passo 3: aplica o modelo final para reconstruir (a) periodos de
    idle/curtailment e (b) picos pontuais que divergem demasiado da
    previsao (outliers na serie completa, nao so no conjunto de treino).
    """
    df_predictable = df.dropna(subset=features).copy()
    p_predicted = model.predict(df_predictable[features])

    # Ruido mecanico para manter a textura organica da serie reconstruida.
    # NOTA: este ruido e sintetico (calibrado a partir do sigma do modelo
    # limpo), nao medido -- ver aviso no topo do ficheiro sobre o impacto
    # na decomposicao v/u do SFA.
    scale_factor = np.clip(p_predicted / ACTIVE_POWER_KW, 0.0, None)
    proportional_noise = np.random.normal(0, base_noise_std * NOISE_MULTIPLIER, len(p_predicted)) * np.sqrt(scale_factor)
    p_predicted_noisy = np.clip(p_predicted + proportional_noise, 0.0, 350.0)

    pred_series = pd.Series(index=df_predictable.index, data=p_predicted_noisy)
    pred_full = pred_series.reindex(df.index)

    df['wec_power_kW_reconstructed'] = df['wec_power_kW']
    df['is_curtailed_or_outlier'] = 0

    # FIX A: idle/curtailment SO conta se o mar tiver energia. Potencia
    # zero com WPF ~ 0 e fisica real (mar em espelho), nao curtailment --
    # nao deve ser reescrita pela previsao do modelo.
    idle_mask = (df['wec_power_kW'] <= ACTIVE_POWER_KW) & (df['WPF_kW_m'] > ACTIVE_WPF_KW_M)
    df.loc[idle_mask, 'wec_power_kW_reconstructed'] = pred_full.loc[idle_mask]
    df.loc[idle_mask, 'is_curtailed_or_outlier'] = 1

    # FIX B: limiar de outlier adaptativo (multiplo do sigma do modelo
    # limpo) em vez de um valor fixo em kW -- adapta-se ao nivel de ruido
    # real de cada boia/periodo, em vez de assumir que 40 kW e sempre a
    # fasquia certa.
    error_vs_prediction = np.abs(df['wec_power_kW'] - pred_full)
    outlier_threshold_kw = OUTLIER_ERROR_SIGMA * base_noise_std
    outlier_mask = (error_vs_prediction > outlier_threshold_kw) & (df['WPF_kW_m'] > ACTIVE_WPF_KW_M)
    df.loc[outlier_mask, 'wec_power_kW_reconstructed'] = pred_full.loc[outlier_mask]
    df.loc[outlier_mask, 'is_curtailed_or_outlier'] = 1

    print(f"-> Limiar de outlier pontual: {outlier_threshold_kw:.2f} kW "
          f"({OUTLIER_ERROR_SIGMA:.1f} x sigma do modelo limpo = {base_noise_std:.2f} kW).")

    df['wec_power__W_reconstructed'] = df['wec_power_kW_reconstructed'] * 1000.0

    # Drop dos buracos irrecuperaveis (falhas longas na Waverider, deixadas
    # de proposito na limpeza). Atencao: isto quebra a frequencia regular
    # de 30 min do index -- qualquer analise seguinte que assuma amostragem
    # continua (rolling windows, etc.) precisa de ter isso em conta.
    df_final = df.dropna(subset=['wec_power__W_reconstructed']).copy()

    return df_final


def plot_reconstruction(df_final, df_clean, df_outliers):
    """
    Passo 4: grafico de validacao -- mostra os outliers identificados (x
    vermelhos) e a serie final reconstruida.
    """
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    ax1.scatter(df_final['WPF_kW_m'], df_final['wec_power_kW_reconstructed'], alpha=0.2, color='tab:green', s=10,
                label='Dados Finais Reconstruidos')
    ax1.scatter(df_clean['WPF_kW_m'], df_clean['wec_power_kW'], alpha=0.8, color='tab:blue', s=15,
                label='Operacao Saudavel (Inliers)')
    ax1.scatter(df_outliers['WPF_kW_m'], df_outliers['wec_power_kW'], alpha=0.8, color='red', marker='x', s=20,
                label='Outliers/Saltos (Removidos no treino)')
    ax1.set_xlabel('Wave Power Flux (kW/m)')
    ax1.set_ylabel('Potencia (kW)')
    ax1.set_title('Fronteira Operacional: ML ignorou os pontos vermelhos para aprender a curva real')
    ax1.legend()

    ax2.plot(df_final.index, df_final['wec_power_kW_reconstructed'], color='tab:green', alpha=0.9, linewidth=1.5,
              label='Potencia Fina (Suavizada)')
    ax2.plot(df_final.index, df_final['wec_power_kW'], color='tab:blue', alpha=0.3, linewidth=1.0,
              label='Potencia Original (Bruta)')
    ax2.set_ylabel('Potencia (kW)')
    ax2.set_title('Serie Temporal: Anomalias e Gaps foram curados organicamente')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('validacao_ajuste_modular.png', dpi=300)
    print("Grafico guardado como 'validacao_ajuste_modular.png'")


def run_pipeline_reconstruction(input_csv="clean_baseline_buoy.csv", output_csv="reconstructed_baseline_buoy.csv"):
    print("\n" + "="*60)
    print("6. RECONSTRUCAO MODULAR (Two-Pass Residual Filter)")
    print("="*60)

    df = pd.read_csv(input_csv, parse_dates=True, index_col=0)

    df_feat, features = engineer_features(df)
    model, base_noise_std, df_clean, df_outliers = detect_outliers_and_train_model(df_feat, features)
    df_final = reconstruct_dataset(df_feat, model, features, base_noise_std)

    df_final.to_csv(output_csv)
    print(f"\nDataset FINAL guardado em: {output_csv}")
    print(f"Total de linhas finais disponiveis: {len(df_final)}")

    plot_reconstruction(df_final, df_clean, df_outliers)


if __name__ == "__main__":
    input_file = "../raw/baseline_buoy_raw.csv"
    clean_file = "clean_baseline_buoy.csv"
    reconstructed_file = "reconstructed_baseline_buoy.csv"

    clean_and_interpolate_baseline(input_csv=input_file, output_csv=clean_file)
    analyze_raw_dataset(csv_path=clean_file)
    analyze_golden_window(csv_path=clean_file)
    plot_golden_window(csv_path=clean_file)

    run_pipeline_reconstruction(input_csv=clean_file, output_csv=reconstructed_file)