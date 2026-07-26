"""
Geracao do dataset simulado da frota de WECs (12 boias) a partir da
Boia 1 real (Ground Truth), reconstruida no passo anterior do pipeline.

Fluxo modular:
  1) load_baseline(): le o reconstructed_baseline_buoy.csv, remove o fuso
     horario (tz-naive), limpa o prefixo 'waverider1.' das colunas para
     uniformizacao, e garante a grelha temporal correta (30 min).
  2) replicate_spatial_fleet(): gera as Boias 2 a 12 a partir da Boia 1,
     simulando a propagacao espacial da onda no parque (6 km) com um
     shift temporal iterativo (np.roll), um fator de atenuacao espacial
     sobre a potencia ideal, e ruido branco microscopico em Hs e Te.
     Qualquer outra coluna numerica existente no CSV base (ex: variaveis
     espectrais adicionais) e propagada de forma generica, para que o
     script nao dependa de nomes de colunas hardcoded.
  3) apply_epoch_degradation(): aplica o modelo fisico de 3 epocas
     baseado na decomposicao SFA (P_final = P_ideal * base_eff + v - u).
  4) assign_split_roles(): rotula cada linha como IGNORE / TRAIN /
     TEST_NOMINAL / TEST_ANOMALY, para que a Fase 1 (Phase 1) deixe de
     precisar de imputar valores em falta ou de fazer um corte
     cronologico ingenuo -- ver nota mais abaixo.
  5) compile_and_export(): junta a frota completa num unico ficheiro CSV.

Nota sobre o shift espacial: o np.roll desloca os valores por posicao na
serie (numero de passos de 30 min), nao por tempo decorrido real. Isto e
uma aproximacao (tal como no script legado), suficiente para gerar
decorrelacao espacial sintetica entre boias, mas nao e uma propagacao de
onda fisicamente rigorosa.

Nota sobre Split_Role: os buracos longos de dados que o pipeline de
limpeza da Boia 1 deixou de proposito (NaN, ver clean_and_interpolate_
baseline) propagam-se para a frota via np.roll. Sem um mecanismo
explicito para os excluir, a Fase 1 acabava a imputa-los com a mediana,
o que destroi a fisica (mar "mediano" emparelhado com potencia real
altamente variavel) e, pior ainda, se o filtro de outliers da Fase 1
correr tambem sobre a coluna alvo, pode apagar exatamente os eventos de
degradacao da Epoca 3 que a dissertacao quer detetar. Split_Role resolve
isto na origem: qualquer linha sem estado de mar valido e marcada
IGNORE e a Fase 1 descarta-a fisicamente antes de tocar em imputacao ou
em modelos.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# --- Constantes gerais -------------------------------------------------------
BASELINE_CSV = "../clean/reconstructed_baseline_buoy.csv"
OUTPUT_CSV = "final_wec_fleet_2026.csv"
N_BUOYS = 12
RANDOM_SEED = 42

# Colunas core esperadas no CSV base (produzido pelo pipeline anterior).
BASE_HS_COL = "Hs__m"
BASE_TE_COL = "Te__s"
BASE_IDEAL_POWER_COL = "wec_power_kW_reconstructed"

# Colunas auxiliares do pipeline anterior que nao devem ser propagadas
# para a frota (sao artefactos de limpeza/reconstrucao, nao variaveis
# fisicas de interesse para a simulacao).
DROP_COLS_EXACT = [
    "wec_power__W",
    "is_interpolated",
    "WPF_kW_m",
    "wec_power_kW",
    "wec_power__W_reconstructed",
    "is_curtailed_or_outlier",
]

# --- Replicacao espacial ------------------------------------------------------
HS_NOISE_STD = 0.02
TE_NOISE_STD = 0.05
PASSTHROUGH_NOISE_FRACTION = 0.01  # fracao do desvio-padrao de cada coluna extra

# --- Epocas temporais ---------------------------------------------------------
EPOCH2_START = "2026-05-01"
EPOCH3_START = "2026-06-01"

FAILING_BUOYS = [f"Boia_{i}" for i in (9, 10, 11, 12)]

# --- Parametros da decomposicao SFA (v = ruido simetrico, u = ineficiencia) ---
V_SIGMA = 4.0

U_SIGMA_EPOCH1 = 2.0
U_SIGMA_EPOCH2 = 5.0
U_SIGMA_EPOCH3_HEALTHY = 2.0
U_SIGMA_EPOCH3_FAILING = 35.0

EFF_EPOCH1 = 1.0
EFF_EPOCH2 = 0.85
EFF_EPOCH3_HEALTHY = 1.0
EFF_EPOCH3_FAILING = 0.45

POWER_CLIP_MIN = 0.0
POWER_CLIP_MAX = 350.0

# --- Split_Role (consumido pela Fase 1) ---------------------------------------
# Fatia "dourada" de treino: periodo da Epoca 1 em que temos a certeza
# absoluta de que a Waverider nao teve falhas. TEM de bater certo com os
# TRAIN_WINDOW_START/END usados no script da Fase 1 (sao dois ficheiros
# separados, por isso a constante existe duplicada -- ver aviso no main).
TRAIN_WINDOW_START = "2026-03-22"
TRAIN_WINDOW_END = "2026-04-30 23:30:00"
TRAIN_FRACTION_WITHIN_WINDOW = 0.8
# -----------------------------------------------------------------------------


def load_baseline(input_csv=BASELINE_CSV):
    print("A carregar baseline da Boia 1: " + input_csv)

    df = pd.read_csv(input_csv, parse_dates=True, index_col=0)

    # 1. Remover fuso horario (tz-naive) para compatibilidade com a pipeline
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    # 2. Uniformizar nomes das colunas (remover o prefixo da API)
    df = df.rename(columns=lambda x: x.replace('waverider1.', ''))

    # 3. Garantir que a coluna Hmax__m existe para o XGBoost
    if "Hmax__m" not in df.columns and "HTmax__m" in df.columns:
        df["Hmax__m"] = df["HTmax__m"]

    required_cols = [BASE_HS_COL, BASE_TE_COL, BASE_IDEAL_POWER_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias em falta no CSV base: {missing}")

    # Garantir grelha temporal regular de 30 min. Isto nao inventa valores,
    # apenas garante que qualquer falha longa deixada de proposito no
    # pipeline anterior fica representada como NaN numa posicao correta,
    # em vez de desalinhar o np.roll usado no passo seguinte.
    df = df.asfreq("30min")

    print(f"Baseline carregada: {len(df)} registos entre {df.index.min()} e {df.index.max()}.")
    return df


def replicate_spatial_fleet(df_baseline, n_buoys=N_BUOYS):
    print(f"A replicar a frota espacial: {n_buoys} boias a partir da Boia 1.")

    timestamps = df_baseline.index
    n_records = len(df_baseline)

    hs_base = df_baseline[BASE_HS_COL].to_numpy()
    te_base = df_baseline[BASE_TE_COL].to_numpy()
    ideal_power_base = df_baseline[BASE_IDEAL_POWER_COL].to_numpy()

    passthrough_cols = [
        c for c in df_baseline.columns
        if c not in DROP_COLS_EXACT
        and "_lag_" not in c
        and c not in (BASE_HS_COL, BASE_TE_COL, BASE_IDEAL_POWER_COL)
    ]

    if passthrough_cols:
        print(f"Colunas extra detetadas no CSV base e propagadas para a frota: {passthrough_cols}")
    else:
        print("Nenhuma coluna extra alem das core foi detetada no CSV base.")

    fleet_frames = []
    for i in range(n_buoys):
        buoy_id = f"Boia_{i + 1}"
        shift_steps = i
        spatial_factor = 0.95 + i * 0.01

        hs_local = (np.roll(hs_base, shift_steps) * spatial_factor) + np.random.normal(0, HS_NOISE_STD, n_records)
        te_local = np.roll(te_base, shift_steps) + np.random.normal(0, TE_NOISE_STD, n_records)
        wpf_local = 0.49 * (hs_local ** 2) * te_local

        ideal_power_local = np.roll(ideal_power_base, shift_steps) * spatial_factor

        df_buoy = pd.DataFrame(index=timestamps)
        df_buoy.index.name = "PCTimeStamp"
        df_buoy["Buoy_ID"] = buoy_id
        df_buoy["Hs__m"] = hs_local
        df_buoy["Te__s"] = te_local
        df_buoy["Wave_Power_Flux"] = wpf_local
        df_buoy["Ideal_Power_kW"] = ideal_power_local

        for col in passthrough_cols:
            base_values = df_baseline[col].to_numpy()
            rolled_values = np.roll(base_values, shift_steps)

            if np.issubdtype(base_values.dtype, np.number):
                col_std = np.nanstd(base_values)
                noise_std = col_std * PASSTHROUGH_NOISE_FRACTION if col_std > 0 else 0.0
                noisy_values = rolled_values + np.random.normal(0, noise_std, n_records)
                if pd.api.types.is_integer_dtype(df_baseline[col]):
                    noisy_values = np.round(noisy_values).astype(int)
                df_buoy[col] = noisy_values
            else:
                df_buoy[col] = rolled_values

        fleet_frames.append(df_buoy)

    print("Replicacao espacial concluida.")
    return fleet_frames


def apply_epoch_degradation(fleet_frames):
    print("A aplicar o modelo de degradacao SFA por epocas.")

    degraded_frames = []
    for df_buoy in fleet_frames:
        buoy_id = df_buoy["Buoy_ID"].iloc[0]
        n_records = len(df_buoy)
        timestamps = df_buoy.index

        mask_epoch1 = timestamps < EPOCH2_START
        mask_epoch2 = (timestamps >= EPOCH2_START) & (timestamps < EPOCH3_START)
        mask_epoch3 = timestamps >= EPOCH3_START

        epoch_labels = np.ones(n_records, dtype=int)
        epoch_labels[mask_epoch2] = 2
        epoch_labels[mask_epoch3] = 3

        base_eff = np.ones(n_records)
        u = np.zeros(n_records)

        # Ruido simetrico (ambiental/sensor), presente em todas as epocas.
        v = np.random.normal(0, V_SIGMA, n_records)

        # Epoca 1: operacao nominal, todas as boias saudaveis.
        base_eff[mask_epoch1] = EFF_EPOCH1
        u[mask_epoch1] = np.abs(np.random.normal(0, U_SIGMA_EPOCH1, int(np.sum(mask_epoch1))))

        # Epoca 2: falso positivo ambiental (spectral spreading), toda a
        # frota cai em simultaneo, com um ligeiro aumento da ineficiencia
        # tecnica.
        base_eff[mask_epoch2] = EFF_EPOCH2
        u[mask_epoch2] = np.abs(np.random.normal(0, U_SIGMA_EPOCH2, int(np.sum(mask_epoch2))))

        # Epoca 3: avaria mecanica isolada. Boias 1-8 recuperam a eficiencia
        # nominal; Boias 9-12 sofrem falha severa no PTO.
        if buoy_id in FAILING_BUOYS:
            base_eff[mask_epoch3] = EFF_EPOCH3_FAILING
            u[mask_epoch3] = np.abs(np.random.normal(0, U_SIGMA_EPOCH3_FAILING, int(np.sum(mask_epoch3))))
        else:
            base_eff[mask_epoch3] = EFF_EPOCH3_HEALTHY
            u[mask_epoch3] = np.abs(np.random.normal(0, U_SIGMA_EPOCH3_HEALTHY, int(np.sum(mask_epoch3))))

        # Decomposicao SFA: P_final = (P_ideal * base_eff) + v - u
        energy = (df_buoy["Ideal_Power_kW"].to_numpy() * base_eff) + v - u
        energy = np.clip(energy, POWER_CLIP_MIN, POWER_CLIP_MAX)

        df_buoy = df_buoy.copy()
        df_buoy["base_eff"] = base_eff
        df_buoy["sfa_u"] = u
        df_buoy["sfa_v"] = v
        df_buoy["Energy_Generation_kW"] = energy
        df_buoy["Epoch_Marker"] = epoch_labels

        degraded_frames.append(df_buoy)

    print("Degradacao por epocas aplicada a todas as boias.")
    return degraded_frames


def assign_split_roles(
    fleet_frames,
    train_window_start=TRAIN_WINDOW_START,
    train_window_end=TRAIN_WINDOW_END,
    train_fraction=TRAIN_FRACTION_WITHIN_WINDOW,
):
    """
    Rotula cada linha com Split_Role:

      IGNORE       -- sem estado de mar valido (Hs ou Te em NaN, herdado
                       do buraco longo deixado de proposito na limpeza).
                       Descartado fisicamente pela Fase 1 antes de
                       qualquer imputacao ou treino.
      TRAIN         -- amostra aleatoria de TRAIN_FRACTION_WITHIN_WINDOW
                       (80%) da fatia dourada da Epoca 1 (dentro da janela
                       [train_window_start, train_window_end]).
      TEST_NOMINAL  -- os restantes 20% da fatia dourada, mais qualquer
                       outro dia valido da Epoca 1 fora dessa janela (ex:
                       inicio de Marco).
      TEST_ANOMALY  -- todas as linhas validas da Epoca 2 e da Epoca 3.
    """
    print("A atribuir Split_Role (IGNORE / TRAIN / TEST_NOMINAL / TEST_ANOMALY).")

    window_start = pd.to_datetime(train_window_start)
    window_end = pd.to_datetime(train_window_end)

    labeled_frames = []
    for i, df_buoy in enumerate(fleet_frames):
        df_buoy = df_buoy.copy()
        timestamps = df_buoy.index

        is_missing_sea_state = (df_buoy["Hs__m"].isna() | df_buoy["Te__s"].isna()).to_numpy()
        is_epoch1 = (df_buoy["Epoch_Marker"] == 1).to_numpy()
        in_golden_window = ((timestamps >= window_start) & (timestamps <= window_end))

        split_role = np.full(len(df_buoy), "TEST_ANOMALY", dtype=object)

        # Epoca 1 fora da janela dourada (ex: inicio de Marco) -> TEST_NOMINAL.
        split_role[is_epoch1 & ~in_golden_window] = "TEST_NOMINAL"

        # Dentro da janela dourada: 80/20 aleatorio entre TRAIN e TEST_NOMINAL.
        golden_slice_idx = np.where(is_epoch1 & in_golden_window)[0]
        rng = np.random.default_rng(RANDOM_SEED + i)
        n_train = int(len(golden_slice_idx) * train_fraction)
        train_idx = rng.choice(golden_slice_idx, size=n_train, replace=False) if len(golden_slice_idx) > 0 else golden_slice_idx

        split_role[golden_slice_idx] = "TEST_NOMINAL"
        split_role[train_idx] = "TRAIN"

        # Falha de sensor (sem dados do mar) sobrepoe tudo -> IGNORE.
        split_role[is_missing_sea_state] = "IGNORE"

        df_buoy["Split_Role"] = split_role
        labeled_frames.append(df_buoy)

    print("Split_Role atribuido a todas as boias.")
    return labeled_frames


def compile_and_export(fleet_frames, output_csv=OUTPUT_CSV):
    print("A compilar o dataset final da frota.")

    final_dataset = pd.concat(fleet_frames, axis=0)
    final_dataset = final_dataset.reset_index()

    final_dataset.to_csv(output_csv, index=False)

    print(f"Dataset final guardado em: {output_csv}")
    print(f"Total de registos: {len(final_dataset)}")
    print(f"Boias incluidas: {sorted(final_dataset['Buoy_ID'].unique())}")
    print("Distribuicao de Split_Role:")
    print(final_dataset["Split_Role"].value_counts().to_string())

    return final_dataset


if __name__ == "__main__":
    df_baseline = load_baseline(BASELINE_CSV)
    fleet_frames = replicate_spatial_fleet(df_baseline, n_buoys=N_BUOYS)
    fleet_frames = apply_epoch_degradation(fleet_frames)
    fleet_frames = assign_split_roles(fleet_frames)
    compile_and_export(fleet_frames, OUTPUT_CSV)