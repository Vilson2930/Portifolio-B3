"""
PORTIFOLIO-B3 — RETORNOS ANUAIS OPERACIONAIS

Objetivo
--------
Construir os retornos anuais operacionais usados pela continuidade temporal
do motor, sem alterar nenhum arquivo histórico em data/.

Entrada
-------
data_live/daily_prices_operational.csv

Saída
-----
data_live/returns_operational.csv

Metodologia preservada a partir da Célula 12 histórica
------------------------------------------------------
1. Preço base = Close bruto / PREULT.
2. RET_RAW = PREULT_t / PREULT_t-1 - 1.
3. Link temporal válido somente quando GAP_DAYS <= 10.
4. Provável evento corporativo/mecânico:
   - |RET_RAW| >= 60%;
   - razão PREULT_t / PREULT_t-1 próxima de fator mecânico;
   - fatores testados:
       * inteiros de 2 a 100;
       * recíprocos de 2 a 100;
       * 1.6666666667;
   - FACTOR_ERROR < 2%.
5. Evento corporativo provável recebe RET_CLEAN = 0.
6. Link inválido recebe RET_CLEAN = NaN.
7. Demais retornos válidos recebem RET_CLEAN = RET_RAW.
8. RET_ANNUAL = produto(1 + RET_CLEAN) - 1.
9. Ano/ticker válido exige pelo menos 10 RET_CLEAN não nulos.
10. Ano corrente é tratado como PARTIAL_YEAR e NÃO recebe RET_ANNUAL_VALID.
11. Apenas anos anteriores ao ano corrente podem alimentar TOP4_1Y.

Observação metodológica
-----------------------
A configuração abaixo reproduziu exatamente os 876 eventos mecânicos presentes
no checkpoint histórico cell12_daily_returns_clean.csv utilizado na auditoria
de reconstrução da metodologia.

Este arquivo NÃO altera data/returns.csv.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
DATA_LIVE = ROOT / "data_live"

DAILY_PRICES_FILE = (
    DATA_LIVE
    / "daily_prices_operational.csv"
)

OUTPUT_RETURNS_FILE = (
    DATA_LIVE
    / "returns_operational.csv"
)

OUTPUT_DAILY_AUDIT_FILE = (
    DATA_LIVE
    / "daily_returns_operational_audit.csv"
)

HISTORICAL_RETURNS_FILE = (
    DATA
    / "returns.csv"
)

CURRENT_YEAR = datetime.now().year

# O histórico congelado termina em 2025.
# A continuidade operacional começa em 2026.
FIRST_OPERATIONAL_YEAR = 2026

MAX_GAP_DAYS = 10
MIN_VALID_RETURNS = 10

# Reconstrução empírica da Célula 12.
MIN_EVENT_ABS_RETURN = 0.60
MAX_FACTOR_ERROR = 0.02

SPECIAL_MECHANICAL_FACTOR = 1.6666666667

VALID_MACRO_SECTORS = {
    "UTILITIES",
    "INDUSTRIALS",
    "FINANCIALS",
    "CONSUMER",
    "MATERIALS",
    "TECHNOLOGY",
    "ENERGY",
    "HEALTHCARE",
    "COMMUNICATION",
}


# =============================================================================
# UTILIDADES
# =============================================================================

def require_columns(
    df: pd.DataFrame,
    required: list[str],
    name: str,
):
    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{name} sem colunas obrigatórias: "
            f"{missing}"
        )


def normalize_ticker(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )


def normalize_sector(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )


# =============================================================================
# 1. CARREGAMENTO DA SÉRIE DIÁRIA
# =============================================================================

def load_daily_prices() -> pd.DataFrame:

    if not DAILY_PRICES_FILE.exists():
        raise RuntimeError(
            "Arquivo diário operacional não encontrado: "
            f"{DAILY_PRICES_FILE}"
        )

    daily = pd.read_csv(
        DAILY_PRICES_FILE,
        low_memory=False,
    )

    require_columns(
        daily,
        [
            "DATE",
            "TICKER",
            "MACRO_SECTOR",
            "PREULT",
        ],
        "daily_prices_operational.csv",
    )

    daily["DATE"] = pd.to_datetime(
        daily["DATE"],
        errors="coerce",
    )

    daily["TICKER"] = normalize_ticker(
        daily["TICKER"]
    )

    daily["MACRO_SECTOR"] = normalize_sector(
        daily["MACRO_SECTOR"]
    )

    daily["PREULT"] = pd.to_numeric(
        daily["PREULT"],
        errors="coerce",
    )

    daily = daily[
        daily["DATE"].notna()
        & daily["TICKER"].ne("")
        & daily["PREULT"].notna()
        & (daily["PREULT"] > 0)
    ].copy()

    daily = (
        daily
        .drop_duplicates(
            ["DATE", "TICKER"],
            keep="last",
        )
        .sort_values(
            ["TICKER", "DATE"]
        )
        .reset_index(drop=True)
    )

    duplicates = int(
        daily.duplicated(
            ["DATE", "TICKER"]
        ).sum()
    )

    if duplicates:
        raise RuntimeError(
            "Duplicidades DATE × TICKER na série diária: "
            f"{duplicates}"
        )

    if daily.empty:
        raise RuntimeError(
            "Série diária operacional vazia."
        )

    return daily


# =============================================================================
# 2. FATOR MECÂNICO MAIS PRÓXIMO
# =============================================================================

def nearest_mechanical_factor(
    ratio: float,
) -> tuple[float, float]:
    """
    Retorna:
        (MECHANICAL_FACTOR, FACTOR_ERROR)

    FACTOR_ERROR = abs(ratio / factor - 1)

    Fatores da reconstrução histórica:
      - 2, 3, ..., 100
      - 1/2, 1/3, ..., 1/100
      - 1.6666666667
    """

    if (
        not np.isfinite(ratio)
        or ratio <= 0
    ):
        return np.nan, np.nan

    candidates = []

    # -------------------------------------------------------------------------
    # Razões >= 1: inteiros próximos e limites.
    # -------------------------------------------------------------------------

    lower = int(np.floor(ratio))
    upper = int(np.ceil(ratio))

    for n in {
        lower,
        upper,
        2,
        100,
    }:
        if 2 <= n <= 100:
            candidates.append(float(n))

    # -------------------------------------------------------------------------
    # Razões < 1: recíprocos de inteiros próximos e limites.
    # -------------------------------------------------------------------------

    inverse = 1.0 / ratio

    lower_inv = int(np.floor(inverse))
    upper_inv = int(np.ceil(inverse))

    for n in {
        lower_inv,
        upper_inv,
        2,
        100,
    }:
        if 2 <= n <= 100:
            candidates.append(
                1.0 / float(n)
            )

    # Fator observado na regra histórica.
    candidates.append(
        SPECIAL_MECHANICAL_FACTOR
    )

    best_factor = np.nan
    best_error = np.inf

    for factor in candidates:

        if (
            not np.isfinite(factor)
            or factor <= 0
        ):
            continue

        error = abs(
            ratio / factor
            - 1.0
        )

        if error < best_error:
            best_error = error
            best_factor = factor

    if not np.isfinite(best_error):
        return np.nan, np.nan

    return (
        float(best_factor),
        float(best_error),
    )


# =============================================================================
# 3. RETORNOS DIÁRIOS LIMPOS
# =============================================================================

def build_daily_returns(
    daily: pd.DataFrame,
) -> pd.DataFrame:

    out = daily.copy()

    out["YEAR"] = (
        out["DATE"]
        .dt.year
        .astype(int)
    )

    out["PREV_DATE"] = (
        out
        .groupby(
            "TICKER",
            sort=False,
        )["DATE"]
        .shift(1)
    )

    out["PREV_PRICE"] = (
        out
        .groupby(
            "TICKER",
            sort=False,
        )["PREULT"]
        .shift(1)
    )

    out["GAP_DAYS"] = (
        out["DATE"]
        -
        out["PREV_DATE"]
    ).dt.days

    out["RET_RAW"] = (
        out["PREULT"]
        /
        out["PREV_PRICE"]
        -
        1.0
    )

    invalid_price = (
        out["PREV_PRICE"].isna()
        | (out["PREV_PRICE"] <= 0)
        | ~np.isfinite(out["RET_RAW"])
    )

    out.loc[
        invalid_price,
        "RET_RAW",
    ] = np.nan

    valid_time_link = (
        out["RET_RAW"].notna()
        & out["GAP_DAYS"].notna()
        & (out["GAP_DAYS"] <= MAX_GAP_DAYS)
    )

    mechanical_factor = np.full(
        len(out),
        np.nan,
        dtype=float,
    )

    factor_error = np.full(
        len(out),
        np.nan,
        dtype=float,
    )

    candidate_positions = np.flatnonzero(
        (
            valid_time_link
            & (
                out["RET_RAW"].abs()
                >=
                MIN_EVENT_ABS_RETURN
            )
        ).to_numpy()
    )

    for pos in candidate_positions:

        ratio = (
            float(out.iloc[pos]["PREULT"])
            /
            float(out.iloc[pos]["PREV_PRICE"])
        )

        factor, error = (
            nearest_mechanical_factor(
                ratio
            )
        )

        mechanical_factor[pos] = factor
        factor_error[pos] = error

    out["MECHANICAL_FACTOR"] = (
        mechanical_factor
    )

    out["FACTOR_ERROR"] = (
        factor_error
    )

    out["LIKELY_CORPORATE_ACTION"] = (
        valid_time_link
        & (
            out["RET_RAW"].abs()
            >=
            MIN_EVENT_ABS_RETURN
        )
        & out["FACTOR_ERROR"].notna()
        & (
            out["FACTOR_ERROR"]
            <
            MAX_FACTOR_ERROR
        )
    )

    # -------------------------------------------------------------------------
    # RET_CLEAN — regra histórica
    # -------------------------------------------------------------------------

    out["RET_CLEAN"] = np.nan

    normal_valid = (
        valid_time_link
        & ~out["LIKELY_CORPORATE_ACTION"]
    )

    out.loc[
        normal_valid,
        "RET_CLEAN",
    ] = out.loc[
        normal_valid,
        "RET_RAW",
    ]

    out.loc[
        out["LIKELY_CORPORATE_ACTION"],
        "RET_CLEAN",
    ] = 0.0

    # Controle explícito:
    # gaps > 10 dias jamais carregam retorno.
    invalid_link_with_return = (
        out["RET_RAW"].notna()
        & (
            out["GAP_DAYS"].isna()
            | (
                out["GAP_DAYS"]
                >
                MAX_GAP_DAYS
            )
        )
    )

    if (
        out.loc[
            invalid_link_with_return,
            "RET_CLEAN",
        ]
        .notna()
        .any()
    ):
        raise RuntimeError(
            "RET_CLEAN encontrado em link temporal inválido."
        )

    # Todo evento mecânico deve ser neutralizado em zero.
    event_values = out.loc[
        out["LIKELY_CORPORATE_ACTION"],
        "RET_CLEAN",
    ]

    if (
        not event_values.empty
        and not np.allclose(
            event_values.to_numpy(
                dtype=float
            ),
            0.0,
            rtol=0.0,
            atol=0.0,
        )
    ):
        raise RuntimeError(
            "Evento corporativo não foi neutralizado corretamente."
        )

    return out


# =============================================================================
# 4. AGREGAÇÃO ANUAL
# =============================================================================

def compound_returns(
    series: pd.Series,
) -> float:

    valid = (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
    )

    if valid.empty:
        return np.nan

    return float(
        np.prod(
            1.0
            +
            valid.to_numpy(
                dtype=float
            )
        )
        -
        1.0
    )


def build_annual_returns(
    daily_returns: pd.DataFrame,
) -> pd.DataFrame:

    operational = daily_returns[
        daily_returns["YEAR"]
        >=
        FIRST_OPERATIONAL_YEAR
    ].copy()

    if operational.empty:
        raise RuntimeError(
            "Não há observações para anos operacionais."
        )

    rows = []

    for (
        year,
        ticker,
    ), group in operational.groupby(
        [
            "YEAR",
            "TICKER",
        ],
        sort=True,
    ):

        group = (
            group
            .sort_values("DATE")
        )

        sectors = (
            group["MACRO_SECTOR"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )

        valid_sectors = (
            sectors[
                sectors.isin(
                    VALID_MACRO_SECTORS
                )
            ]
        )

        if valid_sectors.empty:
            macro_sector = "UNCLASSIFIED"
        else:
            # A série persistida é regravada a cada coleta.
            # Usa a classificação mais recente disponível no ano.
            macro_sector = (
                valid_sectors.iloc[-1]
            )

        valid_returns = int(
            group["RET_CLEAN"]
            .notna()
            .sum()
        )

        corporate_actions = int(
            group[
                "LIKELY_CORPORATE_ACTION"
            ]
            .fillna(False)
            .sum()
        )

        ret_annual = compound_returns(
            group["RET_CLEAN"]
        )

        is_complete_year = (
            int(year)
            <
            CURRENT_YEAR
        )

        if not is_complete_year:
            return_status = (
                "PARTIAL_YEAR"
            )
            ret_annual_valid = np.nan

        elif (
            valid_returns
            <
            MIN_VALID_RETURNS
        ):
            return_status = (
                "INSUFFICIENT_OBSERVATIONS"
            )
            ret_annual_valid = np.nan

        else:
            return_status = "VALID"
            ret_annual_valid = (
                ret_annual
            )

        rows.append(
            {
                "YEAR":
                    int(year),

                "TICKER":
                    ticker,

                "MACRO_SECTOR":
                    macro_sector,

                "FIRST_DATE":
                    group["DATE"]
                    .min()
                    .date()
                    .isoformat(),

                "LAST_DATE":
                    group["DATE"]
                    .max()
                    .date()
                    .isoformat(),

                "VALID_RETURNS":
                    valid_returns,

                "RET_ANNUAL":
                    ret_annual,

                "CORPORATE_ACTIONS":
                    corporate_actions,

                "RETURN_STATUS":
                    return_status,

                "RET_ANNUAL_VALID":
                    ret_annual_valid,

                "IS_COMPLETE_YEAR":
                    bool(
                        is_complete_year
                    ),
            }
        )

    annual = pd.DataFrame(
        rows
    )

    annual = (
        annual
        .sort_values(
            [
                "YEAR",
                "TICKER",
            ]
        )
        .reset_index(drop=True)
    )

    duplicated = int(
        annual.duplicated(
            [
                "YEAR",
                "TICKER",
            ]
        ).sum()
    )

    if duplicated:
        raise RuntimeError(
            "Duplicidades YEAR × TICKER nos retornos operacionais: "
            f"{duplicated}"
        )

    return annual


# =============================================================================
# 5. AUDITORIA DE NÃO CONTAMINAÇÃO DO HISTÓRICO
# =============================================================================

def audit_frozen_history():

    if not HISTORICAL_RETURNS_FILE.exists():
        raise RuntimeError(
            "Histórico congelado não encontrado: "
            f"{HISTORICAL_RETURNS_FILE}"
        )

    historical = pd.read_csv(
        HISTORICAL_RETURNS_FILE,
        low_memory=False,
    )

    require_columns(
        historical,
        [
            "YEAR",
            "TICKER",
            "MACRO_SECTOR",
            "RET_ANNUAL_VALID",
        ],
        "data/returns.csv",
    )

    years = pd.to_numeric(
        historical["YEAR"],
        errors="coerce",
    )

    max_historical_year = int(
        years.dropna().max()
    )

    if (
        max_historical_year
        >=
        FIRST_OPERATIONAL_YEAR
    ):
        raise RuntimeError(
            "Sobreposição temporal inesperada: "
            f"data/returns.csv termina em {max_historical_year}, "
            f"mas FIRST_OPERATIONAL_YEAR={FIRST_OPERATIONAL_YEAR}."
        )

    return max_historical_year


# =============================================================================
# 6. AUDITORIA OPERACIONAL
# =============================================================================

def audit_operational_returns(
    daily_returns: pd.DataFrame,
    annual: pd.DataFrame,
    max_historical_year: int,
):

    print()
    print("=" * 78)
    print(
        "AUDITORIA — RETORNOS OPERACIONAIS"
    )
    print("=" * 78)

    daily_duplicates = int(
        daily_returns.duplicated(
            [
                "DATE",
                "TICKER",
            ]
        ).sum()
    )

    annual_duplicates = int(
        annual.duplicated(
            [
                "YEAR",
                "TICKER",
            ]
        ).sum()
    )

    invalid_gap_clean = int(
        (
            daily_returns[
                "RET_CLEAN"
            ].notna()
            &
            daily_returns[
                "RET_RAW"
            ].notna()
            &
            (
                daily_returns[
                    "GAP_DAYS"
                ]
                >
                MAX_GAP_DAYS
            )
        ).sum()
    )

    events = int(
        daily_returns[
            "LIKELY_CORPORATE_ACTION"
        ]
        .fillna(False)
        .sum()
    )

    valid_annual = int(
        annual[
            "RET_ANNUAL_VALID"
        ]
        .notna()
        .sum()
    )

    partial_rows = int(
        (
            annual[
                "RETURN_STATUS"
            ]
            ==
            "PARTIAL_YEAR"
        ).sum()
    )

    current_valid = int(
        annual.loc[
            annual["YEAR"]
            ==
            CURRENT_YEAR,
            "RET_ANNUAL_VALID",
        ]
        .notna()
        .sum()
    )

    if daily_duplicates != 0:
        raise RuntimeError(
            "Duplicidades DATE × TICKER."
        )

    if annual_duplicates != 0:
        raise RuntimeError(
            "Duplicidades YEAR × TICKER."
        )

    if invalid_gap_clean != 0:
        raise RuntimeError(
            "RET_CLEAN atravessou gap >10 dias."
        )

    if current_valid != 0:
        raise RuntimeError(
            "Ano corrente recebeu RET_ANNUAL_VALID."
        )

    if (
        annual["YEAR"].min()
        <
        FIRST_OPERATIONAL_YEAR
    ):
        raise RuntimeError(
            "returns_operational.csv contém ano histórico congelado."
        )

    print(
        f"Histórico congelado termina ............ "
        f"{max_historical_year}"
    )

    print(
        f"Início operacional ...................... "
        f"{FIRST_OPERATIONAL_YEAR}"
    )

    print(
        f"Observações diárias ..................... "
        f"{len(daily_returns):,}"
    )

    print(
        f"Eventos mecânicos neutralizados ......... "
        f"{events:,}"
    )

    print(
        f"Linhas anuais operacionais .............. "
        f"{len(annual):,}"
    )

    print(
        f"RET_ANNUAL_VALID disponíveis ............ "
        f"{valid_annual:,}"
    )

    print(
        f"Linhas do ano corrente como parcial ..... "
        f"{partial_rows:,}"
    )

    print(
        "RET_RAW = PREULT_t/PREULT_t-1 - 1 ..... PASS"
    )

    print(
        f"GAP_DAYS <= {MAX_GAP_DAYS} ......................... PASS"
    )

    print(
        "Evento mecânico -> RET_CLEAN = 0 ........ PASS"
    )

    print(
        f"Mínimo {MIN_VALID_RETURNS} retornos anuais .............. PASS"
    )

    print(
        "Ano corrente excluído de RET_ANNUAL_VALID  PASS"
    )

    print(
        "Histórico data/returns.csv ............... PRESERVADO"
    )

    print(
        "STATUS ................................... PASS"
    )


# =============================================================================
# EXECUÇÃO
# =============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "PORTIFOLIO-B3 — "
        "RETORNOS ANUAIS OPERACIONAIS"
    )
    print("=" * 78)

    DATA_LIVE.mkdir(
        parents=True,
        exist_ok=True,
    )

    max_historical_year = (
        audit_frozen_history()
    )

    daily_prices = (
        load_daily_prices()
    )

    print()
    print(
        f"Série diária recebida : "
        f"{daily_prices['DATE'].min().date()} a "
        f"{daily_prices['DATE'].max().date()}"
    )

    print(
        f"Tickers na série      : "
        f"{daily_prices['TICKER'].nunique():,}"
    )

    daily_returns = (
        build_daily_returns(
            daily_prices
        )
    )

    annual = (
        build_annual_returns(
            daily_returns
        )
    )

    audit_operational_returns(
        daily_returns,
        annual,
        max_historical_year,
    )

    annual.to_csv(
        OUTPUT_RETURNS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # Auditoria diária operacional: apenas anos da continuidade,
    # evitando duplicar no artefato a janela de 2025 usada só como ponte.
    daily_audit = daily_returns[
        daily_returns["YEAR"]
        >=
        FIRST_OPERATIONAL_YEAR
    ].copy()

    daily_audit.to_csv(
        OUTPUT_DAILY_AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
        date_format="%Y-%m-%d",
    )

    print()
    print("=" * 78)
    print("ARQUIVOS GERADOS")
    print("=" * 78)

    print(
        f"Retornos anuais : "
        f"{OUTPUT_RETURNS_FILE}"
    )

    print(
        f"Auditoria diária: "
        f"{OUTPUT_DAILY_AUDIT_FILE}"
    )

    print()
    print(
        "STATUS: CONTINUIDADE DE "
        "RETORNOS OPERACIONAIS CONSTRUÍDA"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
