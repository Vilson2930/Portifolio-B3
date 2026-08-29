"""
PORTIFOLIO-B3 — SELEÇÃO SETORIAL OPERACIONAL TOP4_1Y

Objetivo:
    Selecionar dinamicamente os 4 macrosetores do ano corrente
    preservando exatamente a regra histórica congelada FREEZE_TOP4_1Y.

Regra congelada:
    - LOOKBACK = 1 ano completo imediatamente anterior.
    - RANK_METRIC = retorno Equal Weight do macrosetor no ano anterior.
    - EW_RETURN = média de RET_ANNUAL_VALID das ações do setor.
    - setor/ano válido exige pelo menos 5 ações com retorno válido.
    - ranking decrescente por RANK_METRIC.
    - desempate por MACRO_SECTOR em ordem alfabética.
    - seleciona os 4 primeiros.
    - HISTORY_END < YEAR: sem look-ahead.

Fontes:
    data/returns.csv                  — histórico congelado
    data_live/returns_operational.csv — continuidade operacional

Os dois arquivos são combinados somente em memória.
data/returns.csv nunca é alterado.

Auditoria:
    Recalcula os rankings históricos LOOKBACK=1 e compara
    com data/sector_rankings.csv antes de produzir a seleção atual.

Saídas:
    data_live/sector_rankings_current.csv
    data_live/selected_sectors_current.csv

Observação:
    Este arquivo NÃO altera nenhum arquivo em data/.
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

HISTORICAL_RETURNS_FILE = DATA / "returns.csv"
OPERATIONAL_RETURNS_FILE = DATA_LIVE / "returns_operational.csv"
HISTORICAL_RANKINGS_FILE = DATA / "sector_rankings.csv"

OUTPUT_RANKING_FILE = (
    DATA_LIVE / "sector_rankings_current.csv"
)

OUTPUT_SELECTION_FILE = (
    DATA_LIVE / "selected_sectors_current.csv"
)

CURRENT_YEAR = datetime.now().year

LOOKBACK = 1
N_SECTORS = 4
MIN_STOCKS_SECTOR_YEAR = 5


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


def normalize_sector(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )


# =============================================================================
# 1. CARREGAMENTO DOS RETORNOS HISTÓRICOS CONGELADOS
# =============================================================================

def _prepare_returns(
    df: pd.DataFrame,
    name: str,
) -> pd.DataFrame:

    require_columns(
        df,
        [
            "YEAR",
            "TICKER",
            "MACRO_SECTOR",
            "RET_ANNUAL_VALID",
        ],
        name,
    )

    df = df.copy()

    df["YEAR"] = pd.to_numeric(
        df["YEAR"],
        errors="coerce",
    )

    df["RET_ANNUAL_VALID"] = pd.to_numeric(
        df["RET_ANNUAL_VALID"],
        errors="coerce",
    )

    df["TICKER"] = (
        df["TICKER"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["MACRO_SECTOR"] = normalize_sector(
        df["MACRO_SECTOR"]
    )

    df = df[
        df["YEAR"].notna()
        & df["RET_ANNUAL_VALID"].notna()
        & df["TICKER"].ne("")
        & df["MACRO_SECTOR"].ne("")
        & df["MACRO_SECTOR"].ne("UNCLASSIFIED")
    ].copy()

    df["YEAR"] = df["YEAR"].astype(int)

    duplicated = int(
        df.duplicated(
            ["YEAR", "TICKER"]
        ).sum()
    )

    if duplicated:
        raise RuntimeError(
            f"{name} possui duplicidades "
            f"YEAR × TICKER: {duplicated}"
        )

    return df


def load_returns() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if not HISTORICAL_RETURNS_FILE.exists():
        raise RuntimeError(
            f"Arquivo histórico não encontrado: "
            f"{HISTORICAL_RETURNS_FILE}"
        )

    historical_raw = pd.read_csv(
        HISTORICAL_RETURNS_FILE,
        low_memory=False,
    )

    historical = _prepare_returns(
        historical_raw,
        "data/returns.csv",
    )

    # O arquivo operacional é obrigatório a partir desta versão.
    if not OPERATIONAL_RETURNS_FILE.exists():
        raise RuntimeError(
            "Arquivo operacional não encontrado: "
            f"{OPERATIONAL_RETURNS_FILE}. "
            "Execute update_returns.py antes de update_sectors.py."
        )

    operational_raw = pd.read_csv(
        OPERATIONAL_RETURNS_FILE,
        low_memory=False,
    )

    operational = _prepare_returns(
        operational_raw,
        "data_live/returns_operational.csv",
    )

    # Proteção estrutural: a camada operacional não pode sobrescrever
    # nenhuma chave YEAR × TICKER já existente no histórico congelado.
    historical_keys = set(
        zip(
            historical["YEAR"],
            historical["TICKER"],
        )
    )

    operational_keys = set(
        zip(
            operational["YEAR"],
            operational["TICKER"],
        )
    )

    overlap = historical_keys.intersection(
        operational_keys
    )

    if overlap:
        sample = sorted(overlap)[:10]
        raise RuntimeError(
            "Sobreposição entre histórico congelado e "
            "retornos operacionais. Exemplos: "
            f"{sample}"
        )

    combined = pd.concat(
        [
            historical,
            operational,
        ],
        ignore_index=True,
    )

    combined = (
        combined
        .sort_values(
            ["YEAR", "TICKER"]
        )
        .reset_index(drop=True)
    )

    duplicated = int(
        combined.duplicated(
            ["YEAR", "TICKER"]
        ).sum()
    )

    if duplicated:
        raise RuntimeError(
            "Base combinada possui duplicidades "
            f"YEAR × TICKER: {duplicated}"
        )

    return historical, operational, combined


# =============================================================================
# 2. PERFORMANCE SETORIAL — MESMA DEFINIÇÃO HISTÓRICA
# =============================================================================

def build_sector_year(
    returns: pd.DataFrame,
) -> pd.DataFrame:

    sector_year = (
        returns
        .groupby(
            [
                "YEAR",
                "MACRO_SECTOR",
            ],
            as_index=False,
        )
        .agg(
            N_STOCKS=(
                "TICKER",
                "nunique",
            ),
            EW_RETURN=(
                "RET_ANNUAL_VALID",
                "mean",
            ),
        )
    )

    sector_year[
        "VALID_SECTOR_YEAR"
    ] = (
        sector_year[
            "N_STOCKS"
        ]
        >=
        MIN_STOCKS_SECTOR_YEAR
    )

    return sector_year


# =============================================================================
# 3. RANKING TOP4_1Y
# =============================================================================

def build_ranking_for_year(
    sector_year: pd.DataFrame,
    target_year: int,
) -> pd.DataFrame:

    history_start = (
        target_year - LOOKBACK
    )

    history_end = (
        target_year - 1
    )

    # LOOKBACK congelado = 1.
    # Logo RANK_METRIC é exatamente o EW_RETURN
    # do ano imediatamente anterior.
    history = sector_year[
        (sector_year["YEAR"] == history_end)
        & sector_year["VALID_SECTOR_YEAR"]
        & sector_year["EW_RETURN"].notna()
    ].copy()

    if history.empty:
        raise RuntimeError(
            "Não há performance setorial válida "
            f"para {history_end}. "
            "Não é possível produzir TOP4_1Y "
            f"para {target_year}."
        )

    ranking = history[
        [
            "MACRO_SECTOR",
            "EW_RETURN",
            "N_STOCKS",
        ]
    ].copy()

    ranking = ranking.rename(
        columns={
            "EW_RETURN":
                "RANK_METRIC",
        }
    )

    ranking.insert(
        0,
        "LOOKBACK",
        LOOKBACK,
    )

    ranking.insert(
        1,
        "YEAR",
        target_year,
    )

    ranking[
        "HISTORY_START"
    ] = history_start

    ranking[
        "HISTORY_END"
    ] = history_end

    ranking = (
        ranking
        .sort_values(
            [
                "RANK_METRIC",
                "MACRO_SECTOR",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    ranking[
        "RANK"
    ] = np.arange(
        1,
        len(ranking) + 1,
    )

    ranking = ranking[
        [
            "LOOKBACK",
            "YEAR",
            "MACRO_SECTOR",
            "RANK_METRIC",
            "HISTORY_START",
            "HISTORY_END",
            "RANK",
            "N_STOCKS",
        ]
    ]

    # Controle explícito de look-ahead.
    if (
        ranking[
            "HISTORY_END"
        ]
        >=
        ranking[
            "YEAR"
        ]
    ).any():
        raise RuntimeError(
            "Look-ahead detectado "
            "no ranking setorial."
        )

    return ranking


# =============================================================================
# 4. AUDITORIA CONTRA O ESTUDO HISTÓRICO
# =============================================================================

def audit_historical_rule(
    sector_year: pd.DataFrame,
):
    if not HISTORICAL_RANKINGS_FILE.exists():
        raise RuntimeError(
            f"Arquivo não encontrado: "
            f"{HISTORICAL_RANKINGS_FILE}"
        )

    frozen = pd.read_csv(
        HISTORICAL_RANKINGS_FILE,
        low_memory=False,
    )

    require_columns(
        frozen,
        [
            "LOOKBACK",
            "YEAR",
            "MACRO_SECTOR",
            "RANK_METRIC",
            "HISTORY_START",
            "HISTORY_END",
            "RANK",
        ],
        "sector_rankings.csv",
    )

    for col in [
        "LOOKBACK",
        "YEAR",
        "HISTORY_START",
        "HISTORY_END",
        "RANK",
    ]:
        frozen[col] = pd.to_numeric(
            frozen[col],
            errors="coerce",
        )

    frozen[
        "RANK_METRIC"
    ] = pd.to_numeric(
        frozen[
            "RANK_METRIC"
        ],
        errors="coerce",
    )

    frozen[
        "MACRO_SECTOR"
    ] = normalize_sector(
        frozen[
            "MACRO_SECTOR"
        ]
    )

    frozen = frozen[
        (frozen["LOOKBACK"] == LOOKBACK)
        & frozen["YEAR"].notna()
        & frozen["RANK"].notna()
        & frozen["RANK_METRIC"].notna()
    ].copy()

    frozen["YEAR"] = (
        frozen["YEAR"].astype(int)
    )

    years = sorted(
        frozen["YEAR"].unique()
    )

    if not years:
        raise RuntimeError(
            "Nenhum ranking histórico LOOKBACK=1."
        )

    mismatches = []

    for year in years:
        rebuilt = build_ranking_for_year(
            sector_year,
            int(year),
        )

        expected = (
            frozen[
                frozen["YEAR"] == year
            ][
                [
                    "MACRO_SECTOR",
                    "RANK_METRIC",
                    "RANK",
                    "HISTORY_START",
                    "HISTORY_END",
                ]
            ]
            .sort_values(
                [
                    "RANK",
                    "MACRO_SECTOR",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        actual = (
            rebuilt[
                [
                    "MACRO_SECTOR",
                    "RANK_METRIC",
                    "RANK",
                    "HISTORY_START",
                    "HISTORY_END",
                ]
            ]
            .sort_values(
                [
                    "RANK",
                    "MACRO_SECTOR",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        # Primeiro verifica estrutura/ranking.
        structure_equal = (
            len(expected) == len(actual)
            and expected[
                [
                    "MACRO_SECTOR",
                    "RANK",
                    "HISTORY_START",
                    "HISTORY_END",
                ]
            ].equals(
                actual[
                    [
                        "MACRO_SECTOR",
                        "RANK",
                        "HISTORY_START",
                        "HISTORY_END",
                    ]
                ]
            )
        )

        metric_equal = False

        if structure_equal:
            metric_equal = bool(
                np.allclose(
                    expected[
                        "RANK_METRIC"
                    ].to_numpy(
                        dtype=float
                    ),
                    actual[
                        "RANK_METRIC"
                    ].to_numpy(
                        dtype=float
                    ),
                    rtol=1e-10,
                    atol=1e-12,
                    equal_nan=False,
                )
            )

        if not (
            structure_equal
            and metric_equal
        ):
            mismatches.append(
                int(year)
            )

    if mismatches:
        raise RuntimeError(
            "Regra operacional não reproduziu "
            "o ranking histórico congelado nos anos: "
            f"{mismatches}"
        )

    print()
    print("=" * 78)
    print(
        "AUDITORIA — REPRODUÇÃO DA REGRA HISTÓRICA"
    )
    print("=" * 78)
    print(
        f"Anos auditados : "
        f"{years[0]}–{years[-1]}"
    )
    print(
        f"LOOKBACK       : {LOOKBACK}"
    )
    print(
        "RANK_METRIC    : EW_RETURN"
    )
    print(
        "Reprodução     : PASS"
    )
    print(
        "FREEZE_TOP4_1Y : PRESERVADO"
    )


# =============================================================================
# 5. SELEÇÃO ATUAL
# =============================================================================

def select_current_top4(
    ranking: pd.DataFrame,
) -> pd.DataFrame:

    selected = (
        ranking
        .sort_values(
            [
                "RANK",
                "MACRO_SECTOR",
            ]
        )
        .head(
            N_SECTORS
        )
        .copy()
    )

    if len(selected) != N_SECTORS:
        raise RuntimeError(
            f"Esperados {N_SECTORS} setores; "
            f"encontrados {len(selected)}."
        )

    if (
        selected[
            "MACRO_SECTOR"
        ]
        .nunique()
        !=
        N_SECTORS
    ):
        raise RuntimeError(
            "TOP4 possui setores duplicados."
        )

    selected[
        "RULE"
    ] = "TOP4_1Y"

    selected[
        "SELECTED"
    ] = True

    return selected[
        [
            "RULE",
            "YEAR",
            "RANK",
            "MACRO_SECTOR",
            "RANK_METRIC",
            "HISTORY_START",
            "HISTORY_END",
            "N_STOCKS",
            "SELECTED",
        ]
    ]


# =============================================================================
# EXECUÇÃO
# =============================================================================

def main():
    print()
    print("=" * 78)
    print(
        "PORTIFOLIO-B3 — "
        "SELEÇÃO SETORIAL OPERACIONAL"
    )
    print("=" * 78)

    historical_returns, operational_returns, returns = load_returns()

    print()
    print(
        f"Retornos históricos : "
        f"{int(historical_returns['YEAR'].min())}–"
        f"{int(historical_returns['YEAR'].max())}"
    )

    if operational_returns.empty:
        print(
            "Retornos operacionais válidos : "
            "0 (ano corrente ainda parcial)"
        )
    else:
        print(
            f"Retornos operacionais válidos : "
            f"{len(operational_returns):,} "
            f"({int(operational_returns['YEAR'].min())}–"
            f"{int(operational_returns['YEAR'].max())})"
        )
    print(
        f"Ano operacional     : "
        f"{CURRENT_YEAR}"
    )
    print(
        f"Ano usado no ranking: "
        f"{CURRENT_YEAR - 1}"
    )

    # A auditoria histórica usa SOMENTE o histórico congelado.
    # Assim, a camada operacional jamais pode alterar a prova
    # de reprodução do FREEZE_TOP4_1Y.
    historical_sector_year = build_sector_year(
        historical_returns
    )

    audit_historical_rule(
        historical_sector_year
    )

    # O ranking corrente usa histórico + continuidade operacional
    # combinados somente em memória.
    combined_sector_year = build_sector_year(
        returns
    )

    current_ranking = (
        build_ranking_for_year(
            combined_sector_year,
            CURRENT_YEAR,
        )
    )

    current_top4 = (
        select_current_top4(
            current_ranking
        )
    )

    DATA_LIVE.mkdir(
        parents=True,
        exist_ok=True,
    )

    current_ranking.to_csv(
        OUTPUT_RANKING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    current_top4.to_csv(
        OUTPUT_SELECTION_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print(
        f"TOP 4 SETORES — {CURRENT_YEAR}"
    )
    print("=" * 78)

    display = current_top4.copy()

    display[
        "RANK_METRIC"
    ] = display[
        "RANK_METRIC"
    ].map(
        lambda x:
        f"{x:.2%}"
    )

    print(
        display[
            [
                "RANK",
                "MACRO_SECTOR",
                "RANK_METRIC",
                "HISTORY_END",
                "N_STOCKS",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("AUDITORIA FINAL")
    print("=" * 78)
    print(
        "Regra ............................. TOP4_1Y"
    )
    print(
        "Lookback .......................... 1 ano"
    )
    print(
        "Ano corrente no ranking ........... NÃO"
    )
    print(
        "HISTORY_END < YEAR ................ PASS"
    )
    print(
        "Quantidade de setores ............. 4"
    )
    print(
        "Regra histórica reproduzida ....... PASS"
    )
    print(
        "Histórico congelado ............... PRESERVADO"
    )
    print(
        "Retornos operacionais ............. INTEGRADOS EM MEMÓRIA"
    )
    print(
        "Fonte ranking corrente ............ HISTÓRICO + OPERACIONAL"
    )
    print(
        "STATUS ............................ PASS"
    )

    print()
    print("=" * 78)
    print("ARQUIVOS GERADOS")
    print("=" * 78)
    print(
        f"Ranking : {OUTPUT_RANKING_FILE}"
    )
    print(
        f"TOP4    : {OUTPUT_SELECTION_FILE}"
    )
    print()
    print(
        "STATUS: SELEÇÃO SETORIAL "
        "OPERACIONAL VALIDADA"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
