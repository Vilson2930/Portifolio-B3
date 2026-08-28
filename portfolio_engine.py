"""
PORTIFOLIO-B3
Motor da metodologia congelada no estudo.

Arquitetura:
    4 setores x 3 ações = 12 ações

Regra setorial:
    TOP4_1Y

Regra de seleção das ações:
    80% Discount + 20% Fundamentals

Versão:
    1.0.0
"""

from __future__ import annotations

import pandas as pd


# ============================================================
# CONFIGURAÇÃO CONGELADA
# ============================================================

N_SECTORS = 4
STOCKS_PER_SECTOR = 3

DISCOUNT_WEIGHT = 0.80
FUNDAMENTAL_WEIGHT = 0.20

SECTOR_RULE = "TOP4_1Y"
STOCK_RULE = "DISCOUNT_80_FUNDAMENTALS_20"


# Componentes utilizados no FUND_SCORE
FUNDAMENTAL_COMPONENTS = [
    "ROE_W",
    "ROA_W",
    "OPERATING_MARGIN_W",
    "NET_MARGIN_W",
]

MIN_FUNDAMENTAL_COMPONENTS = 3


# ============================================================
# VALIDAÇÃO DE COLUNAS
# ============================================================

def require_columns(
    df: pd.DataFrame,
    columns: list[str],
    name: str,
) -> None:
    """
    Interrompe a execução se alguma coluna obrigatória estiver ausente.
    """

    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"{name}: colunas obrigatórias ausentes: {missing}"
        )


# ============================================================
# PERCENTIL POR ANO × SETOR
# ============================================================

def percentile_rank(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Calcula percentile rank exatamente dentro de:

        YEAR × MACRO_SECTOR

    Valores maiores recebem scores maiores.
    """

    return (
        df.groupby(
            ["YEAR", "MACRO_SECTOR"],
            observed=True,
        )[column]
        .rank(
            pct=True,
            method="average",
        )
    )


# ============================================================
# FUNDAMENTOS
# ============================================================

def build_fundamental_score(
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói o FUND_SCORE utilizado pela regra congelada.

    Componentes:

        ROE
        ROA
        Operating Margin
        Net Margin
        Qualidade de alavancagem

    A qualidade de alavancagem é baseada no inverso do
    Debt-to-Equity Proxy.

    O componente de dívida somente é considerado quando
    EQUITY > 0.

    FUND_SCORE exige pelo menos 3 componentes válidos.
    """

    required = [
        "YEAR",
        "TICKER",
        "MACRO_SECTOR",
        "ROE_W",
        "ROA_W",
        "OPERATING_MARGIN_W",
        "NET_MARGIN_W",
        "DEBT_TO_EQUITY_PROXY_W",
        "EQUITY",
    ]

    require_columns(
        fundamentals,
        required,
        "fundamental_factors",
    )

    df = fundamentals.copy()

    # --------------------------------------------------------
    # Percentis dos fundamentos
    # --------------------------------------------------------

    df["ROE_SCORE"] = percentile_rank(df, "ROE_W")
    df["ROA_SCORE"] = percentile_rank(df, "ROA_W")

    df["OPERATING_MARGIN_SCORE"] = percentile_rank(
        df,
        "OPERATING_MARGIN_W",
    )

    df["NET_MARGIN_SCORE"] = percentile_rank(
        df,
        "NET_MARGIN_W",
    )

    # --------------------------------------------------------
    # Qualidade da alavancagem
    #
    # Menor dívida / patrimônio = melhor.
    #
    # No estudo:
    #   leverage_quality = -DEBT_TO_EQUITY_PROXY_W
    #
    # Apenas patrimônio positivo é considerado.
    # --------------------------------------------------------

    df["LEVERAGE_QUALITY"] = pd.NA

    positive_equity = df["EQUITY"] > 0

    df.loc[
        positive_equity,
        "LEVERAGE_QUALITY",
    ] = -df.loc[
        positive_equity,
        "DEBT_TO_EQUITY_PROXY_W",
    ]

    df["LEVERAGE_QUALITY"] = pd.to_numeric(
        df["LEVERAGE_QUALITY"],
        errors="coerce",
    )

    df["LEVERAGE_SCORE"] = percentile_rank(
        df,
        "LEVERAGE_QUALITY",
    )

    # --------------------------------------------------------
    # FUND SCORE
    # --------------------------------------------------------

    score_columns = [
        "ROE_SCORE",
        "ROA_SCORE",
        "OPERATING_MARGIN_SCORE",
        "NET_MARGIN_SCORE",
        "LEVERAGE_SCORE",
    ]

    df["FUND_COMPONENTS_VALID"] = (
        df[score_columns]
        .notna()
        .sum(axis=1)
    )

    df["FUND_SCORE"] = (
        df[score_columns]
        .mean(
            axis=1,
            skipna=True,
        )
    )

    df.loc[
        df["FUND_COMPONENTS_VALID"]
        < MIN_FUNDAMENTAL_COMPONENTS,
        "FUND_SCORE",
    ] = pd.NA

    df["FUND_SCORE"] = pd.to_numeric(
        df["FUND_SCORE"],
        errors="coerce",
    )

    return df


# ============================================================
# DISCOUNT
# ============================================================

def build_discount_score(
    price_factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói o DISCOUNT_SCORE.

    DISCOUNT_52W já vem da metodologia da Célula 26:

        DRAWDOWN_52W =
            PRICE_T0 / MAX_PRICE_52W - 1

        DISCOUNT_52W =
            max(0, -DRAWDOWN_52W)

    Aqui apenas transformamos o fator em percentile rank
    dentro de YEAR × MACRO_SECTOR.
    """

    required = [
        "YEAR",
        "TICKER",
        "MACRO_SECTOR",
        "DISCOUNT_52W",
    ]

    require_columns(
        price_factors,
        required,
        "price_factors",
    )

    df = price_factors.copy()

    df["DISCOUNT_SCORE"] = percentile_rank(
        df,
        "DISCOUNT_52W",
    )

    return df


# ============================================================
# SCORE FINAL
# ============================================================

def build_final_scores(
    fundamentals: pd.DataFrame,
    price_factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Junta fundamentos e preço e aplica a regra vencedora:

        FINAL_SCORE =
            0.80 × DISCOUNT_SCORE
            +
            0.20 × FUND_SCORE
    """

    fund = build_fundamental_score(fundamentals)
    price = build_discount_score(price_factors)

    keys = [
        "YEAR",
        "TICKER",
        "MACRO_SECTOR",
    ]

    price_keep = keys + [
        "DISCOUNT_52W",
        "DISCOUNT_SCORE",
    ]

    df = fund.merge(
        price[price_keep],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    df["FINAL_SCORE"] = (
        DISCOUNT_WEIGHT * df["DISCOUNT_SCORE"]
        +
        FUNDAMENTAL_WEIGHT * df["FUND_SCORE"]
    )

    return df


# ============================================================
# TOP 3 AÇÕES POR SETOR
# ============================================================

def select_top3_per_sector(
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Seleciona exatamente as três ações de maior FINAL_SCORE
    dentro de cada YEAR × MACRO_SECTOR.

    Desempate:
        FINAL_SCORE decrescente
        TICKER crescente

    Essa ordem mantém a seleção determinística.
    """

    required = [
        "YEAR",
        "TICKER",
        "MACRO_SECTOR",
        "FINAL_SCORE",
    ]

    require_columns(
        scores,
        required,
        "scores",
    )

    eligible = scores[
        scores["FINAL_SCORE"].notna()
    ].copy()

    eligible = eligible.sort_values(
        [
            "YEAR",
            "MACRO_SECTOR",
            "FINAL_SCORE",
            "TICKER",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    )

    selected = (
        eligible
        .groupby(
            ["YEAR", "MACRO_SECTOR"],
            observed=True,
            group_keys=False,
        )
        .head(STOCKS_PER_SECTOR)
        .copy()
    )

    selected["RANK_IN_SECTOR"] = (
        selected
        .groupby(
            ["YEAR", "MACRO_SECTOR"],
            observed=True,
        )
        .cumcount()
        + 1
    )

    return selected


# ============================================================
# AUDITORIA
# ============================================================

def audit_selection(
    selected: pd.DataFrame,
) -> None:
    """
    Valida a arquitetura 4 × 3 depois que os quatro setores
    TOP4_1Y já tiverem sido aplicados.
    """

    if selected.empty:
        raise ValueError(
            "AUDITORIA FAIL: nenhuma ação selecionada."
        )

    for year, year_df in selected.groupby("YEAR"):

        sectors = year_df["MACRO_SECTOR"].nunique()
        stocks = len(year_df)

        if sectors != N_SECTORS:
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                f"{sectors} setores encontrados; "
                f"esperados {N_SECTORS}."
            )

        counts = (
            year_df.groupby("MACRO_SECTOR")
            .size()
        )

        invalid = counts[
            counts != STOCKS_PER_SECTOR
        ]

        if not invalid.empty:
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                "nem todos os setores possuem "
                f"{STOCKS_PER_SECTOR} ações."
            )

        expected_total = (
            N_SECTORS
            * STOCKS_PER_SECTOR
        )

        if stocks != expected_total:
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                f"{stocks} ações encontradas; "
                f"esperadas {expected_total}."
            )


# ============================================================
# MOTOR
# ============================================================

def run_stock_selection(
    fundamentals: pd.DataFrame,
    price_factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Executa a etapa de seleção das ações.

    IMPORTANTE:
    Esta função pressupõe que os dados recebidos já estejam
    restritos aos quatro setores definidos pelo TOP4_1Y.

    A implementação do TOP4_1Y será incorporada somente quando
    carregarmos o input histórico correspondente, para não
    inventarmos dados ou regras que não estejam no estudo.
    """

    scores = build_final_scores(
        fundamentals,
        price_factors,
    )

    selected = select_top3_per_sector(scores)

    audit_selection(selected)

    output_columns = [
        "YEAR",
        "MACRO_SECTOR",
        "RANK_IN_SECTOR",
        "TICKER",
        "DISCOUNT_52W",
        "DISCOUNT_SCORE",
        "FUND_SCORE",
        "FINAL_SCORE",
    ]

    return (
        selected[output_columns]
        .sort_values(
            [
                "YEAR",
                "MACRO_SECTOR",
                "RANK_IN_SECTOR",
            ]
        )
        .reset_index(drop=True)
    )
