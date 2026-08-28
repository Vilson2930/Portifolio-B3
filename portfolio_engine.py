"""
PORTIFOLIO-B3
Motor de reprodução da metodologia congelada no estudo.

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
TOTAL_STOCKS = N_SECTORS * STOCKS_PER_SECTOR

DISCOUNT_WEIGHT = 0.80
FUNDAMENTAL_WEIGHT = 0.20

SECTOR_RULE = "TOP4_1Y"
STOCK_RULE = "DISCOUNT_80_FUNDAMENTALS_20"

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

    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

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
    Calcula percentile rank dentro de:

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
# SELEÇÃO DINÂMICA DOS 4 SETORES — TOP4_1Y
# ============================================================

def select_top4_1y(
    sector_rankings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduz a regra setorial congelada na Célula 24.

    A REGRA é congelada.
    Os NOMES dos setores não são congelados.

    Para cada YEAR:

        1. utiliza somente LOOKBACK == 1;
        2. verifica ausência de look-ahead;
        3. ordena por RANK crescente;
        4. usa MACRO_SECTOR como desempate;
        5. seleciona os quatro primeiros setores.
    """

    required = [
        "LOOKBACK",
        "YEAR",
        "MACRO_SECTOR",
        "RANK_METRIC",
        "HISTORY_START",
        "HISTORY_END",
        "RANK",
    ]

    require_columns(
        sector_rankings,
        required,
        "sector_rankings",
    )

    df = sector_rankings.copy()

    # --------------------------------------------------------
    # Conversões
    # --------------------------------------------------------

    for col in [
        "LOOKBACK",
        "YEAR",
        "RANK",
        "HISTORY_START",
        "HISTORY_END",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["MACRO_SECTOR"] = (
        df["MACRO_SECTOR"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Linhas válidas
    # --------------------------------------------------------

    df = df[
        df["LOOKBACK"].notna()
        & df["YEAR"].notna()
        & df["RANK"].notna()
    ].copy()

    df["LOOKBACK"] = (
        df["LOOKBACK"]
        .astype(int)
    )

    df["YEAR"] = (
        df["YEAR"]
        .astype(int)
    )

    df["RANK"] = (
        df["RANK"]
        .astype(int)
    )

    # --------------------------------------------------------
    # AUDITORIA — LOOK-AHEAD
    #
    # Célula 24:
    #
    # HISTORY_END < YEAR
    # --------------------------------------------------------

    lookahead_fail = df[
        df["HISTORY_END"].notna()
        & (
            df["HISTORY_END"]
            >= df["YEAR"]
        )
    ]

    if not lookahead_fail.empty:
        raise ValueError(
            "TOP4_1Y — look-ahead detectado em "
            f"{len(lookahead_fail)} linhas."
        )

    # --------------------------------------------------------
    # AUDITORIA — DUPLICIDADES
    #
    # Uma observação por:
    #
    # LOOKBACK × YEAR × MACRO_SECTOR
    # --------------------------------------------------------

    duplicated = df.duplicated(
        [
            "LOOKBACK",
            "YEAR",
            "MACRO_SECTOR",
        ]
    )

    if duplicated.any():
        raise ValueError(
            "TOP4_1Y — duplicidades encontradas em "
            "LOOKBACK × YEAR × MACRO_SECTOR."
        )

    # --------------------------------------------------------
    # REGRA VENCEDORA
    #
    # TOP4_1Y
    # --------------------------------------------------------

    one_year = df[
        df["LOOKBACK"] == 1
    ].copy()

    if one_year.empty:
        raise ValueError(
            "TOP4_1Y — nenhuma observação com LOOKBACK = 1."
        )

    selected_rows = []

    # --------------------------------------------------------
    # TOP 4 SETORES DE CADA ANO
    # --------------------------------------------------------

    for year, group in one_year.groupby(
        "YEAR",
        sort=True,
    ):

        group = (
            group
            .sort_values(
                [
                    "RANK",
                    "MACRO_SECTOR",
                ],
                ascending=[
                    True,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        selected = (
            group
            .head(N_SECTORS)
            .copy()
        )

        if len(selected) != N_SECTORS:
            raise ValueError(
                f"TOP4_1Y — {year}: "
                f"{len(selected)} setores disponíveis; "
                f"esperados {N_SECTORS}."
            )

        selected["SECTOR_RANK"] = range(
            1,
            N_SECTORS + 1,
        )

        selected_rows.append(
            selected[
                [
                    "YEAR",
                    "SECTOR_RANK",
                    "MACRO_SECTOR",
                    "RANK",
                    "RANK_METRIC",
                    "HISTORY_START",
                    "HISTORY_END",
                ]
            ]
        )

    result = pd.concat(
        selected_rows,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # AUDITORIA DA ARQUITETURA SETORIAL
    # --------------------------------------------------------

    sector_counts = (
        result
        .groupby("YEAR")
        ["MACRO_SECTOR"]
        .nunique()
    )

    invalid = sector_counts[
        sector_counts != N_SECTORS
    ]

    if not invalid.empty:
        raise ValueError(
            "TOP4_1Y — falha estrutural: "
            "nem todos os anos possuem exatamente "
            f"{N_SECTORS} setores."
        )

    return result


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

    A qualidade de alavancagem utiliza:

        -DEBT_TO_EQUITY_PROXY_W

    somente quando:

        EQUITY > 0

    FUND_SCORE exige pelo menos três componentes válidos.
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
    # Padronização das chaves
    # --------------------------------------------------------

    df["YEAR"] = pd.to_numeric(
        df["YEAR"],
        errors="coerce",
    )

    df = df[
        df["YEAR"].notna()
    ].copy()

    df["YEAR"] = (
        df["YEAR"]
        .astype(int)
    )

    df["TICKER"] = (
        df["TICKER"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["MACRO_SECTOR"] = (
        df["MACRO_SECTOR"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # --------------------------------------------------------
    # Percentis dos fundamentos
    # --------------------------------------------------------

    df["ROE_SCORE"] = percentile_rank(
        df,
        "ROE_W",
    )

    df["ROA_SCORE"] = percentile_rank(
        df,
        "ROA_W",
    )

    df["OPERATING_MARGIN_SCORE"] = percentile_rank(
        df,
        "OPERATING_MARGIN_W",
    )

    df["NET_MARGIN_SCORE"] = percentile_rank(
        df,
        "NET_MARGIN_W",
    )

    # --------------------------------------------------------
    # Qualidade de alavancagem
    #
    # Menor dívida/patrimônio = melhor.
    # --------------------------------------------------------

    df["LEVERAGE_QUALITY"] = float("nan")

    positive_equity = (
        pd.to_numeric(
            df["EQUITY"],
            errors="coerce",
        )
        > 0
    )

    debt_equity = pd.to_numeric(
        df["DEBT_TO_EQUITY_PROXY_W"],
        errors="coerce",
    )

    df.loc[
        positive_equity,
        "LEVERAGE_QUALITY",
    ] = -debt_equity.loc[
        positive_equity
    ]

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
    ] = float("nan")

    return df


# ============================================================
# DISCOUNT
# ============================================================

def build_discount_score(
    price_factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Constrói o DISCOUNT_SCORE.

    O DISCOUNT_52W é produzido pela metodologia da Célula 26:

        DRAWDOWN_52W =
            PRICE_T0 / MAX_PRICE_52W - 1

        DISCOUNT_52W =
            max(0, -DRAWDOWN_52W)

    Aqui é calculado o percentile rank dentro de:

        YEAR × MACRO_SECTOR
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

    df["YEAR"] = pd.to_numeric(
        df["YEAR"],
        errors="coerce",
    )

    df = df[
        df["YEAR"].notna()
    ].copy()

    df["YEAR"] = (
        df["YEAR"]
        .astype(int)
    )

    df["TICKER"] = (
        df["TICKER"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["MACRO_SECTOR"] = (
        df["MACRO_SECTOR"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["DISCOUNT_52W"] = pd.to_numeric(
        df["DISCOUNT_52W"],
        errors="coerce",
    )

    df["DISCOUNT_SCORE"] = percentile_rank(
        df,
        "DISCOUNT_52W",
    )

    return df


# ============================================================
# SCORE FINAL DAS AÇÕES
# ============================================================

def build_final_scores(
    fundamentals: pd.DataFrame,
    price_factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Junta fundamentos e fatores de preço.

    Regra vencedora congelada:

        FINAL_SCORE =
            0.80 × DISCOUNT_SCORE
            +
            0.20 × FUND_SCORE
    """

    fund = build_fundamental_score(
        fundamentals
    )

    price = build_discount_score(
        price_factors
    )

    keys = [
        "YEAR",
        "TICKER",
        "MACRO_SECTOR",
    ]

    # --------------------------------------------------------
    # Auditoria de unicidade antes do merge
    # --------------------------------------------------------

    if fund.duplicated(keys).any():
        raise ValueError(
            "fundamental_factors possui duplicidades em "
            "YEAR × TICKER × MACRO_SECTOR."
        )

    if price.duplicated(keys).any():
        raise ValueError(
            "price_factors possui duplicidades em "
            "YEAR × TICKER × MACRO_SECTOR."
        )

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

    if df.empty:
        raise ValueError(
            "Merge fundamentos × preço resultou em base vazia."
        )

    df["FINAL_SCORE"] = (
        DISCOUNT_WEIGHT
        * df["DISCOUNT_SCORE"]
        +
        FUNDAMENTAL_WEIGHT
        * df["FUND_SCORE"]
    )

    return df


# ============================================================
# APLICAR OS SETORES TOP4_1Y
# ============================================================

def apply_selected_sectors(
    scores: pd.DataFrame,
    selected_sectors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mantém somente as ações pertencentes aos quatro setores
    escolhidos dinamicamente pelo TOP4_1Y em cada YEAR.
    """

    required = [
        "YEAR",
        "SECTOR_RANK",
        "MACRO_SECTOR",
    ]

    require_columns(
        selected_sectors,
        required,
        "selected_sectors",
    )

    sector_keys = selected_sectors[
        [
            "YEAR",
            "SECTOR_RANK",
            "MACRO_SECTOR",
        ]
    ].copy()

    if sector_keys.duplicated(
        ["YEAR", "MACRO_SECTOR"]
    ).any():
        raise ValueError(
            "Seleção setorial possui duplicidades em "
            "YEAR × MACRO_SECTOR."
        )

    filtered = scores.merge(
        sector_keys,
        on=[
            "YEAR",
            "MACRO_SECTOR",
        ],
        how="inner",
        validate="many_to_one",
    )

    if filtered.empty:
        raise ValueError(
            "Nenhuma ação permaneceu após aplicar TOP4_1Y."
        )

    return filtered


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
            [
                "YEAR",
                "MACRO_SECTOR",
            ],
            observed=True,
            group_keys=False,
        )
        .head(STOCKS_PER_SECTOR)
        .copy()
    )

    selected["RANK_IN_SECTOR"] = (
        selected
        .groupby(
            [
                "YEAR",
                "MACRO_SECTOR",
            ],
            observed=True,
        )
        .cumcount()
        + 1
    )

    return selected


# ============================================================
# AUDITORIA FINAL 4 × 3
# ============================================================

def audit_selection(
    selected: pd.DataFrame,
) -> None:
    """
    Confirma a arquitetura congelada:

        4 setores
        ×
        3 ações
        =
        12 ações
    """

    if selected.empty:
        raise ValueError(
            "AUDITORIA FAIL: nenhuma ação selecionada."
        )

    for year, year_df in selected.groupby(
        "YEAR",
        sort=True,
    ):

        sectors = (
            year_df["MACRO_SECTOR"]
            .nunique()
        )

        stocks = len(year_df)

        if sectors != N_SECTORS:
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                f"{sectors} setores encontrados; "
                f"esperados {N_SECTORS}."
            )

        counts = (
            year_df
            .groupby("MACRO_SECTOR")
            .size()
        )

        invalid = counts[
            counts != STOCKS_PER_SECTOR
        ]

        if not invalid.empty:
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                "nem todos os setores possuem exatamente "
                f"{STOCKS_PER_SECTOR} ações. "
                f"Contagens: {counts.to_dict()}"
            )

        if stocks != TOTAL_STOCKS:
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                f"{stocks} ações encontradas; "
                f"esperadas {TOTAL_STOCKS}."
            )


# ============================================================
# MOTOR COMPLETO
# ============================================================

def run_portfolio_selection(
    sector_rankings: pd.DataFrame,
    fundamentals: pd.DataFrame,
    price_factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Executa a cadeia de seleção da carteira:

        rankings históricos
                ↓
            TOP4_1Y
                ↓
          4 setores
                ↓
        FUND_SCORE
                +
        DISCOUNT_SCORE
                ↓
        FINAL_SCORE 80/20
                ↓
        Top 3 por setor
                ↓
          12 ações
    """

    # --------------------------------------------------------
    # 1. Selecionar dinamicamente os quatro setores
    # --------------------------------------------------------

    selected_sectors = select_top4_1y(
        sector_rankings
    )

    # --------------------------------------------------------
    # 2. Construir scores de TODAS as ações
    # --------------------------------------------------------

    scores = build_final_scores(
        fundamentals,
        price_factors,
    )

    # --------------------------------------------------------
    # 3. Restringir aos quatro setores TOP4_1Y
    # --------------------------------------------------------

    scores = apply_selected_sectors(
        scores,
        selected_sectors,
    )

    # --------------------------------------------------------
    # 4. Selecionar Top 3 por setor
    # --------------------------------------------------------

    selected = select_top3_per_sector(
        scores
    )

    # --------------------------------------------------------
    # 5. Auditoria 4 × 3
    # --------------------------------------------------------

    audit_selection(
        selected
    )

    # --------------------------------------------------------
    # 6. Ordenação final
    # --------------------------------------------------------

    selected = selected.sort_values(
        [
            "YEAR",
            "SECTOR_RANK",
            "RANK_IN_SECTOR",
            "TICKER",
        ],
        ascending=[
            True,
            True,
            True,
            True,
        ],
    )

    output_columns = [
        "YEAR",
        "SECTOR_RANK",
        "MACRO_SECTOR",
        "RANK_IN_SECTOR",
        "TICKER",
        "DISCOUNT_52W",
        "DISCOUNT_SCORE",
        "FUND_SCORE",
        "FINAL_SCORE",
    ]

    return (
        selected[
            output_columns
        ]
        .reset_index(drop=True)
    )
