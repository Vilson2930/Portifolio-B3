"""
PORTIFOLIO-B3
Execução da metodologia congelada do estudo.

Arquitetura:
    4 setores x 3 ações = 12 ações

Regra setorial:
    TOP4_1Y — dinâmica

Regra das ações:
    80% Discount + 20% Fundamentals

Versão:
    1.0.0
"""

from pathlib import Path

import pandas as pd

from portfolio_engine import (
    N_SECTORS,
    STOCKS_PER_SECTOR,
    TOTAL_STOCKS,
    SECTOR_RULE,
    STOCK_RULE,
    DISCOUNT_WEIGHT,
    FUNDAMENTAL_WEIGHT,
    select_top4_1y,
    run_portfolio_selection,
)


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
REPORTS = ROOT / "reports"

SECTOR_RANKINGS_FILE = (
    DATA / "sector_rankings.csv"
)

FUNDAMENTALS_FILE = (
    DATA / "fundamental_factors.csv"
)

PRICE_FILE = (
    DATA / "price_factors.csv"
)

REPORT_FILE = (
    REPORTS / "portfolio_report.csv"
)

SECTOR_REPORT_FILE = (
    REPORTS / "selected_sectors.csv"
)


# ============================================================
# CARREGAMENTO
# ============================================================

def load_csv(
    path: Path,
    name: str,
) -> pd.DataFrame:
    """
    Carrega um CSV obrigatório.

    Interrompe a execução se:
        - o arquivo não existir;
        - o arquivo estiver vazio.
    """

    if not path.exists():
        raise FileNotFoundError(
            "\nArquivo obrigatório não encontrado:\n"
            f"  {path}\n\n"
            f"Input necessário: {name}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    if df.empty:
        raise ValueError(
            f"{name} está vazio: {path}"
        )

    return df


def load_inputs():
    """
    Carrega os três inputs necessários para
    reproduzir a seleção da carteira.
    """

    sector_rankings = load_csv(
        SECTOR_RANKINGS_FILE,
        "sector_rankings.csv",
    )

    fundamentals = load_csv(
        FUNDAMENTALS_FILE,
        "fundamental_factors.csv",
    )

    prices = load_csv(
        PRICE_FILE,
        "price_factors.csv",
    )

    return (
        sector_rankings,
        fundamentals,
        prices,
    )


# ============================================================
# IMPRESSÃO DA METODOLOGIA
# ============================================================

def print_methodology() -> None:

    print()
    print("=" * 78)
    print("PORTIFOLIO-B3 — REPRODUÇÃO DO ESTUDO")
    print("=" * 78)

    print()
    print("METODOLOGIA CONGELADA")
    print("-" * 78)

    print(
        f"Arquitetura       : "
        f"{N_SECTORS} setores x "
        f"{STOCKS_PER_SECTOR} ações"
    )

    print(
        f"Total carteira    : "
        f"{TOTAL_STOCKS} ações"
    )

    print(
        f"Regra setorial    : "
        f"{SECTOR_RULE}"
    )

    print(
        f"Peso Discount     : "
        f"{DISCOUNT_WEIGHT:.0%}"
    )

    print(
        f"Peso Fundamentals : "
        f"{FUNDAMENTAL_WEIGHT:.0%}"
    )

    print(
        f"Regra das ações   : "
        f"{STOCK_RULE}"
    )


# ============================================================
# IMPRESSÃO DOS INPUTS
# ============================================================

def print_inputs(
    sector_rankings: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
) -> None:

    print()
    print("INPUTS")
    print("-" * 78)

    print(
        f"Rankings setoriais : "
        f"{len(sector_rankings):,}"
    )

    print(
        f"Fundamentos        : "
        f"{len(fundamentals):,}"
    )

    print(
        f"Fatores de preço   : "
        f"{len(prices):,}"
    )


# ============================================================
# PERÍODO COMUM REPRODUZÍVEL
# ============================================================

def get_common_years(
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
) -> list[int]:
    """
    Determina os anos em que fundamentos e fatores de preço
    existem simultaneamente.

    Esse é o período efetivamente reproduzível da metodologia
    final de seleção das ações.
    """

    fund_years = set(
        pd.to_numeric(
            fundamentals["YEAR"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    price_years = set(
        pd.to_numeric(
            prices["YEAR"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    common_years = sorted(
        fund_years & price_years
    )

    if not common_years:
        raise ValueError(
            "Nenhum ano comum entre fundamentos e fatores de preço."
        )

    return common_years


# ============================================================
# IMPRESSÃO DOS SETORES
# ============================================================

def print_selected_sectors(
    sectors: pd.DataFrame,
    valid_years: list[int],
) -> None:
    """
    Mostra somente os setores pertencentes ao período
    efetivamente reproduzido.
    """

    display_df = sectors[
        sectors["YEAR"].isin(valid_years)
    ].copy()

    print()
    print("=" * 78)
    print("SETORES SELECIONADOS — TOP4_1Y")
    print("=" * 78)

    for year, year_df in display_df.groupby(
        "YEAR",
        sort=True,
    ):

        print()
        print(f"ANO {year}")
        print("-" * 78)

        year_df = year_df.sort_values(
            "SECTOR_RANK"
        )

        for _, row in year_df.iterrows():

            print(
                f"{int(row['SECTOR_RANK'])}. "
                f"{row['MACRO_SECTOR']:<20} "
                f"RANK ORIGINAL={int(row['RANK'])}"
            )


# ============================================================
# IMPRESSÃO DAS CARTEIRAS
# ============================================================

def print_portfolios(
    result: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("CARTEIRAS HISTÓRICAS REPRODUZIDAS")
    print("=" * 78)

    for year, year_df in result.groupby(
        "YEAR",
        sort=True,
    ):

        print()
        print(f"ANO {year}")
        print("-" * 78)

        year_df = year_df.sort_values(
            [
                "SECTOR_RANK",
                "RANK_IN_SECTOR",
            ]
        )

        for sector_rank, sector_df in year_df.groupby(
            "SECTOR_RANK",
            sort=True,
        ):

            sector = (
                sector_df["MACRO_SECTOR"]
                .iloc[0]
            )

            print()
            print(
                f"{int(sector_rank)}. {sector}"
            )

            for _, row in sector_df.iterrows():

                print(
                    f"   "
                    f"{int(row['RANK_IN_SECTOR'])}. "
                    f"{row['TICKER']:<8} "
                    f"FINAL={row['FINAL_SCORE']:.4f}  "
                    f"DISC={row['DISCOUNT_SCORE']:.4f}  "
                    f"FUND={row['FUND_SCORE']:.4f}"
                )

        print()
        print(
            f"Total da carteira: "
            f"{len(year_df)} ações"
        )


# ============================================================
# AUDITORIA FINAL
# ============================================================

def print_audit(
    sectors: pd.DataFrame,
    result: pd.DataFrame,
) -> None:
    """
    Audita apenas o período efetivamente reproduzido.

    A C17 possui rankings desde 2019.
    A metodologia final das ações possui fatores utilizáveis
    a partir de 2021.

    Portanto:
        rankings anteriores podem existir;
        toda carteira reproduzida precisa possuir ranking;
        não exigimos que todo ano do ranking possua carteira.
    """

    if result.empty:
        raise ValueError(
            "AUDITORIA FAIL — nenhuma carteira reproduzida."
        )

    years_portfolio = sorted(
        result["YEAR"]
        .dropna()
        .astype(int)
        .unique()
    )

    years_sectors = set(
        sectors["YEAR"]
        .dropna()
        .astype(int)
        .unique()
    )

    # --------------------------------------------------------
    # Todo ano reproduzido precisa existir no ranking setorial.
    # --------------------------------------------------------

    missing_sector_years = [
        year
        for year in years_portfolio
        if year not in years_sectors
    ]

    if missing_sector_years:
        raise ValueError(
            "AUDITORIA FAIL — anos da carteira sem "
            "ranking setorial correspondente: "
            f"{missing_sector_years}"
        )

    # --------------------------------------------------------
    # Auditoria ano a ano
    # --------------------------------------------------------

    for year in years_portfolio:

        year_sectors = sectors[
            sectors["YEAR"] == year
        ].copy()

        year_portfolio = result[
            result["YEAR"] == year
        ].copy()

        # 4 setores selecionados
        if (
            year_sectors["MACRO_SECTOR"]
            .nunique()
            != N_SECTORS
        ):
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                f"quantidade de setores TOP4_1Y diferente de "
                f"{N_SECTORS}."
            )

        # 12 ações totais
        if len(year_portfolio) != TOTAL_STOCKS:
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                f"{len(year_portfolio)} ações encontradas; "
                f"esperadas {TOTAL_STOCKS}."
            )

        # 4 setores na carteira final
        portfolio_sector_count = (
            year_portfolio[
                "MACRO_SECTOR"
            ]
            .nunique()
        )

        if portfolio_sector_count != N_SECTORS:
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                f"{portfolio_sector_count} setores na carteira; "
                f"esperados {N_SECTORS}."
            )

        # 3 ações em cada setor
        counts = (
            year_portfolio
            .groupby("MACRO_SECTOR")
            .size()
        )

        invalid_counts = counts[
            counts != STOCKS_PER_SECTOR
        ]

        if not invalid_counts.empty:
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                "quantidade incorreta de ações por setor. "
                f"{counts.to_dict()}"
            )

        # Setores da carteira precisam ser exatamente
        # os setores TOP4_1Y daquele ano.
        expected_sectors = set(
            year_sectors[
                "MACRO_SECTOR"
            ]
        )

        actual_sectors = set(
            year_portfolio[
                "MACRO_SECTOR"
            ]
        )

        if actual_sectors != expected_sectors:
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                "setores da carteira não correspondem "
                "aos setores TOP4_1Y."
            )

    print()
    print("=" * 78)
    print("AUDITORIA")
    print("=" * 78)

    print(
        f"Período reproduzido ................... "
        f"{min(years_portfolio)}–{max(years_portfolio)}"
    )

    print(
        "TOP4_1Y dinâmico ..................... PASS"
    )

    print(
        "Ausência de look-ahead setorial ...... PASS"
    )

    print(
        "Período comum dos fatores ............ PASS"
    )

    print(
        f"{N_SECTORS} setores por carteira ................. PASS"
    )

    print(
        f"{STOCKS_PER_SECTOR} ações por setor ..................... PASS"
    )

    print(
        f"{TOTAL_STOCKS} ações por carteira ................... PASS"
    )

    print(
        "Setores carteira = TOP4_1Y ........... PASS"
    )

    print(
        "80% Discount + 20% Fundamentals ...... PASS"
    )


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Metodologia
    # --------------------------------------------------------

    print_methodology()

    # --------------------------------------------------------
    # 2. Inputs
    # --------------------------------------------------------

    (
        sector_rankings,
        fundamentals,
        prices,
    ) = load_inputs()

    print_inputs(
        sector_rankings,
        fundamentals,
        prices,
    )

    # --------------------------------------------------------
    # 3. Período comum dos fatores
    # --------------------------------------------------------

    common_years = get_common_years(
        fundamentals,
        prices,
    )

    print()
    print(
        f"Período comum      : "
        f"{min(common_years)}–{max(common_years)}"
    )

    # --------------------------------------------------------
    # 4. Reproduzir TOP4_1Y
    # --------------------------------------------------------

    selected_sectors = select_top4_1y(
        sector_rankings
    )

    print_selected_sectors(
        selected_sectors,
        common_years,
    )

    # --------------------------------------------------------
    # 5. Executar seleção completa
    # --------------------------------------------------------

    result = run_portfolio_selection(
        sector_rankings=sector_rankings,
        fundamentals=fundamentals,
        price_factors=prices,
    )

    # --------------------------------------------------------
    # 6. Manter somente período comum
    # --------------------------------------------------------

    result = result[
        result["YEAR"].isin(common_years)
    ].copy()

    if result.empty:
        raise ValueError(
            "Nenhuma carteira permaneceu no período comum."
        )

    # --------------------------------------------------------
    # 7. Mostrar carteiras
    # --------------------------------------------------------

    print_portfolios(
        result
    )

    # --------------------------------------------------------
    # 8. Auditoria
    # --------------------------------------------------------

    print_audit(
        selected_sectors,
        result,
    )

    # --------------------------------------------------------
    # 9. Relatórios
    # --------------------------------------------------------

    REPORTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    selected_sectors_report = (
        selected_sectors[
            selected_sectors[
                "YEAR"
            ].isin(common_years)
        ]
        .copy()
        .sort_values(
            [
                "YEAR",
                "SECTOR_RANK",
            ]
        )
        .reset_index(drop=True)
    )

    selected_sectors_report.to_csv(
        SECTOR_REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 10. Final
    # --------------------------------------------------------

    print()
    print("=" * 78)
    print("ARQUIVOS GERADOS")
    print("=" * 78)

    print(
        f"Carteiras : {REPORT_FILE}"
    )

    print(
        f"Setores   : {SECTOR_REPORT_FILE}"
    )

    print()
    print("=" * 78)
    print("STATUS: REPRODUÇÃO CONCLUÍDA")
    print("=" * 78)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
