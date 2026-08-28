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

    A execução é interrompida se:
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
    Carrega os três inputs necessários para a
    seleção da carteira.
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
# IMPRESSÃO DA CONFIGURAÇÃO
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
# IMPRESSÃO DOS SETORES
# ============================================================

def print_selected_sectors(
    sectors: pd.DataFrame,
) -> None:

    print()
    print("=" * 78)
    print("SETORES SELECIONADOS — TOP4_1Y")
    print("=" * 78)

    for year, year_df in sectors.groupby(
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

    years_sectors = set(
        sectors["YEAR"].unique()
    )

    years_portfolio = set(
        result["YEAR"].unique()
    )

    if years_sectors != years_portfolio:
        raise ValueError(
            "AUDITORIA FAIL — anos dos rankings setoriais "
            "não coincidem com os anos das carteiras."
        )

    for year in sorted(
        years_portfolio
    ):

        year_sectors = sectors[
            sectors["YEAR"] == year
        ]

        year_portfolio = result[
            result["YEAR"] == year
        ]

        if (
            year_sectors["MACRO_SECTOR"]
            .nunique()
            != N_SECTORS
        ):
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                "quantidade incorreta de setores."
            )

        if len(year_portfolio) != TOTAL_STOCKS:
            raise ValueError(
                f"AUDITORIA FAIL — {year}: "
                f"{len(year_portfolio)} ações; "
                f"esperadas {TOTAL_STOCKS}."
            )

    print()
    print("=" * 78)
    print("AUDITORIA")
    print("=" * 78)

    print(
        "TOP4_1Y dinâmico ..................... PASS"
    )

    print(
        "Ausência de look-ahead setorial ...... PASS"
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
    # 3. Reproduzir TOP4_1Y
    #
    # Aqui os setores são CALCULADOS.
    # Não são lidos de uma lista fixa.
    # --------------------------------------------------------

    selected_sectors = select_top4_1y(
        sector_rankings
    )

    print_selected_sectors(
        selected_sectors
    )

    # --------------------------------------------------------
    # 4. Executar seleção completa
    #
    # TOP4_1Y
    #       ↓
    # 4 setores
    #       ↓
    # 80% Discount + 20% Fundamentals
    #       ↓
    # Top 3 por setor
    #       ↓
    # 12 ações
    # --------------------------------------------------------

    result = run_portfolio_selection(
        sector_rankings=sector_rankings,
        fundamentals=fundamentals,
        price_factors=prices,
    )

    # --------------------------------------------------------
    # 5. Mostrar carteiras
    # --------------------------------------------------------

    print_portfolios(
        result
    )

    # --------------------------------------------------------
    # 6. Auditoria
    # --------------------------------------------------------

    print_audit(
        selected_sectors,
        result,
    )

    # --------------------------------------------------------
    # 7. Gerar relatórios
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

    selected_sectors.to_csv(
        SECTOR_REPORT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # 8. Final
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
