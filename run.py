"""
PORTIFOLIO-B3
Execução da metodologia congelada.

Arquitetura:
    4 setores x 3 ações = 12 ações

Regra setorial:
    TOP4_1Y

Regra das ações:
    80% Discount + 20% Fundamentals
"""

from pathlib import Path

import pandas as pd

from portfolio_engine import (
    N_SECTORS,
    STOCKS_PER_SECTOR,
    SECTOR_RULE,
    STOCK_RULE,
    build_final_scores,
    select_top3_per_sector,
    audit_selection,
)


# ============================================================
# CAMINHOS
# ============================================================

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

FUNDAMENTALS_FILE = DATA / "fundamental_factors.csv"
PRICE_FILE = DATA / "price_factors.csv"
SECTORS_FILE = DATA / "sector_selection.csv"

REPORT_FILE = REPORTS / "portfolio_report.csv"


# ============================================================
# CARREGAMENTO
# ============================================================

def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"\nArquivo obrigatório não encontrado:\n"
            f"  {path}\n\n"
            f"Input necessário: {name}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"{name} está vazio: {path}"
        )

    return df


def load_inputs():
    fundamentals = load_csv(
        FUNDAMENTALS_FILE,
        "fundamental_factors.csv",
    )

    prices = load_csv(
        PRICE_FILE,
        "price_factors.csv",
    )

    sectors = load_csv(
        SECTORS_FILE,
        "sector_selection.csv",
    )

    return fundamentals, prices, sectors


# ============================================================
# VALIDAÇÃO DA SELEÇÃO SETORIAL CONGELADA
# ============================================================

def validate_sector_selection(
    sectors: pd.DataFrame,
) -> pd.DataFrame:

    required = [
        "YEAR",
        "MACRO_SECTOR",
        "SECTOR_RANK",
    ]

    missing = [
        col
        for col in required
        if col not in sectors.columns
    ]

    if missing:
        raise ValueError(
            "sector_selection.csv possui colunas ausentes: "
            f"{missing}"
        )

    df = sectors.copy()

    df["YEAR"] = pd.to_numeric(
        df["YEAR"],
        errors="raise",
    ).astype(int)

    df["SECTOR_RANK"] = pd.to_numeric(
        df["SECTOR_RANK"],
        errors="raise",
    ).astype(int)

    # Não aceitamos duplicação YEAR × setor.
    duplicated = df.duplicated(
        ["YEAR", "MACRO_SECTOR"]
    )

    if duplicated.any():
        raise ValueError(
            "sector_selection.csv possui "
            "YEAR × MACRO_SECTOR duplicado."
        )

    # A metodologia congelada exige exatamente
    # quatro setores em cada ano.
    counts = (
        df.groupby("YEAR")
        ["MACRO_SECTOR"]
        .nunique()
    )

    invalid = counts[counts != N_SECTORS]

    if not invalid.empty:
        raise ValueError(
            "Seleção setorial inválida.\n"
            "Cada ano precisa possuir exatamente "
            f"{N_SECTORS} setores.\n"
            f"{invalid.to_dict()}"
        )

    # Os ranks precisam ser exatamente 1, 2, 3 e 4.
    expected_ranks = set(
        range(1, N_SECTORS + 1)
    )

    for year, group in df.groupby("YEAR"):

        actual_ranks = set(
            group["SECTOR_RANK"].tolist()
        )

        if actual_ranks != expected_ranks:
            raise ValueError(
                f"{year}: SECTOR_RANK inválido. "
                f"Encontrado {sorted(actual_ranks)}; "
                f"esperado {sorted(expected_ranks)}."
            )

    return df


# ============================================================
# FILTRAR SOMENTE OS TOP4_1Y CONGELADOS
# ============================================================

def apply_frozen_sector_selection(
    scores: pd.DataFrame,
    sectors: pd.DataFrame,
) -> pd.DataFrame:

    sector_keys = sectors[
        [
            "YEAR",
            "MACRO_SECTOR",
            "SECTOR_RANK",
        ]
    ].copy()

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
            "Nenhuma ação permaneceu após aplicar "
            "a seleção setorial congelada."
        )

    return filtered


# ============================================================
# EXECUÇÃO
# ============================================================

def main():

    print()
    print("=" * 72)
    print("PORTIFOLIO-B3 — REPRODUÇÃO DO ESTUDO")
    print("=" * 72)

    print()
    print("Metodologia congelada")
    print("-" * 72)
    print(
        f"Arquitetura     : "
        f"{N_SECTORS} setores x "
        f"{STOCKS_PER_SECTOR} ações"
    )
    print(
        f"Total carteira  : "
        f"{N_SECTORS * STOCKS_PER_SECTOR} ações"
    )
    print(
        f"Regra setorial  : {SECTOR_RULE}"
    )
    print(
        f"Regra das ações : {STOCK_RULE}"
    )

    # --------------------------------------------------------
    # 1. Carregar inputs
    # --------------------------------------------------------

    fundamentals, prices, sectors = load_inputs()

    print()
    print("Inputs")
    print("-" * 72)
    print(
        f"Fundamentos     : {len(fundamentals):,}"
    )
    print(
        f"Preço/fatores   : {len(prices):,}"
    )
    print(
        f"Setores TOP4_1Y : {len(sectors):,}"
    )

    # --------------------------------------------------------
    # 2. Validar TOP4_1Y congelado
    # --------------------------------------------------------

    sectors = validate_sector_selection(
        sectors
    )

    # --------------------------------------------------------
    # 3. Construir scores das ações
    #
    # IMPORTANTE:
    # Os percentis são calculados ANTES de restringirmos
    # aos quatro setores selecionados.
    #
    # Isso preserva o cálculo dentro de
    # YEAR × MACRO_SECTOR.
    # --------------------------------------------------------

    scores = build_final_scores(
        fundamentals,
        prices,
    )

    # --------------------------------------------------------
    # 4. Aplicar os quatro setores selecionados
    # --------------------------------------------------------

    scores = apply_frozen_sector_selection(
        scores,
        sectors,
    )

    # --------------------------------------------------------
    # 5. Selecionar Top 3 de cada setor
    # --------------------------------------------------------

    selected = select_top3_per_sector(
        scores
    )

    # --------------------------------------------------------
    # 6. Auditoria 4 × 3
    # --------------------------------------------------------

    audit_selection(selected)

    # --------------------------------------------------------
    # 7. Organizar resultado
    # --------------------------------------------------------

    selected = selected.sort_values(
        [
            "YEAR",
            "SECTOR_RANK",
            "FINAL_SCORE",
            "TICKER",
        ],
        ascending=[
            True,
            True,
            False,
            True,
        ],
    ).copy()

    selected["RANK_IN_SECTOR"] = (
        selected
        .groupby(
            [
                "YEAR",
                "MACRO_SECTOR",
            ]
        )
        .cumcount()
        + 1
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

    result = selected[
        output_columns
    ].reset_index(drop=True)

    # --------------------------------------------------------
    # 8. Mostrar resultado
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("CARTEIRAS HISTÓRICAS REPRODUZIDAS")
    print("=" * 72)

    for year, year_df in result.groupby(
        "YEAR",
        sort=True,
    ):

        print()
        print(f"ANO {year}")
        print("-" * 72)

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
                    f"   {int(row['RANK_IN_SECTOR'])}. "
                    f"{row['TICKER']:<8} "
                    f"FINAL={row['FINAL_SCORE']:.4f}  "
                    f"DISC={row['DISCOUNT_SCORE']:.4f}  "
                    f"FUND={row['FUND_SCORE']:.4f}"
                )

        print()
        print(
            f"Total: {len(year_df)} ações"
        )

    # --------------------------------------------------------
    # 9. Relatório CSV
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

    # --------------------------------------------------------
    # 10. Resultado final da auditoria
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("AUDITORIA")
    print("=" * 72)

    print("Arquitetura 4 x 3 .............. PASS")
    print("4 setores por ano .............. PASS")
    print("3 ações por setor .............. PASS")
    print("12 ações por carteira .......... PASS")
    print("Regra setorial TOP4_1Y ......... PASS")
    print("Regra 80% Discount + 20% Fund .. PASS")

    print()
    print(
        f"Relatório: {REPORT_FILE}"
    )

    print()
    print("=" * 72)
    print("STATUS: REPRODUÇÃO CONCLUÍDA")
    print("=" * 72)
    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
