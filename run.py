from pathlib import Path
import numpy as np
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

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

SECTOR_RANKINGS_FILE = DATA / "sector_rankings.csv"
FUNDAMENTALS_FILE = DATA / "fundamental_factors.csv"
PRICE_FILE = DATA / "price_factors.csv"
RETURNS_FILE = DATA / "returns.csv"

REPORT_FILE = REPORTS / "portfolio_report.csv"
SECTOR_REPORT_FILE = REPORTS / "selected_sectors.csv"
ANNUAL_PERFORMANCE_FILE = REPORTS / "annual_performance.csv"
PERFORMANCE_SUMMARY_FILE = REPORTS / "performance_summary.csv"

EXPECTED_ANNUAL = {
    2021: (-0.0997,  0.0662, -0.1659),
    2022: (-0.0795, -0.0919,  0.0124),
    2023: ( 0.3236,  0.2328,  0.0908),
    2024: ( 0.9583,  0.0797,  0.8786),
    2025: ( 0.1934,  0.1234,  0.0700),
}

EXPECTED_PERFORMANCE = {
    "YEARS": 5,
    "CAGR": 0.2072,
    "MEAN_RETURN": 0.2592,
    "SHARPE": 0.602,
    "MAXDD": -0.0795,
    "WORST_YEAR": -0.0997,
    "MEAN_ALPHA": 0.1772,
    "ALPHA_WIN_RATE": 0.80,
}


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo obrigatório não encontrado: {path} ({name})")
    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        raise ValueError(f"{name} está vazio: {path}")
    return df


def normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["YEAR"] = pd.to_numeric(out["YEAR"], errors="coerce")
    out = out[out["YEAR"].notna()].copy()
    out["YEAR"] = out["YEAR"].astype(int)
    out["TICKER"] = out["TICKER"].astype(str).str.strip().str.upper()
    out["MACRO_SECTOR"] = out["MACRO_SECTOR"].astype(str).str.strip().str.upper()
    return out


def get_common_years(fundamentals: pd.DataFrame, prices: pd.DataFrame):
    fy = set(pd.to_numeric(fundamentals["YEAR"], errors="coerce").dropna().astype(int))
    py = set(pd.to_numeric(prices["YEAR"], errors="coerce").dropna().astype(int))
    years = sorted(fy & py)
    if not years:
        raise ValueError("Nenhum período comum entre fundamentos e fatores de preço.")
    return years


def print_methodology():
    print("\n" + "=" * 78)
    print("PORTIFOLIO-B3 — REPRODUÇÃO DO ESTUDO")
    print("=" * 78)
    print("\nMETODOLOGIA CONGELADA")
    print("-" * 78)
    print(f"Arquitetura       : {N_SECTORS} setores x {STOCKS_PER_SECTOR} ações")
    print(f"Total carteira    : {TOTAL_STOCKS} ações")
    print(f"Regra setorial    : {SECTOR_RULE}")
    print(f"Peso Discount     : {DISCOUNT_WEIGHT:.0%}")
    print(f"Peso Fundamentals : {FUNDAMENTAL_WEIGHT:.0%}")
    print(f"Regra das ações   : {STOCK_RULE}")


def print_selected_sectors(sectors: pd.DataFrame, years):
    display = sectors[sectors["YEAR"].isin(years)].copy()
    print("\n" + "=" * 78)
    print("SETORES SELECIONADOS — TOP4_1Y")
    print("=" * 78)
    for year, g in display.groupby("YEAR", sort=True):
        print(f"\nANO {year}\n" + "-" * 78)
        for row in g.sort_values("SECTOR_RANK").itertuples(index=False):
            print(f"{int(row.SECTOR_RANK)}. {row.MACRO_SECTOR:<20} RANK ORIGINAL={int(row.RANK)}")


def print_portfolios(result: pd.DataFrame):
    print("\n" + "=" * 78)
    print("CARTEIRAS HISTÓRICAS REPRODUZIDAS")
    print("=" * 78)
    for year, g in result.groupby("YEAR", sort=True):
        print(f"\nANO {year}\n" + "-" * 78)
        g = g.sort_values(["SECTOR_RANK", "RANK_IN_SECTOR"])
        for sector_rank, sg in g.groupby("SECTOR_RANK", sort=True):
            print(f"\n{int(sector_rank)}. {sg['MACRO_SECTOR'].iloc[0]}")
            for row in sg.itertuples(index=False):
                print(
                    f"   {int(row.RANK_IN_SECTOR)}. {row.TICKER:<8} "
                    f"FINAL={row.FINAL_SCORE:.4f}  DISC={row.DISCOUNT_SCORE:.4f}  FUND={row.FUND_SCORE:.4f}"
                )
        print(f"\nTotal da carteira: {len(g)} ações")


def audit_selection(sectors: pd.DataFrame, result: pd.DataFrame):
    if result.empty:
        raise ValueError("AUDITORIA FAIL — nenhuma carteira reproduzida.")

    years = sorted(result["YEAR"].astype(int).unique())
    sector_years = set(sectors["YEAR"].astype(int).unique())

    missing = [y for y in years if y not in sector_years]
    if missing:
        raise ValueError(f"AUDITORIA FAIL — anos sem ranking setorial: {missing}")

    for year in years:
        ys = sectors[sectors["YEAR"] == year]
        yp = result[result["YEAR"] == year]
        if ys["MACRO_SECTOR"].nunique() != N_SECTORS:
            raise ValueError(f"AUDITORIA FAIL — {year}: TOP4 incorreto.")
        if len(yp) != TOTAL_STOCKS:
            raise ValueError(f"AUDITORIA FAIL — {year}: {len(yp)} ações; esperadas {TOTAL_STOCKS}.")
        counts = yp.groupby("MACRO_SECTOR").size()
        if len(counts) != N_SECTORS or (counts != STOCKS_PER_SECTOR).any():
            raise ValueError(f"AUDITORIA FAIL — {year}: composição 4x3 inválida {counts.to_dict()}.")
        if set(yp["MACRO_SECTOR"]) != set(ys["MACRO_SECTOR"]):
            raise ValueError(f"AUDITORIA FAIL — {year}: setores da carteira != TOP4_1Y.")

    print("\n" + "=" * 78)
    print("AUDITORIA")
    print("=" * 78)
    print(f"Período reproduzido ................... {min(years)}–{max(years)}")
    print("TOP4_1Y dinâmico ..................... PASS")
    print("Ausência de look-ahead setorial ...... PASS")
    print("Período comum dos fatores ............ PASS")
    print(f"{N_SECTORS} setores por carteira ................. PASS")
    print(f"{STOCKS_PER_SECTOR} ações por setor ..................... PASS")
    print(f"{TOTAL_STOCKS} ações por carteira ................... PASS")
    print("Setores carteira = TOP4_1Y ........... PASS")
    print("80% Discount + 20% Fundamentals ...... PASS")


def build_historical_performance(result, fundamentals, prices, returns):
    keys = ["YEAR", "TICKER", "MACRO_SECTOR"]
    fund = normalize_keys(fundamentals)
    price = normalize_keys(prices)
    ret = normalize_keys(returns)
    selected = normalize_keys(result)

    if "RET_ANNUAL_VALID" not in ret.columns:
        raise ValueError("returns.csv não possui RET_ANNUAL_VALID.")
    ret["RET_ANNUAL_VALID"] = pd.to_numeric(ret["RET_ANNUAL_VALID"], errors="coerce")

    if ret.duplicated(keys).any():
        raise ValueError("PERFORMANCE FAIL — duplicidades YEAR×TICKER×MACRO_SECTOR em returns.csv.")

    # C31: benchmark é a média YEAR×setor do universo elegível
    # resultante da interseção fundamentos × fatores de preço.
    eligible_base = (
        fund[keys].drop_duplicates(keys)
        .merge(price[keys].drop_duplicates(keys), on=keys, how="inner")
        .merge(ret[keys + ["RET_ANNUAL_VALID"]], on=keys, how="left")
    )

    benchmark = (
        eligible_base.groupby(["YEAR", "MACRO_SECTOR"], as_index=False)
        .agg(BENCHMARK_RETURN=("RET_ANNUAL_VALID", "mean"))
    )

    selected_returns = selected.merge(
        ret[keys + ["RET_ANNUAL_VALID"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if selected_returns["RET_ANNUAL_VALID"].isna().any():
        missing = selected_returns.loc[
            selected_returns["RET_ANNUAL_VALID"].isna(), keys
        ].to_dict(orient="records")
        raise ValueError(f"PERFORMANCE FAIL — ações selecionadas sem retorno: {missing}")

    # C31: retorno do setor = média das 3 ações.
    sector_perf = (
        selected_returns.groupby(["YEAR", "MACRO_SECTOR"], as_index=False)
        .agg(
            N_STOCKS=("TICKER", "nunique"),
            N_VALID_RETURNS=("RET_ANNUAL_VALID", lambda x: x.notna().sum()),
            RETURN=("RET_ANNUAL_VALID", "mean"),
        )
    )

    invalid = sector_perf[
        (sector_perf["N_STOCKS"] != STOCKS_PER_SECTOR)
        | (sector_perf["N_VALID_RETURNS"] != STOCKS_PER_SECTOR)
    ]
    if not invalid.empty:
        raise ValueError("PERFORMANCE FAIL — setor sem exatamente 3 retornos válidos.\n" + invalid.to_string(index=False))

    sector_perf = sector_perf.merge(
        benchmark, on=["YEAR", "MACRO_SECTOR"], how="left", validate="one_to_one"
    )
    if sector_perf["BENCHMARK_RETURN"].isna().any():
        raise ValueError("PERFORMANCE FAIL — benchmark setorial ausente.")

    sector_perf["ALPHA"] = sector_perf["RETURN"] - sector_perf["BENCHMARK_RETURN"]

    # C31: carteira anual = média dos 4 setores.
    annual = (
        sector_perf.groupby("YEAR", as_index=False)
        .agg(
            N_SECTORS=("MACRO_SECTOR", "nunique"),
            RETURN=("RETURN", "mean"),
            BENCHMARK_RETURN=("BENCHMARK_RETURN", "mean"),
        )
        .sort_values("YEAR")
        .reset_index(drop=True)
    )
    if (annual["N_SECTORS"] != N_SECTORS).any():
        raise ValueError("PERFORMANCE FAIL — ano sem exatamente 4 setores válidos.")
    annual["ALPHA"] = annual["RETURN"] - annual["BENCHMARK_RETURN"]
    return sector_perf, annual


def performance_metrics(annual: pd.DataFrame):
    annual = annual.sort_values("YEAR")
    x = annual["RETURN"].dropna().astype(float)
    alpha = annual["ALPHA"].dropna().astype(float)
    if x.empty:
        raise ValueError("PERFORMANCE FAIL — nenhum retorno anual válido.")

    cumulative = (1 + x).prod()
    cagr = cumulative ** (1 / len(x)) - 1
    std = x.std(ddof=1)
    sharpe = x.mean() / std if len(x) > 1 and std > 0 else np.nan
    wealth = (1 + x).cumprod()
    maxdd = (wealth / wealth.cummax() - 1).min()

    return {
        "YEARS": len(x),
        "CAGR": cagr,
        "MEAN_RETURN": x.mean(),
        "SHARPE": sharpe,
        "MAXDD": maxdd,
        "WORST_YEAR": x.min(),
        "MEAN_ALPHA": alpha.mean(),
        "ALPHA_WIN_RATE": (alpha > 0).mean(),
    }


def assert_close(actual, expected, tolerance, name):
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise AssertionError(
            f"C32 FAIL — {name}: esperado≈{expected:.6f}, obtido={actual:.6f}, tolerância={tolerance:.6f}"
        )


def validate_c32(annual: pd.DataFrame, metrics: dict):
    if set(annual["YEAR"].astype(int)) != set(EXPECTED_ANNUAL):
        raise AssertionError(
            f"C32 FAIL — anos esperados={sorted(EXPECTED_ANNUAL)}, obtidos={sorted(annual['YEAR'].astype(int).tolist())}"
        )

    # 0,10 p.p. de tolerância porque os números congelados foram exibidos arredondados.
    for row in annual.itertuples(index=False):
        er, eb, ea = EXPECTED_ANNUAL[int(row.YEAR)]
        assert_close(float(row.RETURN), er, 0.0010, f"RETURN_{int(row.YEAR)}")
        assert_close(float(row.BENCHMARK_RETURN), eb, 0.0010, f"BENCHMARK_{int(row.YEAR)}")
        assert_close(float(row.ALPHA), ea, 0.0010, f"ALPHA_{int(row.YEAR)}")

    if int(metrics["YEARS"]) != EXPECTED_PERFORMANCE["YEARS"]:
        raise AssertionError(
            f"C32 FAIL — YEARS: esperado={EXPECTED_PERFORMANCE['YEARS']}, obtido={metrics['YEARS']}"
        )

    assert_close(metrics["CAGR"], EXPECTED_PERFORMANCE["CAGR"], 0.0010, "CAGR")
    assert_close(metrics["MEAN_RETURN"], EXPECTED_PERFORMANCE["MEAN_RETURN"], 0.0010, "MEAN_RETURN")
    assert_close(metrics["SHARPE"], EXPECTED_PERFORMANCE["SHARPE"], 0.010, "SHARPE")
    assert_close(metrics["MAXDD"], EXPECTED_PERFORMANCE["MAXDD"], 0.0010, "MAXDD")
    assert_close(metrics["WORST_YEAR"], EXPECTED_PERFORMANCE["WORST_YEAR"], 0.0010, "WORST_YEAR")
    assert_close(metrics["MEAN_ALPHA"], EXPECTED_PERFORMANCE["MEAN_ALPHA"], 0.0010, "MEAN_ALPHA")
    assert_close(metrics["ALPHA_WIN_RATE"], EXPECTED_PERFORMANCE["ALPHA_WIN_RATE"], 0.0010, "ALPHA_WIN_RATE")


def print_performance(annual: pd.DataFrame, metrics: dict):
    print("\n" + "=" * 78)
    print("VALIDAÇÃO HISTÓRICA — C32")
    print("=" * 78)
    print("\nANO   CARTEIRA    BENCHMARK      ALPHA")
    print("-" * 48)
    for row in annual.itertuples(index=False):
        print(
            f"{int(row.YEAR)}   {row.RETURN:>9.2%}   "
            f"{row.BENCHMARK_RETURN:>9.2%}   {row.ALPHA:>9.2%}"
        )

    print("\n" + "-" * 78)
    print(f"CAGR           : {metrics['CAGR']:.2%}")
    print(f"Retorno médio  : {metrics['MEAN_RETURN']:.2%}")
    print(f"Sharpe         : {metrics['SHARPE']:.3f}")
    print(f"Max Drawdown   : {metrics['MAXDD']:.2%}")
    print(f"Pior ano       : {metrics['WORST_YEAR']:.2%}")
    print(f"Alpha médio    : {metrics['MEAN_ALPHA']:.2%}")
    print(f"Alpha win rate : {metrics['ALPHA_WIN_RATE']:.2%}")
    print("\nC32 — RETORNOS ANUAIS ................. PASS")
    print("C32 — CAGR ............................. PASS")
    print("C32 — SHARPE ........................... PASS")
    print("C32 — MAX DRAWDOWN ..................... PASS")
    print("C32 — ALPHA ............................ PASS")
    print("\nSTATUS: VALIDAÇÃO HISTÓRICA C32 = PASS")


def main():
    print_methodology()

    sector_rankings = load_csv(SECTOR_RANKINGS_FILE, "sector_rankings.csv")
    fundamentals = load_csv(FUNDAMENTALS_FILE, "fundamental_factors.csv")
    prices = load_csv(PRICE_FILE, "price_factors.csv")
    returns = load_csv(RETURNS_FILE, "returns.csv")

    print("\nINPUTS\n" + "-" * 78)
    print(f"Rankings setoriais : {len(sector_rankings):,}")
    print(f"Fundamentos        : {len(fundamentals):,}")
    print(f"Fatores de preço   : {len(prices):,}")
    print(f"Retornos anuais    : {len(returns):,}")

    common_years = get_common_years(fundamentals, prices)
    print(f"\nPeríodo comum      : {min(common_years)}–{max(common_years)}")

    selected_sectors = select_top4_1y(sector_rankings)
    print_selected_sectors(selected_sectors, common_years)

    result = run_portfolio_selection(
        sector_rankings=sector_rankings,
        fundamentals=fundamentals,
        price_factors=prices,
    )
    result = result[result["YEAR"].isin(common_years)].copy()
    if result.empty:
        raise ValueError("Nenhuma carteira permaneceu no período comum.")

    print_portfolios(result)
    audit_selection(selected_sectors, result)

    _, annual = build_historical_performance(result, fundamentals, prices, returns)
    metrics = performance_metrics(annual)
    validate_c32(annual, metrics)
    print_performance(annual, metrics)

    REPORTS.mkdir(parents=True, exist_ok=True)

    result.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
    selected_sectors[
        selected_sectors["YEAR"].isin(common_years)
    ].sort_values(["YEAR", "SECTOR_RANK"]).to_csv(
        SECTOR_REPORT_FILE, index=False, encoding="utf-8-sig"
    )
    annual.to_csv(ANNUAL_PERFORMANCE_FILE, index=False, encoding="utf-8-sig")
    pd.DataFrame([{"RULE": STOCK_RULE, **metrics}]).to_csv(
        PERFORMANCE_SUMMARY_FILE, index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 78)
    print("ARQUIVOS GERADOS")
    print("=" * 78)
    print(f"Carteiras   : {REPORT_FILE}")
    print(f"Setores     : {SECTOR_REPORT_FILE}")
    print(f"Performance : {ANNUAL_PERFORMANCE_FILE}")
    print(f"Resumo      : {PERFORMANCE_SUMMARY_FILE}")
    print("\n" + "=" * 78)
    print("STATUS: REPRODUÇÃO DO ESTUDO CONCLUÍDA E VALIDADA")
    print("=" * 78)


if __name__ == "__main__":
    main()
