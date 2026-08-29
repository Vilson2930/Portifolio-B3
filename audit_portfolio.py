from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA_LIVE = ROOT / "data_live"

PORTFOLIO_FILE = (
    DATA_LIVE
    / "portfolio_current.csv"
)

PRICE_FACTORS_FILE = (
    DATA_LIVE
    / "price_factors_current.csv"
)

OUT_FILE = (
    DATA_LIVE
    / "portfolio_extreme_audit.csv"
)

EXTREME_DISCOUNT = 0.80
MIN_OBS_52W = 60
STALE_BUSINESS_DAYS = 7

# Diferença interna praticamente zero.
ENGINE_TOLERANCE = 1e-10

# Diferença contra nova coleta independente do Yahoo.
EXTERNAL_DISCOUNT_TOLERANCE = 0.02


# =============================================================================
# INÍCIO
# =============================================================================

print()
print("=" * 78)
print(
    "PORTIFOLIO-B3 — "
    "AUDITORIA FINAL DE PREÇOS"
)
print("=" * 78)


# =============================================================================
# 1. CARREGAMENTO DO PORTFÓLIO
# =============================================================================

if not PORTFOLIO_FILE.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: "
        f"{PORTFOLIO_FILE}"
    )


portfolio = pd.read_csv(
    PORTFOLIO_FILE,
    low_memory=False,
)


required_portfolio = {
    "TICKER",
    "MACRO_SECTOR",
    "DISCOUNT_52W",
    "FINAL_SCORE",
}


missing = (
    required_portfolio
    -
    set(portfolio.columns)
)


if missing:
    raise RuntimeError(
        "Colunas ausentes em "
        "portfolio_current.csv: "
        f"{sorted(missing)}"
    )


portfolio["TICKER"] = (
    portfolio["TICKER"]
    .astype(str)
    .str.upper()
    .str.strip()
)


portfolio["MACRO_SECTOR"] = (
    portfolio["MACRO_SECTOR"]
    .astype(str)
    .str.upper()
    .str.strip()
)


portfolio["DISCOUNT_52W"] = (
    pd.to_numeric(
        portfolio["DISCOUNT_52W"],
        errors="coerce",
    )
)


portfolio["FINAL_SCORE"] = (
    pd.to_numeric(
        portfolio["FINAL_SCORE"],
        errors="coerce",
    )
)


duplicates_portfolio = (
    portfolio
    .duplicated("TICKER")
    .sum()
)


if duplicates_portfolio:
    raise RuntimeError(
        "portfolio_current.csv possui "
        f"{duplicates_portfolio} "
        "tickers duplicados."
    )


# =============================================================================
# 2. CARREGAMENTO DA FONTE INTERNA DO MOTOR
# =============================================================================

if not PRICE_FACTORS_FILE.exists():
    raise FileNotFoundError(
        f"Arquivo não encontrado: "
        f"{PRICE_FACTORS_FILE}"
    )


price_factors = pd.read_csv(
    PRICE_FACTORS_FILE,
    low_memory=False,
)


required_price = {
    "TICKER",
    "DISCOUNT_52W",
    "PRICE_QUALITY_STATUS",
}


missing_price = (
    required_price
    -
    set(price_factors.columns)
)


if missing_price:
    raise RuntimeError(
        "Colunas ausentes em "
        "price_factors_current.csv: "
        f"{sorted(missing_price)}"
    )


price_factors["TICKER"] = (
    price_factors["TICKER"]
    .astype(str)
    .str.upper()
    .str.strip()
)


price_factors["DISCOUNT_52W"] = (
    pd.to_numeric(
        price_factors["DISCOUNT_52W"],
        errors="coerce",
    )
)


price_factors["PRICE_QUALITY_STATUS"] = (
    price_factors[
        "PRICE_QUALITY_STATUS"
    ]
    .astype(str)
    .str.upper()
    .str.strip()
)


duplicates_price = (
    price_factors
    .duplicated("TICKER")
    .sum()
)


if duplicates_price:
    raise RuntimeError(
        "price_factors_current.csv possui "
        f"{duplicates_price} "
        "tickers duplicados."
    )


engine_map = (
    price_factors
    .set_index("TICKER")
)


# =============================================================================
# 3. AUDITORIA
# =============================================================================

rows = []


for ticker in portfolio["TICKER"]:

    print(
        f"Auditando {ticker}..."
    )

    portfolio_row = (
        portfolio.loc[
            portfolio["TICKER"] == ticker
        ]
        .iloc[0]
    )


    macro_sector = (
        portfolio_row[
            "MACRO_SECTOR"
        ]
    )


    portfolio_discount = (
        float(
            portfolio_row[
                "DISCOUNT_52W"
            ]
        )
        if pd.notna(
            portfolio_row[
                "DISCOUNT_52W"
            ]
        )
        else np.nan
    )


    # =========================================================================
    # 3.1 INTEGRIDADE INTERNA
    # =========================================================================

    internal_reasons = []


    if ticker not in engine_map.index:

        engine_discount = np.nan
        engine_quality_status = "MISSING"

        internal_reasons.append(
            "TICKER_AUSENTE_PRICE_FACTORS"
        )

    else:

        engine_row = (
            engine_map.loc[ticker]
        )


        engine_discount = (
            float(
                engine_row[
                    "DISCOUNT_52W"
                ]
            )
            if pd.notna(
                engine_row[
                    "DISCOUNT_52W"
                ]
            )
            else np.nan
        )


        engine_quality_status = (
            str(
                engine_row[
                    "PRICE_QUALITY_STATUS"
                ]
            )
            .upper()
            .strip()
        )


    if (
        pd.notna(portfolio_discount)
        and pd.notna(engine_discount)
    ):

        internal_diff = abs(
            portfolio_discount
            -
            engine_discount
        )

    else:

        internal_diff = np.nan


    if (
        pd.notna(internal_diff)
        and
        internal_diff
        >
        ENGINE_TOLERANCE
    ):

        internal_reasons.append(
            "DIVERGENCIA_INTERNA_ENGINE"
        )


    if pd.isna(portfolio_discount):

        internal_reasons.append(
            "DISCOUNT_PORTFOLIO_INVALIDO"
        )


    if pd.isna(engine_discount):

        internal_reasons.append(
            "DISCOUNT_ENGINE_INVALIDO"
        )


    if (
        ticker in engine_map.index
        and
        engine_quality_status
        !=
        "PASS"
    ):

        internal_reasons.append(
            "PRICE_QUALITY_NAO_PASS"
        )


    engine_status = (
        "FAIL"
        if internal_reasons
        else
        "PASS"
    )


    # =========================================================================
    # 3.2 DIAGNÓSTICO EXTERNO — NOVA COLETA YAHOO
    # =========================================================================

    external_reasons = []

    last_date = None

    obs = np.nan

    current_price = np.nan

    high_52w = np.nan

    discount_calc = np.nan

    external_diff = np.nan

    median_turnover_60d = np.nan

    stale_bdays = np.nan


    symbol = (
        f"{ticker}.SA"
    )


    try:

        hist = yf.download(
            symbol,
            period="18mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )


        if hist.empty:

            external_reasons.append(
                "SEM_DADOS_YAHOO"
            )

        else:

            if isinstance(
                hist.columns,
                pd.MultiIndex,
            ):

                hist.columns = (
                    hist.columns
                    .get_level_values(0)
                )


            if "Close" not in hist.columns:

                external_reasons.append(
                    "SEM_COLUNA_CLOSE"
                )

            else:

                hist = (
                    hist
                    .dropna(
                        subset=["Close"]
                    )
                    .copy()
                )


                if hist.empty:

                    external_reasons.append(
                        "SEM_FECHAMENTOS_VALIDOS"
                    )

                else:

                    last252 = (
                        hist
                        .tail(252)
                        .copy()
                    )


                    close_series = (
                        pd.to_numeric(
                            last252["Close"],
                            errors="coerce",
                        )
                        .replace(
                            [
                                np.inf,
                                -np.inf,
                            ],
                            np.nan,
                        )
                        .dropna()
                    )


                    obs = int(
                        close_series
                        .notna()
                        .sum()
                    )


                    if obs == 0:

                        external_reasons.append(
                            "SEM_FECHAMENTOS_VALIDOS"
                        )

                    else:

                        current_price = float(
                            close_series.iloc[-1]
                        )


                        high_52w = float(
                            close_series.max()
                        )


                        if (
                            high_52w > 0
                            and
                            current_price > 0
                        ):

                            discount_calc = (
                                high_52w
                                -
                                current_price
                            ) / high_52w

                        else:

                            external_reasons.append(
                                "PRECO_INVALIDO"
                            )


                        last_date = (
                            pd.Timestamp(
                                close_series.index[-1]
                            )
                            .tz_localize(None)
                        )


                        today = (
                            pd.Timestamp
                            .today()
                            .normalize()
                        )


                        if (
                            last_date.date()
                            <
                            today.date()
                        ):

                            stale_bdays = int(
                                np.busday_count(
                                    last_date.date(),
                                    today.date(),
                                )
                            )

                        else:

                            stale_bdays = 0


                        if (
                            obs
                            <
                            MIN_OBS_52W
                        ):

                            external_reasons.append(
                                "HISTORICO_INSUFICIENTE"
                            )


                        if (
                            stale_bdays
                            >
                            STALE_BUSINESS_DAYS
                        ):

                            external_reasons.append(
                                "COTACAO_DESATUALIZADA"
                            )


                        if (
                            pd.notna(
                                discount_calc
                            )
                            and
                            discount_calc
                            >=
                            EXTREME_DISCOUNT
                        ):

                            external_reasons.append(
                                "DESCONTO_EXTREMO_80PCT"
                            )


                        if (
                            pd.notna(
                                discount_calc
                            )
                            and
                            pd.notna(
                                portfolio_discount
                            )
                        ):

                            external_diff = abs(
                                discount_calc
                                -
                                portfolio_discount
                            )


                            if (
                                external_diff
                                >
                                EXTERNAL_DISCOUNT_TOLERANCE
                            ):

                                external_reasons.append(
                                    "DIVERGENCIA_YAHOO_RAW"
                                )


                        if (
                            "Volume"
                            in
                            last252.columns
                        ):

                            turnover = (
                                pd.to_numeric(
                                    last252[
                                        "Close"
                                    ],
                                    errors="coerce",
                                )
                                *
                                pd.to_numeric(
                                    last252[
                                        "Volume"
                                    ],
                                    errors="coerce",
                                )
                            )


                            median_turnover_60d = (
                                float(
                                    turnover
                                    .tail(60)
                                    .median()
                                )
                            )


    except Exception as exc:

        external_reasons.append(
            "ERRO_COLETA_YAHOO:"
            f"{type(exc).__name__}"
        )


    # =========================================================================
    # 3.3 RESULTADO EXTERNO
    # =========================================================================

    external_status = (
        "REVIEW"
        if external_reasons
        else
        "PASS"
    )


    rows.append(
        {
            "TICKER":
                ticker,

            "MACRO_SECTOR":
                macro_sector,

            "DISCOUNT_PORTFOLIO":
                portfolio_discount,

            "DISCOUNT_ENGINE_SOURCE":
                engine_discount,

            "ENGINE_DIFF":
                internal_diff,

            "PRICE_QUALITY_STATUS":
                engine_quality_status,

            "ENGINE_STATUS":
                engine_status,

            "ENGINE_REASON":
                (
                    "|".join(
                        internal_reasons
                    )
                    if internal_reasons
                    else
                    "OK"
                ),

            "LAST_DATE_YAHOO":
                (
                    last_date
                    .date()
                    .isoformat()
                    if last_date is not None
                    else
                    None
                ),

            "OBS_52W_YAHOO":
                obs,

            "CURRENT_PRICE_YAHOO_RAW":
                current_price,

            "HIGH_52W_YAHOO_RAW":
                high_52w,

            "DISCOUNT_YAHOO_RAW":
                discount_calc,

            "EXTERNAL_DIFF":
                external_diff,

            "MEDIAN_TURNOVER_60D":
                median_turnover_60d,

            "STALE_BUSINESS_DAYS":
                stale_bdays,

            "EXTERNAL_STATUS":
                external_status,

            "EXTERNAL_DIAGNOSTIC":
                (
                    "|".join(
                        external_reasons
                    )
                    if external_reasons
                    else
                    "OK"
                ),
        }
    )


# =============================================================================
# 4. RESULTADO
# =============================================================================

audit = pd.DataFrame(
    rows
)


audit = (
    audit
    .sort_values(
        [
            "MACRO_SECTOR",
            "TICKER",
        ]
    )
    .reset_index(
        drop=True
    )
)


audit.to_csv(
    OUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


print()

print("=" * 78)

print(
    "RESULTADO — "
    "INTEGRIDADE INTERNA"
)

print("=" * 78)


internal_cols = [
    "TICKER",
    "MACRO_SECTOR",
    "DISCOUNT_PORTFOLIO",
    "DISCOUNT_ENGINE_SOURCE",
    "ENGINE_DIFF",
    "PRICE_QUALITY_STATUS",
    "ENGINE_STATUS",
    "ENGINE_REASON",
]


print(
    audit[
        internal_cols
    ]
    .to_string(
        index=False
    )
)


print()

print("=" * 78)

print(
    "RESULTADO — "
    "DIAGNÓSTICO EXTERNO"
)

print("=" * 78)


external_cols = [
    "TICKER",
    "MACRO_SECTOR",
    "DISCOUNT_PORTFOLIO",
    "DISCOUNT_YAHOO_RAW",
    "EXTERNAL_DIFF",
    "CURRENT_PRICE_YAHOO_RAW",
    "HIGH_52W_YAHOO_RAW",
    "OBS_52W_YAHOO",
    "MEDIAN_TURNOVER_60D",
    "EXTERNAL_STATUS",
    "EXTERNAL_DIAGNOSTIC",
]


print(
    audit[
        external_cols
    ]
    .to_string(
        index=False
    )
)


# =============================================================================
# 5. AUDITORIA FINAL
# =============================================================================

n_engine_pass = int(
    (
        audit[
            "ENGINE_STATUS"
        ]
        ==
        "PASS"
    ).sum()
)


n_engine_fail = int(
    (
        audit[
            "ENGINE_STATUS"
        ]
        !=
        "PASS"
    ).sum()
)


n_external_pass = int(
    (
        audit[
            "EXTERNAL_STATUS"
        ]
        ==
        "PASS"
    ).sum()
)


n_external_review = int(
    (
        audit[
            "EXTERNAL_STATUS"
        ]
        ==
        "REVIEW"
    ).sum()
)


print()

print("=" * 78)

print(
    "AUDITORIA FINAL"
)

print("=" * 78)


print(
    f"Ações analisadas ...................... "
    f"{len(audit)}"
)


print(
    f"ENGINE PASS ........................... "
    f"{n_engine_pass}"
)


print(
    f"ENGINE FAIL ........................... "
    f"{n_engine_fail}"
)


print(
    f"Diagnóstico externo PASS .............. "
    f"{n_external_pass}"
)


print(
    f"Diagnóstico externo REVIEW ............ "
    f"{n_external_review}"
)


print(
    "Preço base = Close bruto (PREULT) ..... "
    "PRESERVADO"
)


print(
    "Regra de seleção 80/20 ................ "
    "NÃO ALTERADA"
)


print(
    "Portfólio atual ....................... "
    "NÃO ALTERADO"
)


print(
    "Histórico congelado ................... "
    "PRESERVADO"
)


print(
    "Yahoo ................................ "
    "DIAGNÓSTICO INDEPENDENTE"
)


# =============================================================================
# 6. STATUS ESTRUTURAL
# =============================================================================

if n_engine_fail != 0:

    print(
        "Integridade interna do motor .......... "
        "FAIL"
    )

    print()

    print(
        f"Arquivo: {OUT_FILE}"
    )

    print()

    print(
        "STATUS: AUDITORIA INTERNA FALHOU"
    )

    print("=" * 78)

    raise RuntimeError(
        "Auditoria detectou divergência "
        "interna no motor."
    )


print(
    "Integridade interna do motor .......... "
    "PASS"
)


print(
    "Reviews externos alteram metodologia .. "
    "NÃO"
)


print()

print(
    f"Arquivo: {OUT_FILE}"
)


print()

print(
    "STATUS: AUDITORIA INTERNA DO "
    "PORTFÓLIO VALIDADA"
)


print("=" * 78)
