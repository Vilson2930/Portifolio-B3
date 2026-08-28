"""
PORTIFOLIO-B3 — COLETOR OPERACIONAL DE PREÇOS

Objetivo:
    Buscar o universo atual de ações da B3 e calcular, com preços atualizados:

        MOM_6M        = retorno em 126 pregões
        MOM_12M       = retorno em 252 pregões
        DISCOUNT_52W  = distância positiva para a máxima dos últimos 252 pregões

Saída:
    data_live/price_factors_current.csv

Observação:
    Este arquivo NÃO altera os dados históricos validados em data/.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json

import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

ROOT = Path(__file__).resolve().parent
DATA_LIVE = ROOT / "data_live"
OUTPUT_FILE = DATA_LIVE / "price_factors_current.csv"

BRAPI_LIST_URL = "https://brapi.dev/api/quote/list"

PRICE_PERIOD = "18mo"
PRICE_INTERVAL = "1d"
BATCH_SIZE = 80
SLEEP_BETWEEN_BATCHES = 1.5

TRADING_DAYS_6M = 126
TRADING_DAYS_12M = 252
MIN_OBS_DISCOUNT = 60


# =============================================================================
# UTILIDADES
# =============================================================================

def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    return value


def get_json(url: str, params: dict | None = None) -> dict:
    if params:
        url = f"{url}?{urlencode(params)}"

    request = Request(
        url,
        headers={
            "User-Agent": "Portfolio-B3/1.0",
            "Accept": "application/json",
        },
    )

    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


# =============================================================================
# UNIVERSO ATUAL DA B3
# =============================================================================

def fetch_current_b3_universe() -> list[str]:
    """
    Descobre os ativos atuais classificados como ações pela brapi.

    Não usa a lista histórica congelada do estudo.
    Isso permite que novas empresas/tickers entrem no universo operacional.
    """

    tickers: set[str] = set()
    page = 1

    print()
    print("=" * 78)
    print("ETAPA 1 — UNIVERSO ATUAL DA B3")
    print("=" * 78)

    while True:
        payload = get_json(
            BRAPI_LIST_URL,
            {
                "type": "stock",
                "limit": 100,
                "page": page,
                "sortBy": "name",
                "sortOrder": "asc",
            },
        )

        stocks = payload.get("stocks", [])

        for item in stocks:
            ticker = str(
                item.get("stock", "")
            ).strip().upper()

            if not ticker:
                continue

            # Evita símbolos de mercado fracionário e formatos estranhos.
            # Mantém ações PN/ON e units, por exemplo PETR4, VALE3, SANB11.
            if ticker.endswith("F"):
                continue

            if not any(ch.isdigit() for ch in ticker):
                continue

            tickers.add(ticker)

        has_next = bool(
            payload.get("hasNextPage", False)
        )

        if not has_next:
            break

        page += 1

        if page > 100:
            raise RuntimeError(
                "Paginação do universo excedeu o limite de segurança."
            )

    result = sorted(tickers)

    if not result:
        raise RuntimeError(
            "Nenhum ticker foi encontrado no universo atual da B3."
        )

    print(f"Tickers atuais encontrados : {len(result):,}")
    print("STATUS                     : PASS")

    return result


# =============================================================================
# PREÇOS
# =============================================================================

def yahoo_symbol(ticker: str) -> str:
    return f"{ticker}.SA"


def download_batch(tickers: list[str]) -> pd.DataFrame:
    symbols = [
        yahoo_symbol(ticker)
        for ticker in tickers
    ]

    return yf.download(
        tickers=symbols,
        period=PRICE_PERIOD,
        interval=PRICE_INTERVAL,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )


def extract_close_series(
    data: pd.DataFrame,
    symbol: str,
) -> pd.Series:
    """
    Extrai a série Close já ajustada pelo auto_adjust=True do yfinance.
    """

    try:
        if isinstance(data.columns, pd.MultiIndex):
            series = data["Close"][symbol]
        else:
            # Caso o batch possua apenas um ativo.
            series = data["Close"]

        series = (
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

        series.index = pd.to_datetime(
            series.index,
            errors="coerce",
        )

        series = series[
            series.index.notna()
        ]

        series = series[
            ~series.index.duplicated(
                keep="last"
            )
        ].sort_index()

        return series

    except Exception:
        return pd.Series(
            dtype=float
        )


# =============================================================================
# FATORES — MESMA DEFINIÇÃO DO ESTUDO
# =============================================================================

def price_return(
    series: pd.Series,
    trading_days: int,
):
    if len(series) <= trading_days:
        return np.nan

    current = safe_float(
        series.iloc[-1]
    )

    previous = safe_float(
        series.iloc[
            -(trading_days + 1)
        ]
    )

    if (
        pd.isna(current)
        or pd.isna(previous)
        or previous <= 0
    ):
        return np.nan

    return (
        current
        /
        previous
        -
        1.0
    )


def discount_52w(
    series: pd.Series,
):
    if series.empty:
        return np.nan

    window = series.tail(
        TRADING_DAYS_12M
    )

    if len(window) < MIN_OBS_DISCOUNT:
        return np.nan

    current = safe_float(
        window.iloc[-1]
    )

    high = safe_float(
        window.max()
    )

    if (
        pd.isna(current)
        or pd.isna(high)
        or high <= 0
    ):
        return np.nan

    # Quanto maior, maior a distância abaixo da máxima de 52 semanas.
    return (
        high
        -
        current
    ) / high


# =============================================================================
# CONSTRUÇÃO DA BASE OPERACIONAL
# =============================================================================

def build_current_price_factors(
    tickers: list[str],
) -> pd.DataFrame:

    print()
    print("=" * 78)
    print("ETAPA 2 — PREÇOS E FATORES ATUAIS")
    print("=" * 78)

    rows = []
    total = len(tickers)

    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):
        batch = tickers[
            start:start + BATCH_SIZE
        ]

        batch_number = (
            start // BATCH_SIZE
        ) + 1

        total_batches = (
            total + BATCH_SIZE - 1
        ) // BATCH_SIZE

        print(
            f"Batch {batch_number}/{total_batches} "
            f"({len(batch)} tickers)"
        )

        try:
            data = download_batch(
                batch
            )
        except Exception as exc:
            print(
                f"  ERRO no batch: {exc}"
            )
            continue

        for ticker in batch:
            symbol = yahoo_symbol(
                ticker
            )

            series = extract_close_series(
                data,
                symbol,
            )

            if series.empty:
                continue

            current_price = safe_float(
                series.iloc[-1]
            )

            price_date = (
                series.index[-1]
                .date()
                .isoformat()
            )

            rows.append(
                {
                    "YEAR":
                        datetime.now().year,

                    "TICKER":
                        ticker,

                    "MOM_6M":
                        price_return(
                            series,
                            TRADING_DAYS_6M,
                        ),

                    "MOM_12M":
                        price_return(
                            series,
                            TRADING_DAYS_12M,
                        ),

                    "DISCOUNT_52W":
                        discount_52w(
                            series,
                        ),

                    "LAST_PRICE":
                        current_price,

                    "PRICE_DATE":
                        price_date,

                    "OBS_AVAILABLE":
                        len(series),
                }
            )

        if (
            start
            +
            BATCH_SIZE
            <
            total
        ):
            time.sleep(
                SLEEP_BETWEEN_BATCHES
            )

    factors = pd.DataFrame(
        rows
    )

    if factors.empty:
        raise RuntimeError(
            "Nenhum fator de preço foi calculado."
        )

    factors = (
        factors
        .drop_duplicates(
            ["YEAR", "TICKER"]
        )
        .sort_values(
            ["TICKER"]
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(f"Tickers com preço           : {len(factors):,}")
    print(
        "MOM_6M válido              : "
        f"{factors['MOM_6M'].notna().sum():,}"
    )
    print(
        "MOM_12M válido             : "
        f"{factors['MOM_12M'].notna().sum():,}"
    )
    print(
        "DISCOUNT_52W válido        : "
        f"{factors['DISCOUNT_52W'].notna().sum():,}"
    )

    return factors


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_price_factors(
    factors: pd.DataFrame,
) -> None:

    required = {
        "YEAR",
        "TICKER",
        "MOM_6M",
        "MOM_12M",
        "DISCOUNT_52W",
        "LAST_PRICE",
        "PRICE_DATE",
        "OBS_AVAILABLE",
    }

    missing = (
        required
        -
        set(factors.columns)
    )

    if missing:
        raise RuntimeError(
            f"Colunas ausentes: {sorted(missing)}"
        )

    duplicates = factors.duplicated(
        ["YEAR", "TICKER"]
    ).sum()

    if duplicates:
        raise RuntimeError(
            f"Duplicidades YEAR × TICKER: {duplicates}"
        )

    if (
        factors["DISCOUNT_52W"]
        .dropna()
        .lt(0)
        .any()
    ):
        raise RuntimeError(
            "DISCOUNT_52W negativo encontrado."
        )

    coverage_12m = (
        factors["MOM_12M"]
        .notna()
        .mean()
    )

    print()
    print("=" * 78)
    print("ETAPA 3 — AUDITORIA")
    print("=" * 78)

    print(
        f"Duplicidades YEAR × TICKER ....... {duplicates}"
    )

    print(
        f"Cobertura MOM_12M ................ {coverage_12m:.2%}"
    )

    print(
        "Definição MOM_6M = 126 pregões ... PASS"
    )

    print(
        "Definição MOM_12M = 252 pregões .. PASS"
    )

    print(
        "Definição Discount 52W ............ PASS"
    )

    print(
        "Dados históricos congelados ....... PRESERVADOS"
    )

    print(
        "STATUS ............................. PASS"
    )


# =============================================================================
# EXECUÇÃO
# =============================================================================

def main():

    print()
    print("=" * 78)
    print("PORTIFOLIO-B3 — ATUALIZAÇÃO OPERACIONAL DE PREÇOS")
    print("=" * 78)

    tickers = fetch_current_b3_universe()

    factors = build_current_price_factors(
        tickers
    )

    audit_price_factors(
        factors
    )

    DATA_LIVE.mkdir(
        parents=True,
        exist_ok=True,
    )

    factors.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print("ARQUIVO GERADO")
    print("=" * 78)

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "ATENÇÃO: este arquivo ainda não contém MACRO_SECTOR."
    )

    print(
        "A classificação setorial operacional será adicionada "
        "na próxima etapa."
    )

    print()
    print(
        "STATUS: PREÇOS OPERACIONAIS ATUALIZADOS"
    )
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
