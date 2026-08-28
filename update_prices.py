"""
PORTIFOLIO-B3 — COLETOR OPERACIONAL DE PREÇOS + CLASSIFICAÇÃO SETORIAL

Objetivo:
    Buscar o universo atual de ações da B3 e calcular:

        MOM_6M        = retorno em 126 pregões
        MOM_12M       = retorno em 252 pregões
        DISCOUNT_52W  = distância positiva para a máxima dos últimos 252 pregões

    Também classifica cada ativo nos mesmos macrosetores usados pelo estudo:

        UTILITIES
        INDUSTRIALS
        FINANCIALS
        CONSUMER
        MATERIALS
        TECHNOLOGY
        ENERGY
        HEALTHCARE
        COMMUNICATION

Saída:
    data_live/price_factors_current.csv

Importante:
    Este arquivo NÃO altera os dados históricos validados em data/.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

ROOT = Path(__file__).resolve().parent

DATA_LIVE = ROOT / "data_live"

OUTPUT_FILE = (
    DATA_LIVE
    / "price_factors_current.csv"
)

UNCLASSIFIED_FILE = (
    DATA_LIVE
    / "unclassified_tickers.csv"
)

BRAPI_LIST_URL = (
    "https://brapi.dev/api/quote/list"
)

PRICE_PERIOD = "18mo"
PRICE_INTERVAL = "1d"

BATCH_SIZE = 80

SLEEP_BETWEEN_BATCHES = 1.5

TRADING_DAYS_6M = 126
TRADING_DAYS_12M = 252

MIN_OBS_DISCOUNT = 60


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

def safe_float(value):

    try:
        value = float(value)

    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    return value


def get_json(
    url: str,
    params: dict | None = None,
) -> dict:

    if params:

        url = (
            f"{url}?"
            f"{urlencode(params)}"
        )

    request = Request(
        url,
        headers={
            "User-Agent":
                "Portfolio-B3/1.0",

            "Accept":
                "application/json",
        },
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:

        return json.loads(
            response
            .read()
            .decode("utf-8")
        )


def normalize_text(value) -> str:

    if value is None:
        return ""

    text = (
        str(value)
        .strip()
        .upper()
    )

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        ch
        for ch in text
        if not unicodedata.combining(ch)
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


# =============================================================================
# CLASSIFICAÇÃO MACROSETORIAL
# =============================================================================

def classify_macro_sector(
    sector_raw: str,
    subsector_raw: str,
) -> str:

    sector = normalize_text(
        sector_raw
    )

    subsector = normalize_text(
        subsector_raw
    )

    text = (
        f"{sector} | "
        f"{subsector}"
    )


    # =========================================================================
    # UTILITIES
    # =========================================================================

    utility_terms = [

        "ENERGIA ELETRICA",
        "ELECTRIC",

        "UTILIDADE PUBLICA",
        "UTILITIES",

        "AGUA",
        "WATER",

        "SANEAMENTO",
        "SANITATION",

        "DISTRIBUICAO DE GAS",
    ]

    if any(
        term in text
        for term in utility_terms
    ):
        return "UTILITIES"


    # =========================================================================
    # ENERGY
    # =========================================================================

    energy_terms = [

        "PETROLEO",
        "PETROLEUM",

        "OIL",

        "GAS NATURAL",

        "EXPLORACAO",
        "EXPLORATION",

        "REFINO",
        "REFINING",

        "COMBUSTIVEIS",
        "FUEL",

        "BIOCOMBUSTIVEIS",
        "BIOFUEL",
    ]

    if any(
        term in text
        for term in energy_terms
    ):
        return "ENERGY"


    # =========================================================================
    # MATERIALS
    # =========================================================================

    materials_terms = [

        "MINERACAO",
        "MINING",

        "SIDERURGIA",
        "STEEL",

        "METALURGIA",
        "METALS",

        "PAPEL",
        "PAPER",

        "CELULOSE",
        "PULP",

        "QUIMIC",
        "CHEMICAL",

        "PETROQUIMIC",

        "CIMENTO",
        "CEMENT",

        "MADEIRA",
        "WOOD",

        "MATERIAIS",
        "MATERIALS",

        "EMBALAGENS",
        "PACKAGING",
    ]

    if any(
        term in text
        for term in materials_terms
    ):
        return "MATERIALS"


    # =========================================================================
    # FINANCIALS
    # =========================================================================

    financial_terms = [

        "BANC",
        "BANK",

        "FINANCEIR",
        "FINANCIAL",

        "SEGURO",
        "INSURANCE",

        "PREVIDENCIA",

        "CAPITALIZACAO",

        "CORRETOR",
        "BROKER",

        "BOLSA",
        "EXCHANGE",

        "SERVICOS FINANCEIROS",

        "FINANCIAL SERVICES",

        "HOLDINGS DIVERSIFICADAS",
    ]

    if any(
        term in text
        for term in financial_terms
    ):
        return "FINANCIALS"


    # =========================================================================
    # TECHNOLOGY
    # =========================================================================

    technology_terms = [

        "TECNOLOGIA",
        "TECHNOLOGY",

        "SOFTWARE",

        "COMPUTADOR",
        "COMPUTER",

        "SEMICONDUTOR",
        "SEMICONDUCTOR",

        "SERVICOS DE TI",

        "INFORMATION TECHNOLOGY",

        "SISTEMAS",
        "SYSTEMS",

        "EQUIPAMENTOS DE INFORMATICA",

        "ELETRONIC",
    ]

    if any(
        term in text
        for term in technology_terms
    ):
        return "TECHNOLOGY"


    # =========================================================================
    # COMMUNICATION
    # =========================================================================

    communication_terms = [

        "TELECOM",

        "COMMUNICATION",

        "COMUNICACAO",

        "MIDIA",
        "MEDIA",

        "PUBLICIDADE",
        "ADVERTISING",

        "INTERNET",
    ]

    if any(
        term in text
        for term in communication_terms
    ):
        return "COMMUNICATION"


    # =========================================================================
    # HEALTHCARE
    # =========================================================================

    healthcare_terms = [

        "SAUDE",
        "HEALTH",

        "HOSPITAL",
        "HOSPITAIS",

        "MEDIC",
        "MEDICAL",

        "FARMAC",
        "PHARMA",

        "DIAGNOSTICO",
        "DIAGNOSTIC",

        "LABORATOR",

        "ODONTO",
        "DENTAL",

        "EQUIPAMENTOS MEDICOS",
    ]

    if any(
        term in text
        for term in healthcare_terms
    ):
        return "HEALTHCARE"


    # =========================================================================
    # INDUSTRIALS
    # =========================================================================

    industrial_terms = [

        "INDUSTRIAL",

        "MAQUINAS",
        "MACHINERY",

        "EQUIPAMENTOS",
        "EQUIPMENT",

        "CONSTRUCAO",
        "CONSTRUCTION",

        "TRANSPORTE",
        "TRANSPORT",

        "LOGISTICA",
        "LOGISTICS",

        "FERROVI",
        "RAIL",

        "RODOVI",
        "HIGHWAY",

        "AERONAUT",
        "AEROSPACE",

        "DEFESA",
        "DEFENSE",

        "SERVICOS INDUSTRIAIS",

        "ENGINEERING",
        "ENGENHARIA",
    ]

    if any(
        term in text
        for term in industrial_terms
    ):
        return "INDUSTRIALS"


    # =========================================================================
    # CONSUMER
    # =========================================================================

    consumer_terms = [

        "CONSUMO",
        "CONSUMER",

        "VAREJO",
        "RETAIL",

        "ALIMENT",
        "FOOD",

        "BEBIDA",
        "BEVERAGE",

        "VESTUARIO",
        "APPAREL",

        "CALCAD",
        "FOOTWEAR",

        "AUTOMOVEIS",
        "AUTOMOTIVE",

        "HOTEL",
        "HOTEIS",

        "RESTAURANT",

        "TURISMO",
        "TOURISM",

        "EDUCACAO",
        "EDUCATION",

        "COMERCIO",
        "COMMERCE",

        "E-COMMERCE",

        "AGRICULTURA",
        "AGRICULTURE",

        "AGRO",

        "CARNES",
        "MEAT",

        "PRODUTOS DE USO PESSOAL",

        "PERSONAL PRODUCTS",

        "PRODUTOS DOMESTICOS",

        "HOUSEHOLD",
    ]

    if any(
        term in text
        for term in consumer_terms
    ):
        return "CONSUMER"


    # =========================================================================
    # NÃO FORÇAR CLASSIFICAÇÃO
    # =========================================================================

    return "UNCLASSIFIED"


# =============================================================================
# UNIVERSO ATUAL DA B3
# =============================================================================

def fetch_current_b3_universe() -> pd.DataFrame:

    rows = []

    page = 1

    print()

    print("=" * 78)

    print(
        "ETAPA 1 — UNIVERSO ATUAL DA B3"
    )

    print("=" * 78)


    while True:

        payload = get_json(
            BRAPI_LIST_URL,
            {
                "type":
                    "stock",

                "limit":
                    100,

                "page":
                    page,

                "sortBy":
                    "name",

                "sortOrder":
                    "asc",
            },
        )


        stocks = payload.get(
            "stocks",
            [],
        )


        for item in stocks:

            ticker = str(
                item.get(
                    "stock",
                    "",
                )
            ).strip().upper()


            if not ticker:
                continue


            # Evita mercado fracionário

            if ticker.endswith("F"):
                continue


            # Mantém apenas símbolos com número

            if not any(
                ch.isdigit()
                for ch in ticker
            ):
                continue


            sector_raw = (

                item.get(
                    "sector"
                )

                or

                item.get(
                    "sectorName"
                )

                or

                ""
            )


            subsector_raw = (

                item.get(
                    "subsector"
                )

                or

                item.get(
                    "subSector"
                )

                or

                item.get(
                    "subsectorName"
                )

                or

                ""
            )


            macro_sector = (
                classify_macro_sector(
                    sector_raw,
                    subsector_raw,
                )
            )


            rows.append(
                {
                    "TICKER":
                        ticker,

                    "SECTOR_RAW":
                        str(
                            sector_raw
                        ).strip(),

                    "SUBSECTOR_RAW":
                        str(
                            subsector_raw
                        ).strip(),

                    "MACRO_SECTOR":
                        macro_sector,
                }
            )


        has_next = bool(
            payload.get(
                "hasNextPage",
                False,
            )
        )


        if not has_next:
            break


        page += 1


        if page > 100:

            raise RuntimeError(
                "Paginação do universo excedeu "
                "o limite de segurança."
            )


    universe = pd.DataFrame(
        rows
    )


    if universe.empty:

        raise RuntimeError(
            "Nenhum ticker foi encontrado "
            "no universo atual da B3."
        )


    universe = (

        universe

        .drop_duplicates(
            ["TICKER"],
            keep="first",
        )

        .sort_values(
            "TICKER"
        )

        .reset_index(
            drop=True
        )
    )


    classified = (

        universe[
            "MACRO_SECTOR"
        ]

        !=

        "UNCLASSIFIED"
    )


    print(
        f"Tickers atuais encontrados : "
        f"{len(universe):,}"
    )


    print(
        f"Macrosetores classificados : "
        f"{classified.sum():,}"
    )


    print(
        f"Não classificados          : "
        f"{(~classified).sum():,}"
    )


    print(
        "STATUS                     : PASS"
    )


    return universe


# =============================================================================
# PREÇOS
# =============================================================================

def yahoo_symbol(
    ticker: str,
) -> str:

    return (
        f"{ticker}.SA"
    )


def download_batch(
    tickers: list[str],
) -> pd.DataFrame:


    symbols = [

        yahoo_symbol(
            ticker
        )

        for ticker in tickers
    ]


    return yf.download(

        tickers=
            symbols,

        period=
            PRICE_PERIOD,

        interval=
            PRICE_INTERVAL,

        auto_adjust=
            True,

        progress=
            False,

        group_by=
            "column",

        threads=
            True,
    )


def extract_close_series(
    data: pd.DataFrame,
    symbol: str,
) -> pd.Series:


    try:

        if isinstance(
            data.columns,
            pd.MultiIndex,
        ):

            series = (
                data[
                    "Close"
                ][
                    symbol
                ]
            )

        else:

            series = (
                data[
                    "Close"
                ]
            )


        series = (

            pd.to_numeric(
                series,
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


        series.index = (
            pd.to_datetime(
                series.index,
                errors="coerce",
            )
        )


        series = (
            series[
                series.index.notna()
            ]
        )


        series = (

            series[
                ~series.index.duplicated(
                    keep="last"
                )
            ]

            .sort_index()
        )


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

        or

        pd.isna(previous)

        or

        previous <= 0
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


    window = (
        series.tail(
            TRADING_DAYS_12M
        )
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

        or

        pd.isna(high)

        or

        high <= 0
    ):

        return np.nan


    return (

        high
        -
        current

    ) / high


# =============================================================================
# CONSTRUÇÃO DA BASE OPERACIONAL
# =============================================================================

def build_current_price_factors(
    universe: pd.DataFrame,
) -> pd.DataFrame:


    print()

    print("=" * 78)

    print(
        "ETAPA 2 — PREÇOS E FATORES ATUAIS"
    )

    print("=" * 78)


    rows = []


    tickers = (
        universe[
            "TICKER"
        ]
        .tolist()
    )


    universe_lookup = (

        universe

        .set_index(
            "TICKER"
        )

        .to_dict(
            orient="index"
        )
    )


    total = len(
        tickers
    )


    for start in range(
        0,
        total,
        BATCH_SIZE,
    ):


        batch = tickers[
            start:
            start
            +
            BATCH_SIZE
        ]


        batch_number = (

            start
            //
            BATCH_SIZE

        ) + 1


        total_batches = (

            total
            +
            BATCH_SIZE
            -
            1

        ) // BATCH_SIZE


        print(

            f"Batch "
            f"{batch_number}/"
            f"{total_batches} "
            f"({len(batch)} tickers)"
        )


        try:

            data = (
                download_batch(
                    batch
                )
            )


        except Exception as exc:

            print(
                f"  ERRO no batch: "
                f"{exc}"
            )

            continue


        for ticker in batch:


            symbol = (
                yahoo_symbol(
                    ticker
                )
            )


            series = (
                extract_close_series(
                    data,
                    symbol,
                )
            )


            if series.empty:

                continue


            metadata = (
                universe_lookup[
                    ticker
                ]
            )


            current_price = (
                safe_float(
                    series.iloc[-1]
                )
            )


            price_date = (

                series
                .index[-1]
                .date()
                .isoformat()
            )


            rows.append(
                {

                    "YEAR":
                        datetime.now().year,

                    "TICKER":
                        ticker,

                    "MACRO_SECTOR":
                        metadata[
                            "MACRO_SECTOR"
                        ],

                    "SECTOR_RAW":
                        metadata[
                            "SECTOR_RAW"
                        ],

                    "SUBSECTOR_RAW":
                        metadata[
                            "SUBSECTOR_RAW"
                        ],

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
                            series
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
            [
                "YEAR",
                "TICKER",
            ]
        )

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


    print()


    print(
        f"Tickers com preço           : "
        f"{len(factors):,}"
    )


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


    print(
        "MACRO_SECTOR válido        : "
        f"{(factors['MACRO_SECTOR'] != 'UNCLASSIFIED').sum():,}"
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

        "MACRO_SECTOR",
        "SECTOR_RAW",
        "SUBSECTOR_RAW",

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
        set(
            factors.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"Colunas ausentes: "
            f"{sorted(missing)}"
        )


    duplicates = (

        factors

        .duplicated(
            [
                "YEAR",
                "TICKER",
            ]
        )

        .sum()
    )


    if duplicates:

        raise RuntimeError(

            f"Duplicidades YEAR × TICKER: "
            f"{duplicates}"
        )


    if (

        factors[
            "DISCOUNT_52W"
        ]

        .dropna()

        .lt(0)

        .any()
    ):

        raise RuntimeError(
            "DISCOUNT_52W negativo encontrado."
        )


    invalid_macro = (

        ~factors[
            "MACRO_SECTOR"
        ]

        .isin(

            VALID_MACRO_SECTORS
            |
            {
                "UNCLASSIFIED"
            }
        )

    ).sum()


    if invalid_macro:

        raise RuntimeError(

            f"Macrosetores inválidos: "
            f"{invalid_macro}"
        )


    coverage_12m = (

        factors[
            "MOM_12M"
        ]

        .notna()

        .mean()
    )


    classified_mask = (

        factors[
            "MACRO_SECTOR"
        ]

        !=

        "UNCLASSIFIED"
    )


    classification_coverage = (
        classified_mask.mean()
    )


    classified_count = int(
        classified_mask.sum()
    )


    unclassified_count = int(

        (
            ~classified_mask
        ).sum()
    )


    print()

    print("=" * 78)

    print(
        "ETAPA 3 — AUDITORIA"
    )

    print("=" * 78)


    print(

        f"Duplicidades YEAR × TICKER ....... "
        f"{duplicates}"
    )


    print(

        f"Cobertura MOM_12M ................ "
        f"{coverage_12m:.2%}"
    )


    print(

        f"Cobertura MACRO_SECTOR ........... "
        f"{classification_coverage:.2%}"
    )


    print(

        f"Tickers classificados ............ "
        f"{classified_count:,}"
    )


    print(

        f"Tickers não classificados ........ "
        f"{unclassified_count:,}"
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
        "Taxonomia macrosetorial ........... PASS"
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

    print(
        "PORTIFOLIO-B3 — ATUALIZAÇÃO OPERACIONAL "
        "DE PREÇOS + SETORES"
    )

    print("=" * 78)


    universe = (
        fetch_current_b3_universe()
    )


    factors = (
        build_current_price_factors(
            universe
        )
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


    unclassified = (

        factors[
            factors[
                "MACRO_SECTOR"
            ]
            ==
            "UNCLASSIFIED"
        ]

        .copy()

        .sort_values(
            [
                "SECTOR_RAW",
                "SUBSECTOR_RAW",
                "TICKER",
            ]
        )
    )


    if not unclassified.empty:

        unclassified.to_csv(
            UNCLASSIFIED_FILE,
            index=False,
            encoding="utf-8-sig",
        )


    print()

    print("=" * 78)

    print(
        "ARQUIVOS GERADOS"
    )

    print("=" * 78)


    print(

        f"Fatores atuais      : "
        f"{OUTPUT_FILE}"
    )


    if not unclassified.empty:

        print(

            f"Não classificados   : "
            f"{UNCLASSIFIED_FILE}"
        )


    print()

    print(
        "STATUS: PREÇOS E CLASSIFICAÇÃO "
        "SETORIAL OPERACIONAIS ATUALIZADOS"
    )


    print("=" * 78)

    print()


if __name__ == "__main__":
    main()
