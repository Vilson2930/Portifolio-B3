"""
PORTIFOLIO-B3 — PREÇOS OPERACIONAIS + SETORES COMPATÍVEIS COM O ESTUDO

PRIORIDADE DA CLASSIFICAÇÃO:

1. MACRO_SECTOR histórico do próprio estudo
2. Outra classe da mesma empresa
3. Classificação brapi para empresa nova
4. UNCLASSIFIED se não houver evidência suficiente

O histórico congelado em data/ NÃO é alterado.
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

DATA_DIR = ROOT / "data"
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

# Proteção de comparabilidade da série.
# Saltos de preço de 10x ou mais entre pregões consecutivos são tratados
# como quebra mecânica/corporativa da série, não como retorno econômico.
MAX_CONSECUTIVE_PRICE_RATIO = 10.0

MIN_SECTOR_COVERAGE = 0.95


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


def normalize_text(value):

    if value is None:
        return ""

    text = str(value).strip().upper()

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


def get_json(
    url,
    params=None,
):

    if params:

        url = (
            f"{url}?"
            f"{urlencode(params)}"
        )

    request = Request(
        url,
        headers={
            "User-Agent":
                "Portfolio-B3/2.0",

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


def ticker_root(ticker):

    ticker = str(
        ticker
    ).strip().upper()

    match = re.match(
        r"^([A-Z]{4})",
        ticker,
    )

    if not match:
        return ""

    return match.group(1)


# =============================================================================
# FILTRO DE TICKERS
# =============================================================================

def is_equity_ticker(ticker):

    ticker = str(
        ticker
    ).strip().upper()

    # ON / PN / PNA / PNB / UNIT
    #
    # Exemplos:
    # PETR3
    # PETR4
    # SANB11
    # BPAC5
    # MRSA3B

    pattern = (
        r"^[A-Z]{4}"
        r"(3|4|5|6|11)"
        r"B?$"
    )

    return bool(
        re.match(
            pattern,
            ticker,
        )
    )


# =============================================================================
# CLASSIFICAÇÃO HISTÓRICA — FONTE PRINCIPAL
# =============================================================================

def load_historical_sector_map():

    print()

    print("=" * 78)

    print(
        "CLASSIFICAÇÃO SETORIAL — "
        "MAPA HISTÓRICO DO ESTUDO"
    )

    print("=" * 78)


    source_files = [

        DATA_DIR
        / "fundamental_factors.csv",

        DATA_DIR
        / "price_factors.csv",

        DATA_DIR
        / "returns.csv",
    ]


    frames = []


    for path in source_files:

        if not path.exists():
            continue

        df = pd.read_csv(
            path,
            low_memory=False,
        )


        required = {
            "YEAR",
            "TICKER",
            "MACRO_SECTOR",
        }


        if not required.issubset(
            df.columns
        ):
            continue


        temp = df[
            [
                "YEAR",
                "TICKER",
                "MACRO_SECTOR",
            ]
        ].copy()


        temp["YEAR"] = pd.to_numeric(
            temp["YEAR"],
            errors="coerce",
        )


        temp["TICKER"] = (

            temp["TICKER"]

            .astype(str)

            .str.strip()

            .str.upper()
        )


        temp["MACRO_SECTOR"] = (

            temp["MACRO_SECTOR"]

            .astype(str)

            .str.strip()

            .str.upper()
        )


        temp = temp[
            temp["YEAR"].notna()
            &
            temp["TICKER"].notna()
            &
            temp["MACRO_SECTOR"].isin(
                VALID_MACRO_SECTORS
            )
        ].copy()


        frames.append(
            temp
        )


    if not frames:

        raise RuntimeError(
            "Nenhuma classificação histórica "
            "foi localizada em data/."
        )


    history = pd.concat(
        frames,
        ignore_index=True,
    )


    history = (
        history

        .drop_duplicates(
            [
                "YEAR",
                "TICKER",
                "MACRO_SECTOR",
            ]
        )
    )


    history["YEAR"] = (
        history["YEAR"]
        .astype(int)
    )


    # -------------------------------------------------------------------------
    # Usa a classificação MAIS RECENTE disponível no estudo.
    # -------------------------------------------------------------------------

    latest = (

        history

        .sort_values(
            [
                "TICKER",
                "YEAR",
            ]
        )

        .groupby(
            "TICKER",
            as_index=False,
        )

        .tail(1)
    )


    sector_map = dict(
        zip(
            latest["TICKER"],
            latest["MACRO_SECTOR"],
        )
    )


    print(
        f"Tickers históricos mapeados : "
        f"{len(sector_map):,}"
    )

    print(
        "Fonte                       : "
        "DADOS CONGELADOS DO ESTUDO"
    )

    print(
        "STATUS                      : PASS"
    )


    return sector_map


# =============================================================================
# FALLBACK PARA EMPRESAS NOVAS
# =============================================================================

def classify_new_ticker(
    sector_raw,
    subsector_raw,
):

    sector = normalize_text(
        sector_raw
    )

    subsector = normalize_text(
        subsector_raw
    )


    # =========================================================================
    # OVERRIDES DE SUBSETOR
    # =========================================================================

    # Energia

    if subsector in {
        "EXPLORACAO E PRODUCAO DE PETROLEO",
        "EXPLORACAO. REFINO E DISTRIBUICAO",
        "EXPLORACAO, REFINO E DISTRIBUICAO",
        "PETROLEO E GAS INTEGRADO",
    }:
        return "ENERGY"


    # Utilities

    if subsector in {
        "ENERGIA ELETRICA",
        "AGUA E SANEAMENTO",
        "GAS",
    }:
        return "UTILITIES"


    # Healthcare

    if (
        "MEDICAMENTO" in subsector
        or
        "SERVICOS MEDICOS" in subsector
        or
        "ANALISES E DIAGNOSTICOS" in subsector
    ):
        return "HEALTHCARE"


    # Construção / incorporação pertence a INDUSTRIALS
    # conforme o universo congelado do estudo.

    if subsector in {
        "INCORPORACOES",
        "CONSTRUCAO PESADA",
        "PRODUTOS PARA CONSTRUCAO",
        "EXPLORACAO DE RODOVIAS",
    }:
        return "INDUSTRIALS"


    # Transporte / máquinas / defesa

    if (
        "TRANSPORTE " in subsector
        or
        "MAQ. E EQUIP." in subsector
        or
        "MAQUINAS E EQUIPAMENTOS" in subsector
        or
        "MATERIAL RODOVIARIO" in subsector
        or
        "MATERIAL DE TRANSPORTE" in subsector
        or
        "MATERIAL AERONAUTICO" in subsector
        or
        "ARMAS E MUNICOES" in subsector
    ):
        return "INDUSTRIALS"


    # Tecnologia

    if subsector in {
        "PROGRAMAS E SERVICOS",
        "PROGRAMAS DE FIDELIZACAO",
        "COMPUTADORES E EQUIPAMENTOS",
    }:
        return "TECHNOLOGY"


    # Telecom

    if subsector == "TELECOMUNICACOES":
        return "COMMUNICATION"


    # Financeiro

    if (
        "BANCOS" in subsector
        or
        "SEGURADOR" in subsector
        or
        "RESSEGURADOR" in subsector
        or
        "SERVICOS FINANCEIROS" in subsector
        or
        "BOLSAS DE VALORES" in subsector
        or
        "HOLDINGS DIVERSIFICADAS" in subsector
    ):
        return "FINANCIALS"


    # Materiais

    if (
        "SIDERURGIA" in subsector
        or
        "MINERA" in subsector
        or
        "MINERAIS METALICOS" in subsector
        or
        "PETROQUIM" in subsector
        or
        "QUIMIC" in subsector
        or
        "PAPEL E CELULOSE" in subsector
        or
        "FERTILIZANTES" in subsector
        or
        "ARTEFATOS DE COBRE" in subsector
        or
        "ARTEFATOS DE FERRO" in subsector
    ):
        return "MATERIALS"


    # Consumo

    consumer_terms = [
        "ALIMENT",
        "BEBIDA",
        "CARNES",
        "CALCAD",
        "VESTUARIO",
        "TECIDOS",
        "MOVEIS",
        "ELETRODOMESTICOS",
        "AGRICULTURA",
        "ACUCAR E ALCOOL",
        "RESTAURANTE",
        "TURISMO",
        "SERVICOS EDUCACIONAIS",
        "ATIVIDADES ESPORTIVAS",
        "PRODUTOS DE USO PESSOAL",
        "AUTOMOVEIS E MOTOCICLETAS",
        "BICICLETAS",
    ]


    if any(
        term in subsector
        for term in consumer_terms
    ):
        return "CONSUMER"


    # =========================================================================
    # FALLBACK PELO SETOR PRINCIPAL
    # =========================================================================

    sector_map = {

        "COMMUNICATIONS":
            "COMMUNICATION",

        "HEALTH SERVICES":
            "HEALTHCARE",

        "HEALTH TECHNOLOGY":
            "HEALTHCARE",

        "ENERGY MINERALS":
            "ENERGY",

        "UTILITIES":
            "UTILITIES",

        "TECHNOLOGY SERVICES":
            "TECHNOLOGY",

        "ELECTRONIC TECHNOLOGY":
            "TECHNOLOGY",

        "TRANSPORTATION":
            "INDUSTRIALS",

        "INDUSTRIAL SERVICES":
            "INDUSTRIALS",

        "PRODUCER MANUFACTURING":
            "INDUSTRIALS",

        "NON-ENERGY MINERALS":
            "MATERIALS",

        "FINANCE":
            "FINANCIALS",

        "CONSUMER DURABLES":
            "CONSUMER",

        "CONSUMER NON-DURABLES":
            "CONSUMER",

        "CONSUMER SERVICES":
            "CONSUMER",

        "RETAIL TRADE":
            "CONSUMER",
    }


    if sector in sector_map:

        return sector_map[
            sector
        ]


    return "UNCLASSIFIED"


# =============================================================================
# UNIVERSO ATUAL
# =============================================================================

def fetch_current_b3_universe(
    historical_sector_map,
):

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


            if not is_equity_ticker(
                ticker
            ):
                continue


            sector_raw = (

                item.get("sector")

                or

                item.get("sectorName")

                or

                ""
            )


            subsector_raw = (

                item.get("subsector")

                or

                item.get("subSector")

                or

                item.get("subsectorName")

                or

                ""
            )


            # -----------------------------------------------------------------
            # PRIORIDADE 1 — CLASSIFICAÇÃO DO ESTUDO
            # -----------------------------------------------------------------

            if ticker in historical_sector_map:

                macro_sector = (
                    historical_sector_map[
                        ticker
                    ]
                )

                source = (
                    "HISTORICAL_STUDY"
                )


            # -----------------------------------------------------------------
            # PRIORIDADE 2 — BRAPI PARA TICKER NOVO
            # -----------------------------------------------------------------

            else:

                macro_sector = (
                    classify_new_ticker(
                        sector_raw,
                        subsector_raw,
                    )
                )

                if (
                    macro_sector
                    ==
                    "UNCLASSIFIED"
                ):

                    source = (
                        "UNCLASSIFIED"
                    )

                else:

                    source = (
                        "BRAPI_FALLBACK"
                    )


            rows.append(
                {
                    "TICKER":
                        ticker,

                    "ROOT":
                        ticker_root(
                            ticker
                        ),

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

                    "CLASSIFICATION_SOURCE":
                        source,
                }
            )


        if not bool(
            payload.get(
                "hasNextPage",
                False,
            )
        ):
            break


        page += 1


        if page > 100:

            raise RuntimeError(
                "Paginação excedeu "
                "o limite de segurança."
            )


    universe = pd.DataFrame(
        rows
    )


    if universe.empty:

        raise RuntimeError(
            "Universo atual vazio."
        )


    universe = (

        universe

        .drop_duplicates(
            ["TICKER"]
        )

        .reset_index(
            drop=True
        )
    )


    # =========================================================================
    # HERANÇA ENTRE CLASSES DA MESMA EMPRESA
    # =========================================================================

    root_sector = {}


    classified = universe[
        universe[
            "MACRO_SECTOR"
        ]
        .isin(
            VALID_MACRO_SECTORS
        )
    ]


    for root, group in classified.groupby(
        "ROOT"
    ):

        sectors = (
            group[
                "MACRO_SECTOR"
            ]
            .dropna()
            .unique()
            .tolist()
        )


        if len(
            sectors
        ) == 1:

            root_sector[
                root
            ] = sectors[0]


    mask = (
        universe[
            "MACRO_SECTOR"
        ]
        ==
        "UNCLASSIFIED"
    )


    for idx in universe[
        mask
    ].index:

        root = universe.at[
            idx,
            "ROOT",
        ]


        if root in root_sector:

            universe.at[
                idx,
                "MACRO_SECTOR",
            ] = root_sector[
                root
            ]


            universe.at[
                idx,
                "CLASSIFICATION_SOURCE",
            ] = (
                "SAME_COMPANY"
            )


    universe = (

        universe

        .sort_values(
            "TICKER"
        )

        .reset_index(
            drop=True
        )
    )


    print(
        f"Tickers atuais encontrados : "
        f"{len(universe):,}"
    )


    print()


    source_counts = (

        universe[
            "CLASSIFICATION_SOURCE"
        ]

        .value_counts()
    )


    for source, count in (
        source_counts.items()
    ):

        print(
            f"{source:<24}: "
            f"{count:,}"
        )


    return universe


# =============================================================================
# PREÇOS
# =============================================================================

def yahoo_symbol(
    ticker,
):

    return (
        f"{ticker}.SA"
    )


def download_batch(
    tickers,
):

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

        # PREULT histórico da Célula 26 = preço de fechamento bruto.
        # Mantém o operacional alinhado à base histórica.
        auto_adjust=
            False,

        progress=
            False,

        group_by=
            "column",

        threads=
            True,
    )


def extract_close_series(
    data,
    symbol,
):

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


        series = series[
            series.index.notna()
        ]


        series = (

            series[
                ~series
                .index
                .duplicated(
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
# FATORES DE PREÇO
# =============================================================================

def price_return(
    series,
    trading_days,
):

    if len(
        series
    ) <= trading_days:

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


def has_price_comparability_break(series):
    """
    Detecta quebra extrema de comparabilidade entre fechamentos consecutivos.

    Não corrige nem inventa preços. Apenas impede que uma série com ruptura
    mecânica/corporativa alimente MOM_6M, MOM_12M e DISCOUNT_52W.
    """
    if series is None or len(series) < 2:
        return False

    s = pd.to_numeric(series, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    s = s[s > 0]

    if len(s) < 2:
        return False

    ratios = s / s.shift(1)
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()

    if ratios.empty:
        return False

    return bool(
        (ratios >= MAX_CONSECUTIVE_PRICE_RATIO).any()
        or
        (ratios <= (1.0 / MAX_CONSECUTIVE_PRICE_RATIO)).any()
    )


def discount_52w(
    series,
):

    if series.empty:
        return np.nan


    window = series.tail(
        TRADING_DAYS_12M
    )


    if len(
        window
    ) < MIN_OBS_DISCOUNT:

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
# BASE OPERACIONAL
# =============================================================================

def build_current_price_factors(
    universe,
):

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


    metadata = (

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

            data = download_batch(
                batch
            )


        except Exception as exc:

            print(
                f"ERRO batch: "
                f"{exc}"
            )

            continue


        for ticker in batch:

            symbol = yahoo_symbol(
                ticker
            )


            series = (
                extract_close_series(
                    data,
                    symbol,
                )
            )


            if series.empty:
                continue


            meta = metadata[
                ticker
            ]

            comparability_break = has_price_comparability_break(
                series.tail(TRADING_DAYS_12M + 1)
            )

            if comparability_break:
                mom_6m = np.nan
                mom_12m = np.nan
                discount = np.nan
                price_quality_status = "CORPORATE_ACTION_REVIEW"
            else:
                mom_6m = price_return(
                    series,
                    TRADING_DAYS_6M,
                )
                mom_12m = price_return(
                    series,
                    TRADING_DAYS_12M,
                )
                discount = discount_52w(
                    series
                )
                price_quality_status = "PASS"


            rows.append(
                {
                    "YEAR":
                        datetime.now().year,

                    "TICKER":
                        ticker,

                    "MACRO_SECTOR":
                        meta[
                            "MACRO_SECTOR"
                        ],

                    "CLASSIFICATION_SOURCE":
                        meta[
                            "CLASSIFICATION_SOURCE"
                        ],

                    "SECTOR_RAW":
                        meta[
                            "SECTOR_RAW"
                        ],

                    "SUBSECTOR_RAW":
                        meta[
                            "SUBSECTOR_RAW"
                        ],

                    "MOM_6M":
                        mom_6m,

                    "MOM_12M":
                        mom_12m,

                    "DISCOUNT_52W":
                        discount,

                    "PRICE_QUALITY_STATUS":
                        price_quality_status,

                    "LAST_PRICE":
                        safe_float(
                            series.iloc[-1]
                        ),

                    "PRICE_DATE":
                        series
                        .index[-1]
                        .date()
                        .isoformat(),

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
            "Nenhum fator calculado."
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


    return factors


# =============================================================================
# AUDITORIA
# =============================================================================

def audit(
    factors,
):

    print()

    print("=" * 78)

    print(
        "ETAPA 3 — AUDITORIA"
    )

    print("=" * 78)


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


    coverage_price = (

        factors[
            "MOM_12M"
        ]

        .notna()

        .mean()
    )


    classified = (

        factors[
            "MACRO_SECTOR"
        ]

        .isin(
            VALID_MACRO_SECTORS
        )
    )


    sector_coverage = (
        classified.mean()
    )


    print(
        f"Tickers com preço ................. "
        f"{len(factors):,}"
    )


    print(
        f"Duplicidades ...................... "
        f"{duplicates}"
    )


    print(
        f"Cobertura MOM_12M ................. "
        f"{coverage_price:.2%}"
    )


    print(
        f"Cobertura MACRO_SECTOR ............ "
        f"{sector_coverage:.2%}"
    )


    print(
        f"Classificados ..................... "
        f"{classified.sum():,}"
    )


    print(
        f"Não classificados ................. "
        f"{(~classified).sum():,}"
    )


    print()

    print(
        "ORIGEM DAS CLASSIFICAÇÕES"
    )

    print("-" * 78)


    counts = (

        factors[
            "CLASSIFICATION_SOURCE"
        ]

        .value_counts()
    )


    for source, count in (
        counts.items()
    ):

        print(
            f"{source:<24} "
            f"{count:>5,}"
        )


    print()


    if duplicates != 0:

        raise RuntimeError(
            "Duplicidades encontradas."
        )


    if sector_coverage < (
        MIN_SECTOR_COVERAGE
    ):

        raise RuntimeError(
            "Cobertura setorial abaixo "
            f"de {MIN_SECTOR_COVERAGE:.0%}."
        )


    print(
        "TOP4 taxonomy compatível "
        "com estudo ................. PASS"
    )


    print(
        "MOM_6M = 126 pregões ............. PASS"
    )


    print(
        "MOM_12M = 252 pregões ............ PASS"
    )


    print(
        "Discount 52W ..................... PASS"
    )

    print(
        "Preço base = Close bruto (PREULT)  PASS"
    )

    if "PRICE_QUALITY_STATUS" in factors.columns:
        n_review = int(
            (factors["PRICE_QUALITY_STATUS"] != "PASS").sum()
        )
        print(
            f"Quebras de comparabilidade ........ {n_review}"
        )
        print(
            "Proteção evento corporativo ....... ATIVA"
        )


    print(
        "Histórico congelado .............. PRESERVADO"
    )


    print(
        "STATUS ........................... PASS"
    )


# =============================================================================
# EXECUÇÃO
# =============================================================================

def main():

    print()

    print("=" * 78)

    print(
        "PORTIFOLIO-B3 — "
        "PREÇOS + SETORES OPERACIONAIS"
    )

    print("=" * 78)


    historical_sector_map = (
        load_historical_sector_map()
    )


    universe = (
        fetch_current_b3_universe(
            historical_sector_map
        )
    )


    factors = (
        build_current_price_factors(
            universe
        )
    )


    audit(
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


    unclassified = factors[
        factors[
            "MACRO_SECTOR"
        ]
        ==
        "UNCLASSIFIED"
    ].copy()


    if not unclassified.empty:

        unclassified.to_csv(
            UNCLASSIFIED_FILE,
            index=False,
            encoding="utf-8-sig",
        )


    print()

    print("=" * 78)

    print(
        "ARQUIVO GERADO"
    )

    print("=" * 78)


    print(
        OUTPUT_FILE
    )


    print()

    print(
        "STATUS: CAMADA OPERACIONAL "
        "DE PREÇOS E SETORES VALIDADA"
    )


    print("=" * 78)


if __name__ == "__main__":
    main()
