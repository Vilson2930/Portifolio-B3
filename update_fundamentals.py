"""
PORTIFOLIO-B3 — FUNDAMENTOS OPERACIONAIS CVM

Objetivo:
    Atualizar automaticamente os fundamentos das ações atuais
    preservando a metodologia validada no estudo histórico.

Fonte:
    Portal Dados Abertos CVM
    DFP + ITR

Metodologia preservada:

    ROE =
        NET_INCOME / EQUITY

    ROA =
        NET_INCOME / TOTAL_ASSETS

    OPERATING_MARGIN =
        EBIT / REVENUE

    NET_MARGIN =
        NET_INCOME / REVENUE

    TOTAL_LIABILITIES_PROXY =
        CURRENT_LIABILITIES
        +
        NONCURRENT_LIABILITIES

    DEBT_TO_EQUITY_PROXY =
        TOTAL_LIABILITIES_PROXY / EQUITY

Saída:
    data_live/fundamental_factors_current.csv
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

ROOT = Path(__file__).resolve().parent

DATA_LIVE = ROOT / "data_live"

PRICE_FILE = (
    DATA_LIVE
    / "price_factors_current.csv"
)

OUTPUT_FILE = (
    DATA_LIVE
    / "fundamental_factors_current.csv"
)

AUDIT_FILE = (
    DATA_LIVE
    / "fundamental_audit.csv"
)


CURRENT_YEAR = datetime.now().year


CVM_URLS = {

    "DFP":
        (
            "https://dados.cvm.gov.br/"
            "dados/CIA_ABERTA/DOC/DFP/DADOS/"
            f"dfp_cia_aberta_{CURRENT_YEAR}.zip"
        ),

    "ITR":
        (
            "https://dados.cvm.gov.br/"
            "dados/CIA_ABERTA/DOC/ITR/DADOS/"
            f"itr_cia_aberta_{CURRENT_YEAR}.zip"
        ),
}


REQUIRED_STATEMENTS = {
    "DRE",
    "BPA",
    "BPP",
}


# =============================================================================
# UTILIDADES
# =============================================================================

def safe_numeric(value):

    try:

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:

        return np.nan


def safe_divide(
    numerator,
    denominator,
):

    numerator = safe_numeric(
        numerator
    )

    denominator = safe_numeric(
        denominator
    )

    if (
        pd.isna(numerator)
        or
        pd.isna(denominator)
        or
        denominator == 0
    ):
        return np.nan

    return (
        numerator
        /
        denominator
    )


def clean_cnpj(value):

    if pd.isna(value):
        return ""

    return re.sub(
        r"\D",
        "",
        str(value),
    ).zfill(14)


def normalize_ticker(value):

    return (
        str(value)
        .strip()
        .upper()
    )


# =============================================================================
# UNIVERSO OPERACIONAL
# =============================================================================

def load_current_universe():

    if not PRICE_FILE.exists():

        raise RuntimeError(
            f"Arquivo não encontrado: "
            f"{PRICE_FILE}"
        )


    df = pd.read_csv(
        PRICE_FILE,
        low_memory=False,
    )


    required = {
        "TICKER",
        "MACRO_SECTOR",
    }


    missing = (
        required
        -
        set(df.columns)
    )


    if missing:

        raise RuntimeError(
            f"price_factors_current.csv "
            f"sem colunas: {sorted(missing)}"
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


    df = df[
        df["MACRO_SECTOR"]
        !=
        "UNCLASSIFIED"
    ].copy()


    df = (
        df[
            [
                "TICKER",
                "MACRO_SECTOR",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )


    print()

    print("=" * 78)

    print(
        "ETAPA 1 — UNIVERSO OPERACIONAL"
    )

    print("=" * 78)

    print(
        f"Tickers elegíveis : "
        f"{len(df):,}"
    )

    print(
        "STATUS            : PASS"
    )


    return df


# =============================================================================
# DOWNLOAD CVM
# =============================================================================

def download_zip(
    dataset,
):

    url = CVM_URLS[
        dataset
    ]


    print(
        f"Baixando {dataset} {CURRENT_YEAR}..."
    )


    response = requests.get(
        url,
        timeout=120,
        headers={
            "User-Agent":
                "Portfolio-B3/1.0"
        },
    )


    if response.status_code != 200:

        print(
            f"{dataset}: HTTP "
            f"{response.status_code}"
        )

        return None


    try:

        zf = zipfile.ZipFile(
            io.BytesIO(
                response.content
            )
        )

    except zipfile.BadZipFile:

        print(
            f"{dataset}: ZIP inválido"
        )

        return None


    print(
        f"{dataset}: "
        f"{len(response.content):,} bytes | "
        f"ZIP PASS"
    )


    return zf


# =============================================================================
# IDENTIFICAÇÃO DOS ARQUIVOS
# =============================================================================

def statement_type(
    filename,
):

    name = (
        filename
        .upper()
    )


    if "_DRE_" in name:
        return "DRE"

    if "_BPA_" in name:
        return "BPA"

    if "_BPP_" in name:
        return "BPP"

    if "_DFC_" in name:
        return "DFC"

    return None


def is_consolidated(
    filename,
):

    return (
        "_CON_" in
        filename.upper()
    )


# =============================================================================
# LEITURA DOS DEMONSTRATIVOS
# =============================================================================

def read_csv_from_zip(
    zf,
    filename,
):

    raw = zf.read(
        filename
    )


    for encoding in [
        "latin-1",
        "utf-8",
        "cp1252",
    ]:

        try:

            return pd.read_csv(
                io.BytesIO(raw),
                sep=";",
                encoding=encoding,
                low_memory=False,
            )

        except UnicodeDecodeError:

            continue


    raise RuntimeError(
        f"Não foi possível ler "
        f"{filename}"
    )


def load_dataset(
    dataset,
    zf,
):

    frames = {}


    if zf is None:
        return frames


    for filename in (
        zf.namelist()
    ):

        stype = statement_type(
            filename
        )


        if stype is None:
            continue


        # =============================================================
        # Mantém demonstrativo consolidado
        # =============================================================

        if not is_consolidated(
            filename
        ):
            continue


        try:

            df = read_csv_from_zip(
                zf,
                filename,
            )

        except Exception as exc:

            print(
                f"WARN leitura "
                f"{filename}: {exc}"
            )

            continue


        required = {
            "CNPJ_CIA",
            "DT_REFER",
            "VERSAO",
            "CD_CONTA",
            "VL_CONTA",
        }


        if not required.issubset(
            df.columns
        ):
            continue


        df["DATASET"] = (
            dataset
        )


        df["STATEMENT"] = (
            stype
        )


        frames[
            stype
        ] = df


    return frames


# =============================================================================
# BASE CVM
# =============================================================================

def prepare_statement(
    df,
):

    if df is None:
        return pd.DataFrame()


    df = df.copy()


    df["CNPJ_CIA"] = (
        df["CNPJ_CIA"]
        .apply(
            clean_cnpj
        )
    )


    df["DT_REFER"] = (
        pd.to_datetime(
            df["DT_REFER"],
            errors="coerce",
        )
    )


    df["VERSAO"] = (
        pd.to_numeric(
            df["VERSAO"],
            errors="coerce",
        )
    )


    df["VL_CONTA"] = (
        pd.to_numeric(
            df["VL_CONTA"],
            errors="coerce",
        )
    )


    df["CD_CONTA"] = (
        df["CD_CONTA"]
        .astype(str)
        .str.strip()
    )


    df = df[
        df["CNPJ_CIA"]
        .str.len()
        ==
        14
    ].copy()


    # =============================================================
    # Para cada companhia:
    # maior DT_REFER e maior versão
    # =============================================================

    latest_ref = (
        df.groupby(
            "CNPJ_CIA"
        )[
            "DT_REFER"
        ]
        .transform(
            "max"
        )
    )


    df = df[
        df["DT_REFER"]
        ==
        latest_ref
    ].copy()


    latest_version = (
        df.groupby(
            "CNPJ_CIA"
        )[
            "VERSAO"
        ]
        .transform(
            "max"
        )
    )


    df = df[
        df["VERSAO"]
        ==
        latest_version
    ].copy()


    return df


# =============================================================================
# EXTRAÇÃO DE CONTA
# =============================================================================

def account_value(
    df,
    cnpj,
    account,
):

    if df.empty:
        return np.nan


    subset = df[
        (
            df["CNPJ_CIA"]
            ==
            cnpj
        )
        &
        (
            df["CD_CONTA"]
            ==
            account
        )
    ]


    if subset.empty:
        return np.nan


    values = (
        subset[
            "VL_CONTA"
        ]
        .dropna()
    )


    if values.empty:
        return np.nan


    return safe_numeric(
        values.iloc[-1]
    )


# =============================================================================
# CADASTRO CVM
# =============================================================================

def load_cvm_registry():

    url = (
        "https://dados.cvm.gov.br/"
        "dados/CIA_ABERTA/CAD/DADOS/"
        "cad_cia_aberta.csv"
    )


    response = requests.get(
        url,
        timeout=120,
        headers={
            "User-Agent":
                "Portfolio-B3/1.0"
        },
    )


    response.raise_for_status()


    registry = pd.read_csv(
        io.BytesIO(
            response.content
        ),
        sep=";",
        encoding="latin-1",
        low_memory=False,
    )


    required = {
        "CNPJ_CIA",
        "DENOM_SOCIAL",
        "DENOM_COMERC",
        "CD_CVM",
    }


    missing = (
        required
        -
        set(
            registry.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"Cadastro CVM sem colunas: "
            f"{sorted(missing)}"
        )


    registry["CNPJ_CIA"] = (
        registry[
            "CNPJ_CIA"
        ]
        .apply(
            clean_cnpj
        )
    )


    return registry


# =============================================================================
# MAPA TICKER -> CNPJ
# =============================================================================

def build_ticker_cnpj_map(
    universe,
):

    """
    Usa Yahoo apenas como ponte cadastral para obter
    o nome da companhia.

    NÃO usa fundamentos Yahoo.

    O fundamento continua vindo exclusivamente da CVM.
    """

    import yfinance as yf


    registry = (
        load_cvm_registry()
    )


    registry[
        "DENOM_SOCIAL_NORM"
    ] = (
        registry[
            "DENOM_SOCIAL"
        ]
        .astype(str)
        .str.upper()
        .str.replace(
            r"[^A-Z0-9]",
            "",
            regex=True,
        )
    )


    registry[
        "DENOM_COMERC_NORM"
    ] = (
        registry[
            "DENOM_COMERC"
        ]
        .astype(str)
        .str.upper()
        .str.replace(
            r"[^A-Z0-9]",
            "",
            regex=True,
        )
    )


    rows = []


    print()

    print("=" * 78)

    print(
        "ETAPA 2 — IDENTIDADE TICKER → CVM"
    )

    print("=" * 78)


    for i, ticker in enumerate(
        universe["TICKER"],
        start=1,
    ):


        yahoo = (
            f"{ticker}.SA"
        )


        long_name = ""


        try:

            info = (
                yf.Ticker(
                    yahoo
                ).info
            )


            long_name = (
                info.get(
                    "longName"
                )
                or
                info.get(
                    "shortName"
                )
                or
                ""
            )


        except Exception:

            long_name = ""


        norm = re.sub(
            r"[^A-Z0-9]",
            "",
            str(
                long_name
            ).upper(),
        )


        cnpj = None
        status = (
            "UNRESOLVED"
        )


        if norm:

            exact = registry[
                (
                    registry[
                        "DENOM_SOCIAL_NORM"
                    ]
                    ==
                    norm
                )
                |
                (
                    registry[
                        "DENOM_COMERC_NORM"
                    ]
                    ==
                    norm
                )
            ]


            if len(
                exact
            ) == 1:

                cnpj = (
                    exact
                    .iloc[0][
                        "CNPJ_CIA"
                    ]
                )

                status = (
                    "EXACT_NAME"
                )


        rows.append(
            {
                "TICKER":
                    ticker,

                "COMPANY_NAME":
                    long_name,

                "CNPJ_CIA":
                    cnpj,

                "IDENTITY_STATUS":
                    status,
            }
        )


        if (
            i % 25 == 0
            or
            i == len(
                universe
            )
        ):

            print(
                f"Identidades: "
                f"{i:,}/"
                f"{len(universe):,}"
            )


    identity = pd.DataFrame(
        rows
    )


    resolved = (
        identity[
            "CNPJ_CIA"
        ]
        .notna()
        .sum()
    )


    print()

    print(
        f"Identidades resolvidas : "
        f"{resolved:,}"
    )

    print(
        f"Não resolvidas          : "
        f"{len(identity)-resolved:,}"
    )


    return identity


# =============================================================================
# ESCOLHA ENTRE DFP E ITR
# =============================================================================

def company_reference(
    cnpj,
    datasets,
):

    candidates = []


    for dataset_name in [
        "DFP",
        "ITR",
    ]:


        statements = datasets.get(
            dataset_name,
            {}
        )


        dre = statements.get(
            "DRE",
            pd.DataFrame(),
        )


        if dre.empty:
            continue


        company = dre[
            dre[
                "CNPJ_CIA"
            ]
            ==
            cnpj
        ]


        if company.empty:
            continue


        ref = (
            company[
                "DT_REFER"
            ]
            .max()
        )


        if pd.isna(
            ref
        ):
            continue


        candidates.append(
            (
                ref,
                dataset_name,
            )
        )


    if not candidates:
        return None


    # =============================================================
    # usa o demonstrativo MAIS RECENTE disponível
    # =============================================================

    candidates.sort(
        reverse=True
    )


    return candidates[0]


# =============================================================================
# CONSTRUÇÃO DOS FUNDAMENTOS
# =============================================================================

def build_fundamentals(
    universe,
    identity,
    datasets,
):


    base = universe.merge(
        identity,
        on="TICKER",
        how="left",
        validate="one_to_one",
    )


    rows = []


    print()

    print("=" * 78)

    print(
        "ETAPA 3 — FUNDAMENTOS CVM"
    )

    print("=" * 78)


    for row in (
        base.itertuples()
    ):


        ticker = row.TICKER

        sector = (
            row.MACRO_SECTOR
        )

        cnpj = (
            row.CNPJ_CIA
        )


        result = {

            "YEAR":
                CURRENT_YEAR,

            "TICKER":
                ticker,

            "MACRO_SECTOR":
                sector,

            "ROE_W":
                np.nan,

            "ROA_W":
                np.nan,

            "OPERATING_MARGIN_W":
                np.nan,

            "NET_MARGIN_W":
                np.nan,

            "DEBT_TO_EQUITY_PROXY_W":
                np.nan,

            "EQUITY":
                np.nan,

            "SOURCE_DATASET":
                None,

            "REFERENCE_DATE":
                None,

            "CNPJ_CIA":
                cnpj,

            "IDENTITY_STATUS":
                row.IDENTITY_STATUS,
        }


        if pd.isna(
            cnpj
        ):
            rows.append(
                result
            )
            continue


        reference = (
            company_reference(
                cnpj,
                datasets,
            )
        )


        if reference is None:

            rows.append(
                result
            )

            continue


        ref_date, dataset_name = (
            reference
        )


        statements = (
            datasets[
                dataset_name
            ]
        )


        dre = statements.get(
            "DRE",
            pd.DataFrame(),
        )

        bpa = statements.get(
            "BPA",
            pd.DataFrame(),
        )

        bpp = statements.get(
            "BPP",
            pd.DataFrame(),
        )


        revenue = account_value(
            dre,
            cnpj,
            "3.01",
        )


        ebit = account_value(
            dre,
            cnpj,
            "3.05",
        )


        net_income = account_value(
            dre,
            cnpj,
            "3.11",
        )


        total_assets = (
            account_value(
                bpa,
                cnpj,
                "1",
            )
        )


        current_liabilities = (
            account_value(
                bpp,
                cnpj,
                "2.01",
            )
        )


        noncurrent_liabilities = (
            account_value(
                bpp,
                cnpj,
                "2.02",
            )
        )


        equity = account_value(
            bpp,
            cnpj,
            "2.03",
        )


        total_liabilities = (
            current_liabilities
            +
            noncurrent_liabilities
        )


        roe = safe_divide(
            net_income,
            equity,
        )


        roa = safe_divide(
            net_income,
            total_assets,
        )


        operating_margin = (
            safe_divide(
                ebit,
                revenue,
            )
        )


        net_margin = (
            safe_divide(
                net_income,
                revenue,
            )
        )


        debt_to_equity = (
            safe_divide(
                total_liabilities,
                equity,
            )
        )


        result.update(
            {

                "ROE_W":
                    roe,

                "ROA_W":
                    roa,

                "OPERATING_MARGIN_W":
                    operating_margin,

                "NET_MARGIN_W":
                    net_margin,

                "DEBT_TO_EQUITY_PROXY_W":
                    debt_to_equity,

                "EQUITY":
                    equity,

                "SOURCE_DATASET":
                    dataset_name,

                "REFERENCE_DATE":
                    ref_date,
            }
        )


        rows.append(
            result
        )


    return pd.DataFrame(
        rows
    )


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_fundamentals(
    fundamentals,
):


    print()

    print("=" * 78)

    print(
        "ETAPA 4 — AUDITORIA"
    )

    print("=" * 78)


    duplicates = (
        fundamentals
        .duplicated(
            [
                "YEAR",
                "TICKER",
            ]
        )
        .sum()
    )


    components = [

        "ROE_W",

        "ROA_W",

        "OPERATING_MARGIN_W",

        "NET_MARGIN_W",

        "DEBT_TO_EQUITY_PROXY_W",
    ]


    fundamentals[
        "VALID_COMPONENTS"
    ] = (
        fundamentals[
            components
        ]
        .notna()
        .sum(
            axis=1
        )
    )


    eligible = (
        fundamentals[
            "VALID_COMPONENTS"
        ]
        >=
        3
    )


    print(
        f"Tickers processados ............... "
        f"{len(fundamentals):,}"
    )


    print(
        f"Duplicidades ...................... "
        f"{duplicates}"
    )


    print(
        f"ROE válido ........................ "
        f"{fundamentals['ROE_W'].notna().sum():,}"
    )


    print(
        f"ROA válido ........................ "
        f"{fundamentals['ROA_W'].notna().sum():,}"
    )


    print(
        f"Operating Margin válido ........... "
        f"{fundamentals['OPERATING_MARGIN_W'].notna().sum():,}"
    )


    print(
        f"Net Margin válido ................. "
        f"{fundamentals['NET_MARGIN_W'].notna().sum():,}"
    )


    print(
        f"Debt/Equity válido ................ "
        f"{fundamentals['DEBT_TO_EQUITY_PROXY_W'].notna().sum():,}"
    )


    print(
        f"Fund Score elegível (>=3) ......... "
        f"{eligible.sum():,}"
    )


    coverage = (
        eligible.mean()
    )


    print(
        f"Cobertura fundamental ............. "
        f"{coverage:.2%}"
    )


    print()


    print(
        "Fonte DFP ........................ "
        f"{(fundamentals['SOURCE_DATASET']=='DFP').sum():,}"
    )


    print(
        "Fonte ITR ........................ "
        f"{(fundamentals['SOURCE_DATASET']=='ITR').sum():,}"
    )


    print()


    if duplicates:

        raise RuntimeError(
            "Duplicidades encontradas."
        )


    print(
        "Metodologia CVM ................... PASS"
    )


    print(
        "Regra >= 3 componentes ........... PASS"
    )


    print(
        "Histórico congelado .............. PRESERVADO"
    )


    print(
        "STATUS ........................... PASS"
    )


    return fundamentals


# =============================================================================
# EXECUÇÃO
# =============================================================================

def main():


    print()

    print("=" * 78)

    print(
        "PORTIFOLIO-B3 — "
        "ATUALIZAÇÃO FUNDAMENTALISTA CVM"
    )

    print("=" * 78)


    universe = (
        load_current_universe()
    )


    identity = (
        build_ticker_cnpj_map(
            universe
        )
    )


    datasets = {}


    for dataset_name in [
        "DFP",
        "ITR",
    ]:


        zf = download_zip(
            dataset_name
        )


        statements = (
            load_dataset(
                dataset_name,
                zf,
            )
        )


        prepared = {}


        for stype, df in (
            statements.items()
        ):

            prepared[
                stype
            ] = (
                prepare_statement(
                    df
                )
            )


        datasets[
            dataset_name
        ] = prepared


    fundamentals = (
        build_fundamentals(
            universe,
            identity,
            datasets,
        )
    )


    fundamentals = (
        audit_fundamentals(
            fundamentals
        )
    )


    DATA_LIVE.mkdir(
        parents=True,
        exist_ok=True,
    )


    output_columns = [

        "YEAR",

        "TICKER",

        "MACRO_SECTOR",

        "ROE_W",

        "ROA_W",

        "OPERATING_MARGIN_W",

        "NET_MARGIN_W",

        "DEBT_TO_EQUITY_PROXY_W",

        "EQUITY",

        "SOURCE_DATASET",

        "REFERENCE_DATE",

        "CNPJ_CIA",

        "IDENTITY_STATUS",

        "VALID_COMPONENTS",
    ]


    fundamentals[
        output_columns
    ].to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


    fundamentals[
        [
            "TICKER",
            "MACRO_SECTOR",
            "SOURCE_DATASET",
            "REFERENCE_DATE",
            "IDENTITY_STATUS",
            "VALID_COMPONENTS",
        ]
    ].to_csv(
        AUDIT_FILE,
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
        f"Fundamentos : "
        f"{OUTPUT_FILE}"
    )


    print(
        f"Auditoria   : "
        f"{AUDIT_FILE}"
    )


    print()

    print(
        "STATUS: CAMADA FUNDAMENTALISTA "
        "OPERACIONAL ATUALIZADA"
    )


    print("=" * 78)


if __name__ == "__main__":

    main()
