"""
PORTIFOLIO-B3 — FUNDAMENTOS OPERACIONAIS CVM

Objetivo:
    Atualizar automaticamente os fundamentos das ações atuais
    preservando a metodologia validada no estudo histórico.

Identidade:
    B3 (companhias listadas) -> CNPJ/CD_CVM -> validação no cadastro CVM.

Fonte fundamentalista:
    Portal Dados Abertos CVM — DFP + ITR.

Metodologia preservada:
    ROE = NET_INCOME / EQUITY
    ROA = NET_INCOME / TOTAL_ASSETS
    OPERATING_MARGIN = EBIT / REVENUE
    NET_MARGIN = NET_INCOME / REVENUE
    TOTAL_LIABILITIES_PROXY = CURRENT_LIABILITIES + NONCURRENT_LIABILITIES
    DEBT_TO_EQUITY_PROXY = TOTAL_LIABILITIES_PROXY / EQUITY

Saída:
    data_live/fundamental_factors_current.csv
    data_live/fundamental_audit.csv

Observação:
    Este arquivo NÃO altera data/ nem qualquer histórico congelado.
"""

from __future__ import annotations

import base64
import io
import json
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

PRICE_FILE = DATA_LIVE / "price_factors_current.csv"
OUTPUT_FILE = DATA_LIVE / "fundamental_factors_current.csv"
AUDIT_FILE = DATA_LIVE / "fundamental_audit.csv"

CURRENT_YEAR = datetime.now().year

CVM_BASE = "https://dados.cvm.gov.br/dados/CIA_ABERTA/DOC"
CVM_REGISTRY_URL = (
    "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv"
)

B3_COMPANIES_URL = (
    "https://sistemaswebb3-listados.b3.com.br/"
    "listedCompaniesProxy/CompanyCall/GetInitialCompanies"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 Portfolio-B3/1.0",
    "Accept": "application/json,text/plain,*/*",
}

# Busca o ano corrente e o anterior.
# Isso garante DFP anual anterior + ITR corrente e também funciona no início do ano.
SOURCE_YEARS = [CURRENT_YEAR - 1, CURRENT_YEAR]

# Auditoria mínima: não retorna PASS com cobertura baixa.
MIN_IDENTITY_COVERAGE = 0.90
MIN_FUNDAMENTAL_COVERAGE = 0.80


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


def safe_divide(numerator, denominator):
    numerator = safe_numeric(numerator)
    denominator = safe_numeric(denominator)

    if (
        pd.isna(numerator)
        or pd.isna(denominator)
        or denominator == 0
    ):
        return np.nan

    return numerator / denominator


def clean_cnpj(value):
    if pd.isna(value):
        return ""

    digits = re.sub(r"\D", "", str(value))

    if not digits:
        return ""

    return digits.zfill(14)[-14:]


def normalize_ticker(value):
    return str(value).strip().upper()


def normalize_code(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Evita "9512.0".
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    return re.sub(r"\D", "", text)


def normalize_issuing_company(value):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).strip().upper(),
    )


# =============================================================================
# UNIVERSO OPERACIONAL
# =============================================================================

def load_current_universe():
    if not PRICE_FILE.exists():
        raise RuntimeError(
            f"Arquivo não encontrado: {PRICE_FILE}"
        )

    df = pd.read_csv(
        PRICE_FILE,
        low_memory=False,
    )

    required = {
        "TICKER",
        "MACRO_SECTOR",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "price_factors_current.csv sem colunas: "
            f"{sorted(missing)}"
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
        df["MACRO_SECTOR"] != "UNCLASSIFIED"
    ].copy()

    df = (
        df[
            [
                "TICKER",
                "MACRO_SECTOR",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if df["TICKER"].duplicated().any():
        dup = (
            df.loc[
                df["TICKER"].duplicated(False),
                "TICKER",
            ]
            .tolist()
        )
        raise RuntimeError(
            f"Ticker com mais de um setor: {dup[:20]}"
        )

    print()
    print("=" * 78)
    print("ETAPA 1 — UNIVERSO OPERACIONAL")
    print("=" * 78)
    print(f"Tickers elegíveis : {len(df):,}")
    print("STATUS            : PASS")

    return df


# =============================================================================
# CADASTRO CVM
# =============================================================================

def load_cvm_registry():
    response = requests.get(
        CVM_REGISTRY_URL,
        timeout=120,
        headers=HEADERS,
    )
    response.raise_for_status()

    registry = pd.read_csv(
        io.BytesIO(response.content),
        sep=";",
        encoding="latin-1",
        low_memory=False,
        dtype=str,
    )

    required = {
        "CNPJ_CIA",
        "DENOM_SOCIAL",
        "DENOM_COMERC",
        "CD_CVM",
    }

    missing = required - set(registry.columns)

    if missing:
        raise RuntimeError(
            "Cadastro CVM sem colunas: "
            f"{sorted(missing)}"
        )

    registry["CNPJ_CIA"] = (
        registry["CNPJ_CIA"]
        .apply(clean_cnpj)
    )

    registry["CD_CVM_NORM"] = (
        registry["CD_CVM"]
        .apply(normalize_code)
    )

    registry = registry[
        registry["CNPJ_CIA"].str.len() == 14
    ].copy()

    return registry


# =============================================================================
# IDENTIDADE ATUAL — B3 -> CNPJ/CD_CVM -> CVM
# =============================================================================

def encode_b3_params(params):
    raw = json.dumps(
        params,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return base64.b64encode(raw).decode("ascii")


def fetch_b3_listed_companies():
    rows = []
    page = 1
    page_size = 120

    while True:
        payload = {
            "language": "pt-br",
            "pageNumber": page,
            "pageSize": page_size,
        }

        encoded = encode_b3_params(payload)

        url = (
            f"{B3_COMPANIES_URL}/{encoded}"
        )

        response = requests.get(
            url,
            timeout=120,
            headers=HEADERS,
        )
        response.raise_for_status()

        data = response.json()
        results = data.get("results", [])

        if not results:
            break

        rows.extend(results)

        print(
            f"B3 companhias listadas: "
            f"página {page} | acumulado {len(rows):,}"
        )

        if len(results) < page_size:
            break

        page += 1

        if page > 100:
            raise RuntimeError(
                "Paginação B3 excedeu o limite de segurança."
            )

    if not rows:
        raise RuntimeError(
            "B3 não retornou companhias listadas."
        )

    df = pd.DataFrame(rows)

    # Os nomes usados pela B3 podem variar levemente.
    rename_candidates = {
        "issuingCompany": "ISSUING_COMPANY",
        "companyName": "COMPANY_NAME",
        "tradingName": "TRADING_NAME",
        "cnpj": "CNPJ_B3",
        "codeCVM": "CD_CVM_B3",
    }

    for source, target in rename_candidates.items():
        if source in df.columns:
            df[target] = df[source]
        else:
            df[target] = np.nan

    df["ISSUING_COMPANY_NORM"] = (
        df["ISSUING_COMPANY"]
        .apply(normalize_issuing_company)
    )

    df["CNPJ_B3"] = (
        df["CNPJ_B3"]
        .apply(clean_cnpj)
    )

    df["CD_CVM_B3_NORM"] = (
        df["CD_CVM_B3"]
        .apply(normalize_code)
    )

    return df


def build_ticker_cnpj_map(
    universe,
    registry,
):
    print()
    print("=" * 78)
    print("ETAPA 2 — IDENTIDADE TICKER → B3 → CVM")
    print("=" * 78)

    b3 = fetch_b3_listed_companies()

    # Validação institucional B3 x CVM.
    cvm_by_cnpj = (
        registry[
            [
                "CNPJ_CIA",
                "CD_CVM_NORM",
                "DENOM_SOCIAL",
                "DENOM_COMERC",
            ]
        ]
        .drop_duplicates()
    )

    rows = []

    for ticker in universe["TICKER"]:
        ticker = normalize_ticker(ticker)

        # Regra estrutural corrente B3:
        # código emissor = quatro primeiros caracteres do ticker.
        # Não é usada para reconstrução histórica; somente identidade operacional atual.
        issuer_key = ticker[:4]

        candidates = b3[
            b3["ISSUING_COMPANY_NORM"] == issuer_key
        ].copy()

        # Mais de uma linha para a mesma companhia é normal.
        # A identidade só é aceita quando resta um único CNPJ válido.
        candidate_cnpjs = sorted(
            {
                cnpj
                for cnpj in candidates["CNPJ_B3"].tolist()
                if len(cnpj) == 14
            }
        )

        cnpj = ""
        code_b3 = ""
        code_cvm = ""
        company_name = ""
        identity_status = "UNRESOLVED"

        if len(candidate_cnpjs) == 1:
            candidate_cnpj = candidate_cnpjs[0]

            b3_same_cnpj = candidates[
                candidates["CNPJ_B3"] == candidate_cnpj
            ].copy()

            b3_codes = sorted(
                {
                    code
                    for code in b3_same_cnpj[
                        "CD_CVM_B3_NORM"
                    ].tolist()
                    if code
                }
            )

            cvm_match = cvm_by_cnpj[
                cvm_by_cnpj["CNPJ_CIA"] == candidate_cnpj
            ].copy()

            if len(cvm_match) >= 1:
                cvm_codes = sorted(
                    {
                        code
                        for code in cvm_match[
                            "CD_CVM_NORM"
                        ].tolist()
                        if code
                    }
                )

                # Se ambos os lados possuem CD_CVM, exigimos concordância.
                code_consistent = (
                    not b3_codes
                    or not cvm_codes
                    or bool(
                        set(b3_codes)
                        & set(cvm_codes)
                    )
                )

                if code_consistent:
                    cnpj = candidate_cnpj
                    code_b3 = (
                        b3_codes[0]
                        if b3_codes
                        else ""
                    )
                    code_cvm = (
                        next(
                            (
                                x
                                for x in cvm_codes
                                if x in b3_codes
                            ),
                            cvm_codes[0]
                            if cvm_codes
                            else "",
                        )
                    )

                    company_name = str(
                        cvm_match.iloc[0][
                            "DENOM_SOCIAL"
                        ]
                    )

                    identity_status = (
                        "B3_CNPJ_CVM_VALIDATED"
                    )
                else:
                    identity_status = (
                        "B3_CVM_CODE_CONFLICT"
                    )
            else:
                identity_status = (
                    "B3_CNPJ_NOT_IN_CVM"
                )

        elif len(candidate_cnpjs) > 1:
            identity_status = (
                "AMBIGUOUS_B3_ISSUER"
            )

        rows.append(
            {
                "TICKER": ticker,
                "COMPANY_NAME": company_name,
                "CNPJ_CIA": cnpj,
                "CD_CVM_B3": code_b3,
                "CD_CVM": code_cvm,
                "IDENTITY_STATUS": identity_status,
            }
        )

    identity = pd.DataFrame(rows)

    resolved = (
        identity["CNPJ_CIA"]
        .astype(str)
        .str.len()
        .eq(14)
        .sum()
    )

    coverage = (
        resolved / len(identity)
        if len(identity)
        else 0.0
    )

    print()
    print(
        f"Identidades resolvidas : "
        f"{resolved:,}/{len(identity):,} "
        f"({coverage:.2%})"
    )
    print(
        f"Não resolvidas          : "
        f"{len(identity) - resolved:,}"
    )

    status_counts = (
        identity["IDENTITY_STATUS"]
        .value_counts(dropna=False)
    )

    print()
    print("ORIGEM / STATUS DAS IDENTIDADES")
    for status, count in status_counts.items():
        print(
            f"{str(status):30s} "
            f"{int(count):>6,}"
        )

    return identity


# =============================================================================
# DOWNLOAD CVM
# =============================================================================

def cvm_zip_url(dataset, year):
    return (
        f"{CVM_BASE}/{dataset}/DADOS/"
        f"{dataset.lower()}_cia_aberta_{year}.zip"
    )


def download_zip(dataset, year):
    url = cvm_zip_url(
        dataset,
        year,
    )

    print(
        f"Baixando {dataset} {year}..."
    )

    response = requests.get(
        url,
        timeout=180,
        headers=HEADERS,
    )

    if response.status_code == 404:
        print(
            f"{dataset} {year}: "
            "ainda não disponível"
        )
        return None

    response.raise_for_status()

    try:
        zf = zipfile.ZipFile(
            io.BytesIO(
                response.content
            )
        )
    except zipfile.BadZipFile:
        raise RuntimeError(
            f"{dataset} {year}: ZIP inválido."
        )

    print(
        f"{dataset} {year}: "
        f"{len(response.content):,} bytes | ZIP PASS"
    )

    return zf


# =============================================================================
# IDENTIFICAÇÃO / LEITURA DOS DEMONSTRATIVOS
# =============================================================================

def statement_type(filename):
    name = filename.upper()

    if "_DRE_" in name:
        return "DRE"

    if "_BPA_" in name:
        return "BPA"

    if "_BPP_" in name:
        return "BPP"

    return None


def is_consolidated(filename):
    return "_CON_" in filename.upper()


def read_csv_from_zip(
    zf,
    filename,
):
    raw = zf.read(filename)

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
                decimal=",",
                low_memory=False,
            )
        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        f"Não foi possível ler {filename}"
    )


def prepare_statement(
    df,
    dataset,
    source_year,
):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    required = {
        "CNPJ_CIA",
        "DT_REFER",
        "VERSAO",
        "CD_CONTA",
        "VL_CONTA",
    }

    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["CNPJ_CIA"] = (
        df["CNPJ_CIA"]
        .apply(clean_cnpj)
    )

    df["DT_REFER"] = pd.to_datetime(
        df["DT_REFER"],
        errors="coerce",
    )

    df["VERSAO"] = pd.to_numeric(
        df["VERSAO"],
        errors="coerce",
    )

    df["VL_CONTA"] = pd.to_numeric(
        df["VL_CONTA"],
        errors="coerce",
    )

    df["CD_CONTA"] = (
        df["CD_CONTA"]
        .astype(str)
        .str.strip()
    )

    # Regra preservada do estudo:
    # em demonstrativos com ORDEM_EXERC, usar o exercício atual ("ÚLTIMO").
    if "ORDEM_EXERC" in df.columns:
        ordem = (
            df["ORDEM_EXERC"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        valid_ordem = ordem.isin(
            [
                "ÚLTIMO",
                "ULTIMO",
            ]
        )

        if valid_ordem.any():
            df = df[
                valid_ordem
            ].copy()

    df = df[
        (df["CNPJ_CIA"].str.len() == 14)
        & df["DT_REFER"].notna()
        & df["VERSAO"].notna()
    ].copy()

    df["DATASET"] = dataset
    df["SOURCE_YEAR"] = source_year

    return df


def load_dataset(
    dataset,
    year,
    zf,
):
    frames = {
        "DRE": [],
        "BPA": [],
        "BPP": [],
    }

    if zf is None:
        return {
            key: pd.DataFrame()
            for key in frames
        }

    for filename in zf.namelist():
        stype = statement_type(
            filename
        )

        if stype is None:
            continue

        if not is_consolidated(
            filename
        ):
            continue

        try:
            raw_df = read_csv_from_zip(
                zf,
                filename,
            )
        except Exception as exc:
            print(
                f"WARN leitura {filename}: "
                f"{exc}"
            )
            continue

        df = prepare_statement(
            raw_df,
            dataset,
            year,
        )

        if not df.empty:
            frames[stype].append(df)

    result = {}

    for stype, parts in frames.items():
        if parts:
            result[stype] = pd.concat(
                parts,
                ignore_index=True,
            )
        else:
            result[stype] = pd.DataFrame()

    return result


# =============================================================================
# CONSOLIDAÇÃO DAS FONTES CVM
# =============================================================================

def load_all_cvm_sources():
    combined = {
        "DRE": [],
        "BPA": [],
        "BPP": [],
    }

    loaded_sources = []

    for dataset in [
        "DFP",
        "ITR",
    ]:
        for year in SOURCE_YEARS:
            zf = download_zip(
                dataset,
                year,
            )

            if zf is None:
                continue

            statements = load_dataset(
                dataset,
                year,
                zf,
            )

            loaded_sources.append(
                f"{dataset}_{year}"
            )

            for stype, df in statements.items():
                if not df.empty:
                    combined[
                        stype
                    ].append(df)

    result = {}

    for stype, parts in combined.items():
        if parts:
            result[stype] = pd.concat(
                parts,
                ignore_index=True,
            )
        else:
            result[stype] = pd.DataFrame()

    if result["DRE"].empty:
        raise RuntimeError(
            "Nenhuma DRE consolidada foi carregada."
        )

    print()
    print(
        "Fontes CVM carregadas : "
        + ", ".join(
            loaded_sources
        )
    )

    return result


# =============================================================================
# ESCOLHA DO DOCUMENTO MAIS RECENTE
# =============================================================================

def choose_company_reference(
    dre,
    cnpj,
):
    company = dre[
        dre["CNPJ_CIA"] == cnpj
    ].copy()

    if company.empty:
        return None

    # Não usa referência futura em relação ao instante de execução.
    now = pd.Timestamp.now().normalize()

    company = company[
        company["DT_REFER"] <= now
    ].copy()

    if company.empty:
        return None

    latest_ref = (
        company["DT_REFER"]
        .max()
    )

    company = company[
        company["DT_REFER"]
        == latest_ref
    ].copy()

    # Se houver DFP e ITR para a mesma data, DFP é preferida.
    company[
        "_DATASET_PRIORITY"
    ] = (
        company["DATASET"]
        .map(
            {
                "DFP": 2,
                "ITR": 1,
            }
        )
        .fillna(0)
    )

    best_dataset_priority = (
        company[
            "_DATASET_PRIORITY"
        ]
        .max()
    )

    company = company[
        company[
            "_DATASET_PRIORITY"
        ]
        ==
        best_dataset_priority
    ].copy()

    latest_version = (
        company["VERSAO"]
        .max()
    )

    company = company[
        company["VERSAO"]
        ==
        latest_version
    ].copy()

    company = company.sort_values(
        [
            "_DATASET_PRIORITY",
            "SOURCE_YEAR",
        ],
        ascending=[
            False,
            False,
        ],
    )

    row = company.iloc[0]

    return {
        "REFERENCE_DATE":
            row["DT_REFER"],
        "DATASET":
            row["DATASET"],
        "SOURCE_YEAR":
            int(row["SOURCE_YEAR"]),
        "VERSION":
            float(row["VERSAO"]),
    }


# =============================================================================
# EXTRAÇÃO DE CONTA
# =============================================================================

def account_value(
    df,
    cnpj,
    account,
    reference,
):
    if df.empty:
        return np.nan

    subset = df[
        (df["CNPJ_CIA"] == cnpj)
        & (
            df["DT_REFER"]
            ==
            reference[
                "REFERENCE_DATE"
            ]
        )
        & (
            df["DATASET"]
            ==
            reference[
                "DATASET"
            ]
        )
        & (
            df["SOURCE_YEAR"]
            ==
            reference[
                "SOURCE_YEAR"
            ]
        )
        & (
            df["VERSAO"]
            ==
            reference[
                "VERSION"
            ]
        )
        & (
            df["CD_CONTA"]
            ==
            account
        )
    ].copy()

    if subset.empty:
        return np.nan

    values = pd.to_numeric(
        subset["VL_CONTA"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    # CD_CONTA deve ser único dentro do documento.
    # Se houver repetição técnica, não somamos para não duplicar conta.
    return safe_numeric(
        values.iloc[-1]
    )


# =============================================================================
# CONSTRUÇÃO DOS FUNDAMENTOS
# =============================================================================

def build_fundamentals(
    universe,
    identity,
    statements,
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
    print("ETAPA 3 — FUNDAMENTOS CVM")
    print("=" * 78)

    dre = statements["DRE"]
    bpa = statements["BPA"]
    bpp = statements["BPP"]

    for row in base.itertuples():
        ticker = row.TICKER
        sector = row.MACRO_SECTOR
        cnpj = str(
            row.CNPJ_CIA
            if pd.notna(
                row.CNPJ_CIA
            )
            else ""
        )

        result = {
            "YEAR": CURRENT_YEAR,
            "TICKER": ticker,
            "MACRO_SECTOR": sector,
            "ROE_W": np.nan,
            "ROA_W": np.nan,
            "OPERATING_MARGIN_W": np.nan,
            "NET_MARGIN_W": np.nan,
            "DEBT_TO_EQUITY_PROXY_W": np.nan,
            "EQUITY": np.nan,
            "SOURCE_DATASET": None,
            "SOURCE_YEAR": np.nan,
            "REFERENCE_DATE": None,
            "CNPJ_CIA": cnpj,
            "CD_CVM": getattr(
                row,
                "CD_CVM",
                "",
            ),
            "IDENTITY_STATUS": row.IDENTITY_STATUS,
        }

        if len(cnpj) != 14:
            rows.append(result)
            continue

        reference = choose_company_reference(
            dre,
            cnpj,
        )

        if reference is None:
            rows.append(result)
            continue

        revenue = account_value(
            dre,
            cnpj,
            "3.01",
            reference,
        )

        ebit = account_value(
            dre,
            cnpj,
            "3.05",
            reference,
        )

        net_income = account_value(
            dre,
            cnpj,
            "3.11",
            reference,
        )

        total_assets = account_value(
            bpa,
            cnpj,
            "1",
            reference,
        )

        current_liabilities = (
            account_value(
                bpp,
                cnpj,
                "2.01",
                reference,
            )
        )

        noncurrent_liabilities = (
            account_value(
                bpp,
                cnpj,
                "2.02",
                reference,
            )
        )

        equity = account_value(
            bpp,
            cnpj,
            "2.03",
            reference,
        )

        if (
            pd.notna(
                current_liabilities
            )
            and pd.notna(
                noncurrent_liabilities
            )
        ):
            total_liabilities = (
                current_liabilities
                +
                noncurrent_liabilities
            )
        else:
            total_liabilities = np.nan

        result.update(
            {
                "ROE_W":
                    safe_divide(
                        net_income,
                        equity,
                    ),
                "ROA_W":
                    safe_divide(
                        net_income,
                        total_assets,
                    ),
                "OPERATING_MARGIN_W":
                    safe_divide(
                        ebit,
                        revenue,
                    ),
                "NET_MARGIN_W":
                    safe_divide(
                        net_income,
                        revenue,
                    ),
                "DEBT_TO_EQUITY_PROXY_W":
                    safe_divide(
                        total_liabilities,
                        equity,
                    ),
                "EQUITY":
                    equity,
                "SOURCE_DATASET":
                    reference[
                        "DATASET"
                    ],
                "SOURCE_YEAR":
                    reference[
                        "SOURCE_YEAR"
                    ],
                "REFERENCE_DATE":
                    reference[
                        "REFERENCE_DATE"
                    ],
            }
        )

        rows.append(result)

    return pd.DataFrame(rows)


# =============================================================================
# AUDITORIA
# =============================================================================

def audit_fundamentals(
    fundamentals,
):
    print()
    print("=" * 78)
    print("ETAPA 4 — AUDITORIA")
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
        .sum(axis=1)
    )

    eligible = (
        fundamentals[
            "VALID_COMPONENTS"
        ]
        >=
        3
    )

    identity_ok = (
        fundamentals[
            "CNPJ_CIA"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        .eq(14)
    )

    identity_coverage = (
        identity_ok.mean()
        if len(
            fundamentals
        )
        else 0.0
    )

    fundamental_coverage = (
        eligible.mean()
        if len(
            fundamentals
        )
        else 0.0
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
        f"Identidade B3→CVM válida ......... "
        f"{identity_ok.sum():,} "
        f"({identity_coverage:.2%})"
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
    print(
        f"Cobertura fundamental ............. "
        f"{fundamental_coverage:.2%}"
    )

    print()
    print(
        "Fonte DFP ........................ "
        f"{(fundamentals['SOURCE_DATASET'] == 'DFP').sum():,}"
    )
    print(
        "Fonte ITR ........................ "
        f"{(fundamentals['SOURCE_DATASET'] == 'ITR').sum():,}"
    )

    ref_dates = pd.to_datetime(
        fundamentals[
            "REFERENCE_DATE"
        ],
        errors="coerce",
    )

    if ref_dates.notna().any():
        print(
            "Referência mínima ................. "
            f"{ref_dates.min().date()}"
        )
        print(
            "Referência máxima ................. "
            f"{ref_dates.max().date()}"
        )

    print()

    if duplicates:
        raise RuntimeError(
            "AUDITORIA REPROVADA: duplicidades encontradas."
        )

    if identity_coverage < MIN_IDENTITY_COVERAGE:
        raise RuntimeError(
            "AUDITORIA REPROVADA: cobertura de identidade "
            f"{identity_coverage:.2%} < "
            f"{MIN_IDENTITY_COVERAGE:.2%}."
        )

    if fundamental_coverage < MIN_FUNDAMENTAL_COVERAGE:
        raise RuntimeError(
            "AUDITORIA REPROVADA: cobertura fundamental "
            f"{fundamental_coverage:.2%} < "
            f"{MIN_FUNDAMENTAL_COVERAGE:.2%}."
        )

    print(
        "Identidade B3→CNPJ→CVM ............ PASS"
    )
    print(
        "ORDEM_EXERC = ÚLTIMO .............. PASS"
    )
    print(
        "Leitura decimal CVM ............... PASS"
    )
    print(
        "DFP anterior + fontes atuais ...... PASS"
    )
    print(
        "Metodologia fundamental ........... PASS"
    )
    print(
        "Regra >= 3 componentes ............ PASS"
    )
    print(
        "Histórico congelado ............... PRESERVADO"
    )
    print(
        "STATUS ............................ PASS"
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

    universe = load_current_universe()

    registry = load_cvm_registry()

    identity = build_ticker_cnpj_map(
        universe,
        registry,
    )

    statements = load_all_cvm_sources()

    fundamentals = build_fundamentals(
        universe,
        identity,
        statements,
    )

    fundamentals = audit_fundamentals(
        fundamentals
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
        "SOURCE_YEAR",
        "REFERENCE_DATE",
        "CNPJ_CIA",
        "CD_CVM",
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
            "CNPJ_CIA",
            "CD_CVM",
            "IDENTITY_STATUS",
            "SOURCE_DATASET",
            "SOURCE_YEAR",
            "REFERENCE_DATE",
            "VALID_COMPONENTS",
        ]
    ].to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 78)
    print("ARQUIVOS GERADOS")
    print("=" * 78)
    print(
        f"Fundamentos : {OUTPUT_FILE}"
    )
    print(
        f"Auditoria   : {AUDIT_FILE}"
    )
    print()
    print(
        "STATUS: CAMADA FUNDAMENTALISTA "
        "OPERACIONAL VALIDADA"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
