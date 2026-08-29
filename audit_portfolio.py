from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA_LIVE = ROOT / "data_live"

PORTFOLIO_FILE = DATA_LIVE / "portfolio_current.csv"
OUT_FILE = DATA_LIVE / "portfolio_extreme_audit.csv"

EXTREME_DISCOUNT = 0.80
MIN_OBS_52W = 60
STALE_BUSINESS_DAYS = 7

print("=" * 78)
print("PORTIFOLIO-B3 — AUDITORIA DE PREÇOS EXTREMOS")
print("=" * 78)

if not PORTFOLIO_FILE.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {PORTFOLIO_FILE}")

portfolio = pd.read_csv(PORTFOLIO_FILE)
required = {"TICKER", "MACRO_SECTOR", "DISCOUNT_52W", "FINAL_SCORE"}

missing = required - set(portfolio.columns)
if missing:
    raise RuntimeError(f"Colunas ausentes em portfolio_current.csv: {sorted(missing)}")

portfolio["TICKER"] = portfolio["TICKER"].astype(str).str.upper().str.strip()
portfolio["DISCOUNT_52W"] = pd.to_numeric(portfolio["DISCOUNT_52W"], errors="coerce")

rows = []

for ticker in portfolio["TICKER"]:
    symbol = f"{ticker}.SA"
    print(f"Auditando {ticker}...")

    try:
        hist = yf.download(
            symbol,
            period="18mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if hist.empty:
            rows.append({
                "TICKER": ticker,
                "STATUS": "REVIEW",
                "REASON": "SEM_DADOS_YAHOO"
            })
            continue

        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)

        hist = hist.dropna(subset=["Close"]).copy()

        if hist.empty:
            rows.append({
                "TICKER": ticker,
                "STATUS": "REVIEW",
                "REASON": "SEM_FECHAMENTOS_VALIDOS"
            })
            continue

        last252 = hist.tail(252).copy()
        current_price = float(last252["Close"].iloc[-1])
        high_52w = float(last252["Close"].max())
        discount_calc = (
            (high_52w - current_price) / high_52w
            if high_52w > 0 else np.nan
        )

        last_date = pd.Timestamp(last252.index[-1]).tz_localize(None)
        today = pd.Timestamp.today().normalize()
        stale_bdays = int(np.busday_count(
            last_date.date(),
            today.date()
        )) if last_date.date() < today.date() else 0

        obs = int(last252["Close"].notna().sum())

        if "Volume" in last252.columns:
            turnover = (
                pd.to_numeric(last252["Close"], errors="coerce")
                * pd.to_numeric(last252["Volume"], errors="coerce")
            )
            median_turnover_60d = float(turnover.tail(60).median())
        else:
            median_turnover_60d = np.nan

        original_discount = float(
            portfolio.loc[
                portfolio["TICKER"] == ticker, "DISCOUNT_52W"
            ].iloc[0]
        )

        diff = (
            abs(discount_calc - original_discount)
            if pd.notna(discount_calc) and pd.notna(original_discount)
            else np.nan
        )

        reasons = []

        if obs < MIN_OBS_52W:
            reasons.append("HISTORICO_INSUFICIENTE")

        if stale_bdays > STALE_BUSINESS_DAYS:
            reasons.append("COTACAO_DESATUALIZADA")

        if pd.notna(discount_calc) and discount_calc >= EXTREME_DISCOUNT:
            reasons.append("DESCONTO_EXTREMO_80PCT")

        if pd.notna(diff) and diff > 0.02:
            reasons.append("DIVERGENCIA_DISCOUNT")

        if current_price <= 0:
            reasons.append("PRECO_INVALIDO")

        status = "REVIEW" if reasons else "PASS"

        rows.append({
            "TICKER": ticker,
            "LAST_DATE": last_date.date().isoformat(),
            "OBS_52W": obs,
            "CURRENT_PRICE_RAW": current_price,
            "HIGH_52W_RAW": high_52w,
            "DISCOUNT_ENGINE": original_discount,
            "DISCOUNT_RECALCULATED": discount_calc,
            "DISCOUNT_DIFF": diff,
            "MEDIAN_TURNOVER_60D": median_turnover_60d,
            "STALE_BUSINESS_DAYS": stale_bdays,
            "STATUS": status,
            "REASON": "|".join(reasons) if reasons else "OK"
        })

    except Exception as e:
        rows.append({
            "TICKER": ticker,
            "STATUS": "REVIEW",
            "REASON": f"ERRO_COLETA:{type(e).__name__}"
        })

audit = pd.DataFrame(rows)

base = portfolio[
    ["TICKER", "MACRO_SECTOR", "DISCOUNT_52W", "FINAL_SCORE"]
].copy()

audit = base.merge(
    audit.drop(columns=["DISCOUNT_ENGINE"], errors="ignore"),
    on="TICKER",
    how="left",
    validate="one_to_one"
)

audit.to_csv(OUT_FILE, index=False)

print()
print("=" * 78)
print("RESULTADO")
print("=" * 78)

cols = [
    "TICKER", "MACRO_SECTOR", "DISCOUNT_52W",
    "DISCOUNT_RECALCULATED", "CURRENT_PRICE_RAW",
    "HIGH_52W_RAW", "OBS_52W", "MEDIAN_TURNOVER_60D",
    "STATUS", "REASON"
]

print(audit[cols].to_string(index=False))

n_pass = int((audit["STATUS"] == "PASS").sum())
n_review = int((audit["STATUS"] == "REVIEW").sum())

print()
print("=" * 78)
print("AUDITORIA")
print("=" * 78)
print(f"Ações analisadas .................. {len(audit)}")
print(f"PASS .............................. {n_pass}")
print(f"REVIEW ............................ {n_review}")
print("Preço base = Close bruto (PREULT) . PRESERVADO")
print("Regra de seleção 80/20 ............ NÃO ALTERADA")
print("Portfólio atual ................... NÃO ALTERADO")
print("Histórico congelado ............... PRESERVADO")
print()
print(f"Arquivo: {OUT_FILE}")
print()
print("STATUS: AUDITORIA DIAGNÓSTICA CONCLUÍDA")
print("=" * 78)
