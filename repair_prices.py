from pathlib import Path
import time
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DATA_LIVE = ROOT / "data_live"

PRICE_FILE = DATA_LIVE / "price_factors_current.csv"
AUDIT_FILE = DATA_LIVE / "price_repair_audit.csv"

MIN_OBS_DISCOUNT = 60

print("=" * 78)
print("PORTIFOLIO-B3 — REPARO DE PREÇOS OPERACIONAIS")
print("=" * 78)
print("Fonte      : Yahoo Finance / yfinance repair=True")
print("Metodologia: PRESERVADA")
print("Histórico   : NÃO ALTERADO")
print("=" * 78)

if not PRICE_FILE.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {PRICE_FILE}")

base = pd.read_csv(PRICE_FILE)

required = {"TICKER", "MOM_6M", "MOM_12M", "DISCOUNT_52W"}
missing = required - set(base.columns)
if missing:
    raise RuntimeError(f"Colunas ausentes: {sorted(missing)}")

base["TICKER"] = base["TICKER"].astype(str).str.upper().str.strip()

audit_rows = []
repaired_rows = []

def extract_series(hist, name):
    if isinstance(hist.columns, pd.MultiIndex):
        # yfinance pode devolver MultiIndex mesmo para um ticker.
        level0 = hist.columns.get_level_values(0)
        if name not in level0:
            return pd.Series(dtype=float)
        x = hist[name]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]
        return pd.to_numeric(x, errors="coerce")
    if name not in hist.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(hist[name], errors="coerce")

for i, row in base.iterrows():
    ticker = row["TICKER"]
    symbol = f"{ticker}.SA"

    try:
        hist = yf.download(
            symbol,
            period="18mo",
            interval="1d",
            auto_adjust=True,
            repair=True,
            progress=False,
            threads=False
        )

        close = extract_series(hist, "Close").dropna()

        if close.empty:
            repaired_rows.append({
                "TICKER": ticker,
                "MOM_6M_REPAIRED": np.nan,
                "MOM_12M_REPAIRED": np.nan,
                "DISCOUNT_52W_REPAIRED": np.nan
            })
            audit_rows.append({
                "TICKER": ticker,
                "STATUS": "NO_DATA",
                "OLD_DISCOUNT_52W": row["DISCOUNT_52W"],
                "NEW_DISCOUNT_52W": np.nan,
                "ABS_DIFF": np.nan
            })
            continue

        current = float(close.iloc[-1])

        mom6 = (
            current / float(close.iloc[-127]) - 1.0
            if len(close) >= 127 else np.nan
        )

        mom12 = (
            current / float(close.iloc[-253]) - 1.0
            if len(close) >= 253 else np.nan
        )

        last252 = close.tail(252)
        discount = np.nan
        if len(last252) >= MIN_OBS_DISCOUNT:
            high52 = float(last252.max())
            if high52 > 0:
                discount = (high52 - current) / high52

        old_discount = pd.to_numeric(
            pd.Series([row["DISCOUNT_52W"]]), errors="coerce"
        ).iloc[0]

        diff = (
            abs(float(discount) - float(old_discount))
            if pd.notna(discount) and pd.notna(old_discount)
            else np.nan
        )

        repaired_rows.append({
            "TICKER": ticker,
            "MOM_6M_REPAIRED": mom6,
            "MOM_12M_REPAIRED": mom12,
            "DISCOUNT_52W_REPAIRED": discount
        })

        audit_rows.append({
            "TICKER": ticker,
            "STATUS": "PASS",
            "OLD_DISCOUNT_52W": old_discount,
            "NEW_DISCOUNT_52W": discount,
            "ABS_DIFF": diff
        })

    except Exception as exc:
        repaired_rows.append({
            "TICKER": ticker,
            "MOM_6M_REPAIRED": np.nan,
            "MOM_12M_REPAIRED": np.nan,
            "DISCOUNT_52W_REPAIRED": np.nan
        })
        audit_rows.append({
            "TICKER": ticker,
            "STATUS": f"ERROR:{type(exc).__name__}",
            "OLD_DISCOUNT_52W": row["DISCOUNT_52W"],
            "NEW_DISCOUNT_52W": np.nan,
            "ABS_DIFF": np.nan
        })

    if (i + 1) % 50 == 0:
        print(f"Processados: {i + 1}/{len(base)}")

    time.sleep(0.02)

repair = pd.DataFrame(repaired_rows)
audit = pd.DataFrame(audit_rows)

out = base.merge(repair, on="TICKER", how="left", validate="one_to_one")

# Substitui somente quando o Yahoo reparado devolveu valor válido.
for original, repaired in [
    ("MOM_6M", "MOM_6M_REPAIRED"),
    ("MOM_12M", "MOM_12M_REPAIRED"),
    ("DISCOUNT_52W", "DISCOUNT_52W_REPAIRED"),
]:
    mask = out[repaired].notna()
    out.loc[mask, original] = out.loc[mask, repaired]

out = out.drop(columns=[
    "MOM_6M_REPAIRED",
    "MOM_12M_REPAIRED",
    "DISCOUNT_52W_REPAIRED"
])

if out["TICKER"].duplicated().any():
    raise RuntimeError("Duplicidades após reparo.")

if len(out) != len(base):
    raise RuntimeError("Quantidade de linhas mudou após reparo.")

# O arquivo operacional é atualizado; data/ histórico permanece intocado.
out.to_csv(PRICE_FILE, index=False)
audit.to_csv(AUDIT_FILE, index=False)

changed = int((audit["ABS_DIFF"] > 0.02).fillna(False).sum())
no_data = int((audit["STATUS"] == "NO_DATA").sum())
errors = int(audit["STATUS"].astype(str).str.startswith("ERROR").sum())

print()
print("=" * 78)
print("AUDITORIA")
print("=" * 78)
print(f"Tickers processados ............... {len(base)}")
print(f"Alterações relevantes (>2 p.p.) ... {changed}")
print(f"Sem dados reparados ............... {no_data}")
print(f"Erros ............................. {errors}")
print("MOM_6M = 126 pregões .............. PRESERVADO")
print("MOM_12M = 252 pregões ............. PRESERVADO")
print("DISCOUNT_52W ...................... PRESERVADO")
print("Classificação setorial ............ PRESERVADA")
print("Histórico congelado ............... PRESERVADO")
print()
print(f"Preços corrigidos : {PRICE_FILE}")
print(f"Auditoria          : {AUDIT_FILE}")
print()
print("STATUS: REPARO OPERACIONAL DE PREÇOS CONCLUÍDO")
print("=" * 78)
