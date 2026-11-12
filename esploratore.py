# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "pandas",
#     "torch",
#     "timestral @ git+https://github.com/mistralai/timestral.git@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff",
# ]
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Esploratore: un bot che prova, misura e si corregge da solo. Conto demo Alpaca (Paper), nel cloud (GitHub Actions).

L'idea in parole semplici: più strategie sono in gara. Ogni 15 minuti, su ogni mercato senza operazioni aperte, ogni
strategia dice "entrerei" oppure no. Fra quelle che entrerebbero il bot ne sceglie una pescando a sorte, ma con una
sorte truccata dai risultati: ogni strategia ha un punteggio (quanto ha reso in media, costi pagati) e un'incertezza
(quante volte l'ha provata). Pesca un numero vicino al punteggio, tanto più sparso quanto meno la conosce
(campionamento di Thompson). Vince il numero più alto; se è sotto zero, il bot resta fuori.
Così all'inizio entra quasi sempre (non sa niente, prova tutto) e col passare delle operazioni usa di più le strategie
che guadagnano e sempre meno quelle che perdono, senza spegnerle mai del tutto: se il mercato cambia se ne accorge.
Ogni operazione chiusa entra nel registro, e il registro è la sua memoria: al giro dopo riparte da lì.

Strategie (sulle candele da 15 minuti fatte coi minuti di Alpaca):
- tendenza: prezzo sopra la media di 20 candele, e quella sopra la media di 50 (il mercato sale da un po');
- rimbalzo: RSI sotto 30 (è sceso tanto e in fretta, spesso rimbalza);
- rottura:  prezzo sopra il massimo delle ultime 20 candele (esce dalla gabbia in su);
- slancio:  prezzo più alto di un'ora fa;
- timestral: Timestral-5M (Mistral) guarda gli ultimi 1024 minuti e prevede il prezzo fra 60 minuti: entra se la
  previsione centrale sta sopra prezzo + costi;
- caso: entra sempre. È il metro: una strategia che non batte "caso" non sa niente.
Uscita uguale per tutte, così si confrontano alla pari: stop a 1,5 volte il movimento medio di una candela (ATR) sotto
l'entrata, obiettivo a 2 volte sopra, e comunque dopo 60 minuti. Sulle azioni si chiude anche 15 minuti prima della
chiusura della Borsa.

Mercati: solo simboli che gli altri bot non toccano (Alpaca tiene una sola posizione per simbolo e il conto è lo stesso):
QQQ (Nasdaq) e GLD (oro) a Borsa USA aperta, niente commissioni; SOL/USD (crypto) 24 ore su 24, 0,25% a ordine.
Ogni operazione investe il 2% del conto; al massimo tre insieme (6%). Kill switch: se nel giorno (UTC) l'esploratore
ha perso più dell'1% del conto, niente operazioni nuove fino al giorno dopo. Solo acquisti.
Il punteggio è "netto": risultato dai prezzi eseguiti meno i costi stimati (0,6% andata e ritorno su SOL, 0,01% sulle
azioni), così impara anche quanto costa entrare, comunque il paper registri le commissioni.

    uv run esploratore.py --prova   # prove della logica, niente rete
    uv run esploratore.py           # il bot (servono le chiavi APCA_API_KEY_ID / APCA_API_SECRET_KEY)
"""
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd

import bot  # helper del bot Timestral: chiamate ad Alpaca, ordini con ID unico, minuti, salvataggio del registro

MARKETS = {  # simbolo: (azione?, costi stimati andata e ritorno in %, oscillazione tipica di un'ora in %)
    "QQQ": (True, 0.01, 0.3),
    "GLD": (True, 0.01, 0.3),
    "SOL/USD": (False, 0.6, 1.0),
}
ARMS = ["tendenza", "rimbalzo", "rottura", "slancio", "timestral", "caso"]
SIZE, MAX_DAILY_LOSS = 0.02, 0.01
HOLD = 60          # minuti massimi di un'operazione (= 4 candele da 15)
N0 = 2             # quanto pesa il "non so niente" iniziale, in operazioni finte a risultato zero
MINUTES = int(os.environ.get("MINUTI", 330))
OPS, OPEN = bot.LOGS / "esploratore.csv", bot.LOGS / "esploratore_aperte.json"
FIELDS = ["apertura", "chiusura", "mercato", "strategia", "segnali", "punteggio", "investito", "entrata", "uscita",
          "motivo", "lordo_pct", "costi_pct", "netto_pct", "netto_usd"]


def ema(x, n):
    return x.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff()
    up, dn = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(), (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def atr(b, n=14):
    prev = b.c.shift()
    tr = pd.concat([b.h - b.l, (b.h - prev).abs(), (b.l - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean().iloc[-1]


def bars(closes, hl):
    """Candele da 15 minuti chiuse, dai minuti (i minuti senza scambi prendono il prezzo di prima)."""
    df = pd.DataFrame({"c": closes, "h": hl.h.fillna(closes), "l": hl.l.fillna(closes)})
    return df.resample("15min").agg({"c": "last", "h": "max", "l": "min"}).dropna()


def signals(b):
    """Quali strategie (tranne timestral) entrerebbero adesso, sulle candele chiuse b."""
    c = b.c
    e20, e50 = ema(c, 20).iloc[-1], ema(c, 50).iloc[-1]
    return {"tendenza": bool(c.iloc[-1] > e20 > e50), "rimbalzo": bool(rsi(c).iloc[-1] < 30),
            "rottura": bool(c.iloc[-1] > b.h.iloc[-21:-1].max()), "slancio": bool(c.iloc[-1] > c.iloc[-5]),
            "caso": True}


def history():
    try:
        with OPS.open() as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def choose(market, candidates, rows, rng):
    """Campionamento di Thompson sul rendimento netto medio (%). Ritorna (strategia, numero pescato); strategia None
    se il numero più alto è sotto zero (meglio stare fuori)."""
    sd0 = MARKETS[market][2]
    best, pick = -np.inf, None
    for a in candidates:
        r = [float(x["netto_pct"]) for x in rows if x["mercato"] == market and x["strategia"] == a]
        n = len(r)
        sd = max(float(np.std(r)) if n > 1 else 0.0, sd0)
        x = rng.normal(sum(r) / (n + N0), sd / np.sqrt(n + N0))
        if x > best:
            best, pick = x, a
    return (pick if best > 0 else None), best


def exit_reason(t, h, l, now, closing):
    """Perché chiudere l'operazione t col minuto chiuso (massimo h, minimo l), o "" per restare dentro."""
    if l <= t["stop"]:
        return "stop"
    if h >= t["obiettivo"]:
        return "obiettivo"
    if now >= pd.Timestamp(t["apertura"], tz="UTC") + timedelta(minutes=HOLD):
        return "tempo"
    return "chiusura Borsa" if closing else ""


def order(sym, side, cid, **amount):
    o = {"symbol": sym, "side": side, "type": "market", "time_in_force": "day" if MARKETS[sym][0] else "gtc",
         "client_order_id": cid, **amount}
    return bot.execute(o)


def held(sym):
    p = bot.api("GET", f"/v2/positions/{sym.replace('/', '')}", missing_ok=True)
    return float(p["qty"]) if p and float(p["market_value"]) > 1 else 0.0


def save_open(op):
    bot.LOGS.mkdir(exist_ok=True)
    OPEN.write_text(json.dumps(op, indent=1))


def close(sym, t, why, now, op):
    q = held(sym)
    x = order(sym, "sell", f"es-{sym.replace('/', '')}-out-{now:%Y%m%d%H%M}", qty=f"{q:.9f}") if q else None
    if x and x["status"] != "filled":
        print(f"{sym}: vendita non eseguita ({x['status']}), riprovo al prossimo minuto", flush=True)
        return
    out = float(x["filled_avg_price"]) if x else t["entrata"]
    gross = (out / t["entrata"] - 1) * 100
    cost = MARKETS[sym][1]
    row = {"apertura": t["apertura"], "chiusura": f"{now:%Y-%m-%d %H:%M}", "mercato": sym, "strategia": t["strategia"],
           "segnali": t["segnali"], "punteggio": round(t["punteggio"], 4), "investito": round(t["investito"], 2),
           "entrata": t["entrata"], "uscita": out, "motivo": why if x else f"{why} (posizione già chiusa)",
           "lordo_pct": round(gross, 4), "costi_pct": cost, "netto_pct": round(gross - cost, 4),
           "netto_usd": round(t["investito"] * (gross - cost) / 100, 2)}
    new = not OPS.exists()
    with OPS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(row, flush=True)
    del op[sym]
    save_open(op)
    bot.push(f"esploratore {sym} {t['strategia']} {row['netto_pct']:+.2f}%")


def enter(sym, now, equity, cash, op, rows, rng, model):
    stock = MARKETS[sym][0]
    closes, hl = bot.minutes(now, bot.CTX, sym, stock)
    if len(closes) < bot.CTX:
        raise RuntimeError(f"solo {len(closes)} minuti di storia")
    b = bars(closes, hl)
    sig = signals(b)
    P = float(closes.iloc[-1])
    q = np.asarray(model.forecast(np.ascontiguousarray(closes.to_numpy()[None], dtype=np.float32),
                                  prediction_length=HOLD).quantiles[:, 0])[0, HOLD - 1]
    sig["timestral"] = bool(q[bot.Q50] > P * (1 + MARKETS[sym][1] / 100))
    cand = [a for a in ARMS if sig[a]]
    arm, score = choose(sym, cand, rows, rng)
    print(f"{now:%H:%M} {sym} {P:.2f} segnali {','.join(cand)} → {arm or 'fuori'} ({score:+.3f})", flush=True)
    if not arm:
        return
    amount = min(SIZE * equity, cash * 0.9)
    o = order(sym, "buy", f"es-{sym.replace('/', '')}-in-{arm}-{now:%Y%m%d%H%M}", notional=f"{amount:.2f}")
    if o["status"] != "filled":
        print(f"{sym}: acquisto non eseguito ({o['status']})", flush=True)
        return
    E, a = float(o["filled_avg_price"]), atr(b)
    op[sym] = {"apertura": f"{now:%Y-%m-%d %H:%M}", "strategia": arm, "segnali": " ".join(cand), "punteggio": score,
               "investito": float(o["filled_qty"]) * E, "entrata": E, "stop": E - 1.5 * a, "obiettivo": E + 2 * a}
    save_open(op)
    bot.push(f"esploratore {sym} entra: {arm}")


def notify(msg):
    print(msg, flush=True)
    if os.environ.get("GITHUB_ACTIONS"):
        subprocess.run(["gh", "issue", "create", "--title", f"Esploratore: {msg}"[:250], "--body", msg,
                        "--assignee", os.environ["GITHUB_REPOSITORY_OWNER"]], capture_output=True)


def main():
    if not os.environ.get("APCA_API_KEY_ID"):
        sys.exit("Mancano le chiavi APCA_API_KEY_ID / APCA_API_SECRET_KEY (nel cloud: Secrets del repository).")
    for sg in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sg, lambda *_: bot.STOP.append(1))
    from timestral import TimestralForecaster
    model = TimestralForecaster.from_pretrained("mistralai/Timestral-5M")
    rng = np.random.default_rng()
    # Riconciliazione: lo stato vero è quello del broker. Un'operazione del giro prima si riprende; un simbolo
    # nostro comprato senza registro (giro interrotto prima di salvare) si vende.
    try:
        op = json.loads(OPEN.read_text())
    except (FileNotFoundError, ValueError):
        op = {}
    for sym in MARKETS:
        for o in bot.api("GET", f"/v2/orders?status=open&symbols={sym}") or []:
            bot.api("DELETE", f"/v2/orders/{o['id']}")
        if sym in op and not held(sym):
            close(sym, op[sym], "riavvio", pd.Timestamp.now("UTC"), op)
        elif sym not in op and held(sym):
            q = held(sym)
            order(sym, "sell", f"es-{sym.replace('/', '')}-start-{pd.Timestamp.now('UTC'):%Y%m%d%H%M}", qty=f"{q:.9f}")
            notify(f"all'avvio c'erano {sym} senza registro: venduti.")
    end, stopped_day = time.time() + MINUTES * 60, None
    print(f"Esploratore avviato su {bot.BASE}, mercati {', '.join(MARKETS)}, {MINUTES} minuti; aperte: {list(op)}", flush=True)
    while not bot.STOP and time.time() < end:
        bot.wait_minute()
        if bot.STOP:
            break
        now = pd.Timestamp.now("UTC")
        try:
            clock = bot.api("GET", "/v2/clock")
            acct = bot.api("GET", "/v2/account")
            equity, cash = float(acct["equity"]), float(acct["non_marginable_buying_power"])
            rows = history()
            lost = sum(float(r["netto_usd"]) for r in rows if r["chiusura"].startswith(f"{now:%Y-%m-%d}"))
            if lost < -MAX_DAILY_LOSS * equity and stopped_day != f"{now:%Y-%m-%d}":
                stopped_day = f"{now:%Y-%m-%d}"
                notify(f"kill switch: oggi {lost:.2f} $. Niente operazioni nuove fino a domani (UTC).")
        except Exception as e:
            print(f"ERRORE: {e}", flush=True)
            continue
        closing = pd.Timestamp(clock["next_close"]) - now < timedelta(minutes=15)
        for sym, (stock, _, _) in MARKETS.items():
            try:
                if sym in op:
                    _, hl = bot.minutes(now, 3, sym, stock)
                    bar = hl.iloc[-1]
                    if not bar.isna().any():
                        why = exit_reason(op[sym], bar.h, bar.l, now, stock and closing)
                        if why:
                            close(sym, op[sym], why, now, op)
                    continue
                if now.minute % 15 or stopped_day == f"{now:%Y-%m-%d}":
                    continue
                if stock and (not clock["is_open"] or now.tz_convert(bot.NY).time() < pd.Timestamp("09:45").time()
                              or pd.Timestamp(clock["next_close"]) - now < timedelta(minutes=30)):
                    continue
                enter(sym, now, equity, cash, op, rows, rng, model)
            except Exception as e:  # rete, dati, broker: si salta questo mercato e si riprova al prossimo minuto
                print(f"ERRORE {sym}: {e}", flush=True)
    print(f"Esploratore fermo; operazioni aperte che il giro dopo riprende: {list(op)}", flush=True)


def prova():
    rng = np.random.default_rng(0)
    # Senza storia non sa niente: entra quasi sempre (prova tutto).
    picks = [choose("QQQ", ARMS, [], rng)[0] for _ in range(1000)]
    assert sum(p is not None for p in picks) > 950, "senza storia deve provare quasi sempre"
    # Con una strategia che guadagna e una che perde, sceglie quasi sempre quella che guadagna.
    rows = [{"mercato": "QQQ", "strategia": "tendenza", "netto_pct": str(v)} for v in rng.normal(0.3, 0.3, 30)]
    rows += [{"mercato": "QQQ", "strategia": "caso", "netto_pct": str(v)} for v in rng.normal(-0.3, 0.3, 30)]
    picks = [choose("QQQ", ["tendenza", "caso"], rows, rng)[0] for _ in range(1000)]
    assert picks.count("tendenza") > 950 and picks.count("caso") < 20, picks.count("caso")
    # Se tutto quello che ha provato perde, quasi sempre resta fuori; un'altra moneta non c'entra.
    picks = [choose("QQQ", ["caso"], rows, rng)[0] for _ in range(1000)]
    assert picks.count(None) > 980
    assert sum(choose("GLD", ["caso"], rows, rng)[0] is not None for _ in range(1000)) > 400
    # Strategie su candele finte: salita regolare → tendenza, slancio, rottura; crollo → rimbalzo.
    idx = pd.date_range("2026-11-02 00:00", periods=1024, freq="min", tz="UTC")
    up = pd.Series(np.linspace(100, 110, 1024), idx)
    s = signals(bars(up, pd.DataFrame({"h": up + 0.01, "l": up - 0.01})))
    assert s["tendenza"] and s["slancio"] and s["rottura"] and not s["rimbalzo"] and s["caso"]
    down = pd.Series(np.r_[np.full(900, 100.0), np.linspace(100, 90, 124)], idx)
    s = signals(bars(down, pd.DataFrame({"h": down, "l": down})))
    assert s["rimbalzo"] and not s["tendenza"] and not s["slancio"] and not s["rottura"]
    b = bars(up, pd.DataFrame({"h": up + 0.01, "l": up - 0.01}))
    assert len(b) == 69 and b.index[-1] == pd.Timestamp("2026-11-02 17:00", tz="UTC") and 0 < atr(b) < 1
    # Uscite.
    t = {"apertura": "2026-11-12 10:00", "stop": 99.0, "obiettivo": 102.0}
    at = pd.Timestamp("2026-11-12 10:20", tz="UTC")
    assert exit_reason(t, 101, 99.5, at, False) == ""
    assert exit_reason(t, 102.5, 98, at, False) == "stop"            # stop e obiettivo insieme: vince lo stop
    assert exit_reason(t, 102.5, 100, at, False) == "obiettivo"
    assert exit_reason(t, 101, 100, at + timedelta(minutes=40), False) == "tempo"
    assert exit_reason(t, 101, 100, at, True) == "chiusura Borsa"
    print("prove superate")


if __name__ == "__main__":
    prova() if "--prova" in sys.argv else main()
