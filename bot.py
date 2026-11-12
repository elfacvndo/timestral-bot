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
"""Bot Timestral: microoperazioni su ETH/USD, conto demo Alpaca (Paper). Gira nel cloud (GitHub Actions).

Ogni minuto, sulla candela da 1 minuto appena chiusa, Timestral-5M (Mistral) guarda gli ultimi 1024 minuti di prezzo e
dice dove potrebbe essere ETH fra 15 minuti, come una fascia di 21 livelli. Regole fisse (le stesse dell'esperimento
su SPY, vedi DIARIO.md "Esperimento Timestral su SPY"):
- compra se la probabilità che fra 15 minuti il prezzo stia sopra entrata + costi è almeno il 50% (su SPY era l'80%;
  su ETH all'80% non entra mai: nella settimana fino al 2026-11-12 il massimo è stato 56%, 9 minuti su 10.065 sopra il 50%);
- stop al livello 10% della fascia; tre pezzi con TP1 alla mediana, TP2 al livello 75%, TP3 al 90%;
  preso TP1 lo stop va all'entrata, preso TP2 va su TP1; se nello stesso minuto si toccano stop e obiettivo, vince lo stop;
- dopo 15 minuti si chiude comunque.
Solo acquisti: Alpaca non permette di vendere allo scoperto le crypto. Costi: 0,25% a ordine (Alpaca, ordini a mercato)
più 0,05% di slittamento, cioè 0,55% andata e ritorno.
Lo stesso conto del bot orario (che opera su BTC): questo usa solo ETH e al massimo il 10% del conto, rischia l'1% di
quella quota a stop pieno, e si ferma fino al giorno dopo se nel giorno ha perso il 5% della quota.

    uv run bot.py --prova     # prove delle regole, niente rete
    uv run bot.py             # il bot (servono le chiavi APCA_API_KEY_ID / APCA_API_SECRET_KEY)
"""
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL, POSITION = "ETH/USD", "/v2/positions/ETHUSD"
BASE = "https://paper-api.alpaca.markets"  # PAPER: solo soldi finti. Non cambiare.
BARS = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
H, THETA, CTX = 15, 0.5, 1024   # soglia 50%: scelta per vederlo operare, NON misurata (all'80% non entra mai)
COST = 0.0055                 # andata e ritorno, in frazione del prezzo
FEE = 0.0025                  # commissione Alpaca su ogni ordine a mercato
SHARE, RISK, MAX_DAILY_LOSS = 0.10, 0.01, 0.05
LEVELS = np.array([0.01, *np.round(np.arange(0.05, 0.951, 0.05), 2), 0.99])
Q10, Q50, Q75, Q90 = 2, 10, 15, 18
MINUTES = int(os.environ.get("MINUTI", 330))  # quanto dura un giro: GitHub ferma un lavoro dopo 6 ore
LOGS = Path(__file__).parent / "logs"
OPS, DAY = LOGS / "operazioni.csv", LOGS / "giorno.json"
FIELDS = ["apertura", "chiusura", "probabilita", "investito", "entrata", "stop", "tp1", "tp2", "tp3", "uscite",
          "risultato_usd", "risultato_pct", "motivo"]
STOP = []


def get_json(req, tries=3):
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                raise
            err = e
        except (OSError, ValueError) as e:
            err = e
        if attempt < tries - 1:
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"nessuna risposta dopo {tries} tentativi: {err!r}")


def keys():
    return {"APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"]}


def api(method, path, body=None, missing_ok=False, tries=3):
    req = urllib.request.Request(BASE + path, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={**keys(), "Content-Type": "application/json"})
    try:
        return get_json(req, tries)
    except urllib.error.HTTPError as e:
        if e.code == 404 and missing_ok:
            return None
        raise RuntimeError(f"{method} {path} → {e.code} {e.read().decode()}") from e


def execute(order):
    """Ordine a mercato con ID unico. Se la risposta si perde, prima di riprovare chiede ad Alpaca se è arrivato."""
    for attempt in range(3):
        try:
            o = api("POST", "/v2/orders", order, tries=1)
            break
        except RuntimeError:
            o = api("GET", f"/v2/orders:by_client_order_id?client_order_id={order['client_order_id']}", missing_ok=True)
            if o:
                break
            if attempt == 2:
                raise
            time.sleep(5)
    for _ in range(10):
        if o["status"] in ("filled", "canceled", "expired", "rejected"):
            break
        time.sleep(2)
        o = api("GET", f"/v2/orders/{o['id']}")
    return o


def market(side, cid, qty=None, notional=None):
    o = {"symbol": SYMBOL, "side": side, "type": "market", "time_in_force": "gtc", "client_order_id": cid}
    o.update({"qty": qty} if qty else {"notional": notional})
    return execute(o)


def held():
    p = api("GET", POSITION, missing_ok=True)
    return float(p["qty"]) if p and float(p["market_value"]) > 10 else 0.0


def minutes(now, n):
    """Le ultime n candele da 1 minuto CHIUSE prima di `now` (il minuto in corso si scarta). I minuti senza scambi
    non hanno candela: si riempiono col prezzo di prima. Se i dati sono vecchi, meglio non decidere."""
    end = now.floor("min")
    start = end - timedelta(minutes=n + 30)
    rows, token = [], None
    while True:
        p = {"symbols": SYMBOL, "timeframe": "1Min", "start": f"{start:%Y-%m-%dT%H:%M:%SZ}", "limit": 10000}
        if token:
            p["page_token"] = token
        page = get_json(urllib.request.Request(f"{BARS}?{urllib.parse.urlencode(p)}", headers=keys()))
        rows += page["bars"].get(SYMBOL, [])
        token = page.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(df.t)
    df = df[df.index < end]
    if df.empty or df.index[-1] < end - timedelta(minutes=10):
        raise RuntimeError(f"dati vecchi: ultima candela {df.index[-1] if len(df) else 'nessuna'}")
    full = pd.date_range(end - timedelta(minutes=n), end - timedelta(minutes=1), freq="min", tz="UTC")
    c = df.c.reindex(full).ffill()
    return c.dropna(), df[["h", "l"]].reindex(full)


def decide(P, q):
    """Il ragionamento sulla fascia q (21 prezzi crescenti) col prezzo attuale P.
    Ritorna (probabilità, stop in dollari sotto l'entrata, obiettivi in multipli dello stop) o None per stare fuori."""
    up = 1 - np.interp(P * (1 + COST), q, LEVELS)   # probabilità di guadagnare comprando, costi pagati
    if up < THETA:
        return up, None
    s = max(P - q[Q10], 0.0001 * P)
    return up, (s, tuple((q[[Q50, Q75, Q90]] - P) / s))


def step(plan, hi, lo):
    """Un minuto chiuso con massimo hi e minimo lo. Ritorna quanti dei tre pezzi vendere e perché; aggiorna plan."""
    k = plan["k"]
    stop = (plan["entrata"] - plan["s"], plan["entrata"], plan["tp"][0])[k]
    if lo <= stop:
        plan["k"] = 3
        return 3 - k, "stop"
    while plan["k"] < 3 and hi >= plan["tp"][plan["k"]]:
        plan["k"] += 1
    return plan["k"] - k, f"TP{plan['k']}" if plan["k"] > k else ""


def wait_minute():
    """Fino a 5 secondi dopo il prossimo minuto pieno: la candela appena finita è chiusa e pubblicata."""
    t = time.time()
    while not STOP and time.time() < t - t % 60 + 65:
        time.sleep(1)


def push(msg):
    """Nel cloud salva il registro nel repository: così si legge su GitHub mentre il bot gira."""
    if not os.environ.get("GITHUB_ACTIONS"):
        return
    for cmd in (["git", "add", "logs"], ["git", "commit", "-q", "-m", msg], ["git", "pull", "-q", "--rebase"], ["git", "push", "-q"]):
        if subprocess.run(cmd, capture_output=True).returncode:
            return


def notify(msg):
    print(msg, flush=True)
    if os.environ.get("GITHUB_ACTIONS"):
        subprocess.run(["gh", "issue", "create", "--title", f"Timestral: {msg}"[:250], "--body", msg,
                        "--assignee", os.environ["GITHUB_REPOSITORY_OWNER"]], capture_output=True)


def today(now):
    try:
        d = json.loads(DAY.read_text())
    except (FileNotFoundError, ValueError):
        d = {}
    return d if d.get("giorno") == f"{now:%Y-%m-%d}" else {"giorno": f"{now:%Y-%m-%d}", "risultato": 0.0, "fermo": False}


def record(row):
    LOGS.mkdir(exist_ok=True)
    new = not OPS.exists()
    with OPS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(row, flush=True)


def trade(now, P, prob, s, tps, equity, cash):
    """Un'operazione intera: compra, segue stop e obiettivi minuto per minuto, chiude entro 15 minuti."""
    budget = equity * SHARE
    amount = min(budget, RISK * budget * P / s, cash * 0.98)
    tag = f"{now:%Y%m%d%H%M}"
    o = market("buy", f"ts-buy-{tag}", notional=f"{amount:.2f}")
    if o["status"] != "filled":
        print(f"acquisto non eseguito: {o['status']}", flush=True)
        return 0.0
    E, spent = float(o["filled_avg_price"]), float(o["filled_qty"]) * float(o["filled_avg_price"])
    plan = {"entrata": E, "s": s, "tp": [E + t * s for t in tps], "k": 0}
    qty, piece = held(), round(float(o["filled_qty"]) / 3, 9)
    got, exits, why = 0.0, [], "tempo"
    entered = now.floor("min")
    for minute in range(1, H + 1):
        wait_minute()
        if STOP:
            n, why = 3 - plan["k"], "bot fermato"
        else:
            _, hl = minutes(pd.Timestamp.now("UTC"), 3)
            bar = hl.iloc[-1]
            n, why = (0, "") if bar.isna().any() else step(plan, bar.h, bar.l)
            if minute == H and plan["k"] < 3:   # 15 minuti passati: si chiude il resto
                n, why, plan["k"] = n + 3 - plan["k"], why or "tempo", 3
        if n:
            last = plan["k"] == 3 or STOP
            q = f"{held():.9f}" if last else f"{min(n * piece, qty):.9f}"
            x = market("sell", f"ts-sell-{tag}-{minute}", qty=q)
            if x["status"] == "filled":
                got += float(x["filled_qty"]) * float(x["filled_avg_price"]) * (1 - FEE)
                exits.append(f"{why} {float(x['filled_avg_price']):.2f}")
        if plan["k"] == 3 or STOP:
            break
    pnl = got - spent
    record({"apertura": f"{entered:%Y-%m-%d %H:%M}", "chiusura": f"{pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M}",
            "probabilita": round(prob, 3), "investito": round(spent, 2), "entrata": round(E, 2), "stop": round(E - s, 2),
            "tp1": round(plan["tp"][0], 2), "tp2": round(plan["tp"][1], 2), "tp3": round(plan["tp"][2], 2),
            "uscite": "; ".join(exits), "risultato_usd": round(pnl, 2), "risultato_pct": round(pnl / spent * 100, 3),
            "motivo": why})
    return pnl


def main():
    if not os.environ.get("APCA_API_KEY_ID"):
        sys.exit("Mancano le chiavi APCA_API_KEY_ID / APCA_API_SECRET_KEY (nel cloud: Secrets del repository).")
    for sg in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sg, lambda *_: STOP.append(1))
    from timestral import TimestralForecaster
    model = TimestralForecaster.from_pretrained("mistralai/Timestral-5M")
    # Riconciliazione: lo stato vero è quello del broker. ETH rimasti da un giro interrotto si vendono.
    for o in api("GET", "/v2/orders?status=open&limit=500"):
        if o["symbol"].replace("/", "") == "ETHUSD":
            api("DELETE", f"/v2/orders/{o['id']}")
    if held():
        x = market("sell", f"ts-start-{pd.Timestamp.now('UTC'):%Y%m%d%H%M}", qty=f"{held():.9f}")
        notify(f"All'avvio c'erano ETH di un giro interrotto: venduti ({x['status']}).")
    end = time.time() + MINUTES * 60
    print(f"Bot Timestral avviato su {BASE} — {SYMBOL}, {MINUTES} minuti.", flush=True)
    while not STOP and time.time() < end:
        wait_minute()
        if STOP:
            break
        now = pd.Timestamp.now("UTC")
        try:
            acct = api("GET", "/v2/account")
            equity, day = float(acct["equity"]), today(now)
            if day["fermo"]:
                continue
            closes, _ = minutes(now, CTX)
            if len(closes) < CTX:
                raise RuntimeError(f"solo {len(closes)} minuti di storia")
            ctx = np.ascontiguousarray(closes.to_numpy()[None], dtype=np.float32)
            q = np.asarray(model.forecast(ctx, prediction_length=H).quantiles[:, 0])[0, H - 1]
            P = float(closes.iloc[-1])
            prob, x = decide(P, q)
            print(f"{closes.index[-1]:%H:%M} prezzo {P:.2f} probabilità {prob:.0%} → {'COMPRA' if x else 'fuori'}", flush=True)
            if not x:
                continue
            pnl = trade(now, P, prob, *x, equity, float(acct["non_marginable_buying_power"]))
            day["risultato"] += pnl
            if day["risultato"] < -MAX_DAILY_LOSS * equity * SHARE:
                day["fermo"] = True
                notify(f"Kill switch: oggi {day['risultato']:.2f} $. Fermo fino a domani (UTC).")
            LOGS.mkdir(exist_ok=True)
            DAY.write_text(json.dumps(day))
            push(f"operazione {now:%Y-%m-%d %H:%M} UTC: {pnl:+.2f} $")
        except Exception as e:  # rete, dati, broker: si salta il minuto e si riprova al prossimo
            print(f"ERRORE: {e}", flush=True)
            if held() and not STOP:  # mai lasciare ETH senza stop: se un'operazione si è rotta a metà, si vende
                market("sell", f"ts-err-{now:%Y%m%d%H%M}", qty=f"{held():.9f}")
    if held():
        market("sell", f"ts-end-{pd.Timestamp.now('UTC'):%Y%m%d%H%M}", qty=f"{held():.9f}")
    print("Bot Timestral fermo.", flush=True)


def prova():
    q = 100 + np.linspace(-1.5, 9, 21)            # fascia spostata in su: mediana 103,75
    prob, (s, tps) = decide(100, q)
    assert prob >= 0.8 and abs(s - (100 - q[Q10])) < 1e-9 and 0 < tps[0] < tps[1] < tps[2]
    assert decide(100, 100 + np.linspace(-2, 2, 21))[1] is None          # fascia centrata: fuori
    assert decide(100, 100 + np.linspace(-0.2, 0.6, 21))[1] is None      # sale, ma meno dei costi: fuori
    plan = {"entrata": 100, "s": 1, "tp": [101, 102, 103], "k": 0}
    assert step(plan, 100.5, 99.5) == (0, "") and plan["k"] == 0       # niente tocca niente
    assert step(plan, 101.2, 99.5) == (1, "TP1") and plan["k"] == 1       # TP1, lo stop sale all'entrata
    assert step(plan, 100.9, 100) == (2, "stop") and plan["k"] == 3       # stop all'entrata: chiude gli altri due
    plan = {"entrata": 100, "s": 1, "tp": [101, 102, 103], "k": 0}
    assert step(plan, 105, 98) == (3, "stop")                              # stop e obiettivi nello stesso minuto: vince lo stop
    plan = {"entrata": 100, "s": 1, "tp": [101, 102, 103], "k": 0}
    assert step(plan, 103.5, 99.5) == (3, "TP3")                           # tutti gli obiettivi in un colpo
    print("prove superate")


if __name__ == "__main__":
    prova() if "--prova" in sys.argv else main()
