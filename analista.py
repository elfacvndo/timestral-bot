# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""Analista AI: un modello linguistico (GPT, Claude, Gemini...) fa l'analista, il bot fa il risk manager. SOL/USD, conto
demo Alpaca (Paper), nel cloud (GitHub Actions). Nato il 2026-11-16 seguendo alla lettera il metodo di un video
(DIARIO.md, "Analista AI").

In parole semplici, ogni 15 minuti:
1. il bot manda al modello le candele chiuse di SOL su quattro scale (15 minuti, 1 ora, 4 ore, 1 giorno), l'operazione
   aperta se c'è, gli ultimi tre risultati e le notizie importanti in arrivo (calendario economico gratuito);
2. il modello risponde come un analista: compra, vendi, aspetta o sposta lo stop, quanto è sicuro e perché, in parole;
3. il bot controlla la risposta e decide QUANTO: l'AI propone l'operazione e lo stop, ma la misura la calcola il bot dal
   rischio fissato qui sotto. L'AI non tocca mai il rischio.
Ogni minuto il bot controlla stop e obiettivo da solo. Ogni decisione finisce nel registro col suo perché.

Livelli (LIVELLO, deciso dall'utente): "suggerisce" = scrive solo cosa farebbe; "apre_chiude" = apre e chiude, stop
fisso; "pieno" = anche sposta lo stop (solo in su) e l'obiettivo.
Modalità prop firm: il bot ha un budget di perdita giornaliero; più ne ha perso oggi, più piccola è ogni operazione
nuova; finito il budget, niente operazioni fino al giorno dopo (UTC). È anche il suo kill switch: conta solo le sue.
Solo acquisti: Alpaca non permette di vendere allo scoperto le crypto. Costi: 0,25% a ordine più slittamento, ~0,6%
andata e ritorno: il modello lo sa, ed è il motivo per cui il bot rifiuta obiettivi troppo vicini.
Modello: Claude (MODELLO = "opus", l'Opus più recente; o "sonnet"), con l'abbonamento Claude dell'utente tramite Claude
Code: nessuna chiave a pagamento, consuma i limiti dell'abbonamento. Cambiare modello = cambiare una riga del workflow, non
il bot; quando esce un Opus nuovo, l'alias lo prende da solo. Se il modello non risponde (limiti finiti), si aspetta.
Non si prova sul passato (il modello ha letto quegli anni: sarebbe un esame con le risposte già viste); si giudica solo
in avanti, sul demo.

    uv run analista.py --prova   # prove della logica, niente rete
    uv run analista.py           # il bot (chiavi Alpaca e Claude Code autenticato)
"""
import csv
import json
import os
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import timedelta

import pandas as pd

import bot  # helper di Timestral: chiamate ad Alpaca, ordini con ID unico, minuti, salvataggio del registro

SYM = "SOL/USD"
MODEL = os.environ.get("MODELLO", "opus")    # alias di Claude Code: sempre l'Opus più recente
CLAUDE = os.environ.get("CLAUDE", "claude")  # il programma Claude Code
LIVELLO = os.environ.get("LIVELLO", "pieno")
RISK, QUOTA, BUDGET = 0.005, 0.20, 0.02  # rischio a operazione, massimo investito, perdita massima del giorno (del conto)
COSTS = 0.6                               # % andata e ritorno
MIN_STOP, MAX_STOP = 0.005, 0.04          # stop fra 0,5% e 4% sotto il prezzo (oltre, l'ordine scende sotto i 12 $)
FRAMES = {"15Min": "15min", "1Hour": "1h", "4Hour": "4h", "1Day": "1D"}
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
MINUTES = int(os.environ.get("MINUTI", 330))
OPS, DEC, OPEN = bot.LOGS / "analista.csv", bot.LOGS / "analista_decisioni.csv", bot.LOGS / "analista_aperta.json"
FIELDS = ["apertura", "chiusura", "modello", "investito", "entrata", "uscita", "motivo", "perche", "lordo_pct",
          "netto_pct", "netto_usd"]
DFIELDS = ["ora", "modello", "prezzo", "azione", "fiducia", "stop", "obiettivo", "perche", "esito"]
NEWS = {"t": None, "v": []}
SYSTEM = f"""Sei un analista di trading su {SYM} (crypto, 24 ore su 24). Ogni 15 minuti ricevi i dati e decidi.
Puoi solo comprare e poi vendere (niente vendite allo scoperto). Costi: circa {COSTS}% andata e ritorno: un'operazione
ha senso solo se il movimento atteso è ben più grande. Lo stop di un acquisto va fra {MIN_STOP:.1%} e {MAX_STOP:.0%}
sotto il prezzo. Non decidi quanto investire: lo calcola il risk manager.
Rispondi SOLO con un oggetto JSON:
{{"azione": "compra" | "vendi" | "aspetta" | "sposta_stop", "fiducia": 0-100, "stop": prezzo, "obiettivo": prezzo,
"perche": "il motivo in due o tre frasi, in italiano"}}
Senza operazione aperta: "compra" o "aspetta". Con un'operazione aperta: "aspetta" (tieni), "vendi" (chiudi) o
"sposta_stop" (nuovo stop e obiettivo; lo stop può solo salire)."""


def table(df):
    return "\n".join(f"{t:%m-%d %H:%M},{r.o:.3f},{r.h:.3f},{r.l:.3f},{r.c:.3f},{r.v:.0f}" for t, r in df.iterrows())


def prompt(frames, P, t, last, events, now):
    """Il messaggio per il modello: tutto quello che serve, niente di più."""
    s = [f"Ora: {now:%Y-%m-%d %H:%M} UTC. Prezzo {SYM}: {P:.3f}."]
    for tf, df in frames.items():
        s.append(f"\nCandele {tf} chiuse (ora UTC,apertura,massimo,minimo,chiusura,volume):\n{table(df)}")
    if t:
        mins = (now - pd.Timestamp(t["apertura"], tz="UTC")) / timedelta(minutes=1)
        s.append(f"\nOperazione aperta da {mins:.0f} minuti: entrata {t['entrata']:.3f}, stop {t['stop']:.3f}, "
                 f"obiettivo {t['obiettivo']:.3f}, ora {(P / t['entrata'] - 1) * 100:+.2f}% (costi esclusi).")
    else:
        s.append("\nNessuna operazione aperta.")
    s.append("\nUltimi risultati: " + ("; ".join(f"{r['chiusura']} {float(r['netto_pct']):+.2f}% ({r['motivo']})"
                                                 for r in last) or "nessuno"))
    s.append("\nNotizie importanti (ora UTC): " + ("; ".join(events) or "nessuna nelle prossime 24 ore"))
    return "\n".join(s)


def check(a, P, t, livello=LIVELLO):
    """La risposta del modello passata dal risk manager: (cosa fare, perché no). La risposta è un testo che viene da
    fuori, quindi ogni numero si controlla."""
    act = str(a.get("azione", "")).strip().lower()
    try:
        stop, tgt = float(a.get("stop") or 0), float(a.get("obiettivo") or 0)
    except (TypeError, ValueError):
        return "aspetta", "numeri non validi"
    if t is None:
        if act != "compra":
            return "aspetta", ""
        if not P * (1 - MAX_STOP) <= stop <= P * (1 - MIN_STOP):
            return "aspetta", f"stop {stop} fuori dai limiti"
        if tgt < P * (1 + 2 * COSTS / 100):
            return "aspetta", "obiettivo troppo vicino per i costi"
        return "compra", ""
    if act == "vendi":
        return "vendi", ""
    if act == "sposta_stop":
        if livello != "pieno":
            return "tieni", f"livello {livello}: lo stop non si sposta"
        if not t["stop"] < stop < P or tgt <= P:
            return "tieni", "spostamento non valido (lo stop sale soltanto e resta sotto il prezzo)"
        return "sposta_stop", ""
    return "tieni", ""


def size(equity, today_usd, stop_pct):
    """Quanto investire: rischio fisso a stop pieno, ridotto da quanto si è già perso oggi (modalità prop firm).
    0 = non si entra (budget finito, o importo sotto il minimo d'ordine di Alpaca)."""
    scale = max(0.0, 1 + min(today_usd, 0) / (BUDGET * equity))
    amount = min(RISK * equity * scale / stop_pct, QUOTA * equity)
    return amount if amount >= bot.CRYPTO_MIN else 0.0


def exit_reason(t, h, l):
    return "stop" if l <= t["stop"] else "obiettivo" if h >= t["obiettivo"] else ""


def rows(path):
    try:
        with path.open() as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def append(path, fields, row):
    bot.LOGS.mkdir(exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def save_open(t):
    bot.LOGS.mkdir(exist_ok=True)
    OPEN.write_text(json.dumps(t, indent=1))


def candles(tf, n, now):
    """Le ultime n candele CHIUSE di una scala (la candela in corso si scarta)."""
    step = pd.Timedelta(FRAMES[tf])
    p = {"symbols": SYM, "timeframe": tf, "start": f"{now - step * (n + 3):%Y-%m-%dT%H:%M:%SZ}", "limit": 1000}
    page = bot.get_json(urllib.request.Request(f"{bot.CRYPTO_BARS}?{urllib.parse.urlencode(p)}", headers=bot.keys()))
    df = pd.DataFrame(page["bars"].get(SYM, []))
    df.index = pd.to_datetime(df.t)
    return df[df.index + step <= now].tail(n)


def news(now):
    """Notizie ad alto impatto sul dollaro dall'ora scorsa alle prossime 24 ore. Il calendario si scarica al massimo
    una volta l'ora (il sito limita le richieste); se non risponde, si usa l'ultimo scaricato."""
    if NEWS["t"] is None or now - NEWS["t"] > timedelta(hours=1):
        try:
            req = urllib.request.Request(CAL_URL, headers={"User-Agent": "Mozilla/5.0"})
            NEWS["v"], NEWS["t"] = bot.get_json(req, tries=1) or [], now
        except Exception as e:
            print(f"calendario non disponibile: {e}", flush=True)
            NEWS["t"] = now
    out = []
    for e in NEWS["v"]:
        when = pd.Timestamp(e["date"]).tz_convert("UTC")
        if e.get("impact") == "High" and e.get("country") == "USD" and now - timedelta(hours=1) <= when <= now + timedelta(hours=24):
            out.append(f"{when:%m-%d %H:%M} {e['title']}")
    return out


def ask(text):
    """Il modello col tuo abbonamento Claude: `claude -p` (Claude Code) senza strumenti, solo testo. Nel cloud si
    autentica col token CLAUDE_CODE_OAUTH_TOKEN (fatto con `claude setup-token`): nessuna chiave a pagamento.
    Ritorna (risposta, nome esatto del modello che ha risposto)."""
    r = subprocess.run([CLAUDE, "-p", "--model", MODEL, "--tools", "", "--system-prompt", SYSTEM, "--output-format", "json"],
                       input=text, capture_output=True, text=True, timeout=600)
    try:
        out = json.loads(r.stdout)
    except ValueError:
        raise RuntimeError(f"claude: {(r.stdout or r.stderr)[-300:]}") from None
    if r.returncode or out.get("is_error"):
        raise RuntimeError(f"claude: {str(out.get('result') or r.stderr)[:300]}")
    s = out["result"]
    return json.loads(s[s.index("{"):s.rindex("}") + 1]), ",".join(out.get("modelUsage") or [MODEL])


def held():
    p = bot.api("GET", f"/v2/positions/{SYM.replace('/', '')}", missing_ok=True)
    return float(p["qty"]) if p and float(p["market_value"]) > 1 else 0.0


def order(side, cid, **amount):
    return bot.execute({"symbol": SYM, "side": side, "type": "market", "time_in_force": "gtc", "client_order_id": cid,
                        **amount})


def close(t, why, now):
    """Vende tutto e scrive il risultato. Ritorna l'operazione ancora aperta (vendita non eseguita) o None."""
    q = held()
    if not q:  # chiusa altrove: niente riga finta
        print("posizione già chiusa, la tolgo senza registrarla", flush=True)
        save_open(None)
        return None
    x = order("sell", f"ai-SOLUSD-out-{now:%Y%m%d%H%M}", qty=f"{q:.9f}")
    if x["status"] != "filled":
        print(f"vendita non eseguita ({x['status']}), riprovo al prossimo minuto", flush=True)
        return t
    out = float(x["filled_avg_price"])
    gross = (out / t["entrata"] - 1) * 100
    row = {"apertura": t["apertura"], "chiusura": f"{now:%Y-%m-%d %H:%M}", "modello": t["modello"],
           "investito": round(t["investito"], 2), "entrata": t["entrata"], "uscita": out, "motivo": why,
           "perche": t["perche"], "lordo_pct": round(gross, 4), "netto_pct": round(gross - COSTS, 4),
           "netto_usd": round(t["investito"] * (gross - COSTS) / 100, 4)}
    append(OPS, FIELDS, row)
    print(row, flush=True)
    save_open(None)
    bot.push(f"analista SOL {row['netto_pct']:+.2f}% ({why})")
    return None


def buy(a, P, now, equity, cash, today_usd, model):
    stop = float(a["stop"])
    amount = size(equity, today_usd, 1 - stop / P)
    if not amount or amount > cash * 0.9:
        return None, f"importo {amount:.2f} $ non possibile (contanti {cash:.2f} $, oggi {today_usd:+.2f} $)"
    o = order("buy", f"ai-SOLUSD-in-{now:%Y%m%d%H%M}", notional=f"{amount:.2f}")
    if o["status"] != "filled":  # non confermato: si annulla e si guarda cosa è stato comprato davvero
        try:
            bot.api("DELETE", f"/v2/orders/{o['id']}")
        except RuntimeError:
            pass
        o = bot.api("GET", f"/v2/orders/{o['id']}")
        if not float(o["filled_qty"] or 0):
            return None, f"acquisto non eseguito ({o['status']})"
    E = float(o["filled_avg_price"])
    t = {"apertura": f"{now:%Y-%m-%d %H:%M}", "modello": model, "investito": float(o["filled_qty"]) * E, "entrata": E,
         "stop": stop, "obiettivo": float(a["obiettivo"]), "perche": str(a.get("perche", ""))[:500]}
    save_open(t)
    return t, f"comprati {t['investito']:.2f} $"


def decide(now, t, equity, cash):
    """Il giro dei 15 minuti: dati → modello → risk manager → (forse) ordine. Ritorna l'operazione aperta."""
    frames = {tf: candles(tf, 24, now) for tf in FRAMES}
    P = float(frames["15Min"].c.iloc[-1])
    done = rows(OPS)
    today = sum(float(r["netto_usd"]) for r in done if r["chiusura"].startswith(f"{now:%Y-%m-%d}"))
    try:
        a, model = ask(prompt(frames, P, t, done[-3:], news(now), now))
        act, why = check(a, P, t)
    except Exception as e:  # modello giù, limiti dell'abbonamento finiti, risposta non JSON: si aspetta
        a, model, act, why = {"azione": "errore", "perche": str(e)[:300]}, MODEL, None, "modello non disponibile"
    if why or act in ("aspetta", "tieni"):
        result = why or act
    elif LIVELLO == "suggerisce":
        result = "solo suggerimento"
    elif act == "compra":
        t2, result = buy(a, P, now, equity, cash, today, model)
        t = t2 or t
    elif act == "vendi":
        t = close(t, "AI", now)
        result = "venduto" if t is None else "vendita non eseguita"
    elif act == "sposta_stop":
        t.update(stop=float(a["stop"]), obiettivo=float(a["obiettivo"]))
        save_open(t)
        result = f"stop {t['stop']:.3f}, obiettivo {t['obiettivo']:.3f}"
    append(DEC, DFIELDS, {"ora": f"{now:%Y-%m-%d %H:%M}", "modello": model, "prezzo": P, "azione": a.get("azione"),
                          "fiducia": a.get("fiducia"), "stop": a.get("stop"), "obiettivo": a.get("obiettivo"),
                          "perche": str(a.get("perche", ""))[:500], "esito": result})
    print(f"{now:%H:%M} {P:.3f} AI: {a.get('azione')} ({a.get('fiducia')}) → {result}. {a.get('perche', '')}", flush=True)
    bot.push(f"analista {now:%H:%M} {a.get('azione')}")
    return t


def main():
    for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        if not os.environ.get(k):
            sys.exit(f"Manca {k} (nel cloud: Secrets del repository).")
    for sg in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sg, lambda *_: bot.STOP.append(1))
    # Riconciliazione: lo stato vero è quello del broker.
    try:
        t = json.loads(OPEN.read_text())
    except (FileNotFoundError, ValueError):
        t = None
    for o in bot.api("GET", "/v2/orders?status=open&limit=500") or []:
        if o["symbol"].replace("/", "") == "SOLUSD":
            bot.api("DELETE", f"/v2/orders/{o['id']}")
    now = pd.Timestamp.now("UTC")
    if t and not held():
        t = close(t, "riavvio", now)
    elif not t and held():
        order("sell", f"ai-SOLUSD-start-{now:%Y%m%d%H%M}", qty=f"{held():.9f}")
        print("all'avvio c'erano SOL senza registro: venduti", flush=True)
    end = time.time() + MINUTES * 60
    print(f"Analista AI avviato su {bot.BASE}, {SYM}, modello {MODEL}, livello {LIVELLO}; aperta: {bool(t)}", flush=True)
    while not bot.STOP and time.time() < end:
        bot.wait_minute()
        if bot.STOP:
            break
        now = pd.Timestamp.now("UTC")
        try:
            if t:
                bar = bot.minutes(now, 3, SYM, False)[1].iloc[-1]
                why = "" if bar.isna().any() else exit_reason(t, bar.h, bar.l)
                if why:
                    t = close(t, why, now)
            if now.minute % 15 == 0:
                acct = bot.api("GET", "/v2/account")
                t = decide(now, t, float(acct["equity"]), float(acct["non_marginable_buying_power"]))
        except Exception as e:  # rete, dati, broker: si riprova al prossimo minuto
            print(f"ERRORE: {e}", flush=True)
    print(f"Analista fermo; operazione aperta che il giro dopo riprende: {bool(t)}", flush=True)


def prova():
    P = 100.0
    t = {"apertura": "2026-11-16 10:00", "entrata": 99.0, "stop": 97.0, "obiettivo": 104.0}
    buy_ok = {"azione": "compra", "stop": 98, "obiettivo": 103, "fiducia": 70}
    # Risk manager: senza operazione si entra solo con stop e obiettivo sensati.
    assert check(buy_ok, P, None) == ("compra", "")
    assert check({**buy_ok, "azione": " Compra "}, P, None)[0] == "compra"
    assert check({**buy_ok, "stop": 99.8}, P, None)[0] == "aspetta"       # stop troppo stretto (0,2%)
    assert check({**buy_ok, "stop": 90}, P, None)[0] == "aspetta"         # stop troppo largo (10%)
    assert check({**buy_ok, "stop": 101}, P, None)[0] == "aspetta"        # stop sopra il prezzo
    assert check({**buy_ok, "obiettivo": 100.5}, P, None)[0] == "aspetta"  # obiettivo dentro i costi
    assert check({**buy_ok, "stop": "tanto"}, P, None) == ("aspetta", "numeri non validi")
    assert check({"azione": "vendi"}, P, None)[0] == "aspetta"            # niente da vendere
    assert check({}, P, None)[0] == "aspetta"
    # Con operazione aperta: vendi, tieni, sposta lo stop solo in su e solo al livello pieno.
    assert check({"azione": "vendi"}, P, t) == ("vendi", "")
    assert check(buy_ok, P, t)[0] == "tieni"
    move = {"azione": "sposta_stop", "stop": 98.5, "obiettivo": 105}
    assert check(move, P, t) == ("sposta_stop", "")
    assert check(move, P, t, "apre_chiude")[0] == "tieni"
    assert check({**move, "stop": 96}, P, t)[0] == "tieni"                # stop che scende
    assert check({**move, "stop": 100.5}, P, t)[0] == "tieni"             # stop sopra il prezzo
    # Misura: rischio 0,5% a stop pieno, al massimo 20% del conto, ridotta da quanto si è perso oggi.
    assert size(100, 0, 0.02) == 20                                     # 0,50 $ / 2% = 25 $, tetto 20 $
    assert size(100, 0, 0.04) == 12.5                                   # 0,50 $ / 4%
    assert size(100, -1, 0.02) == 12.5                                  # metà budget perso → metà misura
    assert size(100, -1.5, 0.02) == 0                                   # 6,25 $: sotto il minimo d'ordine
    assert size(100, -2, 0.01) == 0 and size(100, -5, 0.01) == 0         # budget finito: kill switch
    assert size(100, +3, 0.02) == 20                                    # guadagnare non alza il rischio
    assert exit_reason(t, 105, 96) == "stop" and exit_reason(t, 105, 98) == "obiettivo" and exit_reason(t, 103, 98) == ""
    # Il messaggio al modello ha le quattro scale, l'operazione aperta, gli ultimi risultati e le notizie.
    idx = pd.date_range("2026-11-16 08:00", periods=3, freq="15min", tz="UTC")
    df = pd.DataFrame({"o": [1.0, 2, 3], "h": [1.5, 2.5, 3.5], "l": [0.5, 1.5, 2.5], "c": [1.2, 2.2, 3.2], "v": [10, 20, 30]}, idx)
    last = [{"chiusura": "2026-11-16 09:00", "netto_pct": "-0.8", "motivo": "stop"}]
    s = prompt({tf: df for tf in FRAMES}, P, t, last, ["11-16 19:00 FOMC Statement"], pd.Timestamp("2026-11-16 10:30", tz="UTC"))
    assert all(f"Candele {tf}" in s for tf in FRAMES) and "11-16 08:15,2.000,2.500,1.500,2.200,20" in s
    assert "aperta da 30 minuti" in s and "+1.01%" in s and "-0.80% (stop)" in s and "FOMC" in s
    assert "Nessuna operazione aperta" in prompt({}, P, None, [], [], pd.Timestamp("2026-11-16", tz="UTC"))
    # Notizie: solo alto impatto sul dollaro, dall'ora scorsa alle prossime 24 ore.
    now = pd.Timestamp("2026-11-16 12:00", tz="UTC")
    NEWS.update(t=now, v=[{"title": "FOMC", "country": "USD", "impact": "High", "date": "2026-11-16T14:00:00-05:00"},
                          {"title": "PIL", "country": "EUR", "impact": "High", "date": "2026-11-16T09:00:00-05:00"},
                          {"title": "Vecchia", "country": "USD", "impact": "High", "date": "2026-11-15T09:00:00-05:00"},
                          {"title": "Minore", "country": "USD", "impact": "Low", "date": "2026-11-16T09:00:00-05:00"}])
    assert news(now) == ["11-16 19:00 FOMC"]
    # Posizione già chiusa altrove: si toglie dalle aperte, nessuna riga finta.
    g, saved = globals(), {k: globals()[k] for k in ("held", "save_open", "OPS")}
    g.update(held=lambda: 0.0, save_open=lambda t: None, OPS=bot.LOGS / "prova_inesistente.csv")
    try:
        assert close(t, "stop", now) is None and not g["OPS"].exists()
    finally:
        g.update(saved)
    # Un giro completo con modello e broker finti: compra → misura dal rischio, decisione nel registro col suo perché.
    fake = {"bars": {SYM: [{"t": f"2026-11-16T{h:02d}:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 5} for h in range(8)]}}
    sent, logged = [], []
    g.update(saved)
    saved2 = {k: g[k] for k in ("candles", "ask", "rows", "append", "save_open", "news", "order")}
    g.update(candles=lambda tf, n, now: pd.DataFrame(fake["bars"][SYM]).assign(t=lambda d: pd.to_datetime(d.t)).set_index("t"),
             ask=lambda text: ({"azione": "compra", "fiducia": 65, "stop": 98, "obiettivo": 104, "perche": "rompe in su"}, "claude-x"),
             rows=lambda p: [], append=lambda p, f, r: logged.append(r), save_open=lambda t: None, news=lambda now: [],
             order=lambda side, cid, **a: sent.append((side, a)) or {"status": "filled", "filled_avg_price": "100",
                                                                     "filled_qty": "0.2"})
    try:
        t2 = decide(now, None, 100.0, 100.0)
        assert sent == [("buy", {"notional": "20.00"})] and t2["stop"] == 98 and t2["perche"] == "rompe in su"
        assert t2["modello"] == "claude-x" and logged[-1]["modello"] == "claude-x"
        assert logged[-1]["esito"] == "comprati 20.00 $" and logged[-1]["perche"] == "rompe in su"
        g["ask"] = lambda text: (_ for _ in ()).throw(RuntimeError("429 troppe richieste"))
        assert decide(now, None, 100.0, 100.0) is None and logged[-1]["esito"] == "modello non disponibile"
        g.update(ask=lambda text: ({"azione": "compra", "stop": 98, "obiettivo": 104}, "claude-x"), LIVELLO="suggerisce")
        assert decide(now, None, 100.0, 100.0) is None and logged[-1]["esito"] == "solo suggerimento" and len(sent) == 1
    finally:
        g.update(saved2, LIVELLO=os.environ.get("LIVELLO", "pieno"))
    print("prove superate")


if __name__ == "__main__":
    prova() if "--prova" in sys.argv else main()
