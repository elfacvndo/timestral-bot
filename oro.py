# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""Analista AI sull'oro: come analista.py (il modello fa l'analista, il bot fa il risk manager), ma su un solo mercato,
l'oro come CFD con la leva, sul conto demo di Capital.com da 100 € (capital.py). Nato il 2026-11-18 al posto
dell'Analista su Alpaca, su richiesta dell'utente: con 20 $ a operazione e senza leva i guadagni erano centesimi
(DIARIO.md, "Analista sull'oro").

In parole semplici, ogni 15 minuti con il mercato dell'oro aperto (e subito se scatta un suo allarme):
1. il bot manda al modello le candele chiuse dell'oro su quattro scale, i conti già fatti, l'operazione aperta, i suoi
   risultati e le sue lezioni, gli annunci economici USA e i titoli delle notizie sull'oro;
2. il modello risponde come un analista: compra, vendi, aspetta o sposta lo stop, quanto è sicuro e perché;
3. il bot decide QUANTO: la misura la calcola lui perché arrivare allo stop costi il RISK del conto (2%, ~2 €), con un
   tetto alla perdita del giorno (BUDGET, 6%) e la leva massima europea sull'oro (1:20). L'AI non tocca mai il rischio.
Due versi (dal 2026-11-18, richiesta dell'utente): al rialzo ("compra") o al ribasso ("vendi_allo_scoperto": sui CFD si
vende oro che non si ha e lo si ricompra dopo; si guadagna se scende). Nelle operazioni "verso" = 1 o -1.
Lo stop è depositato presso il broker (vale anche a bot fermo); obiettivo e allarmi li guarda il bot ogni minuto. Il
venerdì sera il bot chiude tutto: nel fine settimana l'oro non si scambia e alla riapertura il prezzo può saltare lo stop.
Costi: spread ~0,013% (dentro i prezzi di entrata e uscita) e, per ogni notte con la posizione aperta, ~0,04%
dell'esposizione pagato da chi compra (~0,03% ricevuto da chi vende allo scoperto). Il netto si calcola dai prezzi veri:
i costi notturni restano fuori dal registro.
Si giudica solo in avanti, sul demo, come l'Analista su Alpaca.

    uv run oro.py --prova                     # prove della logica, niente rete
    uv run --env-file ../.env oro.py          # il bot (chiavi Capital.com e Claude Code autenticato)
"""
import json
import math
import os
import signal
import sys
import threading
import time
from datetime import timedelta

import pandas as pd

import analista as A  # gli aiutanti dell'Analista: conti, controlli, registri, notizie, il modello
import bot
import capital

SYM = "GOLD"
COST = 0.03                        # andata e ritorno in %: spread ~0,013% più slittamento
RISK, BUDGET, LEVA = 0.02, 0.06, 20  # rischio a operazione, perdita massima del giorno (del conto), leva massima
MIN_STOP, MAX_STOP = 0.002, 0.02   # stop fra 0,2% e 2% dal prezzo (con la leva 1:20, il 2% dell'oro è il 40%)
MIN_SIZE = 0.01                    # once: taglia minima e passo di Capital.com
RES = {"15Min": "MINUTE_15", "1Hour": "HOUR", "4Hour": "HOUR_4", "1Day": "DAY"}
MINUTES = int(os.environ.get("MINUTI", 330))
OPS, DEC, OPEN = bot.LOGS / "oro.csv", bot.LOGS / "oro_decisioni.csv", bot.LOGS / "oro_aperta.json"
LESSONS = bot.LOGS / "oro_lezioni.csv"
FIELDS = A.FIELDS[:-1] + ["verso", "once", "netto_eur"]
VERSO = {1: "al rialzo", -1: "al ribasso (allo scoperto)"}
ENTRY = {"compra": 1, "vendi_allo_scoperto": -1}
ALERTS = []              # prezzi che fanno scattare subito una domanda (li sceglie il modello)
LOCK = threading.Lock()  # operazione aperta, registri e ordini: la sorveglianza e la domanda non si pestano i piedi
SYSTEM = f"""Sei un analista di trading sull'oro (GOLD, dollari per oncia). Ogni 15 minuti, e subito quando scatta un tuo
allarme, ricevi i dati e decidi. È un CFD con la leva su un conto demo da circa 100 €: un movimento dell'1% dell'oro vale
diversi euro, nei due sensi. Puoi puntare al rialzo ("compra") o al ribasso ("vendi_allo_scoperto": guadagni se l'oro
scende). Costi: circa {COST}% andata e ritorno; per ogni notte con la posizione aperta chi compra paga circa lo 0,04%
dell'esposizione, chi vende allo scoperto riceve circa lo 0,03%. Il bot chiude da solo il venerdì sera, prima del fine
settimana. Lo stop va fra {MIN_STOP:.1%} e {MAX_STOP:.0%} dal prezzo, dal lato che ti dà torto (sotto per un acquisto,
sopra per una vendita allo scoperto), e il bot lo deposita anche presso il broker; l'obiettivo sta dall'altro lato. Non
decidi quanto investire: lo calcola il risk manager.
Rispondi SOLO con un oggetto JSON:
{{"GOLD": {{"azione": "compra" | "vendi_allo_scoperto" | "aspetta" | "chiudi" | "sposta_stop", "fiducia": 0-100, "stop": prezzo,
"obiettivo": prezzo, "allarmi": [prezzo, prezzo], "perche": "il motivo in due o tre frasi, in italiano"}},
"_lezioni": [{{"simbolo": "GOLD", "chiusura": "AAAA-MM-GG HH:MM", "lezione": "una frase"}}]}}
Senza operazione aperta: "compra", "vendi_allo_scoperto" o "aspetta". Con un'operazione aperta: "aspetta" (tieni),
"chiudi" o "sposta_stop" (nuovo stop e obiettivo; lo stop si avvicina soltanto al prezzo: sale per un acquisto, scende per
una vendita allo scoperto).
Ricevi anche i titoli delle notizie recenti: usali come contesto, non come segnale da soli.
Il bot ti dà i conti già fatti (RSI, movimento medio di una candela, distanza dalla media di 20 candele, massimi e
minimi): usa quelli invece di rifarli. Il movimento medio dice quanto si muove il prezzo di solito: uno stop più vicino
di così scatta per rumore.
"allarmi": fino a due prezzi (lista vuota se non servono). Il bot li guarda ogni minuto e, se il prezzo ne passa uno, ti fa
subito una domanda nuova: mettili dove agiresti o cambieresti idea.
"_lezioni": per ogni operazione chiusa che ti viene data da commentare, una lezione di una frase su cosa ha funzionato o
no. Le tue lezioni di prima le ricevi a ogni domanda: sono la tua esperienza.
Lo scopo è fare operazioni e imparare da come vanno: aspettare non è gratis. Il rischio è già fissato: il risk manager
sceglie la misura in modo che arrivare allo stop costi il {RISK:.0%} del conto, e c'è un tetto alla perdita del giorno.
Non serve la certezza: compra quando hai un motivo concreto e l'obiettivo è almeno una volta e mezza la distanza dello
stop, anche con fiducia media. In laterale si può comprare vicino al supporto o vendere allo scoperto vicino alla
resistenza. Se il quadro è ribassista non limitarti ad aspettare: vendi allo scoperto, con lo stop sopra la resistenza. Un annuncio economico in arrivo non basta per aspettare: entra
con lo stop, oppure aspetta solo se mancano meno di 30 minuti. Orizzonte: da qualche ora a uno-due giorni. Ricevi anche
quello che hai detto al giro prima: se il livello che aspettavi è arrivato, agisci.
"fiducia" è quanto credi nell'operazione che proponi (per "aspetta", quanto credi che entrare adesso sia sbagliato)."""


def candles(tf, n, now):
    """Le ultime n candele CHIUSE di una scala, al prezzo di mezzo fra denaro e lettera."""
    step, p = pd.Timedelta(A.FRAMES[tf]), capital.prices(RES[tf], n + 2)
    mid = lambda k: [(x[k]["bid"] + x[k]["ask"]) / 2 for x in p]
    df = pd.DataFrame({"o": mid("openPrice"), "h": mid("highPrice"), "l": mid("lowPrice"), "c": mid("closePrice"),
                       "v": [x.get("lastTradedVolume", 0) for x in p]},
                      index=pd.to_datetime([x["snapshotTimeUTC"] for x in p], utc=True))
    return df[df.index + step <= now].tail(n)


def quote():
    """(denaro, lettera, si può operare?)."""
    s = capital.snapshot()
    return float(s["bid"]), float(s["offer"]), s["marketStatus"] == "TRADEABLE"


def ny(now):
    return now.tz_convert(bot.NY)


def window(now):
    """Niente operazioni nuove il venerdì dalle 16 di New York (l'oro chiude alle 17 per il fine settimana)."""
    return not (ny(now).weekday() == 4 and ny(now).hour >= 16)


def weekend(now):
    """Il venerdì dalle 16:40 di New York si chiude tutto."""
    return ny(now).weekday() == 4 and ny(now).time() >= pd.Timestamp("16:40").time()


def check(a, P, t, livello=None):
    """La risposta del modello passata dal risk manager: (cosa fare, perché no). Ogni numero si controlla: la risposta è
    un testo che viene da fuori."""
    act = str(a.get("azione", "")).strip().lower()
    try:
        stop, tgt = float(a.get("stop") or 0), float(a.get("obiettivo") or 0)
    except (TypeError, ValueError):
        return "aspetta", "numeri non validi"
    if t is None:
        v = ENTRY.get(act)
        if not v:
            return "aspetta", ""
        if not MIN_STOP <= v * (P - stop) / P <= MAX_STOP:  # lo stop dal lato che dà torto, né troppo vicino né lontano
            return "aspetta", f"stop {stop} fuori dai limiti"
        if v * (tgt - P) / P < 2 * COST / 100:
            return "aspetta", "obiettivo troppo vicino per i costi"
        return act, ""
    if act in ("chiudi", "vendi"):
        return "chiudi", ""
    if act == "sposta_stop":
        livello = livello or A.LIVELLO
        if livello != "pieno":
            return "tieni", f"livello {livello}: lo stop non si sposta"
        v = t.get("verso", 1)
        if not (v * (stop - t["stop"]) > 0 and v * (P - stop) > 0 and v * (tgt - P) > 0):
            return "tieni", "spostamento non valido (lo stop si avvicina soltanto e resta dal lato giusto del prezzo)"
        return "sposta_stop", ""
    return "tieni", ""


def exit_reason(t, bid, offer):
    """Stop o obiettivo, al prezzo a cui si chiuderebbe: denaro per chi ha comprato, lettera per chi ha venduto."""
    v = t.get("verso", 1)
    x = bid if v > 0 else offer
    return "stop" if v * (x - t["stop"]) <= 0 else "obiettivo" if v * (x - t["obiettivo"]) >= 0 else ""


def size(equity, today, stop_pct):
    """L'esposizione in euro: arrivare allo stop costa RISK del conto, ridotto da quanto si è già perso oggi (modalità
    prop firm) e mai più di quanto resta del budget del giorno; tetto: il 90% della leva massima."""
    scale = max(0.0, 1 + min(today, 0) / (BUDGET * equity))
    room = max(0.0, BUDGET * equity + min(today, 0))
    return min(min(RISK * equity * scale, room) / stop_pct, 0.9 * LEVA * equity)


def ounces(expo_eur, fx, P):
    """Esposizione in euro → once, arrotondate in giù al passo di Capital.com."""
    return math.floor(expo_eur * fx / P / MIN_SIZE + 1e-9) * MIN_SIZE


def today_eur(now):
    return sum(float(r["netto_eur"]) for r in A.rows(OPS) if r["chiusura"].startswith(f"{now:%Y-%m-%d}"))


def stats(done):
    if not done:
        return "Finora nessuna operazione chiusa."
    pct = [float(x["netto_pct"]) for x in done]
    return (f"Finora: {len(done)} chiuse, {sum(v > 0 for v in pct)} in guadagno, media {sum(pct) / len(pct):+.2f}% "
            f"dell'oro, totale {sum(float(x['netto_eur']) for x in done):+.2f} € sul conto.")


def prompt(frames, P, t, done, events, now, heads=(), prev=None, lessons=(), why=""):
    """Il messaggio per il modello: l'oro su quattro scale, l'operazione aperta, i risultati, notizie e lezioni."""
    s = [f"Ora: {now:%Y-%m-%d %H:%M} UTC."] + ([f"Domanda anticipata: {why}."] if why else [])
    s.append(f"\n\n=== GOLD (oro, CFD con leva, costi ~{COST}% andata e ritorno). Prezzo adesso: {P:.2f}")
    if frames:
        s.append("\nConti fatti (RSI 14 | movimento medio di una candela | prezzo rispetto alla media di 20 candele): "
                 + "; ".join(f"{tf} {A.fmt(r, '.0f')} | {A.fmt(a, '.2f', '%')} | {A.fmt(m, '+.2f', '%')}"
                             for tf, (r, a, m) in ((tf, A.numbers(df)) for tf, df in frames.items() if len(df))))
    h1, d1 = frames.get("1Hour"), frames.get("1Day")
    if h1 is not None and len(h1) and d1 is not None and len(d1):
        s.append(f"\nMassimo / minimo: ultime 24 candele da 1 ora {h1.h.tail(24).max():.2f} / {h1.l.tail(24).min():.2f}; "
                 f"ultimi 7 giorni {d1.h.tail(7).max():.2f} / {d1.l.tail(7).min():.2f}.")
    for tf, df in frames.items():
        s.append(f"\nCandele {tf} chiuse, le ultime {A.SEND} (ora UTC,apertura,massimo,minimo,chiusura,volume):\n"
                 f"{A.table(df.tail(A.SEND))}")
    if t:
        mins, v = (now - pd.Timestamp(t["apertura"], tz="UTC")) / timedelta(minutes=1), t.get("verso", 1)
        s.append(f"\nOperazione aperta da {mins:.0f} minuti, {VERSO[v]}: {t['once']:.2f} once, entrata {t['entrata']:.2f}, "
                 f"stop {t['stop']:.2f}, obiettivo {t['obiettivo']:.2f}, ora {v * (P / t['entrata'] - 1) * 100:+.2f}% a favore "
                 f"(costi esclusi).")
    else:
        s.append("\nNessuna operazione aperta.")
    if prev:
        s.append(f"\nAl giro prima ({prev['ora']}) avevi detto: {prev['azione']}, fiducia {prev['fiducia']}. {prev['perche']}")
    s.append("\nUltimi risultati: " + ("; ".join(f"{r['chiusura']} {VERSO[int(r['verso'])]} {float(r['netto_pct']):+.2f}% ({r['motivo']})"
                                                 for r in done[-3:]) or "nessuno") + ". " + stats(done))
    s.append("\n\nNotizie importanti (ora UTC): " + ("; ".join(events) or "nessuna nelle prossime 24 ore"))
    s.append(f"\n\nTitoli delle notizie sull'oro delle ultime {A.NEWS_HOURS} ore:\n" + ("\n".join(heads) or "nessuno"))
    s.append("\n\nLe tue lezioni (le più recenti):" + ("".join(f"\n- {r['lezione']}" for r in lessons[-15:])
                                                      or " nessuna, per ora"))
    todo = A.pending(done, lessons)
    if todo:
        s.append('\n\nOperazioni chiuse da commentare in "_lezioni":' + "".join(
            f"\n- chiusura {r['chiusura']}: {VERSO[int(r['verso'])]}, aperta {r['apertura']} a {r['entrata']}, uscita a {r['uscita']} "
            f"({r['motivo']}), netto {float(r['netto_pct']):+.2f}%. Il tuo perché d'entrata: {r['perche']}" for r in todo))
    return "\n".join(s)


def load_open():
    try:
        return json.loads(OPEN.read_text()) or {}
    except (FileNotFoundError, ValueError):
        return {}


def save_open(op):
    bot.LOGS.mkdir(exist_ok=True)
    OPEN.write_text(json.dumps(op, indent=1))


def close(t, why, now, op):
    """Chiude e scrive il risultato. Se la posizione non c'è più l'ha chiusa il broker allo stop: si registra allo stop.
    Il netto viene dai prezzi veri (spread compreso).
    ponytail: costi notturni esclusi dal netto; leggerli da /api/v1/history/transactions se diventano importanti."""
    p = capital.position()
    if p:
        out = capital.close(p["dealId"])
    else:
        out, why = t["stop"], f"{why} (chiusa dal broker)"
    v = t.get("verso", 1)
    gross = v * (out / t["entrata"] - 1) * 100
    row = {"apertura": t["apertura"], "chiusura": f"{now:%Y-%m-%d %H:%M}", "simbolo": SYM, "modello": t["modello"],
           "investito": round(t["investito"], 2), "entrata": t["entrata"], "uscita": out, "motivo": why,
           "perche": t["perche"], "lordo_pct": round(gross, 4), "netto_pct": round(gross - (0 if p else COST), 4),
           "verso": v, "once": t["once"], "netto_eur": round(v * t["once"] * (out - t["entrata"]) / capital.eurusd(), 4)}
    A.append(OPS, FIELDS, row)
    print(row, flush=True)
    op.pop(SYM)
    save_open(op)
    bot.push(f"oro {row['netto_eur']:+.2f} € ({why})")


def enter(a, P, now, op, model, v):
    """Apre al rialzo (v = 1) o allo scoperto (v = -1) quanto dice il risk manager; ritorna l'esito in parole."""
    stop = float(a["stop"])
    bal, avail = capital.balance()
    fx, today = capital.eurusd(), today_eur(now)
    oz = ounces(size(bal, today, abs(1 - stop / P)), fx, P)
    margin = oz * P / LEVA / fx
    if oz < MIN_SIZE or margin > 0.9 * avail:
        return f"misura non possibile ({oz:.2f} once, disponibili {avail:.2f} €, oggi {today:+.2f} €)"
    capital.open("BUY" if v > 0 else "SELL", round(oz, 2), stop)
    p = capital.position()
    E = float(p["level"])
    op[SYM] = {"apertura": f"{now:%Y-%m-%d %H:%M}", "modello": model, "once": round(oz, 2), "investito": oz * E,
               "verso": v, "entrata": E, "stop": stop, "obiettivo": float(a["obiettivo"]), "deal": p["dealId"],
               "perche": str(a.get("perche", ""))[:500]}
    save_open(op)
    return f"{'comprate' if v > 0 else 'vendute allo scoperto'} {oz:.2f} once ({oz * E:.0f} $, margine {margin:.2f} €)"


def decide(now, op, P, why=""):
    """Un giro: dati → una domanda al modello → risk manager → (forse) ordine. Gira accanto alla sorveglianza."""
    try:
        frames = {tf: candles(tf, A.FETCH, now) for tf in A.FRAMES}
    except Exception as e:
        print(f"ERRORE dati: {e}", flush=True)
        return
    with LOCK:
        t, done, lessons = dict(op[SYM]) if SYM in op else None, A.rows(OPS), A.rows(LESSONS)
        prev = next((r for r in A.rows(DEC)[::-1] if r["azione"] != "errore"), None)
    try:
        text = prompt(frames, P, t, done, A.news(now), now, A.headlines(["GLD"], now), prev, lessons, why)
        resp, model, err = *A.ask(text, SYSTEM), ""
    except Exception as e:  # modello giù, limiti dell'abbonamento finiti, risposta non JSON: si aspetta
        resp, model, err = {}, A.MODEL, str(e)[:300]
    with LOCK:
        t = op.get(SYM)
        a = resp.get(SYM) if isinstance(resp.get(SYM), dict) else {}
        if err:
            a, act, why_not = {"azione": "errore", "perche": err}, None, "modello non disponibile"
        else:
            act, why_not = check(a, P, t)
            ALERTS[:] = A.levels(a, P)
            if act in ENTRY and not window(now):
                why_not = "venerdì sera: niente operazioni nuove"
        try:
            if why_not or act in ("aspetta", "tieni"):
                result = why_not or act
            elif A.LIVELLO == "suggerisce":
                result = "solo suggerimento"
            elif act in ENTRY:
                result = enter(a, P, now, op, model, ENTRY[act])
            elif act == "chiudi":
                close(t, "AI", now, op)
                result = "chiusa"
            else:  # sposta_stop: anche presso il broker
                capital.move_stop(t["deal"], float(a["stop"]))
                t.update(stop=float(a["stop"]), obiettivo=float(a["obiettivo"]))
                save_open(op)
                result = f"stop {t['stop']:.2f}, obiettivo {t['obiettivo']:.2f}"
        except Exception as e:  # broker: niente ordine, si riprova al giro dopo
            result = f"errore broker: {str(e)[:200]}"
        A.append(DEC, A.DFIELDS, {"ora": f"{now:%Y-%m-%d %H:%M}", "simbolo": SYM, "modello": model, "prezzo": P,
                                  "azione": a.get("azione"), "fiducia": a.get("fiducia"), "stop": a.get("stop"),
                                  "obiettivo": a.get("obiettivo"), "perche": str(a.get("perche", ""))[:500],
                                  "esito": result, "allarmi": "" if err else ";".join(map(str, ALERTS))})
        print(f"{now:%H:%M} GOLD {P:.2f} AI: {a.get('azione')} ({a.get('fiducia')}) → {result}. {a.get('perche', '')}",
              flush=True)
        todo = {r["chiusura"] for r in A.pending(done, lessons)}
        for x in resp.get("_lezioni") or [] if isinstance(resp.get("_lezioni"), list) else []:
            if isinstance(x, dict) and x.get("chiusura") in todo and x.get("lezione"):
                A.append(LESSONS, A.LFIELDS, {"ora": f"{now:%Y-%m-%d %H:%M}", "simbolo": SYM,
                                              "chiusura": x["chiusura"], "lezione": str(x["lezione"])[:300]})
                todo.discard(x["chiusura"])
        bot.push(f"oro {now:%H:%M} {a.get('azione')}")


def run(*args):
    try:
        decide(*args)
    except Exception as e:
        print(f"ERRORE giro: {e}", flush=True)


def reconcile(op, now):
    """Lo stato vero è quello del broker: operazione registrata senza posizione = chiusa dal broker (si registra);
    posizione senza registro (un giro interrotto) = si chiude."""
    p = capital.position()
    if SYM in op and not p:
        close(op[SYM], "riavvio", now, op)
    elif SYM not in op and p:
        capital.close(p["dealId"])
        print("all'avvio c'era oro senza registro: chiuso", flush=True)


def crossed(a, b):
    """Gli allarmi che il prezzo ha passato andando da a a b."""
    return [x for x in ALERTS if min(a, b) <= x <= max(a, b)]


def main():
    for k in capital.KEYS:
        if not os.environ.get(k):
            sys.exit(f"Manca {k} (nel cloud: Secrets del repository).")
    for sg in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sg, lambda *_: bot.STOP.append(1))
    op = load_open()
    last = next((r for r in A.rows(DEC)[::-1] if r["azione"] != "errore"), None)  # gli allarmi valgono anche dopo un riavvio
    ALERTS[:] = [float(x) for x in ((last or {}).get("allarmi") or "").split(";") if x]
    fresh, end, worker, asked, before = True, time.time() + MINUTES * 60, None, 0.0, None
    print(f"Analista sull'oro avviato ({capital.BASE}), modello {A.MODEL}, livello {A.LIVELLO}; aperta: {SYM in op}", flush=True)
    while not bot.STOP and time.time() < end:
        bot.wait_minute()
        if bot.STOP:
            break
        now = pd.Timestamp.now("UTC")
        try:
            bid, offer, ok = quote()
        except Exception as e:
            print(f"ERRORE: {e}", flush=True)
            continue
        P = (bid + offer) / 2
        rang = []
        with LOCK:  # la sorveglianza di ogni minuto: gira anche mentre il modello pensa
            try:
                if fresh and ok:
                    reconcile(op, now)
                    fresh = False
                if SYM in op and ok and not fresh:
                    t = op[SYM]
                    why = "fine settimana" if weekend(now) else exit_reason(t, bid, offer)
                    if not why and capital.position() is None:
                        why = "stop"
                    if why:
                        close(t, why, now, op)
            except Exception as e:  # rete, broker: si riprova al prossimo minuto
                print(f"ERRORE: {e}", flush=True)
            if ok and before:
                rang = crossed(before, P)
            before = P if ok else None
        due = ok and not fresh and now.minute % 15 == 0
        if (due or rang and time.time() - asked > A.GAP) and not (worker and worker.is_alive()):
            for x in rang:  # un allarme scatta una volta sola; il modello ne mette di nuovi a ogni risposta
                ALERTS.remove(x)
            why = "" if due else "è scattato il tuo allarme a " + ", ".join(map(str, rang))
            worker = threading.Thread(target=run, daemon=True, args=(now, op, P, why))
            worker.start()
            asked = time.time()
    if worker:
        worker.join()
    print(f"Analista sull'oro fermo; aperta: {SYM in op}", flush=True)


def prova():
    now = pd.Timestamp("2026-11-18 12:00", tz="UTC")  # mercoledì
    # Misura: 2% del conto allo stop, ridotto da quanto si è perso oggi, tetto la leva.
    assert round(size(100, 0, 0.005), 9) == 400          # 2 € / 0,5% = 400 € di esposizione
    assert round(size(100, -3, 0.005), 9) == 200         # metà budget perso → metà misura
    assert size(100, -6, 0.005) == 0                     # budget finito: kill switch
    assert size(100, 0, 0.0001) == 1800                  # stop strettissimo: tetto al 90% della leva 1:20
    assert size(100, 5, 0.005) == size(100, 0, 0.005)    # guadagnare non alza il rischio
    assert round(ounces(400, 1.1, 3770), 2) == 0.11 and ounces(30, 1.1, 3770) == 0   # sotto 0,01 once: niente
    # Il venerdì sera niente operazioni nuove, poi si chiude (ore di New York, anche con l'ora legale).
    assert window(now) and not weekend(now)
    assert not window(pd.Timestamp("2026-11-20 21:10", tz="UTC")) and not weekend(pd.Timestamp("2026-11-20 21:30", tz="UTC"))
    assert weekend(pd.Timestamp("2026-11-20 21:45", tz="UTC")) and weekend(pd.Timestamp("2026-07-17 20:45", tz="UTC"))
    # Risk manager, nei due versi: stop dal lato che dà torto, fra 0,2% e 2%; obiettivo oltre i costi, dall'altro lato.
    P, buy, short = 3770.0, {"azione": "compra", "stop": 3751.15, "obiettivo": 3810}, {"azione": "vendi_allo_scoperto", "stop": 3788.85, "obiettivo": 3730}
    assert check(buy, P, None) == ("compra", "") and check(short, P, None) == ("vendi_allo_scoperto", "")
    assert check({**buy, "stop": 3788.85}, P, None)[0] == "aspetta"      # acquisto con lo stop sopra il prezzo
    assert check({**short, "stop": 3751.15}, P, None)[0] == "aspetta"    # scoperto con lo stop sotto
    assert check({**short, "stop": 3771}, P, None)[0] == "aspetta"       # troppo vicino (0,03%)
    assert check({**short, "stop": 3900}, P, None)[0] == "aspetta"       # troppo lontano (3,4%)
    assert check({**short, "obiettivo": 3769}, P, None)[0] == "aspetta"  # obiettivo dentro i costi
    assert check({**short, "obiettivo": 3800}, P, None)[0] == "aspetta"  # obiettivo dal lato sbagliato
    assert check({**buy, "stop": "tanto"}, P, None) == ("aspetta", "numeri non validi") and check({"azione": "chiudi"}, P, None)[0] == "aspetta"
    tl, ts = {"verso": 1, "stop": 3750, "obiettivo": 3800}, {"verso": -1, "stop": 3790, "obiettivo": 3740}
    assert check({"azione": "chiudi"}, P, ts) == ("chiudi", "") and check({"azione": "vendi"}, P, tl) == ("chiudi", "")
    assert check({"azione": "sposta_stop", "stop": 3780, "obiettivo": 3730}, P, ts, "pieno") == ("sposta_stop", "")
    assert check({"azione": "sposta_stop", "stop": 3795, "obiettivo": 3730}, P, ts, "pieno")[0] == "tieni"   # si allontana
    assert check({"azione": "sposta_stop", "stop": 3765, "obiettivo": 3730}, P, ts, "pieno")[0] == "tieni"   # oltre il prezzo
    assert check({"azione": "sposta_stop", "stop": 3760, "obiettivo": 3800}, P, tl, "pieno") == ("sposta_stop", "")
    assert check({"azione": "sposta_stop", "stop": 3760, "obiettivo": 3800}, P, tl, "apre_chiude")[0] == "tieni"
    assert check(buy, P, tl)[0] == "tieni"                                # già dentro: niente seconda operazione
    # Uscite: chi ha comprato esce al denaro, chi ha venduto alla lettera.
    assert exit_reason(tl, 3749, 3751) == "stop" and exit_reason(tl, 3800, 3801) == "obiettivo" and exit_reason(tl, 3770, 3771) == ""
    assert exit_reason(ts, 3789, 3790) == "stop" and exit_reason(ts, 3739, 3740) == "obiettivo" and exit_reason(ts, 3741, 3742) == ""
    # Allarmi: scattano quando il prezzo li passa da un minuto all'altro.
    ALERTS[:] = [3780.0, 3700.0]
    assert crossed(3775, 3781) == [3780.0] and crossed(3781, 3775) == [3780.0] and crossed(3750, 3760) == []
    ALERTS.clear()
    g = globals()
    fake = {"pos": None, "sent": [], "moved": [], "out": 3800.0}
    fakes = {"prices": lambda res, n: [{"snapshotTimeUTC": f"2026-11-18T{h:02d}:00:00", "lastTradedVolume": 5,
                                        **{k: {"bid": 3769.5 + h, "ask": 3770.5 + h} for k in
                                           ("openPrice", "highPrice", "lowPrice", "closePrice")}} for h in range(12)],
             "balance": lambda: (100.0, 100.0), "eurusd": lambda: 1.1, "position": lambda: fake["pos"],
             "open": lambda d, s, st: fake.update(pos={"dealId": "d1", "size": s, "level": 3770.0}) or fake["sent"].append((d, s, st)),
             "close": lambda deal: fake.update(pos=None) or fake["out"],
             "move_stop": lambda deal, st: fake["moved"].append((deal, st))}
    saved_cap = {k: getattr(capital, k) for k in fakes}
    tmp = bot.LOGS / "prova_oro"
    saved = {k: g[k] for k in ("OPS", "DEC", "OPEN", "LESSONS")}
    saved_a = {k: getattr(A, k) for k in ("ask", "news", "headlines")}
    asked = []
    try:
        for k, v in fakes.items():
            setattr(capital, k, v)
        tmp.mkdir(parents=True, exist_ok=True)
        g.update(OPS=tmp / "oro.csv", DEC=tmp / "dec.csv", OPEN=tmp / "aperta.json", LESSONS=tmp / "lezioni.csv")
        A.news, A.headlines = (lambda now: []), (lambda syms, now: [])
        # Candele: si scarta quella in corso (la candela delle 11 da 1 ora finisce alle 12: è chiusa).
        df = candles("1Hour", 60, now)
        assert len(df) == 12 and df.c.iloc[-1] == 3781.0
        assert len(candles("1Hour", 60, pd.Timestamp("2026-11-18 11:30", tz="UTC"))) == 11
        # Un giro: compra 0,11 once con lo stop presso il broker, registra la decisione.
        A.ask = lambda text, system: asked.append(text) or ({"GOLD": {"azione": "compra", "fiducia": 60, "stop": 3751.15,
                                                                      "obiettivo": 3810, "allarmi": [3790], "perche": "supporto"}}, "claude-x")
        op = {}
        decide(now, op, 3770.0)
        assert "=== GOLD" in asked[0] and "Conti fatti" in asked[0] and "Nessuna operazione aperta" in asked[0]
        assert fake["sent"] == [("BUY", 0.11, 3751.15)] and op["GOLD"]["once"] == 0.11 and op["GOLD"]["verso"] == 1 and op["GOLD"]["deal"] == "d1"
        assert A.rows(DEC)[-1]["esito"].startswith("comprate 0.11 once") and ALERTS == [3790.0]
        assert json.loads(OPEN.read_text())["GOLD"]["entrata"] == 3770.0
        # Sposta lo stop: anche presso il broker; poi chiude.
        A.ask = lambda text, system: ({"GOLD": {"azione": "sposta_stop", "stop": 3765, "obiettivo": 3820}}, "claude-x")
        decide(now, op, 3780.0)
        assert fake["moved"] == [("d1", 3765.0)] and op["GOLD"]["stop"] == 3765
        A.ask = lambda text, system: ({"GOLD": {"azione": "chiudi"}}, "claude-x")
        decide(now, op, 3800.0)
        r = A.rows(OPS)[-1]
        assert op == {} and r["motivo"] == "AI" and float(r["netto_eur"]) == round(0.11 * 30 / 1.1, 4) and r["verso"] == "1"
        assert "Finora: 1 chiuse, 1 in guadagno" in prompt({}, 3800, None, A.rows(OPS), [], now)
        # Allo scoperto: vende 0,11 once con lo stop sopra; l'oro scende a 3740 e il guadagno è positivo.
        A.ask = lambda text, system: asked.append(text) or ({"GOLD": {**short, "fiducia": 55, "perche": "resistenza"}}, "claude-x")
        decide(now, op, 3770.0)
        assert fake["sent"][-1] == ("SELL", 0.11, 3788.85) and op["GOLD"]["verso"] == -1
        assert A.rows(DEC)[-1]["esito"].startswith("vendute allo scoperto 0.11 once")
        decide(now, op, 3750.0)                                               # già dentro: tiene
        assert "al ribasso (allo scoperto)" in asked[-1] and "+0.53% a favore" in asked[-1] and len(fake["sent"]) == 2
        fake["out"] = 3740.0
        A.ask = lambda text, system: ({"GOLD": {"azione": "chiudi"}}, "claude-x")
        decide(now, op, 3740.0)
        r = A.rows(OPS)[-1]
        assert r["verso"] == "-1" and float(r["netto_eur"]) == round(0.11 * 30 / 1.1, 4) and float(r["lordo_pct"]) > 0
        # Chiusa dal broker (stop): si registra allo stop, con i costi.
        op = {"GOLD": {**json.loads(json.dumps({"apertura": "2026-11-18 11:00", "modello": "m", "once": 0.1,
                                                 "investito": 377, "entrata": 3770.0, "stop": 3750.0, "obiettivo": 3800.0,
                                                 "deal": "d2", "perche": "x"}))}}
        reconcile(op, now)
        r = A.rows(OPS)[-1]
        assert op == {} and r["uscita"] == "3750.0" and r["motivo"] == "riavvio (chiusa dal broker)"
        assert float(r["netto_pct"]) == round((3750 / 3770 - 1) * 100 - COST, 4) and round(today_eur(now), 4) == round(6.0 - 2 / 1.1, 4)
        # Il venerdì sera il modello non può aprire; il modello giù non apre niente.
        A.ask = lambda text, system: ({"GOLD": {"azione": "compra", "stop": 3760, "obiettivo": 3800}}, "claude-x")
        decide(pd.Timestamp("2026-11-20 21:30", tz="UTC"), op, 3770.0)
        assert A.rows(DEC)[-1]["esito"].startswith("venerdì sera") and len(fake["sent"]) == 2
        A.ask = lambda text, system: (_ for _ in ()).throw(RuntimeError("429"))
        ALERTS[:] = [3790.0]
        decide(now, op, 3770.0)
        assert A.rows(DEC)[-1]["esito"] == "modello non disponibile" and ALERTS == [3790.0]
        # Posizione senza registro (giro interrotto): si chiude.
        fake["pos"] = {"dealId": "d3"}
        reconcile({}, now)
        assert fake["pos"] is None
    finally:
        for k, v in saved_cap.items():
            setattr(capital, k, v)
        for k, v in saved_a.items():
            setattr(A, k, v)
        g.update(saved)
        ALERTS.clear()
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()
    print("prove superate")


if __name__ == "__main__":
    prova() if "--prova" in sys.argv else main()
