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
   Dal 2026-11-18 sera l'utente può scegliere a mano le once dalla dashboard (SETTINGS). Dal 2026-11-20 (richiesta
   dell'utente) la scelta a mano vince sempre: niente tetto del giorno né riduzione della leva, solo il broker può rifiutare.
   La misura scelta a mano vale per OGNI parte (dal 2026-11-20 sera, richiesta dell'utente): 0,30 once = tre operazioni
   da 0,30, non tre da 0,10.
Due versi (dal 2026-11-18, richiesta dell'utente): al rialzo ("compra") o al ribasso ("vendi_allo_scoperto": sui CFD si
vende oro che non si ha e lo si ricompra dopo; si guadagna se scende). Nelle operazioni "verso" = 1 o -1.
Tre parti (dal 2026-11-20, richiesta dell'utente): ogni entrata apre fino a tre operazioni con lo stesso stop e un
obiettivo ciascuna (TP1 vicino, TP2 quello di sempre, TP3 lontano); in automatico la misura in tutto è quella di sempre, divisa in tre.
Lo stop è depositato presso il broker (vale anche a bot fermo); obiettivi e allarmi li guarda il bot ogni minuto. Il
venerdì sera il bot chiude tutto: nel fine settimana l'oro non si scambia e alla riapertura il prezzo può saltare lo stop.
Costi da conto vero (dal 2026-11-20, richiesta dell'utente: "come se fosse un conto reale"): lo spread è già nei prezzi
veri di entrata e uscita, commissioni sui CFD non ce ne sono; quelli che il demo non applica li conta il bot (costs()):
cambio euro/dollaro 0,7% sul risultato, il costo di ogni notte alle 22 UTC al tasso del broker, lo slittamento stimato
degli ordini al mercato. Nel registro: lordo_eur (quello del demo), costi_eur, netto_eur = lordo - costi. Le tasse sono
annuali (26% sul guadagno dell'anno): le stima la dashboard.
Si giudica solo in avanti, sul demo, come l'Analista su Alpaca.

    uv run oro.py --prova                     # prove della logica, niente rete
    uv run --env-file ../.env oro.py          # il bot (chiavi Capital.com e Claude Code autenticato)
"""
import json
import math
import os
import signal
import subprocess
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
CONV = 0.007                       # cambio euro/dollaro di Capital.com sul risultato (conto in €, oro in $)
SLIP = 0.00005                     # slittamento stimato di un ordine al mercato (0,005% ≈ 0,18 $): il demo non slitta
PARTS = {1: [1], 2: [0, 2], 3: [0, 1, 2]}  # quanti pezzi → quali obiettivi (con un pezzo solo: TP2, quello di sempre)
RES = {"15Min": "MINUTE_15", "1Hour": "HOUR", "4Hour": "HOUR_4", "1Day": "DAY"}
MINUTES = int(os.environ.get("MINUTI", 330))
OPS, DEC, OPEN = bot.LOGS / "oro.csv", bot.LOGS / "oro_decisioni.csv", bot.LOGS / "oro_aperta.json"
LESSONS = bot.LOGS / "oro_lezioni.csv"
SETTINGS = bot.LOGS.parent / "oro_impostazioni.json"  # la misura scelta dall'utente nella dashboard (fuori da logs/)
CLOSE = bot.LOGS.parent / "oro_chiudi.json"  # "chiudi adesso" dalla dashboard: {"deal": ...} dell'operazione da chiudere
FIELDS = A.FIELDS[:-1] + ["verso", "once", "netto_eur", "lordo_eur", "costi_eur", "parte"]
VERSO = {1: "al rialzo", -1: "al ribasso (allo scoperto)"}
ENTRY = {"compra": 1, "vendi_allo_scoperto": -1}
ALERTS = []              # prezzi che fanno scattare subito una domanda (li sceglie il modello)
LOCK = threading.Lock()  # operazione aperta, registri e ordini: la sorveglianza e la domanda non si pestano i piedi
SYSTEM = f"""Sei un analista di trading sull'oro (GOLD, dollari per oncia). Ogni 15 minuti, e subito quando scatta un tuo
allarme, ricevi i dati e decidi. È un CFD con la leva su un conto demo da circa 100 €: un movimento dell'1% dell'oro vale
diversi euro, nei due sensi. Puoi puntare al rialzo ("compra") o al ribasso ("vendi_allo_scoperto": guadagni se l'oro
scende). Costi veri, come su un conto reale: spread e slittamento circa {COST}% andata e ritorno, cambio euro/dollaro
{CONV:.1%} sul risultato, e per ogni notte con la posizione aperta (alle 22 UTC) un tasso sull'esposizione che di solito
chi compra paga e chi vende allo scoperto riceve. Il bot chiude da solo il venerdì sera, prima del fine settimana. Lo stop
va fra {MIN_STOP:.1%} e {MAX_STOP:.0%} dal prezzo, dal lato che ti dà torto (sotto per un acquisto, sopra per una vendita
allo scoperto), e il bot lo deposita anche presso il broker; gli obiettivi stanno dall'altro lato. Non decidi quanto
investire: lo calcola il risk manager.
Ogni entrata il bot la divide in tre operazioni uguali con lo stesso stop e un obiettivo ciascuna, dal più vicino al più
lontano: TP1 per incassare presto (circa una volta la distanza dello stop), TP2 il tuo obiettivo di sempre, TP3 dove
arriverebbe il movimento se continua. Quando il prezzo tocca un obiettivo il bot chiude quella parte; le altre restano
aperte con lo stesso stop.
Rispondi SOLO con un oggetto JSON:
{{"GOLD": {{"azione": "compra" | "vendi_allo_scoperto" | "aspetta" | "chiudi" | "sposta_stop", "fiducia": 0-100, "stop": prezzo,
"obiettivi": [TP1, TP2, TP3], "allarmi": [prezzo, prezzo], "perche": "il motivo in due o tre frasi, in italiano"}},
"_lezioni": [{{"simbolo": "GOLD", "chiusura": "AAAA-MM-GG HH:MM", "lezione": "una frase"}}]}}
Senza operazione aperta: "compra", "vendi_allo_scoperto" o "aspetta". Con un'operazione aperta: "aspetta" (tieni),
"chiudi" (tutte le parti) o "sposta_stop" (nuovo stop per tutte le parti e sempre tre obiettivi: quelli delle parti già
chiuse si ignorano; lo stop si avvicina soltanto al prezzo: sale per un acquisto, scende per una vendita allo scoperto).
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
Non serve la certezza: compra quando hai un motivo concreto e TP2 è almeno una volta e mezza la distanza dello stop,
anche con fiducia media. In laterale si può comprare vicino al supporto o vendere allo scoperto vicino alla
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
    """(denaro, lettera, si può operare?, costo notturno del broker)."""
    m = capital.market()
    s = m["snapshot"]
    return float(s["bid"]), float(s["offer"]), s["marketStatus"] == "TRADEABLE", m["instrument"].get("overnightFee")


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
        stop = float(a.get("stop") or 0)
    except (TypeError, ValueError):
        return "aspetta", "numeri non validi"
    if t is None:
        v = ENTRY.get(act)
        if not v:
            return "aspetta", ""
        if not MIN_STOP <= v * (P - stop) / P <= MAX_STOP:  # lo stop dal lato che dà torto, né troppo vicino né lontano
            return "aspetta", f"stop {stop} fuori dai limiti"
        tps = targets(a, v)
        if not tps:
            return "aspetta", "obiettivi non validi (ne servono tre, dal più vicino al più lontano)"
        if v * (tps[0] - P) / P < 2 * COST / 100:
            return "aspetta", "primo obiettivo troppo vicino per i costi"
        return act, ""
    if act in ("chiudi", "vendi"):
        return "chiudi", ""
    if act == "sposta_stop":
        livello = livello or A.LIVELLO
        if livello != "pieno":
            return "tieni", f"livello {livello}: lo stop non si sposta"
        v, tps = t.get("verso", 1), targets(a, t.get("verso", 1))
        if not (tps and v * (stop - t["stop"]) > 0 and v * (P - stop) > 0
                and all(v * (tps[q["n"] - 1] - P) > 0 for q in t["parti"])):
            return "tieni", "spostamento non valido (lo stop si avvicina soltanto e resta dal lato giusto del prezzo)"
        return "sposta_stop", ""
    return "tieni", ""


def targets(a, v):
    """I tre obiettivi del modello, dal più vicino al più lontano nel verso v, o None se non validi (la risposta è un
    testo che viene da fuori)."""
    try:
        x = [float(y) for y in a.get("obiettivi")]
    except (TypeError, ValueError):
        return None
    return x if len(x) == 3 and v * (x[1] - x[0]) > 0 and v * (x[2] - x[1]) > 0 else None


def split(oz):
    """La misura in tutto divisa in tre parti quasi uguali al passo di 0,01 once (meno parti se non bastano)."""
    k = round(oz / MIN_SIZE)
    n = min(3, k)
    return [round((k // n + (i < k % n)) * MIN_SIZE, 2) for i in range(n)] if n else []


def costs(lordo, oz, E, out, notte, fx):
    """I costi veri che il demo non applica, in euro: cambio euro/dollaro sul risultato (CONV), slittamento stimato
    dell'ordine di entrata e di quello di uscita (SLIP) e le notti (notte: negativo = pagato, positivo = ricevuto).
    Commissioni: nessuna sui CFD di Capital.com; lo spread è già nei prezzi veri."""
    return CONV * abs(lordo + notte) + SLIP * oz * (E + out) / fx - notte


def nights(t, fee, P, op):
    """Il costo di ogni notte, come su un conto vero: alle 22 UTC il broker applica all'esposizione il tasso del suo verso
    (fee, letto ogni minuto). Il demo non lo fa: lo conta il bot, una volta per notte e per parte (anche dopo un riavvio
    nella stessa giornata)."""
    if not fee:
        return
    roll = pd.Timestamp(fee["swapChargeTimestamp"] - fee["swapChargeInterval"] * 60000, unit="ms", tz="UTC")
    rate, fx = fee["longRate"] if t["verso"] > 0 else fee["shortRate"], None
    for q in t["parti"]:
        if pd.Timestamp(q.get("notte_fino") or t["apertura"], tz="UTC") < roll:
            fx = fx or capital.eurusd()
            q["notte_eur"] = round(q.get("notte_eur", 0) + q["once"] * P * rate / 100 / fx, 4)
            q["notte_fino"] = f"{roll:%Y-%m-%d %H:%M}"
    if fx:
        save_open(op)


def equity(bal):
    """Il saldo da conto vero: quello del demo meno i costi che il demo non ha applicato."""
    return bal - sum(float(r.get("costi_eur") or 0) for r in A.rows(OPS))


def exit_reason(t, bid, offer):
    """Stop o obiettivo, al prezzo a cui si chiuderebbe: denaro per chi ha comprato, lettera per chi ha venduto."""
    v = t.get("verso", 1)
    x = bid if v > 0 else offer
    return "stop" if v * (x - t["stop"]) <= 0 else "obiettivo" if v * (x - t["obiettivo"]) >= 0 else ""


def limit(equity, today, stop_pct):
    """Il tetto di ogni misura in euro, anche di quella scelta a mano: allo stop non si perde più di quanto resta del
    budget del giorno (kill switch) e la leva resta entro il 90% di 1:20."""
    return min(max(0.0, BUDGET * equity + min(today, 0)) / stop_pct, 0.9 * LEVA * equity)


def size(equity, today, stop_pct):
    """L'esposizione in euro in automatico: arrivare allo stop costa RISK del conto, ridotto da quanto si è già perso
    oggi (modalità prop firm), sempre entro limit()."""
    scale = max(0.0, 1 + min(today, 0) / (BUDGET * equity))
    return min(RISK * equity * scale / stop_pct, limit(equity, today, stop_pct))


def latest(f):
    """Un file che l'utente scrive dalla dashboard, come dict ({} se manca o è rotto). Nel cloud si legge l'ultima
    versione del repository, non la copia del giro (l'utente lo cambia mentre il bot gira)."""
    try:
        if os.environ.get("GITHUB_ACTIONS") and f.parent == bot.LOGS.parent:  # le prove usano un file loro
            subprocess.run(["git", "fetch", "-q", "origin", "main"], capture_output=True, timeout=30)
            text = subprocess.run(["git", "show", f"origin/main:{f.name}"], capture_output=True, text=True,
                                  timeout=30).stdout
        else:
            text = f.read_text()
        x = json.loads(text)
        return x if isinstance(x, dict) else {}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def chosen():
    """Le once scelte a mano dall'utente nella dashboard, o None (automatico: decide il risk manager)."""
    try:
        x = float(latest(SETTINGS).get("once") or 0)
    except (ValueError, TypeError):
        return None  # file rotto: automatico
    return x if x >= MIN_SIZE else None


def asked_close(t):
    """L'utente ha premuto "chiudi adesso" (dashboard su Vercel) per questa operazione? La richiesta vale solo per il
    suo deal: una richiesta vecchia non chiude le operazioni dopo."""
    return latest(CLOSE).get("deal") == t.get("deal")


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
        parts = "; ".join(f"parte {q['n']} (TP{q['n']}): {q['once']:.2f} once, obiettivo {q['obiettivo']:.2f}" for q in t["parti"])
        s.append(f"\nOperazione aperta da {mins:.0f} minuti, {VERSO[v]}: {t['once']:.2f} once ancora aperte, entrata media "
                 f"{t['entrata']:.2f}, stop {t['stop']:.2f} per tutte le parti, ora {v * (P / t['entrata'] - 1) * 100:+.2f}% "
                 f"a favore (costi esclusi). Parti ancora aperte: {parts}.")
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
            f"({r['motivo']}), netto {float(r['netto_pct']):+.2f}%. Il tuo perché d'entrata: {r['perche']}" + by_hand(r, P) for r in todo))
    return "\n".join(s)


def by_hand(r, P):
    """Per un'operazione chiusa dall'utente: cosa avrebbe fatto tenendola fino a adesso (la lezione da trarre)."""
    if r["motivo"] != "a mano":
        return ""
    kept = int(r["verso"]) * (P / float(r["entrata"]) - 1) * 100
    return (f" L'HA CHIUSA L'UTENTE A MANO, prima del tuo stop e del tuo obiettivo: adesso l'oro è a {P:.2f}, tenendola "
            f"saresti a {kept:+.2f}% lordo. Chiedi a te stesso se l'obiettivo era troppo lontano o troppo vicino, se "
            f"conveniva incassare prima o se l'utente ha chiuso troppo presto.")


def load_open():
    try:
        op = json.loads(OPEN.read_text()) or {}
    except (FileNotFoundError, ValueError):
        return {}
    t = op.get(SYM)
    if t and "parti" not in t:  # aperta prima del 2026-11-20: una parte sola, con il suo obiettivo
        t.setdefault("verso", 1)
        t["parti"] = [{"n": 2, "deal": t["deal"], "once": t["once"], "entrata": t["entrata"], "obiettivo": t["obiettivo"]}]
    return op


def regroup(t):
    """Dopo che una parte si chiude: once, entrata media ed esposizione di quelle ancora aperte."""
    oz = sum(q["once"] for q in t["parti"])
    t.update(once=round(oz, 2), entrata=round(sum(q["once"] * q["entrata"] for q in t["parti"]) / oz, 2),
             investito=sum(q["once"] * q["entrata"] for q in t["parti"]), obiettivo=t["parti"][-1]["obiettivo"])


def save_open(op):
    bot.LOGS.mkdir(exist_ok=True)
    OPEN.write_text(json.dumps(op, indent=1))


def close(t, why, now, op, parts=None):
    """Chiude le parti (tutte se parts manca) e scrive una riga per parte. Una parte che presso il broker non c'è più
    l'ha chiusa qualcun altro: si registra al prezzo vero dallo storico del broker ("a mano" se l'ha chiusa l'utente
    prima dello stop), o allo stop se lo storico non la trova. Il risultato è da conto vero: lordo_eur è quello del
    demo (prezzi veri, spread compreso), netto_eur toglie i costi che il demo non applica (costs())."""
    live, fx, v, net = {p["dealId"] for p in capital.positions()}, capital.eurusd(), t.get("verso", 1), 0
    for q in list(parts or t["parti"]):
        w, cost = (f"obiettivo {q['n']}" if why == "obiettivo" else why), 0
        if q["deal"] in live:
            out = capital.close(q["deal"])
        else:
            out, src = capital.closed_at(q["deal"]) or (t["stop"], None)
            cost = 0 if src else COST  # dallo storico il prezzo è vero; allo stop è una stima
            w = "a mano" if src == "USER" and v * (out - t["stop"]) > 0 else f"{w} (chiusa dal broker)"
        E, oz = q["entrata"], q["once"]
        gross, lordo = v * (out / E - 1) * 100, round(v * oz * (out - E) / fx, 4)
        costi = round(costs(lordo, oz, E, out, q.get("notte_eur", 0), fx), 4)  # netto = lordo - costi, al centesimo di centesimo
        row = {"apertura": t["apertura"], "chiusura": f"{now:%Y-%m-%d %H:%M}", "simbolo": SYM, "modello": t["modello"],
               "investito": round(oz * E, 2), "entrata": E, "uscita": out, "motivo": w, "perche": t["perche"],
               "lordo_pct": round(gross, 4), "netto_pct": round(gross - cost - costi * fx / (oz * E) * 100, 4),
               "verso": v, "once": oz, "netto_eur": round(lordo - costi, 4), "lordo_eur": lordo,
               "costi_eur": costi, "parte": q["n"]}
        A.append(OPS, FIELDS, row)
        print(row, flush=True)
        net += row["netto_eur"]
        t["parti"].remove(q)
    if t["parti"]:
        regroup(t)
    else:
        op.pop(SYM)
    save_open(op)
    bot.push(f"oro {net:+.2f} € ({why})")


def enter(a, P, now, op, model, v):
    """Apre al rialzo (v = 1) o allo scoperto (v = -1) quanto dice il risk manager, in tre parti con lo stesso stop e
    un obiettivo ciascuna; ritorna l'esito in parole."""
    stop, tps = float(a["stop"]), targets(a, v)
    bal, avail = capital.balance()
    bal = equity(bal)
    fx, today, want, sp = capital.eurusd(), today_eur(now), chosen(), abs(1 - stop / P)
    if want is None:
        oz = ounces(size(bal, today, sp), fx, P)
        if oz < MIN_SIZE or oz * P / LEVA / fx > 0.9 * avail:
            return f"misura non possibile ({oz:.2f} once, disponibili {avail:.2f} €, oggi {today:+.2f} €)"
    else:  # scelta a mano (dal 2026-11-20, richiesta dell'utente): vince sempre, niente tetti del bot; decide il broker
        oz = math.floor(want / MIN_SIZE + 1e-9) * MIN_SIZE
    sizes = split(oz) if want is None else [round(oz, 2)] * 3  # a mano: tre parti da quella misura ciascuna
    margin, how = sum(sizes) * P / LEVA / fx, "" if want is None else f", scelta a mano: 3 × {oz:.2f}"
    before, err = {p["dealId"] for p in capital.positions()}, ""
    for s in sizes:
        try:
            capital.open("BUY" if v > 0 else "SELL", s, stop)
        except Exception as e:  # una parte rifiutata: si tengono quelle già aperte
            err = f"; errore broker: {str(e)[:150]}"
            break
    new = sorted((p for p in capital.positions() if p["dealId"] not in before), key=lambda p: p.get("createdDateUTC", ""))
    if not new:
        raise RuntimeError(err[2:] or "il broker non mostra la posizione")
    parts = [{"n": n + 1, "deal": p["dealId"], "once": float(p["size"]), "entrata": float(p["level"]), "obiettivo": tps[n]}
             for n, p in zip(PARTS[len(new)], new)]
    fee = capital.market()["instrument"].get("overnightFee") or {}
    op[SYM] = {"apertura": f"{now:%Y-%m-%d %H:%M}", "modello": model, "verso": v, "stop": stop, "deal": parts[0]["deal"],
               "perche": str(a.get("perche", ""))[:500], "parti": parts, "obiettivi": tps,
               "notte_tasso": fee.get("longRate" if v > 0 else "shortRate")}  # per la proiezione della dashboard
    regroup(op[SYM])
    save_open(op)
    t = op[SYM]
    return (f"{'comprate' if v > 0 else 'vendute allo scoperto'} {t['once']:.2f} once in {len(parts)} parti "
            f"({' + '.join(f'{q['once']:.2f}' for q in parts)}; obiettivi {', '.join(f'{q['obiettivo']:.2f}' for q in parts)}), "
            f"{t['investito']:.0f} $, margine {margin:.2f} €{how}{err}")


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
            else:  # sposta_stop: per tutte le parti, anche presso il broker
                tps = targets(a, t["verso"])
                for q in t["parti"]:
                    capital.move_stop(q["deal"], float(a["stop"]))
                    q["obiettivo"] = tps[q["n"] - 1]
                t.update(stop=float(a["stop"]), obiettivi=tps)
                regroup(t)
                save_open(op)
                result = f"stop {t['stop']:.2f}, obiettivi " + ", ".join(f"TP{q['n']} {q['obiettivo']:.2f}" for q in t["parti"])
        except Exception as e:  # broker: niente ordine, si riprova al giro dopo
            result = f"errore broker: {str(e)[:200]}"
        tps = a.get("obiettivi")
        A.append(DEC, A.DFIELDS, {"ora": f"{now:%Y-%m-%d %H:%M}", "simbolo": SYM, "modello": model, "prezzo": P,
                                  "azione": a.get("azione"), "fiducia": a.get("fiducia"), "stop": a.get("stop"),
                                  "obiettivo": ";".join(map(str, tps)) if isinstance(tps, list) else a.get("obiettivo"),
                                  "perche": str(a.get("perche", ""))[:500],
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
    """Lo stato vero è quello del broker: parte registrata senza posizione = chiusa dal broker (si registra);
    posizione senza registro (un giro interrotto) = si chiude."""
    live, t = capital.positions(), op.get(SYM)
    deals = {p["dealId"] for p in live}
    gone = [q for q in (t or {}).get("parti", []) if q["deal"] not in deals]
    if gone:
        close(t, "riavvio", now, op, gone)
    mine = {q["deal"] for q in op[SYM]["parti"]} if SYM in op else set()
    for p in live:
        if p["dealId"] not in mine:
            capital.close(p["dealId"])
            print("all'avvio c'era oro senza registro: chiuso", flush=True)


def watch(op, now, bid, offer, fee):
    """La sorveglianza di ogni minuto: le notti, poi il venerdì sera, lo stop (chiude tutte le parti), "chiudi adesso"
    dalla dashboard, gli obiettivi (ognuno chiude la sua parte) e le parti chiuse dal broker."""
    t = op.get(SYM)
    if not t:
        return
    nights(t, fee, (bid + offer) / 2, op)
    hit = {q["n"]: exit_reason({**t, "obiettivo": q["obiettivo"]}, bid, offer) for q in t["parti"]}
    why = "fine settimana" if weekend(now) else "stop" if "stop" in hit.values() else "a mano" if asked_close(t) else ""
    if why:
        return close(t, why, now, op)
    done = [q for q in t["parti"] if hit[q["n"]] == "obiettivo"]
    if done:
        close(t, "obiettivo", now, op, done)
    if SYM in op:
        deals = {p["dealId"] for p in capital.positions()}
        gone = [q for q in t["parti"] if q["deal"] not in deals]
        if gone:
            close(t, "stop", now, op, gone)


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
            bid, offer, ok, fee = quote()
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
                if ok and not fresh:
                    watch(op, now, bid, offer, fee)
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
    # Risk manager, nei due versi: stop dal lato che dà torto, fra 0,2% e 2%; tre obiettivi in fila dall'altro lato,
    # il primo oltre i costi.
    P = 3770.0
    buy = {"azione": "compra", "stop": 3751.15, "obiettivi": [3790, 3810, 3840]}
    short = {"azione": "vendi_allo_scoperto", "stop": 3788.85, "obiettivi": [3750, 3730, 3700]}
    assert check(buy, P, None) == ("compra", "") and check(short, P, None) == ("vendi_allo_scoperto", "")
    assert check({**buy, "stop": 3788.85}, P, None)[0] == "aspetta"      # acquisto con lo stop sopra il prezzo
    assert check({**short, "stop": 3751.15}, P, None)[0] == "aspetta"    # scoperto con lo stop sotto
    assert check({**short, "stop": 3771}, P, None)[0] == "aspetta"       # troppo vicino (0,03%)
    assert check({**short, "stop": 3900}, P, None)[0] == "aspetta"       # troppo lontano (3,4%)
    assert check({**short, "obiettivi": [3769, 3750, 3730]}, P, None)[0] == "aspetta"  # TP1 dentro i costi
    assert check({**short, "obiettivi": [3800, 3810, 3820]}, P, None)[0] == "aspetta"  # dal lato sbagliato
    assert check({**buy, "obiettivi": [3810, 3790, 3840]}, P, None)[0] == "aspetta"    # non in fila
    for bad in ([3790], None, "3790", [3790, "x", 3840]):
        assert check({**buy, "obiettivi": bad}, P, None)[0] == "aspetta"
    assert check({"azione": "compra", "stop": 3751.15, "obiettivo": 3810}, P, None)[0] == "aspetta"  # un obiettivo solo
    assert check({**buy, "stop": "tanto"}, P, None) == ("aspetta", "numeri non validi") and check({"azione": "chiudi"}, P, None)[0] == "aspetta"
    three = lambda *tp: [{"n": i + 1, "obiettivo": x} for i, x in enumerate(tp)]
    tl = {"verso": 1, "stop": 3750, "obiettivo": 3800, "parti": three(3780, 3790, 3800)}
    ts = {"verso": -1, "stop": 3790, "obiettivo": 3740, "parti": three(3760, 3750, 3740)}
    mv = lambda stop, tps: {"azione": "sposta_stop", "stop": stop, "obiettivi": tps}
    assert check({"azione": "chiudi"}, P, ts) == ("chiudi", "") and check({"azione": "vendi"}, P, tl) == ("chiudi", "")
    assert check(mv(3780, [3760, 3740, 3730]), P, ts, "pieno") == ("sposta_stop", "")
    assert check(mv(3795, [3760, 3740, 3730]), P, ts, "pieno")[0] == "tieni"   # lo stop si allontana
    assert check(mv(3765, [3760, 3740, 3730]), P, ts, "pieno")[0] == "tieni"   # oltre il prezzo
    assert check(mv(3780, [3775, 3740, 3730]), P, ts, "pieno")[0] == "tieni"   # TP1 dal lato sbagliato, parte ancora aperta
    assert check(mv(3780, [3775, 3740, 3730]), P, {**ts, "parti": ts["parti"][1:]}, "pieno") == ("sposta_stop", "")  # TP1 già chiusa
    assert check(mv(3760, [3780, 3800, 3820]), P, tl, "pieno") == ("sposta_stop", "")
    assert check(mv(3760, [3780, 3800, 3820]), P, tl, "apre_chiude")[0] == "tieni"
    assert check(buy, P, tl)[0] == "tieni"                                # già dentro: niente seconda operazione
    # Tre parti quasi uguali, al passo di 0,01 once; meno parti se non bastano.
    assert split(0.11) == [0.04, 0.04, 0.03] and split(0.3) == [0.1, 0.1, 0.1] and split(0.02) == [0.01, 0.01] and split(0.004) == []
    # Costi da conto vero: cambio 0,7% sul risultato, slittamento in entrata e uscita, notti (pagate o ricevute).
    assert round(costs(10, 0.1, 3770, 3800, 0, 1.1), 6) == round(0.07 + SLIP * 0.1 * 7570 / 1.1, 6)
    assert round(costs(-10, 0.1, 3770, 3740, -0.5, 1.1), 6) == round(0.0735 + SLIP * 0.1 * 7510 / 1.1 + 0.5, 6)
    assert costs(0, 0.1, 3770, 3770, 0.5, 1.1) < 0                       # allo scoperto la notte si riceve
    # Uscite: chi ha comprato esce al denaro, chi ha venduto alla lettera.
    assert exit_reason(tl, 3749, 3751) == "stop" and exit_reason(tl, 3800, 3801) == "obiettivo" and exit_reason(tl, 3770, 3771) == ""
    assert exit_reason(ts, 3789, 3790) == "stop" and exit_reason(ts, 3739, 3740) == "obiettivo" and exit_reason(ts, 3741, 3742) == ""
    # Allarmi: scattano quando il prezzo li passa da un minuto all'altro.
    ALERTS[:] = [3780.0, 3700.0]
    assert crossed(3775, 3781) == [3780.0] and crossed(3781, 3775) == [3780.0] and crossed(3750, 3760) == []
    ALERTS.clear()
    g = globals()
    fee = {"longRate": -0.02, "shortRate": 0.01, "swapChargeInterval": 1440,
           "swapChargeTimestamp": int(pd.Timestamp("2026-11-18 22:00", tz="UTC").timestamp() * 1000)}
    fake = {"pos": {}, "sent": [], "moved": [], "out": 3800.0, "n": 0}

    def opened(d, s, st):  # il broker finto: una posizione per ordine, come Capital.com nello stesso verso
        fake["n"] += 1
        deal = f"d{fake['n']}"
        fake["pos"][deal] = {"dealId": deal, "size": s, "level": 3770.0, "createdDateUTC": f"{fake['n']:04d}"}
        fake["sent"].append((d, s, st))

    fakes = {"prices": lambda res, n: [{"snapshotTimeUTC": f"2026-11-18T{h:02d}:00:00", "lastTradedVolume": 5,
                                        **{k: {"bid": 3769.5 + h, "ask": 3770.5 + h} for k in
                                           ("openPrice", "highPrice", "lowPrice", "closePrice")}} for h in range(12)],
             "balance": lambda: (100.0, 100.0), "eurusd": lambda: 1.1, "positions": lambda: list(fake["pos"].values()),
             "open": opened, "close": lambda deal: fake["pos"].pop(deal) and fake["out"],
             "move_stop": lambda deal, st: fake["moved"].append((deal, st)), "closed_at": lambda deal: fake.get("hist"),
             "market": lambda: {"instrument": {"overnightFee": fee}}}
    saved_cap = {k: getattr(capital, k) for k in fakes}
    tmp = bot.LOGS / "prova_oro"
    saved = {k: g[k] for k in ("OPS", "DEC", "OPEN", "LESSONS", "SETTINGS", "CLOSE")}
    saved_a = {k: getattr(A, k) for k in ("ask", "news", "headlines")}
    saved_push = bot.push
    bot.push = lambda msg: None  # le prove non salvano niente nel repository, nemmeno nel cloud
    asked = []
    net = lambda r: round(float(r["lordo_eur"]) - float(r["costi_eur"]), 4) == float(r["netto_eur"])
    try:
        for k, v in fakes.items():
            setattr(capital, k, v)
        tmp.mkdir(parents=True, exist_ok=True)
        g.update(OPS=tmp / "oro.csv", DEC=tmp / "dec.csv", OPEN=tmp / "aperta.json", LESSONS=tmp / "lezioni.csv",
                 SETTINGS=tmp / "impostazioni.json", CLOSE=tmp / "chiudi.json")
        A.news, A.headlines = (lambda now: []), (lambda syms, now: [])
        # Candele: si scarta quella in corso (la candela delle 11 da 1 ora finisce alle 12: è chiusa).
        df = candles("1Hour", 60, now)
        assert len(df) == 12 and df.c.iloc[-1] == 3781.0
        assert len(candles("1Hour", 60, pd.Timestamp("2026-11-18 11:30", tz="UTC"))) == 11
        # Un giro: compra 0,11 once in tre parti con lo stesso stop presso il broker, un obiettivo ciascuna.
        A.ask = lambda text, system: asked.append(text) or ({"GOLD": {**buy, "fiducia": 60, "allarmi": [3790], "perche": "supporto"}}, "claude-x")
        op = {}
        decide(now, op, 3770.0)
        t = op["GOLD"]
        assert "=== GOLD" in asked[0] and "Conti fatti" in asked[0] and "Nessuna operazione aperta" in asked[0]
        assert fake["sent"] == [("BUY", 0.04, 3751.15), ("BUY", 0.04, 3751.15), ("BUY", 0.03, 3751.15)]
        assert [(q["n"], q["deal"], q["once"], q["obiettivo"]) for q in t["parti"]] == [(1, "d1", 0.04, 3790), (2, "d2", 0.04, 3810), (3, "d3", 0.03, 3840)]
        assert t["obiettivi"] == [3790, 3810, 3840] and t["once"] == 0.11 and t["verso"] == 1 and t["deal"] == "d1" and t["entrata"] == 3770.0 and t["notte_tasso"] == -0.02
        assert A.rows(DEC)[-1]["esito"].startswith("comprate 0.11 once in 3 parti") and A.rows(DEC)[-1]["obiettivo"] == "3790;3810;3840"
        assert ALERTS == [3790.0] and json.loads(OPEN.read_text())["GOLD"]["parti"][2]["obiettivo"] == 3840
        assert "parte 3 (TP3): 0.03 once, obiettivo 3840.00" in prompt({}, 3775, t, [], [], now)
        # Ogni minuto: TP1 toccato chiude solo la prima parte; le altre restano, con lo stesso stop.
        watch(op, now, 3785, 3786, None)
        assert len(t["parti"]) == 3 and not A.rows(OPS)
        fake["out"] = 3791.0
        watch(op, now, 3791, 3792, None)
        r = A.rows(OPS)[-1]
        assert [q["n"] for q in t["parti"]] == [2, 3] and t["once"] == 0.07 and "d1" not in fake["pos"]
        assert r["motivo"] == "obiettivo 1" and r["parte"] == "1" and float(r["lordo_eur"]) == round(0.04 * 21 / 1.1, 4) and net(r)
        assert 0 < float(r["costi_eur"]) < 0.05 and float(r["netto_pct"]) < float(r["lordo_pct"])
        # Sposta lo stop: per tutte le parti ancora aperte, anche presso il broker, con i loro obiettivi nuovi.
        A.ask = lambda text, system: ({"GOLD": mv(3765, [3790, 3820, 3850])}, "claude-x")
        decide(now, op, 3780.0)
        assert fake["moved"] == [("d2", 3765.0), ("d3", 3765.0)] and t["stop"] == 3765 and [q["obiettivo"] for q in t["parti"]] == [3820, 3850]
        # Lo stop chiude tutte le parti rimaste.
        fake["out"] = 3764.0
        watch(op, now, 3764, 3765, None)
        R = A.rows(OPS)
        assert op == {} and [r["motivo"] for r in R] == ["obiettivo 1", "stop", "stop"] and [r["parte"] for r in R] == ["1", "2", "3"]
        assert all(net(r) for r in R) and "Finora: 3 chiuse, 1 in guadagno" in prompt({}, 3800, None, R, [], now)
        # Allo scoperto: tre parti, l'oro scende a 3740, l'AI chiude tutto e il guadagno è positivo.
        A.ask = lambda text, system: asked.append(text) or ({"GOLD": {**short, "fiducia": 55, "perche": "resistenza"}}, "claude-x")
        decide(now, op, 3770.0)
        assert [x[0] for x in fake["sent"][-3:]] == ["SELL"] * 3 and op["GOLD"]["verso"] == -1
        assert A.rows(DEC)[-1]["esito"].startswith("vendute allo scoperto 0.11 once in 3 parti")
        decide(now, op, 3750.0)                                               # già dentro: tiene
        assert "al ribasso (allo scoperto)" in asked[-1] and "+0.53% a favore" in asked[-1] and len(fake["sent"]) == 6
        fake["out"] = 3740.0
        A.ask = lambda text, system: ({"GOLD": {"azione": "chiudi"}}, "claude-x")
        decide(now, op, 3740.0)
        R = A.rows(OPS)[-3:]
        assert op == {} and all(r["verso"] == "-1" and r["motivo"] == "AI" and float(r["lordo_eur"]) > 0 and net(r) for r in R)
        assert float(R[0]["lordo_eur"]) == round(0.04 * 30 / 1.1, 4) and not fake["pos"]
        # Le notti: alle 22 UTC il tasso del verso sull'esposizione, una volta sola per notte; poi nei costi.
        op = {"GOLD": {"apertura": "2026-11-18 11:00", "modello": "m", "verso": 1, "stop": 3750.0, "deal": "d7", "perche": "x",
                       "parti": [{"n": 2, "deal": "d7", "once": 0.1, "entrata": 3770.0, "obiettivo": 3800.0}]}}
        regroup(op["GOLD"])
        fake["pos"]["d7"] = {"dealId": "d7", "size": 0.1, "level": 3770.0}
        nights(op["GOLD"], fee, 3770, op)                                     # la notte di ieri: era prima dell'entrata
        assert "notte_eur" not in op["GOLD"]["parti"][0]
        fee2 = {**fee, "swapChargeTimestamp": fee["swapChargeTimestamp"] + 86400000}
        nights(op["GOLD"], fee2, 3770, op)
        nights(op["GOLD"], fee2, 3770, op)
        q = op["GOLD"]["parti"][0]
        assert q["notte_eur"] == round(0.1 * 3770 * -0.02 / 100 / 1.1, 4) and q["notte_fino"] == "2026-11-18 22:00"
        fake["out"] = 3770.0
        close(op["GOLD"], "AI", now, op)
        r = A.rows(OPS)[-1]
        assert float(r["lordo_eur"]) == 0 and float(r["costi_eur"]) > -q["notte_eur"] and net(r)
        # Chiusa dal broker (stop) mentre il bot era fermo, operazione del formato di prima (una parte sola).
        OPEN.write_text(json.dumps({"GOLD": {"apertura": "2026-11-18 11:00", "modello": "m", "once": 0.1, "investito": 377,
                                             "entrata": 3770.0, "stop": 3750.0, "obiettivo": 3800.0, "deal": "d2", "perche": "x"}}))
        op = load_open()
        assert op["GOLD"]["parti"] == [{"n": 2, "deal": "d2", "once": 0.1, "entrata": 3770.0, "obiettivo": 3800.0}]
        reconcile(op, now)
        r = A.rows(OPS)[-1]
        assert op == {} and r["uscita"] == "3750.0" and r["motivo"] == "riavvio (chiusa dal broker)" and r["parte"] == "2" and net(r)
        assert float(r["netto_pct"]) < round((3750 / 3770 - 1) * 100 - COST, 4)
        assert round(equity(100), 4) == round(100 - sum(float(x["costi_eur"]) for x in A.rows(OPS)), 4) < 100
        # Il venerdì sera il modello non può aprire; il modello giù non apre niente.
        A.ask = lambda text, system: ({"GOLD": {**buy}}, "claude-x")
        decide(pd.Timestamp("2026-11-20 21:30", tz="UTC"), op, 3770.0)
        assert A.rows(DEC)[-1]["esito"].startswith("venerdì sera") and len(fake["sent"]) == 6
        A.ask = lambda text, system: (_ for _ in ()).throw(RuntimeError("429"))
        ALERTS[:] = [3790.0]
        decide(now, op, 3770.0)
        assert A.rows(DEC)[-1]["esito"] == "modello non disponibile" and ALERTS == [3790.0]
        # Chiusa a mano dall'utente (dashboard): prezzo vero dallo storico, niente costi stimati dello stop.
        sh = lambda deal: {"GOLD": {"apertura": "2026-11-18 11:00", "modello": "m", "verso": -1, "stop": 3790.0, "deal": deal,
                                    "perche": "x", "parti": [{"n": 1, "deal": deal, "once": 0.1, "entrata": 3770.0, "obiettivo": 3740.0}]}}
        op = sh("d4")
        fake["hist"] = (3760.0, "USER")
        reconcile(op, now)
        r = A.rows(OPS)[-1]
        assert op == {} and r["motivo"] == "a mano" and r["uscita"] == "3760.0" and float(r["lordo_pct"]) == round((1 - 3760 / 3770) * 100, 4)
        fake["hist"] = (3791.0, "USER")                                        # oltre lo stop: è lo stop, non la mano
        op = sh("d5")
        reconcile(op, now)
        assert A.rows(OPS)[-1]["motivo"] == "riavvio (chiusa dal broker)" and A.rows(OPS)[-1]["uscita"] == "3791.0"
        fake["hist"] = None
        # "Chiudi adesso" da Vercel: vale solo per il suo deal e chiude tutte le parti; la lezione sa che l'ha chiusa l'utente.
        assert not asked_close({"deal": "d6"})
        CLOSE.write_text('{"deal": "d6"}')
        assert asked_close({"deal": "d6"}) and not asked_close({"deal": "d7"})
        op = {"GOLD": {**sh("d6")["GOLD"], "parti": [{"n": 1, "deal": "d6", "once": 0.05, "entrata": 3770.0, "obiettivo": 3760.0},
                                                    {"n": 2, "deal": "d8", "once": 0.05, "entrata": 3770.0, "obiettivo": 3740.0}]}}
        fake["pos"].update(d6={"dealId": "d6"}, d8={"dealId": "d8"})
        fake["out"] = 3765.0
        watch(op, now, 3764, 3765, None)
        assert op == {} and [r["motivo"] for r in A.rows(OPS)[-2:]] == ["a mano", "a mano"] and not fake["pos"]
        CLOSE.write_text('rotto')
        assert not asked_close({"deal": "d6"})
        hand = {"chiusura": "x", "simbolo": "GOLD", "verso": "-1", "apertura": "a", "entrata": "3770.0", "uscita": "3760.0",
                "motivo": "a mano", "netto_pct": "0.26", "netto_eur": "0.9", "perche": "p"}
        assert "L'HA CHIUSA L'UTENTE" in by_hand(hand, 3750.0) and "+0.53% lordo" in by_hand(hand, 3750.0)
        assert by_hand({**hand, "motivo": "AI"}, 3750.0) == ""
        assert "L'HA CHIUSA L'UTENTE" in prompt({}, 3750.0, None, [hand], [], now)
        # Venerdì sera: tutte le parti si chiudono.
        op = {"GOLD": {**sh("d9")["GOLD"], "parti": [{"n": 3, "deal": "d9", "once": 0.05, "entrata": 3770.0, "obiettivo": 3700.0}]}}
        fake["pos"]["d9"] = {"dealId": "d9"}
        watch(op, pd.Timestamp("2026-11-20 21:45", tz="UTC"), 3769, 3770, None)
        assert op == {} and A.rows(OPS)[-1]["motivo"] == "fine settimana"
        # Misura scelta a mano: vale per ogni parte (0,20 = tre da 0,20) e vince sempre (niente tetti del bot).
        A.ask = lambda text, system: ({"GOLD": {**buy}}, "claude-x")
        assert chosen() is None
        SETTINGS.write_text('{"once": 0.2}')
        assert chosen() == 0.2
        op = {}
        decide(now, op, 3770.0)
        assert [x[1] for x in fake["sent"][-3:]] == [0.2, 0.2, 0.2] and "scelta a mano: 3 × 0.20" in A.rows(DEC)[-1]["esito"]
        fake["pos"], op = {}, {}
        SETTINGS.write_text('{"once": 5}')                                    # oltre il tetto del giorno: resta 5
        decide(now, op, 3770.0)
        assert [x[1] for x in fake["sent"][-3:]] == [5, 5, 5] and "ridotta" not in A.rows(DEC)[-1]["esito"]
        fake["pos"] = {}
        SETTINGS.write_text('{"once": null}')
        assert chosen() is None
        SETTINGS.write_text('rotto')
        assert chosen() is None
        # Se il broker unisse gli ordini in una posizione sola: una parte, con l'obiettivo di sempre (TP2).
        def merged(d, s, st):
            p = fake["pos"].setdefault("m", {"dealId": "m", "size": 0, "level": 3770.0})
            p["size"] = round(p["size"] + s, 2)
        capital.open, op = merged, {}
        decide(now, op, 3770.0)
        assert [(q["n"], q["once"], q["obiettivo"]) for q in op["GOLD"]["parti"]] == [(2, 0.11, 3810)]
        # Posizione senza registro (giro interrotto): si chiude.
        fake["pos"] = {"d3": {"dealId": "d3"}}
        reconcile({}, now)
        assert not fake["pos"]
    finally:
        for k, v in saved_cap.items():
            setattr(capital, k, v)
        for k, v in saved_a.items():
            setattr(A, k, v)
        bot.push = saved_push
        g.update(saved)
        ALERTS.clear()
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()
    print("prove superate")


if __name__ == "__main__":
    prova() if "--prova" in sys.argv else main()
