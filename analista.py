# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""Analista AI: un modello linguistico fa l'analista, il bot fa il risk manager. Conto demo Alpaca (Paper), nel cloud
(GitHub Actions). Nato il 2026-11-16 su SOL/USD seguendo alla lettera il metodo di un video; dallo stesso giorno opera su
tutti i mercati e ha preso il posto degli altri bot (DIARIO.md, "Analista AI").

In parole semplici, ogni 15 minuti a Borsa USA aperta, ogni ora quando è chiusa (dal 2026-11-17, ci sono solo le crypto:
così si consumano meno limiti dell'abbonamento; stop e obiettivo li controlla comunque il bot ogni minuto):
1. il bot manda al modello, in UNA sola domanda, le candele chiuse di ogni mercato su quattro scale (15 minuti, 1 ora,
   4 ore, 1 giorno), le operazioni aperte, gli ultimi tre risultati di ogni mercato, gli annunci economici importanti
   in arrivo, i titoli delle notizie delle ultime 2 ore sui suoi mercati (servizio notizie di Alpaca, gratis) e quello
   che lui stesso ha detto al giro prima su ogni mercato (così si ricorda i livelli che aspettava). Dal 2026-11-17 anche
   i conti già fatti dal bot (RSI, movimento medio, distanza dalla media, massimi e minimi), il prezzo dell'ultimo
   minuto, i suoi risultati per mercato e il suo quaderno delle lezioni;
2. il modello risponde come un analista, mercato per mercato: compra, vendi, aspetta o sposta lo stop, quanto è sicuro e
   perché, in parole;
3. il bot controlla ogni risposta e decide QUANTO: l'AI propone l'operazione e lo stop, ma la misura la calcola il bot dal
   rischio fissato qui sotto. L'AI non tocca mai il rischio.
Ogni minuto il bot controlla stop e obiettivo da solo, anche mentre il modello pensa (la domanda gira in una corsia a
parte). Ogni decisione finisce nel registro col suo perché.
Allarmi (dal 2026-11-17): il modello indica fino a due prezzi per mercato; se il prezzo ne tocca uno il bot gli fa subito
una domanda nuova, senza aspettare il giro dopo. Lezioni: per ogni operazione chiusa il modello scrive una lezione di una
frase (logs/analista_lezioni.csv) e le rilegge a ogni domanda: è il suo modo di imparare dai propri risultati.

Mercati (MARKETS): crypto 24 ore su 24; fondi di Borsa USA solo a Borsa aperta, e il bot chiude le loro operazioni 15
minuti prima della chiusura (di notte lo stop non si può controllare). Una operazione al massimo per mercato.
Livelli (LIVELLO, deciso dall'utente): "suggerisce" = scrive solo cosa farebbe; "apre_chiude" = apre e chiude, stop
fisso; "pieno" = anche sposta lo stop (solo in su) e l'obiettivo.
Modalità prop firm: il bot ha un budget di perdita giornaliero su tutti i mercati insieme; più ne ha perso oggi, più
piccola è ogni operazione nuova, e gli stop delle operazioni aperte contano come già persi; finito il budget, niente
operazioni fino al giorno dopo (UTC). È anche il suo kill switch: conta solo le sue.
Solo acquisti: Alpaca non permette di vendere allo scoperto le crypto. Costi: crypto 0,25% a ordine più slittamento, ~0,6%
andata e ritorno; fondi senza commissioni, 0,1% di spread e slittamento (stima prudente). Il modello lo sa, ed è il motivo
per cui il bot rifiuta obiettivi troppo vicini.
Modello: Claude (MODELLO = "opus", l'Opus più recente; o "sonnet"), con l'abbonamento Claude dell'utente tramite Claude
Code: nessuna chiave a pagamento, consuma i limiti dell'abbonamento. Cambiare modello = cambiare una riga del workflow, non
il bot. Se il modello non risponde (limiti finiti), si aspetta.
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
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pandas as pd

import bot  # helper di Timestral: chiamate ad Alpaca, ordini con ID unico, minuti, salvataggio del registro

MARKETS = {  # simbolo: (fondo di Borsa?, costi stimati andata e ritorno in %)
    "BTC/USD": (False, 0.6), "ETH/USD": (False, 0.6), "SOL/USD": (False, 0.6),
    "SPY": (True, 0.1),   # le 500 aziende più grandi degli USA
    "QQQ": (True, 0.1),   # Nasdaq
    "GLD": (True, 0.1),   # oro
    "IWM": (True, 0.1),   # piccole aziende USA
    "DIA": (True, 0.1),   # Dow Jones
    "TLT": (True, 0.1),   # titoli di Stato USA a lungo termine
    "SLV": (True, 0.1),   # argento
    "XLE": (True, 0.1),   # energia
    "EFA": (True, 0.1),   # Borse dei Paesi sviluppati fuori dagli USA
    # fondi inversi (dal 2026-11-16): salgono quando il loro mercato scende, così comprando si punta al ribasso
    "SH": (True, 0.1),    # contro l'S&P 500 (SPY)
    "PSQ": (True, 0.1),   # contro il Nasdaq (QQQ)
    "BITI": (True, 0.1),  # contro il Bitcoin; a Borsa aperta, come gli altri fondi
}
MODEL = os.environ.get("MODELLO", "opus")    # alias di Claude Code: sempre l'Opus più recente
CLAUDE = os.environ.get("CLAUDE", "claude")  # il programma Claude Code
OLLAMA = os.environ.get("OLLAMA", "http://127.0.0.1:11434")  # i modelli sul Mac (dal 2026-11-20): MODELLO con ":", es. qwen3-vl:30b
LIVELLO = os.environ.get("LIVELLO", "pieno")
RISK, QUOTA, BUDGET = 0.005, 0.20, 0.02  # rischio a operazione, massimo per operazione, perdita massima del giorno (del conto)
MIN_STOP, MAX_STOP = 0.005, 0.04          # stop fra 0,5% e 4% sotto il prezzo
STOCK_MIN = 1                             # minimo per ordine di un fondo (frazioni comprese); crypto: bot.CRYPTO_MIN
FRAMES = {"15Min": "15min", "1Hour": "1h", "4Hour": "4h", "1Day": "1D"}
FETCH, SEND = 60, 16  # candele scaricate per scala (servono ai conti) e candele mandate al modello
LATEST = {False: ("https://data.alpaca.markets/v1beta3/crypto/us/latest/bars", {}),  # ultima candela da 1 minuto
          True: ("https://data.alpaca.markets/v2/stocks/bars/latest", {"feed": "iex"})}
GAP = 300  # secondi minimi fra due domande quando scatta un allarme (i limiti dell'abbonamento)
CAL_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
NEWS_URL, NEWS_HOURS = "https://data.alpaca.markets/v1beta1/news", 2
MINUTES = int(os.environ.get("MINUTI", 330))
OPS, DEC, OPEN = bot.LOGS / "analista.csv", bot.LOGS / "analista_decisioni.csv", bot.LOGS / "analista_aperta.json"
LESSONS = bot.LOGS / "analista_lezioni.csv"
FIELDS = ["apertura", "chiusura", "simbolo", "modello", "investito", "entrata", "uscita", "motivo", "perche", "lordo_pct",
          "netto_pct", "netto_usd"]
DFIELDS = ["ora", "simbolo", "modello", "prezzo", "azione", "fiducia", "stop", "obiettivo", "perche", "esito", "allarmi"]
LFIELDS = ["ora", "simbolo", "chiusura", "lezione"]
ALERTS = {}              # simbolo: prezzi che fanno scattare subito una domanda (li sceglie il modello)
LOCK = threading.Lock()  # operazioni aperte, registri e ordini: la sorveglianza e la domanda non si pestano i piedi
NEWS = {"t": None, "v": []}
SYSTEM = f"""Sei un analista di trading. Ogni 15 minuti (ogni ora quando la Borsa USA è chiusa, e subito quando scatta un tuo
allarme) ricevi i dati di alcuni mercati e decidi per ciascuno.
Crypto: 24 ore su 24. Fondi di Borsa USA: arrivano solo a Borsa aperta, e le loro operazioni il bot le chiude da solo 15
minuti prima della chiusura. Puoi solo comprare e poi vendere (niente vendite allo scoperto). Per puntare al ribasso
compra un fondo inverso: SH sale quando SPY scende, PSQ quando QQQ scende, BITI quando il Bitcoin scende. Ogni mercato ha i suoi
costi andata e ritorno, scritti nei dati: un'operazione ha senso solo se il movimento atteso è ben più grande. Lo stop di
un acquisto va fra {MIN_STOP:.1%} e {MAX_STOP:.0%} sotto il prezzo. Non decidi quanto investire: lo calcola il risk manager.
Rispondi SOLO con un oggetto JSON con una chiave per ogni mercato ricevuto, scritta come nei dati:
{{"BTC/USD": {{"azione": "compra" | "vendi" | "aspetta" | "sposta_stop", "fiducia": 0-100, "stop": prezzo,
"obiettivo": prezzo, "allarmi": [prezzo, prezzo], "perche": "il motivo in due o tre frasi, in italiano"}}, ...,
"_lezioni": [{{"simbolo": "BTC/USD", "chiusura": "AAAA-MM-GG HH:MM", "lezione": "una frase"}}]}}
Senza operazione aperta: "compra" o "aspetta". Con un'operazione aperta: "aspetta" (tieni), "vendi" (chiudi) o
"sposta_stop" (nuovo stop e obiettivo; lo stop può solo salire).
Ricevi anche i titoli delle notizie recenti: usali come contesto, non come segnale da soli (un titolo può essere già nei
prezzi, o non contare).
Per ogni mercato il bot ti dà i conti già fatti (RSI, movimento medio di una candela, distanza dalla media di 20 candele,
massimi e minimi): usa quelli invece di rifarli sulle candele. Il movimento medio dice quanto si muove il prezzo di
solito: uno stop più vicino di così scatta per rumore.
"allarmi": fino a due prezzi per mercato (lista vuota se non servono). Il bot li guarda ogni minuto e, se il prezzo ne
tocca uno, ti fa subito una domanda nuova invece di aspettare il giro dopo: mettili dove agiresti o cambieresti idea
(la rottura che aspetti, il supporto che non deve cedere).
"_lezioni": per ogni operazione chiusa che ti viene data da commentare, una lezione di una frase su cosa ha funzionato o
no, utile alle prossime. Le tue lezioni di prima le ricevi a ogni domanda: tienine conto, sono la tua esperienza.
Lo scopo è fare operazioni e imparare da come vanno: aspettare non è gratis, perché senza operazioni non si impara niente.
Il rischio è piccolo e già fissato: il risk manager sceglie l'importo in modo che arrivare allo stop costi solo lo
{RISK:.1%} del conto, e c'è un tetto alla perdita del giorno. Non serve la certezza: compra quando hai un motivo concreto e
l'obiettivo, al netto dei costi, è almeno una volta e mezza la distanza dello stop, anche con fiducia media. In un mercato
laterale si può comprare vicino al supporto, con lo stop sotto. Se il quadro è ribassista non limitarti ad aspettare:
a Borsa aperta compra il fondo inverso. Un annuncio economico in arrivo non è da solo un motivo per aspettare: entra con lo
stop, oppure aspetta solo se mancano meno di 30 minuti. Orizzonte: crypto da qualche ora a qualche giorno; fondi entro la
giornata. Ricevi anche quello che hai detto al giro prima su ogni mercato: se il livello che aspettavi è arrivato, agisci.
"fiducia" è quanto credi nell'operazione che proponi (per "aspetta", quanto credi che comprare adesso sia sbagliato)."""


def plain(sym):
    return sym.replace("/", "")


def table(df):
    return "\n".join(f"{t:%m-%d %H:%M},{r.o:.3f},{r.h:.3f},{r.l:.3f},{r.c:.3f},{round(r.v, 3)}" for t, r in df.iterrows())  # volume crypto: frazioni di moneta


def numbers(df):
    """I conti che il modello farebbe a occhio: RSI a 14 candele, movimento medio di una candela (ATR a 14) in % del
    prezzo, distanza del prezzo dalla media delle ultime 20 candele in %. NaN se le candele sono troppo poche."""
    c = df.c
    d = c.diff()
    rsi = 100 - 100 / (1 + d.clip(lower=0).rolling(14).mean() / (-d.clip(upper=0)).rolling(14).mean())
    tr = pd.concat([df.h - df.l, (df.h - c.shift()).abs(), (df.l - c.shift()).abs()], axis=1).max(axis=1)
    return rsi.iloc[-1], (tr.rolling(14).mean() / c * 100).iloc[-1], ((c / c.rolling(20).mean() - 1) * 100).iloc[-1]


def fmt(x, f, unit=""):
    return f"{x:{f}}{unit}" if pd.notna(x) else "n.d."


def stats(done, sym):
    """I risultati del modello su un mercato, tutti insieme."""
    r = [x for x in done if x["simbolo"] == sym]
    if not r:
        return "Finora nessuna operazione chiusa."
    pct = [float(x["netto_pct"]) for x in r]
    return (f"Finora: {len(r)} chiuse, {sum(v > 0 for v in pct)} in guadagno, media {sum(pct) / len(pct):+.2f}%, "
            f"totale {sum(float(x['netto_usd']) for x in r):+.2f} $.")


def prompt(data, op, done, events, now, heads=(), prev=None, lessons=(), why=""):
    """Il messaggio per il modello: tutti i mercati in una domanda, niente di più. data = {simbolo: (candele, prezzo)};
    prev = {simbolo: la sua ultima decisione dal registro}; lessons = le sue lezioni; why = perché la domanda arriva
    prima del giro (un allarme)."""
    s = [f"Ora: {now:%Y-%m-%d %H:%M} UTC."] + ([f"Domanda anticipata: {why}."] if why else [])
    for sym, (frames, P) in data.items():
        stock, cost = MARKETS[sym]
        kind = "fondo di Borsa USA" if stock else "crypto"
        s.append(f"\n\n=== {sym} ({kind}, costi ~{cost}% andata e ritorno). Prezzo (ultimo minuto): {P:.3f}")
        if frames:
            s.append("\nConti fatti (RSI 14 | movimento medio di una candela | prezzo rispetto alla media di 20 candele): "
                     + "; ".join(f"{tf} {fmt(r, '.0f')} | {fmt(a, '.2f', '%')} | {fmt(m, '+.2f', '%')}"
                                 for tf, (r, a, m) in ((tf, numbers(df)) for tf, df in frames.items())))
        h1, d1 = frames.get("1Hour"), frames.get("1Day")
        if h1 is not None and len(h1) and d1 is not None and len(d1):
            s.append(f"\nMassimo / minimo: ultime 24 candele da 1 ora {h1.h.tail(24).max():.3f} / {h1.l.tail(24).min():.3f}; "
                     f"ultimi 7 giorni {d1.h.tail(7).max():.3f} / {d1.l.tail(7).min():.3f}.")
        for tf, df in frames.items():
            s.append(f"\nCandele {tf} chiuse, le ultime {SEND} (ora UTC,apertura,massimo,minimo,chiusura,volume):\n"
                     f"{table(df.tail(SEND))}")
        t = op.get(sym)
        if t:
            mins = (now - pd.Timestamp(t["apertura"], tz="UTC")) / timedelta(minutes=1)
            s.append(f"\nOperazione aperta da {mins:.0f} minuti: entrata {t['entrata']:.3f}, stop {t['stop']:.3f}, "
                     f"obiettivo {t['obiettivo']:.3f}, ora {(P / t['entrata'] - 1) * 100:+.2f}% (costi esclusi).")
        else:
            s.append("\nNessuna operazione aperta.")
        last = [r for r in done if r["simbolo"] == sym][-3:]
        p = (prev or {}).get(sym)
        if p:
            s.append(f"\nAl giro prima ({p['ora']}) avevi detto: {p['azione']}, fiducia {p['fiducia']}. {p['perche']}")
        s.append("\nUltimi risultati: " + ("; ".join(f"{r['chiusura']} {float(r['netto_pct']):+.2f}% ({r['motivo']})"
                                                     for r in last) or "nessuno") + ". " + stats(done, sym))
    s.append("\n\nNotizie importanti (ora UTC): " + ("; ".join(events) or "nessuna nelle prossime 24 ore"))
    s.append(f"\n\nTitoli delle notizie delle ultime {NEWS_HOURS} ore (ora UTC, mercati collegati):\n"
             + ("\n".join(heads) or "nessuno"))
    s.append("\n\nLe tue lezioni (le più recenti):" + ("".join(f"\n- [{r['simbolo']}] {r['lezione']}" for r in lessons[-15:])
                                                      or " nessuna, per ora"))
    todo = pending(done, lessons)
    if todo:
        s.append('\n\nOperazioni chiuse da commentare in "_lezioni":' + "".join(
            f"\n- {r['simbolo']} chiusura {r['chiusura']}: aperta {r['apertura']} a {r['entrata']}, uscita a {r['uscita']} "
            f"({r['motivo']}), netto {float(r['netto_pct']):+.2f}%. Il tuo perché d'entrata: {r['perche']}" for r in todo))
    return "\n".join(s)


def pending(done, lessons):
    """Le operazioni chiuse (al massimo le ultime 5) che non hanno ancora la loro lezione."""
    seen = {(r["simbolo"], r["chiusura"]) for r in lessons}
    return [r for r in done if (r["simbolo"], r["chiusura"]) not in seen][-5:]


def levels(a, P):
    """Gli allarmi proposti dal modello, controllati: al massimo due prezzi, entro il 10% dal prezzo di adesso."""
    raw, out = a.get("allarmi"), []
    for x in raw[:2] if isinstance(raw, list) else []:
        try:
            x = float(x)
        except (TypeError, ValueError):
            continue
        if 0.9 * P <= x <= 1.1 * P:
            out.append(x)
    return out


def ring(px, syms):
    """Gli allarmi che l'ultimo minuto ha toccato: [(simbolo, prezzo)]. px = {simbolo: (massimo, minimo, chiusura)}."""
    return [(s, x) for s in syms if s in px for x in ALERTS.get(s, []) if px[s][1] <= x <= px[s][0]]


def check(a, P, t, cost, livello=LIVELLO, lo=MIN_STOP, hi=MAX_STOP):
    """La risposta del modello per un mercato passata dal risk manager: (cosa fare, perché no). La risposta è un testo
    che viene da fuori, quindi ogni numero si controlla."""
    act = str(a.get("azione", "")).strip().lower()
    try:
        stop, tgt = float(a.get("stop") or 0), float(a.get("obiettivo") or 0)
    except (TypeError, ValueError):
        return "aspetta", "numeri non validi"
    if t is None:
        if act != "compra":
            return "aspetta", ""
        if not P * (1 - hi) <= stop <= P * (1 - lo):
            return "aspetta", f"stop {stop} fuori dai limiti"
        if tgt < P * (1 + 2 * cost / 100):
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


def risk_open(op):
    """Quanto si perderebbe se tutte le operazioni aperte finissero allo stop."""
    return sum(t["investito"] * (1 - t["stop"] / t["entrata"]) for t in op.values())


def size(equity, today_usd, stop_pct, open_risk=0.0, minimum=bot.CRYPTO_MIN):
    """Quanto investire: rischio fisso a stop pieno, ridotto da quanto si è già perso oggi (modalità prop firm), e mai
    più di quanto resta del budget del giorno contando gli stop delle operazioni aperte.
    0 = non si entra (budget finito, o importo sotto il minimo d'ordine di Alpaca)."""
    scale = max(0.0, 1 + min(today_usd, 0) / (BUDGET * equity))
    room = max(0.0, BUDGET * equity + min(today_usd, 0) - open_risk)
    amount = min(min(RISK * equity * scale, room) / stop_pct, QUOTA * equity)
    return amount if amount >= minimum else 0.0


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
    if path.exists():
        with path.open() as f:
            head = f.readline().strip().split(",")
        if head != fields:  # registro di prima (solo SOL, senza colonna simbolo): si riscrive una volta
            old = rows(path)
            with path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows({"simbolo": "SOL/USD", **r} for r in old)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow(row)


def load_open():
    try:
        op = json.loads(OPEN.read_text()) or {}
    except (FileNotFoundError, ValueError):
        op = {}
    return {"SOL/USD": op} if "entrata" in op else op  # file di prima: una sola operazione, su SOL


def save_open(op):
    bot.LOGS.mkdir(exist_ok=True)
    OPEN.write_text(json.dumps(op, indent=1))


def candles(sym, tf, n, now):
    """Le ultime n candele CHIUSE di una scala (la candela in corso si scarta). I fondi hanno candele solo a Borsa
    aperta: si chiede un periodo più lungo (notti e fine settimana)."""
    step, stock = pd.Timedelta(FRAMES[tf]), MARKETS[sym][0]
    url, feed = (bot.STOCK_BARS, {"feed": "iex"}) if stock else (bot.CRYPTO_BARS, {})
    back = max(step * (n + 3) * 7, pd.Timedelta(days=5)) if stock else step * (n + 3)
    p = {"symbols": sym, "timeframe": tf, "start": f"{now - back:%Y-%m-%dT%H:%M:%SZ}", "limit": 10000, **feed}
    page = bot.get_json(urllib.request.Request(f"{url}?{urllib.parse.urlencode(p)}", headers=bot.keys()))
    df = pd.DataFrame(page["bars"].get(sym, []))
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


def pick(items, syms):
    """I titoli che parlano dei nostri mercati, dal più vecchio, con i mercati collegati."""
    names = {plain(s): s for s in syms}
    out = []
    for a in sorted(items, key=lambda a: a["created_at"]):
        mine = [names[x] for x in a.get("symbols", []) if x in names]
        if mine:
            out.append(f"{pd.Timestamp(a['created_at']):%H:%M} [{', '.join(mine)}] {a['headline']}")
    return out


def headlines(syms, now):
    """I titoli delle ultime NEWS_HOURS ore sui mercati di questo giro (servizio notizie di Alpaca, incluso nel conto
    gratuito). Se non risponde, si va avanti senza."""
    p = {"symbols": ",".join(plain(s) for s in syms), "start": f"{now - timedelta(hours=NEWS_HOURS):%Y-%m-%dT%H:%M:%SZ}",
         "limit": 50, "sort": "desc"}
    try:
        page = bot.get_json(urllib.request.Request(f"{NEWS_URL}?{urllib.parse.urlencode(p)}", headers=bot.keys()), tries=1)
        return pick((page or {}).get("news", []), syms)
    except Exception as e:
        print(f"notizie non disponibili: {e}", flush=True)
        return []


def last_prices(syms, now):
    """L'ultima candela da 1 minuto di ogni mercato, in due richieste in tutto (crypto e fondi): {simbolo: (massimo,
    minimo, chiusura)}. Una candela più vecchia di 10 minuti si scarta: dati fermi, meglio non decidere su quelli."""
    out = {}
    for stock, (url, feed) in LATEST.items():
        mine = [s for s in syms if MARKETS[s][0] == stock]
        if mine:
            q = urllib.parse.urlencode({"symbols": ",".join(mine), **feed})
            page = bot.get_json(urllib.request.Request(f"{url}?{q}", headers=bot.keys())) or {}
            for s, b in page.get("bars", {}).items():
                if now - pd.Timestamp(b["t"]) <= timedelta(minutes=10):
                    out[s] = (float(b["h"]), float(b["l"]), float(b["c"]))
    return out


def ask(text, system=SYSTEM):
    """Il modello col tuo abbonamento Claude: `claude -p` (Claude Code) senza strumenti, solo testo. Nel cloud si
    autentica col token CLAUDE_CODE_OAUTH_TOKEN (fatto con `claude setup-token`): nessuna chiave a pagamento.
    Ritorna (risposta, nome esatto del modello che ha risposto).
    Un nome con ":" (es. qwen3-vl:30b) è un modello aperto sul Mac con Ollama: niente termini d'uso di terzi."""
    if ":" in MODEL:
        body = {"model": MODEL, "stream": False, "format": "json", "think": False, "keep_alive": "20m", "options": {"num_ctx": 16384},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}
        req = urllib.request.Request(f"{OLLAMA}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            s = json.load(r)["message"]["content"]
        return json.loads(s[s.index("{"):s.rindex("}") + 1]), MODEL
    r = subprocess.run([CLAUDE, "-p", "--model", MODEL, "--tools", "", "--system-prompt", system, "--output-format", "json"],
                       input=text, capture_output=True, text=True, timeout=600)
    try:
        out = json.loads(r.stdout)
    except ValueError:
        raise RuntimeError(f"claude: {(r.stdout or r.stderr)[-300:]}") from None
    if r.returncode or out.get("is_error"):
        raise RuntimeError(f"claude: {str(out.get('result') or r.stderr)[:300]}")
    s = out["result"]
    return json.loads(s[s.index("{"):s.rindex("}") + 1]), ",".join(out.get("modelUsage") or [MODEL])


def held(sym):
    p = bot.api("GET", f"/v2/positions/{plain(sym)}", missing_ok=True)
    return float(p["qty"]) if p and float(p["market_value"]) > 1 else 0.0


def order(sym, side, cid, **amount):
    return bot.execute({"symbol": sym, "side": side, "type": "market", "client_order_id": cid,
                        "time_in_force": "day" if MARKETS[sym][0] else "gtc", **amount})


def close(sym, t, why, now, op):
    """Vende tutto e scrive il risultato; l'operazione esce da op. Se la vendita non è eseguita resta in op e si riprova."""
    q = held(sym)
    if not q:  # chiusa altrove: niente riga finta
        print(f"{sym}: posizione già chiusa, la tolgo senza registrarla", flush=True)
        op.pop(sym, None)
        save_open(op)
        return
    x = order(sym, "sell", f"ai-{plain(sym)}-out-{now:%Y%m%d%H%M}", qty=f"{q:.9f}")
    if x["status"] != "filled":
        print(f"{sym}: vendita non eseguita ({x['status']}), riprovo al prossimo minuto", flush=True)
        return
    out, cost = float(x["filled_avg_price"]), MARKETS[sym][1]
    gross = (out / t["entrata"] - 1) * 100
    row = {"apertura": t["apertura"], "chiusura": f"{now:%Y-%m-%d %H:%M}", "simbolo": sym, "modello": t["modello"],
           "investito": round(t["investito"], 2), "entrata": t["entrata"], "uscita": out, "motivo": why,
           "perche": t["perche"], "lordo_pct": round(gross, 4), "netto_pct": round(gross - cost, 4),
           "netto_usd": round(t["investito"] * (gross - cost) / 100, 4)}
    append(OPS, FIELDS, row)
    print(row, flush=True)
    op.pop(sym)
    save_open(op)
    bot.push(f"analista {sym} {row['netto_pct']:+.2f}% ({why})")


def buy(sym, a, P, now, equity, cash, today_usd, op, model):
    """Compra quanto dice il risk manager. Ritorna (esito, dollari spesi)."""
    stop = float(a["stop"])
    amount = size(equity, today_usd, 1 - stop / P, risk_open(op), STOCK_MIN if MARKETS[sym][0] else bot.CRYPTO_MIN)
    if not amount or amount > cash * 0.9:
        return f"importo {amount:.2f} $ non possibile (contanti {cash:.2f} $, oggi {today_usd:+.2f} $)", 0.0
    o = order(sym, "buy", f"ai-{plain(sym)}-in-{now:%Y%m%d%H%M}", notional=f"{amount:.2f}")
    if o["status"] != "filled":  # non confermato: si annulla e si guarda cosa è stato comprato davvero
        try:
            bot.api("DELETE", f"/v2/orders/{o['id']}")
        except RuntimeError:
            pass
        o = bot.api("GET", f"/v2/orders/{o['id']}")
        if not float(o["filled_qty"] or 0):
            return f"acquisto non eseguito ({o['status']})", 0.0
    E = float(o["filled_avg_price"])
    t = {"apertura": f"{now:%Y-%m-%d %H:%M}", "modello": model, "investito": float(o["filled_qty"]) * E, "entrata": E,
         "stop": stop, "obiettivo": float(a["obiettivo"]), "perche": str(a.get("perche", ""))[:500]}
    op[sym] = t
    save_open(op)
    return f"comprati {t['investito']:.2f} $", t["investito"]


def stock_window(clock, now):
    """Si apre un'operazione su un fondo solo a Borsa aperta, da 15 minuti dopo l'apertura a 30 prima della chiusura."""
    return clock["is_open"] and now.tz_convert(bot.NY).time() >= pd.Timestamp("09:45").time() \
        and pd.Timestamp(clock["next_close"]) - now > timedelta(minutes=30)


def decide(now, op, equity, cash, clock, skip=(), why=""):
    """Un giro: dati di tutti i mercati (scaricati insieme) → una domanda al modello → risk manager su ogni risposta →
    (forse) ordini. Gira accanto alla sorveglianza: operazioni, registri e ordini si toccano solo col LOCK, così mentre
    il modello pensa gli stop si controllano lo stesso."""
    window = stock_window(clock, now)
    syms = [s for s, (stock, _) in MARKETS.items() if s not in skip and (window or not stock)]
    if not syms:
        return
    with ThreadPoolExecutor(8) as ex:
        jobs = {(s, tf): ex.submit(candles, s, tf, FETCH, now) for s in syms for tf in FRAMES}
    try:
        px = last_prices(syms, now)
    except Exception as e:  # senza l'ultimo minuto si usa la chiusura dell'ultima candela da 15 minuti
        px = {}
        print(f"ERRORE prezzi: {e}", flush=True)
    data = {}
    for sym in syms:
        try:
            frames = {tf: jobs[sym, tf].result() for tf in FRAMES}
            data[sym] = (frames, px[sym][2] if sym in px else float(frames["15Min"].c.iloc[-1]))
        except Exception as e:  # dati che mancano: questo mercato salta il giro
            print(f"ERRORE dati {sym}: {e}", flush=True)
    if not data:
        return
    with LOCK:
        snap, done, lessons = {k: dict(v) for k, v in op.items()}, rows(OPS), rows(LESSONS)
        prev = {r["simbolo"]: r for r in rows(DEC) if r["azione"] != "errore"}  # l'ultima di ogni mercato
    try:
        resp, model, err = *ask(prompt(data, snap, done, news(now), now, headlines(data, now), prev, lessons, why)), ""
    except Exception as e:  # modello giù, limiti dell'abbonamento finiti, risposta non JSON: si aspetta
        resp, model, err = {}, MODEL, str(e)[:300]
    with LOCK:
        done = rows(OPS)
        today = sum(float(r["netto_usd"]) for r in done if r["chiusura"].startswith(f"{now:%Y-%m-%d}"))
        news_push = []
        for sym, (_, P) in data.items():
            t = op.get(sym)
            a = resp.get(sym) if isinstance(resp.get(sym), dict) else {}
            if err:
                a, act, why_not = {"azione": "errore", "perche": err}, None, "modello non disponibile"
            else:
                act, why_not = check(a, P, t, MARKETS[sym][1])
                ALERTS[sym] = levels(a, P)
            if why_not or act in ("aspetta", "tieni"):
                result = why_not or act
            elif LIVELLO == "suggerisce":
                result = "solo suggerimento"
            elif act == "compra":
                result, spent = buy(sym, a, P, now, equity, cash, today, op, model)
                cash -= spent
            elif act == "vendi":
                close(sym, t, "AI", now, op)
                result = "venduto" if sym not in op else "vendita non eseguita"
            elif act == "sposta_stop":
                t.update(stop=float(a["stop"]), obiettivo=float(a["obiettivo"]))
                save_open(op)
                result = f"stop {t['stop']:.3f}, obiettivo {t['obiettivo']:.3f}"
            append(DEC, DFIELDS, {"ora": f"{now:%Y-%m-%d %H:%M}", "simbolo": sym, "modello": model, "prezzo": P,
                                  "azione": a.get("azione"), "fiducia": a.get("fiducia"), "stop": a.get("stop"),
                                  "obiettivo": a.get("obiettivo"), "perche": str(a.get("perche", ""))[:500], "esito": result,
                                  "allarmi": "" if err else ";".join(map(str, ALERTS[sym]))})
            print(f"{now:%H:%M} {sym} {P:.3f} AI: {a.get('azione')} ({a.get('fiducia')}) → {result}. {a.get('perche', '')}",
                  flush=True)
            if a.get("azione") not in ("aspetta", None):
                news_push.append(f"{sym} {a.get('azione')}")
        todo = {(r["simbolo"], r["chiusura"]) for r in pending(done, rows(LESSONS))}
        for x in resp.get("_lezioni") or [] if isinstance(resp.get("_lezioni"), list) else []:
            if isinstance(x, dict) and (x.get("simbolo"), x.get("chiusura")) in todo and x.get("lezione"):
                append(LESSONS, LFIELDS, {"ora": f"{now:%Y-%m-%d %H:%M}", "simbolo": x["simbolo"],
                                          "chiusura": x["chiusura"], "lezione": str(x["lezione"])[:300]})
                todo.discard((x["simbolo"], x["chiusura"]))
        bot.push(f"analista {now:%H:%M} " + (", ".join(news_push) or "aspetta"))


def run(*args):
    """decide() nella sua corsia: un errore lì non deve fermare la sorveglianza."""
    try:
        decide(*args)
    except Exception as e:
        print(f"ERRORE giro: {e}", flush=True)


def reconcile(sym, op, now):
    """Lo stato vero è quello del broker: un'operazione registrata ma senza posizione si toglie; una posizione senza
    registro (di un altro bot di prima, o di un giro interrotto) si vende."""
    if sym in op and not held(sym):
        close(sym, op[sym], "riavvio", now, op)
    elif sym not in op and (q := held(sym)):
        order(sym, "sell", f"ai-{plain(sym)}-start-{now:%Y%m%d%H%M}", qty=f"{q:.9f}")
        print(f"all'avvio c'erano {sym} senza registro: venduti", flush=True)


def main():
    for k in ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"):
        if not os.environ.get(k):
            sys.exit(f"Manca {k} (nel cloud: Secrets del repository).")
    for sg in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sg, lambda *_: bot.STOP.append(1))
    op = load_open()
    for o in bot.api("GET", "/v2/orders?status=open&limit=500") or []:
        if o["symbol"].replace("/", "") in map(plain, MARKETS):
            bot.api("DELETE", f"/v2/orders/{o['id']}")
    for r in rows(DEC):  # gli allarmi dell'ultimo giro valgono anche dopo un riavvio
        if r["azione"] != "errore":
            ALERTS[r["simbolo"]] = [float(x) for x in (r.get("allarmi") or "").split(";") if x]
    fresh = set(MARKETS)  # mercati ancora da riconciliare: i fondi solo a Borsa aperta (prima non si vende)
    end = time.time() + MINUTES * 60
    worker, asked = None, 0.0
    print(f"Analista AI avviato su {bot.BASE}, {len(MARKETS)} mercati, modello {MODEL}, livello {LIVELLO}; aperte: {list(op)}",
          flush=True)
    while not bot.STOP and time.time() < end:
        bot.wait_minute()
        if bot.STOP:
            break
        now = pd.Timestamp.now("UTC")
        try:
            clock = bot.api("GET", "/v2/clock")
        except Exception as e:
            print(f"ERRORE: {e}", flush=True)
            continue
        closing = pd.Timestamp(clock["next_close"]) - now < timedelta(minutes=15)
        window = stock_window(clock, now)
        with LOCK:  # la sorveglianza di ogni minuto: gira anche mentre il modello pensa
            for sym in sorted(fresh):
                if MARKETS[sym][0] and not clock["is_open"]:
                    continue
                try:
                    reconcile(sym, op, now)
                    fresh.discard(sym)
                except Exception as e:
                    print(f"ERRORE {sym}: {e}", flush=True)
            live = [s for s in MARKETS if s not in fresh and (clock["is_open"] or not MARKETS[s][0])]
            try:
                px = last_prices(live, now)
            except Exception as e:  # rete o dati: si riprova al prossimo minuto
                px = {}
                print(f"ERRORE prezzi: {e}", flush=True)
            for sym in list(op):
                stock = MARKETS[sym][0]
                if sym in fresh or (stock and not clock["is_open"]):
                    continue
                try:
                    why = "chiusura Borsa" if stock and closing else exit_reason(op[sym], *px[sym][:2]) if sym in px else ""
                    if why:
                        close(sym, op[sym], why, now, op)
                except Exception as e:  # rete, broker: si salta questo mercato e si riprova al prossimo minuto
                    print(f"ERRORE {sym}: {e}", flush=True)
            rang = ring(px, [s for s in live if window or not MARKETS[s][0]])
        due = now.minute % (15 if clock["is_open"] else 60) == 0  # Borsa chiusa: solo crypto, una domanda all'ora
        if (due or rang and time.time() - asked > GAP) and not (worker and worker.is_alive()):
            for s, x in rang:  # un allarme scatta una volta sola; il modello ne mette di nuovi a ogni risposta
                ALERTS[s].remove(x)
            why = "" if due else "è scattato il tuo allarme su " + ", ".join(f"{s} a {x}" for s, x in rang)
            try:
                acct = bot.api("GET", "/v2/account")
                worker = threading.Thread(target=run, daemon=True, args=(
                    now, op, float(acct["equity"]), float(acct["non_marginable_buying_power"]), clock, frozenset(fresh), why))
                worker.start()
                asked = time.time()
            except Exception as e:
                print(f"ERRORE: {e}", flush=True)
    if worker:
        worker.join()  # la domanda in corso finisce prima di chiudere
    print(f"Analista fermo; operazioni aperte che il giro dopo riprende: {list(op)}", flush=True)


def prova():
    P = 100.0
    t = {"apertura": "2026-11-16 10:00", "entrata": 99.0, "stop": 97.0, "obiettivo": 104.0, "investito": 20.0}
    buy_ok = {"azione": "compra", "stop": 98, "obiettivo": 103, "fiducia": 70}
    # Risk manager: senza operazione si entra solo con stop e obiettivo sensati.
    assert check(buy_ok, P, None, 0.6) == ("compra", "")
    assert check({**buy_ok, "azione": " Compra "}, P, None, 0.6)[0] == "compra"
    assert check({**buy_ok, "stop": 99.8}, P, None, 0.6)[0] == "aspetta"       # stop troppo stretto (0,2%)
    assert check({**buy_ok, "stop": 90}, P, None, 0.6)[0] == "aspetta"         # stop troppo largo (10%)
    assert check({**buy_ok, "stop": 101}, P, None, 0.6)[0] == "aspetta"        # stop sopra il prezzo
    assert check({**buy_ok, "obiettivo": 100.5}, P, None, 0.6)[0] == "aspetta"  # obiettivo dentro i costi crypto...
    assert check({**buy_ok, "obiettivo": 100.5}, P, None, 0.1)[0] == "compra"   # ...ma non dentro quelli di un fondo
    assert check({**buy_ok, "stop": "tanto"}, P, None, 0.6) == ("aspetta", "numeri non validi")
    assert check({"azione": "vendi"}, P, None, 0.6)[0] == "aspetta"            # niente da vendere
    assert check({}, P, None, 0.6)[0] == "aspetta"
    # Con operazione aperta: vendi, tieni, sposta lo stop solo in su e solo al livello pieno.
    assert check({"azione": "vendi"}, P, t, 0.6) == ("vendi", "")
    assert check(buy_ok, P, t, 0.6)[0] == "tieni"
    move = {"azione": "sposta_stop", "stop": 98.5, "obiettivo": 105}
    assert check(move, P, t, 0.6) == ("sposta_stop", "")
    assert check(move, P, t, 0.6, "apre_chiude")[0] == "tieni"
    assert check({**move, "stop": 96}, P, t, 0.6)[0] == "tieni"                # stop che scende
    assert check({**move, "stop": 100.5}, P, t, 0.6)[0] == "tieni"             # stop sopra il prezzo
    # Misura: rischio 0,5% a stop pieno, al massimo 20% del conto, ridotta da quanto si è perso oggi.
    assert size(100, 0, 0.02) == 20                                     # 0,50 $ / 2% = 25 $, tetto 20 $
    assert size(100, 0, 0.04) == 12.5                                   # 0,50 $ / 4%
    assert size(100, -1, 0.02) == 12.5                                  # metà budget perso → metà misura
    assert size(100, -1.5, 0.02) == 0                                   # 6,25 $: sotto il minimo d'ordine crypto...
    assert size(100, -1.5, 0.02, minimum=1) == 6.25                     # ...non di un fondo
    assert size(100, -2, 0.01) == 0 and size(100, -5, 0.01) == 0         # budget finito: kill switch
    assert size(100, +3, 0.02) == 20                                    # guadagnare non alza il rischio
    # Gli stop delle operazioni aperte contano come già persi: 15 mercati non possono perdere più del budget insieme.
    assert risk_open({"A": t}) == 20 * (1 - 97 / 99)
    assert round(size(100, 0, 0.02, open_risk=1.9, minimum=1), 9) == 5            # resta 0,10 $ di budget / 2%
    assert size(100, 0, 0.02, open_risk=2.5, minimum=1) == 0             # budget già tutto negli stop
    assert exit_reason(t, 105, 96) == "stop" and exit_reason(t, 105, 98) == "obiettivo" and exit_reason(t, 103, 98) == ""
    # Il messaggio al modello: ogni mercato con le sue quattro scale, i suoi costi, la sua operazione e i suoi risultati.
    idx = pd.date_range("2026-11-16 08:00", periods=3, freq="15min", tz="UTC")
    df = pd.DataFrame({"o": [1.0, 2, 3], "h": [1.5, 2.5, 3.5], "l": [0.5, 1.5, 2.5], "c": [1.2, 2.2, 3.2], "v": [10, 20, 30]}, idx)
    done = [{"simbolo": "SOL/USD", "apertura": "2026-11-16 08:00", "chiusura": "2026-11-16 09:00", "entrata": "74",
             "uscita": "73.4", "netto_pct": "-0.8", "netto_usd": "-0.16", "motivo": "stop", "perche": "supporto"},
            {"simbolo": "SPY", "apertura": "2026-11-16 09:00", "chiusura": "2026-11-16 09:30", "entrata": "600",
             "uscita": "603", "netto_pct": "0.4", "netto_usd": "0.08", "motivo": "obiettivo", "perche": "rottura"}]
    s = prompt({"SOL/USD": ({tf: df for tf in FRAMES}, P), "SPY": ({tf: df for tf in FRAMES}, 600.0)}, {"SOL/USD": t}, done,
               ["11-16 19:00 FOMC Statement"], pd.Timestamp("2026-11-16 10:30", tz="UTC"))
    sol, spy = s.split("=== SPY")
    assert all(f"Candele {tf}" in sol and f"Candele {tf}" in spy for tf in FRAMES) and "11-16 08:15,2.000,2.500,1.500,2.200,20" in s
    assert "aperta da 30 minuti" in sol and "+1.01%" in sol and "-0.80% (stop)" in sol and "+0.40%" not in sol
    prev = {"SOL/USD": {"ora": "2026-11-16 10:15", "azione": "aspetta", "fiducia": "60", "perche": "compro sopra 73,3"}}
    assert "avevi detto: aspetta, fiducia 60. compro sopra 73,3" in prompt({"SOL/USD": ({}, P)}, {}, [], [], pd.Timestamp(
        "2026-11-16 10:30", tz="UTC"), prev=prev) and "avevi detto" not in spy
    assert "Conti fatti" in sol and "Massimo / minimo" in sol and "Finora: 1 chiuse, 0 in guadagno, media -0.80%" in sol
    assert "Finora: 1 chiuse, 1 in guadagno" in spy and "SOL/USD chiusura 2026-11-16 09:00: aperta" in s
    lesson = [{"simbolo": "SOL/USD", "chiusura": "2026-11-16 09:00", "lezione": "stop troppo stretto"}]
    s2 = prompt({}, {}, done, [], pd.Timestamp("2026-11-16 10:30", tz="UTC"), lessons=lesson, why="allarme su SOL/USD a 99")
    assert "[SOL/USD] stop troppo stretto" in s2 and "SOL/USD chiusura" not in s2 and "SPY chiusura 2026-11-16 09:30" in s2
    assert "Domanda anticipata: allarme su SOL/USD a 99" in s2
    # Conti fatti dal bot: prezzo fermo con candele larghe 2 → movimento medio 2%, prezzo sulla media, RSI non definito.
    flat = pd.DataFrame({"o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0, "v": 1.0}, index=range(30))
    r, a, m = numbers(flat)
    assert pd.isna(r) and round(a, 9) == 2 and m == 0
    assert numbers(flat.assign(c=range(100, 130)))[0] == 100                    # sempre in salita: RSI 100
    assert all(pd.isna(x) for x in numbers(flat.head(3)))                       # troppo poche candele
    # Allarmi: al massimo due, numeri veri, entro il 10% dal prezzo; scattano quando l'ultimo minuto li tocca.
    assert levels({"allarmi": [101, "99.5", 98]}, 100) == [101, 99.5] and levels({"allarmi": [150, "x"]}, 100) == []
    assert levels({"allarmi": 101}, 100) == [] and levels({}, 100) == []
    ALERTS.update({"BTC/USD": [101.0], "SPY": [500.0]})
    assert ring({"BTC/USD": (101.5, 100.2, 101.0), "SPY": (600, 599, 600)}, ["BTC/USD", "SPY"]) == [("BTC/USD", 101.0)]
    assert ring({"BTC/USD": (101.5, 100.2, 101.0)}, ["SPY"]) == []
    ALERTS.clear()
    assert "costi ~0.1%" in spy and "Nessuna operazione aperta" in spy and "+0.40% (obiettivo)" in spy and "FOMC" in spy
    # Notizie: solo alto impatto sul dollaro, dall'ora scorsa alle prossime 24 ore.
    now = pd.Timestamp("2026-11-16 12:00", tz="UTC")
    NEWS.update(t=now, v=[{"title": "FOMC", "country": "USD", "impact": "High", "date": "2026-11-16T14:00:00-05:00"},
                          {"title": "PIL", "country": "EUR", "impact": "High", "date": "2026-11-16T09:00:00-05:00"},
                          {"title": "Vecchia", "country": "USD", "impact": "High", "date": "2026-11-15T09:00:00-05:00"},
                          {"title": "Minore", "country": "USD", "impact": "Low", "date": "2026-11-16T09:00:00-05:00"}])
    assert news(now) == ["11-16 19:00 FOMC"]
    # Titoli: solo quelli dei nostri mercati (BTCUSD → BTC/USD), dal più vecchio; finiscono nella domanda.
    items = [{"created_at": "2026-11-16T15:04:35Z", "symbols": ["BTCUSD", "COIN", "SPY"], "headline": "Bitcoin giù"},
             {"created_at": "2026-11-16T14:46:33Z", "symbols": ["NVDA"], "headline": "Chip"},
             {"created_at": "2026-11-16T14:39:41Z", "symbols": ["VGK", "SPY"], "headline": "Ucraina"}]
    heads = pick(items, ["BTC/USD", "SPY"])
    assert heads == ["14:39 [SPY] Ucraina", "15:04 [BTC/USD, SPY] Bitcoin giù"], heads
    assert "15:04 [BTC/USD, SPY] Bitcoin giù" in prompt({}, {}, [], [], now, heads) and "Titoli" in prompt({}, {}, [], [], now)
    # Fondi: si entra da 15 minuti dopo l'apertura a 30 prima della chiusura (ore di New York).
    clk = {"is_open": True, "next_close": "2026-11-16T16:00:00-05:00"}
    assert stock_window(clk, pd.Timestamp("2026-11-16 15:00", tz="UTC"))           # 10:00 a New York
    assert not stock_window(clk, pd.Timestamp("2026-11-16 14:40", tz="UTC"))       # 9:40: appena aperta
    assert not stock_window(clk, pd.Timestamp("2026-11-16 20:40", tz="UTC"))       # 20 minuti alla chiusura
    assert not stock_window({**clk, "is_open": False}, pd.Timestamp("2026-11-16 15:00", tz="UTC"))
    # Registri e operazioni aperte di prima (solo SOL) si leggono ancora.
    g = globals()
    tmp = bot.LOGS / "prova_analista"
    saved = {k: g[k] for k in ("OPEN",)}
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        g["OPEN"] = tmp / "aperta.json"
        g["OPEN"].write_text(json.dumps(t))
        assert load_open() == {"SOL/USD": t}
        g["OPEN"].write_text("null")
        assert load_open() == {}
        old = tmp / "dec.csv"
        old.write_text("ora,modello,esito\n2026-11-16 13:45,opus,aspetta\n")
        append(old, ["ora", "simbolo", "modello", "esito"], {"ora": "x", "simbolo": "BTC/USD", "modello": "m", "esito": "e"})
        assert [(r["simbolo"], r["esito"]) for r in rows(old)] == [("SOL/USD", "aspetta"), ("BTC/USD", "e")]
    finally:
        g.update(saved)
        for f in tmp.glob("*"):
            f.unlink()
        tmp.rmdir()
    # Posizione già chiusa altrove: si toglie dalle aperte, nessuna riga finta.
    saved = {k: g[k] for k in ("held", "save_open", "OPS")}
    g.update(held=lambda s: 0.0, save_open=lambda op: None, OPS=bot.LOGS / "prova_inesistente.csv")
    try:
        op = {"SOL/USD": t}
        close("SOL/USD", t, "stop", now, op)
        assert op == {} and not g["OPS"].exists()
    finally:
        g.update(saved)
    # Un giro completo con modello e broker finti: una domanda per tutti i mercati, una decisione per mercato.
    bars = [{"t": f"2026-11-16T{h:02d}:00:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 5} for h in range(8)]
    sent, logged, asked = [], [], []
    saved = {k: g[k] for k in ("candles", "ask", "rows", "append", "save_open", "news", "order", "MARKETS", "last_prices")}
    g.update(candles=lambda sym, tf, n, now: pd.DataFrame(bars).assign(t=lambda d: pd.to_datetime(d.t)).set_index("t"),
             ask=lambda text: asked.append(text) or ({"SOL/USD": {"azione": "compra", "fiducia": 65, "stop": 98, "obiettivo": 104,
                                                                  "allarmi": [99, 105], "perche": "rompe in su"}, "BTC/USD": "rotto",
                                                      "_lezioni": [{"simbolo": "SOL/USD", "chiusura": "2026-11-16 09:00",
                                                                    "lezione": "stop sotto il supporto"},
                                                                   {"simbolo": "ETH/USD", "chiusura": "x", "lezione": "inventata"}]},
                                                     "claude-x"),
             last_prices=lambda syms, now: {"SOL/USD": (100.5, 99.5, 100.0)},
             rows=lambda p: done[:1] if p == OPS else [], append=lambda p, f, r: logged.append(r), save_open=lambda op: None, news=lambda now: [],
             order=lambda sym, side, cid, **a: sent.append((sym, side, a)) or {"status": "filled", "filled_avg_price": "100",
                                                                               "filled_qty": "0.2"},
             MARKETS={"BTC/USD": (False, 0.6), "SOL/USD": (False, 0.6), "SPY": (True, 0.1)})
    closed = {"is_open": False, "next_close": "2026-11-17T16:00:00-05:00"}
    try:
        op = {}
        decide(now, op, 100.0, 100.0, closed)
        assert len(asked) == 1 and "=== SOL/USD" in asked[0] and "=== SPY" not in asked[0]   # Borsa chiusa: niente SPY
        assert sent == [("SOL/USD", "buy", {"notional": "20.00"})] and op["SOL/USD"]["perche"] == "rompe in su"
        assert op["SOL/USD"]["modello"] == "claude-x" and len(logged) == 3 and "Conti fatti" in asked[0]
        assert logged[0]["simbolo"] == "BTC/USD" and logged[0]["esito"] == "aspetta"                # risposta rotta
        assert logged[1]["simbolo"] == "SOL/USD" and logged[1]["esito"] == "comprati 20.00 $"
        assert logged[1]["allarmi"] == "99.0;105.0" and ALERTS["SOL/USD"] == [99.0, 105.0] and ALERTS["BTC/USD"] == []
        assert logged[2] == {"ora": "2026-11-16 12:00", "simbolo": "SOL/USD", "chiusura": "2026-11-16 09:00",
                             "lezione": "stop sotto il supporto"}                            # la lezione inventata no
        assert "SOL/USD chiusura 2026-11-16 09:00" in asked[0]
        g["ask"] = lambda text: (_ for _ in ()).throw(RuntimeError("429 troppe richieste"))
        logged.clear()
        decide(now, op, 100.0, 100.0, closed)
        assert [r["esito"] for r in logged] == ["modello non disponibile"] * 2 and len(sent) == 1
        assert ALERTS["SOL/USD"] == [99.0, 105.0]                                      # un errore non cancella gli allarmi
        g.update(ask=lambda text: ({"BTC/USD": {"azione": "compra", "stop": 98, "obiettivo": 104}}, "claude-x"), LIVELLO="suggerisce")
        logged.clear()
        decide(now, {}, 100.0, 100.0, closed, skip={"SOL/USD"})
        assert [(r["simbolo"], r["esito"]) for r in logged] == [("BTC/USD", "solo suggerimento")] and len(sent) == 1
    finally:
        g.update(saved, LIVELLO=os.environ.get("LIVELLO", "pieno"))
        ALERTS.clear()
    print("prove superate")


if __name__ == "__main__":
    prova() if "--prova" in sys.argv else main()
