# /// script
# requires-python = ">=3.12"
# ///
"""Collegamento al conto demo di Capital.com (CFD sull'oro con la leva). Nato il 2026-11-18 (DIARIO.md): lo usa
oro.py, l'Analista AI che opera solo sull'oro.

Solo il conto DEMO, di proposito (BASE). La sessione dura 10 minuti dall'ultima richiesta: se scade se ne apre
un'altra da sola. Chiavi nel file .env (le mette l'utente): CAPITAL_API_KEY, CAPITAL_EMAIL, CAPITAL_PASSWORD (la
password scelta per la chiave API, non quella del conto).

    uv run --env-file ../.env capital.py --check   # sola lettura: saldo, leva, regole dell'oro, spread
"""
import json, os, sys, threading, time, urllib.error, urllib.request

BASE = "https://demo-api-capital.backend-capital.com"  # solo demo
EPIC = "GOLD"
KEYS = ("CAPITAL_API_KEY", "CAPITAL_EMAIL", "CAPITAL_PASSWORD")
_TOK, _LOCK = {}, threading.Lock()


def call(path, method="GET", body=None, tok=None):
    h = {"Content-Type": "application/json", "X-CAP-API-KEY": os.environ["CAPITAL_API_KEY"], **(tok or {})}
    req = urllib.request.Request(BASE + path, method=method, headers=h,
                                 data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r), r.headers


def session():
    _, hd = call("/api/v1/session", "POST", {"identifier": os.environ["CAPITAL_EMAIL"],
                                             "password": os.environ["CAPITAL_PASSWORD"]})
    return {"CST": hd["CST"], "X-SECURITY-TOKEN": hd["X-SECURITY-TOKEN"]}


def api(path, method="GET", body=None):
    """Una richiesta con la sessione in comune (anche fra la sorveglianza e la domanda); se è scaduta, se ne apre
    un'altra e si riprova una volta."""
    for again in (False, True):
        with _LOCK:
            if again or not _TOK:
                _TOK.clear()
                _TOK.update(session())
            tok = dict(_TOK)
        try:
            return call(path, method, body, tok)[0]
        except urllib.error.HTTPError as e:
            if e.code != 401 or again:
                raise RuntimeError(f"capital {method} {path}: {e.code} {e.read()[:300]!r}") from None


def prices(resolution, n):
    """Le ultime n candele (prezzi denaro e lettera), l'ultima può essere quella in corso."""
    return api(f"/api/v1/prices/{EPIC}?resolution={resolution}&max={n}")["prices"]


def snapshot(epic=EPIC):
    return api(f"/api/v1/markets/{epic}")["snapshot"]  # bid, offer, marketStatus ("TRADEABLE" = si può operare)


def eurusd():
    s = snapshot("EURUSD")
    return (s["bid"] + s["offer"]) / 2


def balance():
    """(saldo, disponibile) del conto, in euro."""
    a = next(x for x in api("/api/v1/accounts")["accounts"] if x["preferred"])["balance"]
    return float(a["balance"]), float(a["available"])


def position():
    """La posizione aperta sull'oro ({dealId, size, level, stopLevel, ...}) o None."""
    return next((p["position"] for p in api("/api/v1/positions")["positions"] if p["market"]["epic"] == EPIC), None)


def confirm(ref):
    """L'esito di un ordine: il broker lo conferma qualche istante dopo."""
    for _ in range(10):
        try:
            c = api(f"/api/v1/confirms/{ref}")
            if c.get("dealStatus"):
                return c
        except RuntimeError:
            pass
        time.sleep(1)
    raise RuntimeError(f"capital: nessuna conferma per {ref}")


def open(direction, size, stop):
    """Apre size once d'oro, "BUY" (al rialzo) o "SELL" (allo scoperto, al ribasso), con lo stop depositato presso il
    broker (vale anche se il bot è fermo)."""
    c = confirm(api("/api/v1/positions", "POST", {"direction": direction, "epic": EPIC, "size": size,
                                                   "stopLevel": round(stop, 2)})["dealReference"])
    if c["dealStatus"] != "ACCEPTED":
        raise RuntimeError(f"capital: apertura rifiutata ({c.get('reason') or c['dealStatus']})")
    return c


def close(deal):
    """Chiude la posizione; ritorna il prezzo di uscita."""
    c = confirm(api(f"/api/v1/positions/{deal}", "DELETE")["dealReference"])
    if c["dealStatus"] != "ACCEPTED":
        raise RuntimeError(f"capital: chiusura rifiutata ({c.get('reason') or c['dealStatus']})")
    return float(c["level"])


def move_stop(deal, stop):
    api(f"/api/v1/positions/{deal}", "PUT", {"stopLevel": round(stop, 2)})


def check():
    print(f"Saldo {balance()} EUR; leva:", json.dumps(api("/api/v1/accounts/preferences")["leverages"]["COMMODITIES"]))
    d = api(f"/api/v1/markets/{EPIC}")
    i, r, s = d["instrument"], d["dealingRules"], d["snapshot"]
    print(f"{EPIC}: {i['name']}, valuta {i['currency']}, margine {i['marginFactor']}{i['marginFactorUnit']}")
    print("Regole:", json.dumps(r, ensure_ascii=False))
    print("Costo notturno:", json.dumps(i.get("overnightFee"), ensure_ascii=False))
    print("Orari:", json.dumps(i.get("openingHours"), ensure_ascii=False))
    mid = (s["bid"] + s["offer"]) / 2
    print(f"Prezzo {s['bid']} / {s['offer']}: spread {(s['offer'] - s['bid']) / mid:.3%}, stato {s['marketStatus']}")
    print("Posizione aperta:", position())


if __name__ == "__main__":
    if "--check" not in sys.argv:
        sys.exit("uso: capital.py --check (sola lettura)")
    check()
