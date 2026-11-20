#!/bin/zsh
# L'Analista sull'oro sul Mac (dal 2026-11-20, al posto di oro.yml nel cloud): stesso bot, ma il modello è Qwen3-VL 30B
# con Ollama, sul Mac, senza termini d'uso di terzi. Lo tiene acceso launchd (it.tradingbot.oro.plist): quando oro.py
# finisce il suo giro (~5 ore e mezza) o si ferma per un errore, launchd lo rilancia. caffeinate tiene sveglio il Mac
# mentre gira (col coperchio chiuso il Mac dorme lo stesso; lo stop resta comunque presso il broker).
#   launchctl unload ~/Library/LaunchAgents/it.tradingbot.oro.plist   # spegnere
#   launchctl load ~/Library/LaunchAgents/it.tradingbot.oro.plist     # accendere
#   tail -f ~/Library/Logs/tradingbot-oro.log                          # cosa sta facendo
cd "${0:A:h}"
export MODELLO=qwen3-vl:30b LIVELLO=pieno SALVA_REGISTRO=1 PYTHONUNBUFFERED=1
git pull -q --rebase
caffeinate -i uv run oro.py --prova && caffeinate -i uv run --env-file ../.env oro.py
git add logs && git commit -q -m "registro oro $(date -u '+%Y-%m-%d %H:%M') UTC" && git pull -q --rebase && git push -q
sleep 30  # se qualcosa non va, niente giri a raffica
