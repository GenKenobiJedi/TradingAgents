# TradingAgents fork

## Scanner-Rollenmodelle (2026-11-10)

Bull, Bear und Risk optional getrennt konfigurierbar, Judge bleibt Deep-Modell.
Responses binden alle Rollen; Audit nennt konfigurierte und, falls vom Provider
gemeldet, konkrete Modellkennung. OpenRouter-Alias mit fuehrendem `~` erlaubt,
ungenutzte Rollen fallen nur bei fehlender Einstellung auf Quick zurueck.
Bull/Bear/Risk analysieren zeitgleich unabhaengig; Judge wartet auf alle drei.
Timeout/Fehler bricht Peers ab und sperrt Judge. Barrier- und Abbruchtests gruen.
Exakter GLM-Flash-Latest-Alias fordert Reasoning aus fuer geringere Latenz;
Qualitaetswirkung ungeprueft, Aliasziel kann wechseln.
VPS-Gateway `tradingagents-gateway-roles.service` :18766 aktiv; Bull Mistral
Nemo, Bear GLM-Alias, Risk/Judge Qwen Flash. Zwei isolierte BingX/OpenRouter-
Proben ergaben VETO nach 14,79/12,31 s; 73 VPS-Tests. Erster echter Kandidat
WIF-LONG 08:17 CET: alle Rollen belegt, Qwen-VETO nach 44/45 s, kein Telegram.
Timeoutquote und Trefferquote offen; altes Gateway :18765 bleibt als Standby.

## Scanner gateway (2026-11-09)

`tradingagents.scanner_gateway` adds a bounded OpenRouter/LangGraph review of
specific scanner proposals. Uses up to four Bull/Bear/Risk/decision calls;
requires a fresh signal and closed Binance Spot or BingX Futures 1h data; BingX
quotes require positive bid/ask, correct spread and fresh quoteTime. HTTP header
reads time out; OpenRouter clients are request-loop-bound and closed; calls cap at 25 s/2048 tokens.
Safe logs identify failed stages and codes without exception messages or payloads.
Qwen 3.8 Flash disables reasoning; Qwen 3.8 Max requests minimal reasoning.
Each paid call has private fsync-backed start/end audit records; provider usage
and costs are reported or unknown, with no local budget. Original NOW/LIMIT entry
preserved; BingX currentPrice must equal LONG ask / SHORT bid; NOW quote must
remain between stop and TP1. Corrupt audit tails block calls without deletion.
HTTP replies bind startup code SHA256 and model IDs. Explicit per-request HTTP transports prevent closed global SDK connection reuse. Returns correlated
APPROVE/VETO/HOLD. It never submits orders. Gateway usage, prerequisites and
limits: `SCANNER_GATEWAY.md`.

Live probe: BTC LONG and ETH SHORT synthetic setups with real BingX data returned
HOLD in two consecutive OpenRouter/Flash requests, eight calls, reported total
$0.00374526; no Telegram or orders. The final security fixes still need a live
probe and VPS integration. No win-rate established. LangSmith tracing disabled.
