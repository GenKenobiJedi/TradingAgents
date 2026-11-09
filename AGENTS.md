# TradingAgents fork

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
