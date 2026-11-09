# Scanner gateway

This fork adds `python -m tradingagents.scanner_gateway`: a bounded LangGraph
analysis of one scanner proposal, using TradingAgents' OpenRouter client and
separate quick/deep model names. Bull, bear, risk and decision nodes make at
most four model requests in total. No Yahoo, news, memory, account, exchange
order or execution API is used.

The input must carry an unambiguous Binance Spot or BingX Futures USDT instrument and at least
15 closed, contiguous 1-hour candles; CoinGecko, unknown market data, invalid
price geometry, stale snapshots, failed models and invalid decisions produce
`HOLD`. Spot and Futures data are explicitly identified by source. `signalSource`
and `signalCandleInterval` identify the separately generated scanner signal. A decision is
valid for at most 120 seconds after the older of the signal and market timestamps; callers must independently
verify the decision ID and expiry and recheck the price immediately before use.
`APPROVE` is an experimental model judgment, not a performance guarantee.

Minimal isolated installation (Python >= 3.11):

```sh
uv venv ~/.cache/tradingagents-scanner-gateway
uv pip install --python ~/.cache/tradingagents-scanner-gateway/bin/python \
  -r requirements-scanner-gateway.txt
```

Set `OPENROUTER_API_KEY`, `TRADINGAGENTS_GATE_TOKEN`,
`TRADINGAGENTS_QUICK_THINK_LLM`, and `TRADINGAGENTS_DEEP_THINK_LLM` in a
private environment, then run:

```sh
~/.cache/tradingagents-scanner-gateway/bin/python -m tradingagents.scanner_gateway
```

The HTTP endpoint binds **only** to `127.0.0.1:8765` (port override:
`TRADINGAGENTS_GATE_PORT`) and requires `Authorization: Bearer <token>`.
`POST /v1/scanner/evaluate` accepts JSON with `contractVersion: 1`, `id`,
`createdAt`, `signalAsOf`, `symbol`, `direction`, `currentPrice`, `entry`, `stopLoss`,
`targets`, and `market` containing `source`, `instrument`, `asOf`,
`candleInterval` and `candles` with `timestamp`, `open`, `high`, `low`,
`close`. `signalSource` and `signalCandleInterval` describe the signal source.
BingX Futures uses `source: bingx_futures` and `instrument: BTC-USDT`; Binance
Spot uses `source: binance_spot` and `instrument: BTCUSDT`. BingX also requires
`market.bid`, `market.ask`, and `market.quoteTime`. Its `currentPrice` must be
the ask for LONG and the bid for SHORT, while `entry` remains the original plan.
For NOW (and unknown entry type), the current executable price must also remain
strictly between stop and the first target. LIMIT keeps its planned-entry geometry.
Responses contain `id`, `action` (`APPROVE`, `VETO`, `HOLD`) and the frozen
`gatewaySha256`, `quickModel`, and `deepModel`. The hash identifies the gateway
source loaded at startup. A valid model decision also contains `marketAsOf`, `expiresAt`, and a short
`note`. The model never receives the gateway token or arbitrary Telegram text.

The individual LLM calls are capped at 25 seconds and 2048 output tokens (including reasoning tokens), with
zero SDK retries. Qwen 3.8 Flash requests disable reasoning; Qwen 3.8 Max
requests use minimal reasoning. These request options apply only to those exact
model IDs. Other models receive no Qwen-specific option. Bull/Bear/Risk prompts
request at most 100 words. Disconnection, timeouts, or snapshot expiry prevent further
model calls. A request already sent to OpenRouter may still be billed if the
client disconnects. No live service is installed or started by this package.

Run the focused tests from this checkout:

```sh
uv pip install --python ~/.cache/tradingagents-scanner-gateway/bin/python pytest==9.2.0
~/.cache/tradingagents-scanner-gateway/bin/python -m pytest --noconftest -q tests/test_scanner_gateway.py
```

Each model request appends a durable `start` and then `completed` or `failed`
event to `~/.local/state/tradingagents-scanner/openrouter-usage.jsonl` (override
with `TRADINGAGENTS_AUDIT_DIR`). The directory and file must be owned by the
process user and have modes `0700` and `0600`; a missing or unsafe audit path
blocks the model call. `attempt_id` joins the events; a `start` without an end
remains visible after a crash. A corrupt final audit record blocks further model
calls without deleting or rewriting the record. Prompt and model output are never recorded.
`usage.include=true` requests provider usage: input/output token counts and
`cost_usd` are saved only when supplied by the provider. Missing values remain
null and `cost_status` is `unknown`; no cost is estimated. The audit has no
spending limit and cannot cap OpenRouter costs.

`entryType` is `NOW`, `LIMIT`, or `unknown`. LIMIT preserves the proposed entry
separately from the current price; the gateway does not adjust prices.
`signalSource` also supports `bingx_futures` and `signalCandleInterval` supports
`4h`; the separately identified gateway market snapshot still uses 1h candles.
