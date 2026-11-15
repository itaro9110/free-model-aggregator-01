[![free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/itaro9110/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/itaro9110/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01)

# free-model-aggregator-01

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API, in JSON
and in server-sent events, behind the best free models of ten simulated providers — and answers
each of the trial's twenty requests with the highest-rated free model that can serve it.

[make.fun bounty page](https://make.fun/bounty/free-model-aggregator-01)

## How it routes

- **Quality:** every request is routed to the highest-rated free model meeting its requirements
  (tools, vision, JSON mode, context length), from the brief's sealed quality table.
- **Budgets:** primaries are planned greedily in manifest order, highest rating first, capped
  per provider at the call count its budget is guaranteed to hold — so a provider's budget is
  never overrun by the plan itself.
- **Failover:** a refusal (402, 429, 5xx) is never the answer. The request is retried on the next
  candidate, a different provider first; a provider is treated as spent or outaged only after
  three failures, since a failure schedule holds at most two calls.
- **Wires:** every client wire (OpenAI, Anthropic) can reach every provider wire (OpenAI,
  Anthropic, Gemini) through lossless translation — system text, user text, tool names,
  max_tokens, stop sequences, image data and JSON mode all carry across, so the provider's echo
  matches the request. Streamed answers always close with a final usage chunk.
- **Deterministic:** no wall clock, no randomness — every choice is a function of the brief, the
  probe results and the providers' own answers, so a replay reproduces the same bytes.

## Run it

```sh
node main.ts            # reads $MAKEFUN_PROMPT_FILE, probes the providers, registers, serves
node main.ts --list     # prints the catalog document it registers, byte for byte
```

Node 22 standard library only; nothing is installed. `lib/providers.ts` reads the brief and
probes the catalogs, `lib/wire.ts` translates the OpenAI and Anthropic wires,
`lib/gemini.ts` translates the Gemini wire.

## Practice scores

`grade-local --driver process --instances 3` on the kit's practice trials: **99 / 99 / 99**
(quality 99, served 100, wire 100, failover 100, catalog 100; replay matched byte-for-byte).
The house baseline scores 38.
