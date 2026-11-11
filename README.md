# kodizm-ai-agy-docker

The Docker image of the Kodizm agy-bridge: a small Go service that serves the Anthropic Messages API (`/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`) in front of Google's Antigravity CLI (`agy acp`). It lets a client that speaks Messages use the models the CLI offers (Gemini, plus the third party models it lists) under the Google accounts it is handed.

The image carries the bridge and a pinned `agy` (URL and sha512 in the `Dockerfile`, self-updater off). It is published to `ghcr.io/kodizm/kodizm-ai-agy-docker` (`latest` and `sha-<commit>`) by the `image` workflow on every push to `main`, after `go vet` and `go test -race`.

## How it is used

Kodizm provisions the container on its Docker host, then **connects to the bridge**; the bridge never calls out to Kodizm. Every route needs the key in `AGY_BRIDGE_API_KEY` (`x-api-key` or a bearer token), and the bridge refuses to start without one.

| Route | Purpose |
|---|---|
| `POST /v1/messages`, `POST /v1/messages/count_tokens`, `GET /v1/models` | The Messages front |
| `PUT /v1/admin/accounts` | Replace the account list (access tokens only, refresh tokens never reach the bridge) |
| `GET /v1/admin/outcomes?after=<seq>&boot=<id>` | Outcomes of finished requests; the cursor acknowledges |
| `GET /v1/admin/usage` | The latest quota reading per account |
| `POST /v1/admin/login`, `POST /v1/admin/login/{id}/code`, `DELETE /v1/admin/login/{id}` | A Google sign-in through `agy acp --login` |

Every agy process runs in its own throwaway `HOME` with one account's access-only login. Which account serves a request is chosen from the conversation prefix (prompt cache reuse), then remaining quota, then least recent use; quota, rate and capacity failures cool an account per model group. Every `/v1/messages` response an agy session produced carries `X-Kodizm-Account: <account id>` (the id Kodizm pushed, never the email or a token); a response with no session, such as `503 no agy account available`, does not.

The conversation reaches agy as one ACP prompt. Images (base64 JPEG, PNG, GIF or WebP) and PDFs (base64) are forwarded as ACP image and embedded resource blocks where they occurred, also inside tool results, but only when agy advertises image or embedded context support at `initialize`; otherwise, and for url or file sources, they become `[image omitted]` or `[document omitted]`. A plain text document is forwarded as text. The bridge never fetches a url.

agy reads about 192,000 bytes of one prompt and drops the rest from the end without an error, so a replayed conversation would lose its newest messages. The text of a replay is therefore kept within `AGY_BRIDGE_MAX_PROMPT_BYTES` (default 180000). Over it, the oldest large tool results and tool arguments are cut to their head and tail around a marker, then to the marker alone; only after that is the text of older messages cut to its head and tail (500 bytes each). The prompt then opens with a note saying so, and the newest user text is never cut. A replay that still does not fit answers `400 invalid_request_error` with `prompt is too long: N bytes > M maximum ...`, before any agy process starts. A resumed turn replays nothing and is not measured. The request line logs `prompt_bytes` and `shrunk`.

agy retries a dead upstream for minutes without a word, and until then the request holds its slots. When agy shows no sign of life (a message or thought chunk, a tool call or an update to one, a request of its own) for `AGY_BRIDGE_SILENCE_TIMEOUT` seconds (default 180), the bridge kills it and answers `529 overloaded_error` (a stream whose head has gone out gets it as an `error` event, since a status can no longer be sent), logged as `reason=agy_silent`. The account is not penalised for it. A turn parked on tool results waits for the client and is not counted; its clock restarts when the results arrive.

Without pushed accounts, requests get `503 no agy account available`. For local development, `AGY_BRIDGE_TOKEN_FILE` points at one agy token file instead (the admin routes then answer `409`).

## Settings

`AGY_BRIDGE_LISTEN` (`0.0.0.0:3457` in the image), `AGY_BRIDGE_API_KEY`, `AGY_BIN`, `AGY_BRIDGE_HOMES`, `AGY_BRIDGE_MODELS`, `AGY_BRIDGE_DEFAULT_MODEL`, `AGY_BRIDGE_CLEAN`, `AGY_BRIDGE_WEB`, `AGY_BRIDGE_SCRUB_FILE`, `AGY_BRIDGE_MAX_CONCURRENT`, `AGY_BRIDGE_PER_ACCOUNT`, `AGY_BRIDGE_MAX_PARKED`, `AGY_BRIDGE_MAX_PROMPT_BYTES`, `AGY_BRIDGE_PENDING_TTL`, `AGY_BRIDGE_SLOT_WAIT`, `AGY_BRIDGE_SILENCE_TIMEOUT`, `AGY_BRIDGE_AFFINITY_TTL`, `AGY_BRIDGE_USAGE_INTERVAL`, `AGY_BRIDGE_TOKEN_FILE`. Defaults and meaning are in `cmd/agy-bridge/main.go` and `internal/agybridge/server.go`.

## Build and test

```
go vet ./... && go test -race ./...
docker build -t kodizm-ai-agy-docker .
```

linux/amd64 only. The image runs as uid 10001 with a named volume for the homes; run it read-only with a `/tmp` tmpfs and no capabilities.

## Bumping agy

`agy-version` goes red weekly while the pin is behind the release manifest. Bump `AGY_VERSION`, `AGY_URL` and `AGY_SHA512` together, smoke a real request through a test account, then push.
