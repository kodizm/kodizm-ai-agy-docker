# The agy-bridge and the agy it drives.
#
# Build context is the repository root:
#   docker build -t kodizm-ai-agy-docker .
#
# agy is pinned by URL and sha512, taken from the release manifest
# https://antigravity-cli-auto-updater-974169037036.us-central1.run.app/manifests/linux_amd64.json
# and its self-updater is off (AGY_CLI_DISABLE_AUTO_UPDATE), so what runs is what
# was smoke-tested. A bump edits all three ARGs together; .github/workflows/agy-version.yml
# goes red while AGY_VERSION is behind the manifest. linux/amd64 only, like the host.
FROM golang:1.25.6-bookworm AS build

WORKDIR /src
COPY go.mod ./
COPY cmd ./cmd
COPY internal ./internal

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -buildvcs=false -trimpath -ldflags "-s -w" -o /out/agy-bridge ./cmd/agy-bridge

FROM debian:bookworm-slim

ARG AGY_VERSION=1.5.2
ARG AGY_URL=https://storage.googleapis.com/antigravity-public/antigravity-cli/1.5.2-4951595792859136/linux-x64/cli_linux_x64.tar.gz
ARG AGY_SHA512=aa876c0813572efaf85ed47983cc937414705d40490cbdb54f6e255656a4f8744ae38e457823e924fa8270b348124e0521d3cda3fec6824bb2d14b98a7a05baa

# curl is the build's downloader and the healthcheck's client. tini is PID 1: the
# bridge kills a whole agy process group and the children that outlive it are
# re-parented to PID 1, which must reap them or they pile up as zombies until the
# pids limit stops every spawn. The image carries its own init, so it does not
# depend on how a host starts the container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o /tmp/agy.tar.gz "${AGY_URL}" \
    && echo "${AGY_SHA512}  /tmp/agy.tar.gz" | sha512sum -c - \
    && tar -xzf /tmp/agy.tar.gz -C /tmp antigravity \
    && install -m 0755 /tmp/antigravity /usr/local/bin/agy \
    && rm -f /tmp/agy.tar.gz /tmp/antigravity

# Unprivileged. Every agy process runs in a throwaway HOME below AGY_BRIDGE_HOMES,
# which a named volume keeps so each account's installation id survives a restart.
RUN useradd --system --uid 10001 --create-home --home-dir /home/agybridge agybridge \
    && install -d -o agybridge -g agybridge -m 0700 /var/lib/agy-bridge

COPY --from=build /out/agy-bridge /usr/local/bin/agy-bridge

USER agybridge
ENV AGY_BIN=/usr/local/bin/agy \
    AGY_CLI_DISABLE_AUTO_UPDATE=true \
    AGY_BRIDGE_HOMES=/var/lib/agy-bridge \
    AGY_BRIDGE_LISTEN=0.0.0.0:3457

EXPOSE 3457

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS -H "x-api-key: ${AGY_BRIDGE_API_KEY}" http://127.0.0.1:3457/v1/models >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/agy-bridge"]
