package agybridge

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

// bridge is a Server behind an httptest listener; the fake agy's MCP stub
// calls it back over the server's own unix socket.
type bridge struct {
	srv       *Server
	ts        *httptest.Server
	agyBefore []string
	// home is the HOME of this bridge; homes holds the throwaway HOME of
	// every fake agy it spawns.
	home  string
	homes string
	src   AccountSource
}

// newBridge serves the one account "a", whose token the fake knows.
func newBridge(t *testing.T, env map[string]string, configure ...func(*Config)) *bridge {
	t.Helper()

	return newBridgeWith(t, env, newTestSource(testAccount("a")), configure...)
}

func newBridgeWith(t *testing.T, env map[string]string, src AccountSource, configure ...func(*Config)) *bridge {
	t.Helper()

	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	home := t.TempDir()
	t.Setenv("HOME", home)

	homes := filepath.Join(t.TempDir(), "homes")
	vars := map[string]string{"AGY_BIN": self, "HOME": home, "AGY_BRIDGE_HOMES": homes}
	maps.Copy(vars, env)

	cfg, err := LoadConfig(func(k string) string { return vars[k] })
	if err != nil {
		t.Fatal(err)
	}

	for _, change := range configure {
		change(&cfg)
	}

	srv, err := New(cfg, src)
	if err != nil {
		t.Fatal(err)
	}

	ts := httptest.NewServer(srv)

	b := &bridge{srv: srv, ts: ts, agyBefore: fakePIDs(t, "agy"), home: home, homes: homes, src: src}

	t.Cleanup(func() {
		// Close the bridge first: it releases handlers that hold tool calls.
		srv.Close()
		ts.Close()

		if _, err := os.Stat(srv.root); !errors.Is(err, os.ErrNotExist) {
			t.Errorf("work dir %s left behind: %v", srv.root, err)
		}

		assertProcessesGone(t)
		assertNoGoroutines(t)
		assertHomesGone(t, homes)
	})

	return b
}

// assertHomesGone fails when anything but an account's installation id is
// left under the homes dir: no agy state outlives its process.
func assertHomesGone(t *testing.T, homes string) {
	t.Helper()

	accounts, err := os.ReadDir(homes)
	if err != nil {
		t.Fatal(err)
	}

	for _, account := range accounts {
		if !account.IsDir() {
			t.Errorf("left under the homes dir: %s", account.Name())

			continue
		}

		inside, err := os.ReadDir(filepath.Join(homes, account.Name()))
		if err != nil {
			t.Fatal(err)
		}

		for _, entry := range inside {
			if entry.Name() != installationIDFile || entry.IsDir() {
				t.Errorf("left under %s: %s", account.Name(), entry.Name())
			}
		}
	}
}

func (b *bridge) do(t *testing.T, ctx context.Context, method, path string, body any, header http.Header) *http.Response {
	t.Helper()

	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			t.Fatal(err)
		}

		reader = bytes.NewReader(raw)
	}

	req, err := http.NewRequestWithContext(ctx, method, b.ts.URL+path, reader)
	if err != nil {
		t.Fatal(err)
	}

	maps.Copy(req.Header, header)

	resp, err := b.ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = resp.Body.Close() })

	return resp
}

func (b *bridge) post(t *testing.T, body any) *http.Response {
	t.Helper()

	return b.do(t, context.Background(), http.MethodPost, "/v1/messages", body, nil)
}

// spawned counts the fake agy processes started since the bridge was built.
func (b *bridge) spawned(t *testing.T) int {
	t.Helper()

	return len(b.spawnedTokens(t))
}

// spawnedBy counts the fake agy processes started under an access token.
func (b *bridge) spawnedBy(t *testing.T, token string) int {
	t.Helper()

	n := 0

	for _, got := range b.spawnedTokens(t) {
		if got == token {
			n++
		}
	}

	return n
}

func (b *bridge) spawnedTokens(t *testing.T) []string {
	t.Helper()

	var tokens []string

	for _, name := range fakePIDs(t, "agy") {
		if slices.Contains(b.agyBefore, name) {
			continue
		}

		marker, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}

		tokens = append(tokens, string(marker))
	}

	return tokens
}

// waitIdle waits until no session, held call or fake process is left, which
// the bridge must reach on its own, without Close.
func (b *bridge) waitIdle(t *testing.T) {
	t.Helper()

	eventually(t, "the bridge to go idle", func() bool {
		sessions, calls := b.srv.reg.size()

		return sessions == 0 && calls == 0 && len(livePIDs(t)) == 0
	})
}

func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()

	deadline := time.Now().Add(5 * time.Second)

	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}

		time.Sleep(20 * time.Millisecond)
	}
}

func fakePIDs(t *testing.T, kind string) []string {
	t.Helper()

	matches, err := filepath.Glob(filepath.Join(os.Getenv(fakePIDsEnv), kind+"-*"))
	if err != nil {
		t.Fatal(err)
	}

	return matches
}

// deadPIDs remembers exited fakes so a reused pid can never fail a later test.
var deadPIDs = map[string]bool{}

// livePIDs lists fake agy and stub processes still running.
func livePIDs(t *testing.T) []int {
	t.Helper()

	var live []int

	for _, file := range fakePIDs(t, "*") {
		if deadPIDs[file] {
			continue
		}

		_, raw, _ := strings.Cut(filepath.Base(file), "-")

		pid, err := strconv.Atoi(raw)
		if err != nil {
			t.Fatal(err)
		}

		if errors.Is(syscall.Kill(pid, 0), syscall.ESRCH) {
			deadPIDs[file] = true

			continue
		}

		live = append(live, pid)
	}

	return live
}

func assertProcessesGone(t *testing.T) {
	t.Helper()

	eventually(t, "fake agy and stub processes to exit", func() bool {
		return len(livePIDs(t)) == 0
	})
}

// assertNoGoroutines fails when a goroutine of the package's own code, not
// of a test, is still running.
func assertNoGoroutines(t *testing.T) {
	t.Helper()

	var leaked []string

	eventually(t, "package goroutines to exit", func() bool {
		buf := make([]byte, 1<<20)
		stacks := strings.Split(string(buf[:runtime.Stack(buf, true)]), "\n\n")

		leaked = nil

		for _, stack := range stacks[1:] {
			if strings.Contains(stack, "internal/agybridge.") && !strings.Contains(stack, "_test.go") {
				leaked = append(leaked, stack)
			}
		}

		return len(leaked) == 0
	})
}

func userText(text string) []map[string]any {
	return []map[string]any{{"role": "user", "content": text}}
}

var readFileTool = []map[string]any{{
	"name":         "read_file",
	"description":  "Reads a file",
	"input_schema": map[string]any{"type": "object", "properties": map[string]any{"path": map[string]string{"type": "string"}}},
}}

type messageResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Content []struct {
		Type  string          `json:"type"`
		Text  string          `json:"text"`
		ID    string          `json:"id"`
		Name  string          `json:"name"`
		Input json.RawMessage `json:"input"`
	} `json:"content"`
	StopReason string `json:"stop_reason"`
	Usage      usage  `json:"usage"`
}

func decodeJSON[T any](t *testing.T, resp *http.Response) T {
	t.Helper()

	var v T
	if err := json.NewDecoder(resp.Body).Decode(&v); err != nil {
		t.Fatalf("decode %d response: %v", resp.StatusCode, err)
	}

	return v
}

func expectStatus(t *testing.T, resp *http.Response, status int) {
	t.Helper()

	if resp.StatusCode != status {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d, want %d: %s", resp.StatusCode, status, body)
	}
}

type sseEvent struct {
	name string
	data map[string]any
}

func readSSE(t *testing.T, r io.Reader) []sseEvent {
	t.Helper()

	var events []sseEvent

	reader := bufio.NewReader(r)

	for {
		event, err := nextSSE(reader)
		if errors.Is(err, io.EOF) {
			return events
		}

		if err != nil {
			t.Fatalf("read SSE: %v", err)
		}

		events = append(events, event)
	}
}

func nextSSE(reader *bufio.Reader) (sseEvent, error) {
	var event sseEvent

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			return event, err
		}

		line = strings.TrimRight(line, "\n")

		switch {
		case line == "" && event.name != "":
			return event, nil
		case strings.HasPrefix(line, "event: "):
			event.name = strings.TrimPrefix(line, "event: ")
		case strings.HasPrefix(line, "data: "):
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event.data); err != nil {
				return event, err
			}
		}
	}
}

func eventNames(events []sseEvent) []string {
	names := make([]string, 0, len(events))
	for _, e := range events {
		names = append(names, e.name)
	}

	return names
}

func streamedText(events []sseEvent) string {
	var text strings.Builder

	for _, e := range events {
		if delta, ok := e.data["delta"].(map[string]any); ok && delta["type"] == "text_delta" {
			text.WriteString(delta["text"].(string))
		}
	}

	return text.String()
}

func TestMessagesNonStreamText(t *testing.T) {
	b := newBridge(t, nil)

	resp := b.post(t, map[string]any{"model": "claude-sonnet-4-5", "messages": userText("SCRIPT:plain")})
	expectStatus(t, resp, http.StatusOK)

	got := decodeJSON[messageResponse](t, resp)

	if len(got.Content) != 1 || got.Content[0].Type != "text" || got.Content[0].Text != "Hello, world" {
		t.Fatalf("content = %+v", got.Content)
	}

	if got.Model != "claude-sonnet-4-5" || got.StopReason != "end_turn" ||
		got.Usage != (usage{InputTokens: 11, OutputTokens: 7}) {
		t.Fatalf("message = %+v", got)
	}

	if !strings.HasPrefix(got.ID, "msg_") {
		t.Fatalf("id = %q", got.ID)
	}

	b.waitIdle(t)
}

func TestMessagesStreamText(t *testing.T) {
	b := newBridge(t, nil)

	body := map[string]any{"model": "gemini-3.8-flash-high", "stream": true, "messages": userText("SCRIPT:plain")}
	resp := b.do(t, context.Background(), http.MethodPost, "/v1/messages?beta=true", body, nil)
	expectStatus(t, resp, http.StatusOK)

	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Fatalf("content type %q", ct)
	}

	events := readSSE(t, resp.Body)
	want := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}

	if !slices.Equal(eventNames(events), want) {
		t.Fatalf("events = %v, want %v", eventNames(events), want)
	}

	if text := streamedText(events); text != "Hello, world" {
		t.Fatalf("text = %q", text)
	}

	model := events[0].data["message"].(map[string]any)["model"]
	delta := events[5].data

	if model != "gemini-3.8-flash-high" || delta["delta"].(map[string]any)["stop_reason"] != "end_turn" {
		t.Fatalf("message_start model %v, message_delta %v", model, delta)
	}

	if u := delta["usage"].(map[string]any); u["input_tokens"] != float64(11) || u["output_tokens"] != float64(7) {
		t.Fatalf("usage = %v", u)
	}
}

func TestModelSelection(t *testing.T) {
	tests := []struct {
		name      string
		env       map[string]string
		requested string
		fields    map[string]any
		want      string
	}{
		{name: "listed at the default level", requested: "gemini-4-argon", want: "gemini-4-argon-medium"},
		{
			name:      "effort picks the level",
			requested: "claude-opus-5-5",
			fields:    map[string]any{"output_config": map[string]any{"effort": "high"}},
			want:      "claude-opus-5-5-high",
		},
		{
			name:      "a thinking budget picks the level",
			requested: "claude-sonnet-5-5",
			fields:    map[string]any{"thinking": map[string]any{"type": "enabled", "budget_tokens": 1024}},
			want:      "claude-sonnet-5-5-low",
		},
		{
			name:      "gpt-oss runs at its only level",
			requested: "gpt-oss-120b",
			fields:    map[string]any{"output_config": map[string]any{"effort": "low"}},
			want:      "gpt-oss-120b-medium",
		},
		{name: "unknown falls back", requested: "claude-opus-4-1", want: "gemini-3.8-flash-medium"},
		{
			name:      "a composite id is no model",
			requested: "gemini-4-argon-low",
			fields:    map[string]any{"output_config": map[string]any{"effort": "high"}},
			want:      "gemini-3.8-flash-high",
		},
		{
			name:      "configured list and default",
			env:       map[string]string{"AGY_BRIDGE_MODELS": "m-a, m-b", "AGY_BRIDGE_DEFAULT_MODEL": "m-b"},
			requested: "gemini-4-argon",
			want:      "m-b-medium",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := newBridge(t, tt.env)

			body := map[string]any{"model": tt.requested, "messages": userText("SCRIPT:model")}
			maps.Copy(body, tt.fields)

			resp := b.post(t, body)
			expectStatus(t, resp, http.StatusOK)

			got := decodeJSON[messageResponse](t, resp)

			if got.Content[0].Text != tt.want || got.Model != tt.requested {
				t.Fatalf("agy model %q, echoed %q; want %q, %q", got.Content[0].Text, got.Model, tt.want, tt.requested)
			}
		})
	}
}

func TestMaxTokensStopReason(t *testing.T) {
	b := newBridge(t, nil)

	resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:maxtokens")})
	got := decodeJSON[messageResponse](t, resp)

	if got.StopReason != "max_tokens" || got.Usage != (usage{InputTokens: 3, OutputTokens: 4}) {
		t.Fatalf("message = %+v", got)
	}
}

// agy's output tokens already include its thinking tokens (input plus output
// plus cache reads and writes is its total) and it reports cache reads and
// writes; Anthropic names the cache fields its own way.
func TestUsageCarriesThinkingAndCache(t *testing.T) {
	want := usage{
		InputTokens:              6895,
		CacheCreationInputTokens: 120,
		CacheReadInputTokens:     4000,
		OutputTokens:             43,
	}

	for _, stream := range []bool{false, true} {
		t.Run("stream="+strconv.FormatBool(stream), func(t *testing.T) {
			logs := captureLog(t)
			b := newBridge(t, nil)

			text, _, stop, use := b.turn(t, stream, userText("SCRIPT:usage"))

			if text != "counted" || stop != "end_turn" || use != want {
				t.Fatalf("text %q, stop %q, usage %+v; want usage %+v", text, stop, use, want)
			}

			b.waitIdle(t)

			if line := logs.String(); !strings.Contains(line, " agy_usage=in:6895,out:43,cr:4000,cw:120,th:33,total:11058 ") {
				t.Fatalf("want agy's raw usage in the request line:\n%s", line)
			}
		})
	}
}

func TestReportedUsage(t *testing.T) {
	agy := agyUsage{InputTokens: 11, OutputTokens: 10, ThoughtTokens: 3, CachedReadTokens: 5}
	call := toolCall{name: "read_file", input: json.RawMessage(`{"path":"a.txt"}`)}
	perCall := (len(call.name) + len(call.input)) / 4

	tests := []struct {
		name      string
		end       turnEnd
		exact     bool
		body      int
		priorOut  int
		toolCalls int
		want      agyUsage
	}{
		{"one model call keeps agy's numbers", turnEnd{stopReason: "end_turn", usage: agy}, true, 4000, 0, 0, agy},
		{"a tool_use end reports its request and the calls", turnEnd{stopReason: "tool_use", calls: []toolCall{call}}, true, 4000, 0, 0, agyUsage{InputTokens: 1000 + agyOverheadTokens, OutputTokens: perCall}},
		{"a request bigger than agy's share is the context", turnEnd{stopReason: "end_turn", usage: agy}, false, 4000, 0, 1, agyUsage{InputTokens: 1000 + agyOverheadTokens, OutputTokens: 10}},
		{"agy's total over its calls wins when it read more than the request holds", turnEnd{stopReason: "end_turn", usage: agyUsage{InputTokens: 1761, OutputTokens: 5}}, false, 224, 0, 1, agyUsage{InputTokens: 880, OutputTokens: 5}},
		{"four calls of one context report that context", turnEnd{stopReason: "end_turn", usage: agyUsage{InputTokens: 64363}}, false, 61000, 0, 3, agyUsage{InputTokens: 16090}},
		{"cache tokens count toward agy's total", turnEnd{stopReason: "end_turn", usage: agyUsage{InputTokens: 200, CachedReadTokens: 700, CachedWriteTokens: 100}}, false, 40, 0, 1, agyUsage{InputTokens: 500}},
		{"the output already reported is not counted twice", turnEnd{stopReason: "end_turn", usage: agy}, false, 4000, 4, 1, agyUsage{InputTokens: 1000 + agyOverheadTokens, OutputTokens: 6}},
		{"the output never goes below zero", turnEnd{stopReason: "end_turn", usage: agy}, false, 4000, 99, 1, agyUsage{InputTokens: 1000 + agyOverheadTokens}},
		{"two calls at a tool_use end", turnEnd{stopReason: "tool_use", calls: []toolCall{call, call}}, false, 8, 0, 2, agyUsage{InputTokens: 2 + agyOverheadTokens, OutputTokens: 2 * perCall}},
		{"an empty body still carries agy's own prompt", turnEnd{stopReason: "end_turn"}, false, 0, 0, 0, agyUsage{InputTokens: agyOverheadTokens}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := reportedUsage(tt.end, tt.exact, tt.body, tt.priorOut, tt.toolCalls); got != tt.want {
				t.Fatalf("reportedUsage = %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestMessageStartUsageIsZero(t *testing.T) {
	b := newBridge(t, nil)

	resp := b.post(t, map[string]any{"model": "x", "stream": true, "messages": userText("SCRIPT:usage")})
	expectStatus(t, resp, http.StatusOK)

	start := readSSE(t, resp.Body)[0].data["message"].(map[string]any)["usage"]
	want := map[string]any{
		"input_tokens":                float64(0),
		"cache_creation_input_tokens": float64(0),
		"cache_read_input_tokens":     float64(0),
		"output_tokens":               float64(0),
	}

	if !maps.Equal(start.(map[string]any), want) {
		t.Fatalf("message_start usage = %v", start)
	}
}

func TestRequestLevel(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{name: "absent", body: `{}`, want: "medium"},
		{name: "effort low", body: `{"output_config": {"effort": "low"}}`, want: "low"},
		{name: "effort medium", body: `{"output_config": {"effort": "medium"}}`, want: "medium"},
		{name: "effort high", body: `{"output_config": {"effort": "high"}}`, want: "high"},
		{name: "effort xhigh", body: `{"output_config": {"effort": "xhigh"}}`, want: "high"},
		{name: "effort max", body: `{"output_config": {"effort": "max"}}`, want: "high"},
		{
			name: "unknown effort falls to thinking",
			body: `{"output_config": {"effort": "huge"}, "thinking": {"type": "disabled"}}`,
			want: "low",
		},
		{name: "unknown effort alone", body: `{"output_config": {"effort": "huge"}}`, want: "medium"},
		{name: "thinking disabled", body: `{"thinking": {"type": "disabled"}}`, want: "low"},
		{name: "budget 1024", body: `{"thinking": {"type": "enabled", "budget_tokens": 1024}}`, want: "low"},
		{name: "budget 1025", body: `{"thinking": {"type": "enabled", "budget_tokens": 1025}}`, want: "medium"},
		{name: "budget 8192", body: `{"thinking": {"type": "enabled", "budget_tokens": 8192}}`, want: "medium"},
		{name: "budget 8193", body: `{"thinking": {"type": "enabled", "budget_tokens": 8193}}`, want: "high"},
		{name: "adaptive", body: `{"thinking": {"type": "adaptive", "display": "omitted"}}`, want: "medium"},
		{name: "effort of the wrong type", body: `{"output_config": {"effort": 5}}`, want: "medium"},
		{name: "output_config of the wrong type", body: `{"output_config": "high"}`, want: "medium"},
		{name: "thinking of the wrong type", body: `{"thinking": true}`, want: "medium"},
		{name: "budget of the wrong type", body: `{"thinking": {"type": "enabled", "budget_tokens": "big"}}`, want: "medium"},
		{name: "budget beyond the int range", body: `{"thinking": {"type": "enabled", "budget_tokens": 1e30}}`, want: "high"},
		{name: "negative budget", body: `{"thinking": {"type": "enabled", "budget_tokens": -5}}`, want: "low"},
		{
			name: "effort wins over thinking",
			body: `{"output_config": {"effort": "low"}, "thinking": {"type": "enabled", "budget_tokens": 32000}}`,
			want: "low",
		},
		{
			name: "what Claude Code sends",
			body: `{"output_config": {"effort": "high"}, "thinking": {"type": "adaptive", "display": "omitted"}}`,
			want: "high",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req messagesRequest
			if err := json.Unmarshal([]byte(tt.body), &req); err != nil {
				t.Fatal(err)
			}

			if got := requestLevel(&req); got != tt.want {
				t.Fatalf("requestLevel = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestLoadConfigRefusesIdsThatNameALevel(t *testing.T) {
	for _, vars := range []map[string]string{
		{"AGY_BRIDGE_MODELS": "gemini-3.8-flash-low"},
		{"AGY_BRIDGE_MODELS": "gemini-4-argon, claude-opus-5-5-high"},
		{"AGY_BRIDGE_DEFAULT_MODEL": "gemini-3.8-flash-medium"},
	} {
		if _, err := LoadConfig(func(k string) string { return vars[k] }); err == nil || !strings.Contains(err.Error(), "names a level") {
			t.Fatalf("LoadConfig(%v) error = %v, want a refusal naming the level", vars, err)
		}
	}
}

func TestLoadConfigPoolSettings(t *testing.T) {
	cfg, err := LoadConfig(func(string) string { return "" })
	if err != nil {
		t.Fatalf("the defaults must load: %v", err)
	}

	if cfg.MaxConcurrent != 8 || cfg.PerAccount != 2 || cfg.AffinityTTL != time.Hour ||
		cfg.Homes != filepath.Join(os.TempDir(), "agy-bridge") || cfg.TokenFile != "" ||
		cfg.UsageInterval != 5*time.Minute {
		t.Fatalf("defaults %+v", cfg)
	}

	vars := map[string]string{
		"AGY_BRIDGE_PER_ACCOUNT":    "3",
		"AGY_BRIDGE_AFFINITY_TTL":   "90",
		"AGY_BRIDGE_HOMES":          "/srv/agy//homes/",
		"AGY_BRIDGE_TOKEN_FILE":     "/srv/token",
		"AGY_BRIDGE_MAX_CONCURRENT": "4",
		"AGY_BRIDGE_USAGE_INTERVAL": "90s",
	}

	cfg, err = LoadConfig(func(k string) string { return vars[k] })
	if err != nil {
		t.Fatal(err)
	}

	if cfg.PerAccount != 3 || cfg.AffinityTTL != 90*time.Second || cfg.Homes != "/srv/agy/homes" ||
		cfg.TokenFile != "/srv/token" || cfg.MaxConcurrent != 4 || cfg.UsageInterval != 90*time.Second {
		t.Fatalf("config %+v", cfg)
	}

	if cfg, err := LoadConfig(func(k string) string {
		return map[string]string{"AGY_BRIDGE_USAGE_INTERVAL": "0"}[k]
	}); err != nil || cfg.UsageInterval != 0 {
		t.Fatalf("interval 0: %v, %s", err, cfg.UsageInterval)
	}

	for _, bad := range []map[string]string{
		{"AGY_BRIDGE_PER_ACCOUNT": "0"},
		{"AGY_BRIDGE_AFFINITY_TTL": "soon"},
		{"AGY_BRIDGE_HOMES": "relative/homes"},
		{"AGY_BRIDGE_USAGE_INTERVAL": "10s"},
		{"AGY_BRIDGE_USAGE_INTERVAL": "300"},
		{"AGY_BRIDGE_USAGE_INTERVAL": "-5m"},
	} {
		if _, err := LoadConfig(func(k string) string { return bad[k] }); err == nil {
			t.Fatalf("LoadConfig(%v) accepted", bad)
		}
	}
}

func TestImageBytesCountsImagesInMessagesAndToolResults(t *testing.T) {
	messages := []message{
		{Role: "user", Content: json.RawMessage(`"plain text"`)},
		{Role: "user", Content: json.RawMessage(`[
			{"type":"text","text":"look"},
			{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAABBBB"}}
		]`)},
		{Role: "user", Content: json.RawMessage(`[
			{"type":"tool_result","tool_use_id":"t1","content":[
				{"type":"image","source":{"type":"base64","data":"CCCC"}},
				{"type":"text","text":"shot"}
			]},
			{"type":"tool_result","tool_use_id":"t2","content":"text only"},
			{"type":"image","source":"not an object"}
		]`)},
	}

	if got := imageBytes(messages); got != 12 {
		t.Fatalf("imageBytes = %d, want 12 (8 + 4)", got)
	}
}

func TestAgyID(t *testing.T) {
	tests := []struct {
		model     string
		level     string
		want      string
		wantLevel string
	}{
		{model: "gemini-3.8-flash", level: "low", want: "gemini-3.8-flash-low", wantLevel: "low"},
		{model: "claude-opus-5-5", level: "high", want: "claude-opus-5-5-high", wantLevel: "high"},
		{model: "gpt-oss-120b", level: "low", want: "gpt-oss-120b-medium", wantLevel: "medium"},
		{model: "gpt-oss-120b", level: "high", want: "gpt-oss-120b-medium", wantLevel: "medium"},
		{model: "not-in-catalogue", level: "high", want: "not-in-catalogue-high", wantLevel: "high"},
	}

	for _, tt := range tests {
		t.Run(tt.model+"/"+tt.level, func(t *testing.T) {
			if got, level := lookupModel(tt.model).agyID(tt.level); got != tt.want || level != tt.wantLevel {
				t.Fatalf("agyID = %q, %q; want %q, %q", got, level, tt.want, tt.wantLevel)
			}
		})
	}
}

// The level is fixed when agy opens the session: a continuation asking for
// another one resumes the same turn and logs the level it runs at.
func TestContinuationKeepsTheSessionLevel(t *testing.T) {
	logs := captureLog(t)
	b := newBridge(t, nil)
	first := userText("SCRIPT:tool")

	resp := b.post(t, map[string]any{
		"model":         "gemini-4-argon",
		"output_config": map[string]any{"effort": "low"},
		"tools":         readFileTool,
		"messages":      first,
	})
	got := decodeJSON[messageResponse](t, resp)

	calls := []toolUse{{id: got.Content[1].ID, name: got.Content[1].Name, input: string(got.Content[1].Input)}}
	messages := append(first,
		assistantCalls("Let me look.", calls),
		toolResultMessage(map[string]string{calls[0].id: "ok"}),
	)

	resp = b.post(t, map[string]any{
		"model":         "gemini-4-argon",
		"output_config": map[string]any{"effort": "high"},
		"tools":         readFileTool,
		"messages":      messages,
	})
	expectStatus(t, resp, http.StatusOK)

	b.waitIdle(t)

	line := logs.String()
	if strings.Count(line, `agy_model="gemini-4-argon-low" level=low `) != 2 || strings.Contains(line, "level=high") {
		t.Fatalf("want both requests on the opening level:\n%s", line)
	}
}

func TestWorkspaceStaysEmptyWithTools(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_CLEAN": "false"})

	resp := b.post(t, map[string]any{"model": "x", "tools": readFileTool, "messages": userText("SCRIPT:cwd")})
	got := decodeJSON[messageResponse](t, resp)

	if got.Content[0].Text != "entries=0 err=<nil>" {
		t.Fatalf("workspace: %q", got.Content[0].Text)
	}
}

// launchReport is what the fake agy saw of how it was started.
type launchReport struct {
	Args       []string `json:"args"`
	AutoUpdate string   `json:"autoUpdate"`
	BridgeKey  string   `json:"bridgeKey"`
	Cwd        string   `json:"cwd"`
	Home       string   `json:"home"`
}

func (b *bridge) ask(t *testing.T, script string, tools any) string {
	t.Helper()

	body := map[string]any{"model": "x", "messages": userText("SCRIPT:" + script)}
	if tools != nil {
		body["tools"] = tools
	}

	resp := b.post(t, body)
	expectStatus(t, resp, http.StatusOK)

	return decodeJSON[messageResponse](t, resp).Content[0].Text
}

func TestCleanModeLaunchesAgyAsAPlainModel(t *testing.T) {
	b := newBridge(t, nil)

	var got launchReport
	if err := json.Unmarshal([]byte(b.ask(t, "launch", readFileTool)), &got); err != nil {
		t.Fatal(err)
	}

	want := []string{"acp", "--agent", "clean-chat", "--include-default-customizations=false", "--disable-telemetry"}
	if !slices.Equal(got.Args, want) || got.AutoUpdate != "true" {
		t.Fatalf("agy started with %v, auto update %q", got.Args, got.AutoUpdate)
	}

	// The workspace is private to the session, not the shared empty one.
	if got.Cwd == b.srv.workspace || !strings.HasPrefix(got.Cwd, filepath.Join(b.srv.root, "sessions")) {
		t.Fatalf("cwd %q", got.Cwd)
	}

	// HOME is a throwaway one of the leased account, not the bridge's own.
	if filepath.Dir(got.Home) != filepath.Join(b.homes, "a") {
		t.Fatalf("home %q", got.Home)
	}
}

func TestAgyNeverSeesTheBridgeKey(t *testing.T) {
	t.Setenv("AGY_BRIDGE_API_KEY", "bridge-key-marker")

	b := newBridge(t, nil)

	var got launchReport
	if err := json.Unmarshal([]byte(b.ask(t, "launch", readFileTool)), &got); err != nil {
		t.Fatal(err)
	}

	if got.BridgeKey != "" {
		t.Fatalf("agy inherited the bridge key: %q", got.BridgeKey)
	}
}

func TestBuiltInModeLaunchesAgyUnchanged(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_CLEAN": "false"})

	var got launchReport
	if err := json.Unmarshal([]byte(b.ask(t, "launch", readFileTool)), &got); err != nil {
		t.Fatal(err)
	}

	if want := []string{"acp", "--disable-telemetry"}; !slices.Equal(got.Args, want) || got.AutoUpdate != "true" {
		t.Fatalf("agy started with %v, auto update %q", got.Args, got.AutoUpdate)
	}

	if got.Cwd != b.srv.workspace {
		t.Fatalf("cwd %q, want the shared workspace", got.Cwd)
	}

	var file fakeAgentFile
	if err := json.Unmarshal([]byte(b.ask(t, "agentfile", readFileTool)), &file); err != nil || file.Exists {
		t.Fatalf("an agent file exists without clean mode: %+v, %v", file, err)
	}
}

func TestCleanAgentFileExistsWhenAgySessionOpens(t *testing.T) {
	tools := []map[string]any{
		{"name": "write_file", "input_schema": map[string]any{"type": "object"}},
		{"name": `odd "name": x`, "input_schema": map[string]any{"type": "object"}},
		{"name": "read_file", "input_schema": map[string]any{"type": "object"}},
		{"name": "write_file", "input_schema": map[string]any{"type": "object"}},
	}

	tests := []struct {
		name  string
		env   map[string]string
		tools any
		want  []string
		deny  []string
	}{
		{
			name:  "tools and web",
			tools: tools,
			want: []string{
				"tools: [search_web, read_url_content]\n",
				`mcpServers: [{"name":"client","tools":{"odd \"name\": x":{"eager":true},` +
					`"read_file":{"eager":true},"write_file":{"eager":true}}}]` + "\n",
			},
		},
		{
			name:  "no tools",
			want:  []string{"tools: [search_web, read_url_content]\n"},
			deny:  []string{"mcpServers"},
			tools: nil,
		},
		{
			name:  "web off",
			env:   map[string]string{"AGY_BRIDGE_WEB": "false"},
			tools: readFileTool,
			want:  []string{"\ntools: []\n", `"read_file":{"eager":true}`},
			deny:  []string{"search_web", "read_url_content"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := newBridge(t, tt.env)

			var file fakeAgentFile
			if err := json.Unmarshal([]byte(b.ask(t, "agentfile", tt.tools)), &file); err != nil || !file.Exists {
				t.Fatalf("no agent file at session/new: %+v, %v", file, err)
			}

			for _, want := range tt.want {
				if !strings.Contains(file.Content, want) {
					t.Fatalf("agent file lacks %q:\n%s", want, file.Content)
				}
			}

			for _, deny := range tt.deny {
				if strings.Contains(file.Content, deny) {
					t.Fatalf("agent file has %q:\n%s", deny, file.Content)
				}
			}

			if file.FileMode != "-rw-------" || !slices.Equal(file.DirModes, []string{"-rwx------", "-rwx------", "-rwx------"}) {
				t.Fatalf("modes: file %s, dirs %v", file.FileMode, file.DirModes)
			}
		})
	}
}

func TestAgyStateIsRemovedWhenTheSessionEnds(t *testing.T) {
	for _, clean := range []string{"true", "false"} {
		t.Run("clean="+clean, func(t *testing.T) {
			b := newBridge(t, map[string]string{"AGY_BRIDGE_CLEAN": clean})

			// 1. While the session runs, agy has persisted its files beside the
			// login it was given.
			var during []string
			if err := json.Unmarshal([]byte(b.ask(t, "state", readFileTool)), &during); err != nil || len(during) != 9 ||
				!slices.Contains(during, "antigravity-oauth-token") || !slices.Contains(during, "installation_id") {
				t.Fatalf("state during the session: %v, %v", during, err)
			}

			// 2. Once it ends, its whole home is gone, and the bridge's own
			// HOME was never touched.
			b.waitIdle(t)
			assertHomesGone(t, b.homes)

			if left := filesUnder(t, b.home); len(left) != 0 {
				t.Fatalf("the bridge's HOME has %v", left)
			}
		})
	}
}

// toolUse is a tool_use block as either response shape carries it.
type toolUse struct {
	id    string
	name  string
	input string
}

func TestToolHandBackContinuesTheSameTurn(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run("stream="+strconv.FormatBool(stream), func(t *testing.T) {
			b := newBridge(t, nil)
			first := userText("SCRIPT:tool")

			// 1. The first response ends at the tool call, with the text before it.
			text, calls, stop, use := b.turn(t, stream, first)

			// A tool_use end has no agy usage yet: it reports its own request.
			if text != "Let me look." || stop != "tool_use" || use.InputTokens != bodyTokens(t, stream, first) || use.OutputTokens < 1 {
				t.Fatalf("first response: text %q, stop %q, usage %+v", text, stop, use)
			}

			firstOut := use.OutputTokens

			if len(calls) != 1 || calls[0].name != "read_file" || calls[0].input != `{"path":"a.txt"}` {
				t.Fatalf("tool calls = %+v", calls)
			}

			if !strings.HasPrefix(calls[0].id, "toolu_") || len(calls[0].id) != len("toolu_")+24 {
				t.Fatalf("tool id %q", calls[0].id)
			}

			// 2. The tool result resumes the held agy turn, which finishes.
			messages := append(first,
				assistantCalls(text, calls),
				toolResultMessage(map[string]string{calls[0].id: "file body"}),
			)

			text, calls, stop, use = b.turn(t, stream, messages)

			// agy's 20 input tokens add up both of its model calls, so the client
			// is told this request's own size instead; agy's 9 output tokens
			// include the call the first response already reported.
			if text != "Tool said: file body" || len(calls) != 0 || stop != "end_turn" ||
				use != (usage{InputTokens: bodyTokens(t, stream, messages), OutputTokens: 9 - firstOut}) {
				t.Fatalf("second response: text %q, calls %+v, stop %q, usage %+v", text, calls, stop, use)
			}

			if n := b.spawned(t); n != 1 {
				t.Fatalf("agy started %d times, want once for the whole turn", n)
			}

			b.waitIdle(t)
		})
	}
}

// Claude Code follows a tool result with system messages of its own. They must
// not cost the turn a replay: the held agy turn resumes and reads them after
// the result.
func TestTrailingSystemMessagesDoNotBreakContinuation(t *testing.T) {
	b := newBridge(t, nil)
	first := userText("SCRIPT:tool")

	text, calls, _, _ := b.turn(t, false, first)

	messages := append(first,
		assistantCalls(text, calls),
		toolResultMessage(map[string]string{calls[0].id: "file body"}),
		map[string]any{"role": "system", "content": "token budget reminder"},
	)

	got, _, stop, _ := b.turn(t, false, messages)

	if got != "Tool said: file body\n\ntoken budget reminder" || stop != "end_turn" {
		t.Fatalf("second response: text %q, stop %q", got, stop)
	}

	if n := b.spawned(t); n != 1 {
		t.Fatalf("agy started %d times, want once for the whole turn", n)
	}

	b.waitIdle(t)
}

func TestParallelToolCallsLeaveAsOneMessage(t *testing.T) {
	b := newBridge(t, nil)
	first := userText("SCRIPT:tool2")

	_, calls, stop, _ := b.turn(t, false, first)

	if len(calls) != 2 || stop != "tool_use" {
		t.Fatalf("calls %+v, stop %q", calls, stop)
	}

	results := map[string]string{}
	for _, call := range calls {
		results[call.id] = "r" + strings.Trim(call.input, `{}"n:`)
	}

	messages := append(first, assistantCalls("", calls), toolResultMessage(results))
	text, _, stop, _ := b.turn(t, false, messages)

	if text != "Results: r1 | r2" || stop != "end_turn" {
		t.Fatalf("final text %q, stop %q", text, stop)
	}

	if n := b.spawned(t); n != 1 {
		t.Fatalf("agy started %d times, want 1", n)
	}
}

func TestUnknownToolResultsReplayTheConversation(t *testing.T) {
	b := newBridge(t, nil)

	messages := append(userText("SCRIPT:prompt start"),
		assistantCalls("", []toolUse{{id: "toolu_unknown", name: "read_file", input: `{}`}}),
		toolResultMessage(map[string]string{"toolu_unknown": "late"}),
	)

	text, _, stop, _ := b.turn(t, false, messages)

	for _, want := range []string{"<conversation_history>", "[tool_use read_file {}]", "[tool_result toolu_unknown]\nlate"} {
		if !strings.Contains(text, want) {
			t.Fatalf("replayed prompt lacks %q:\n%s", want, text)
		}
	}

	if stop != "end_turn" {
		t.Fatalf("stop %q", stop)
	}
}

func TestUpstreamFailuresMapToAnthropicErrors(t *testing.T) {
	tests := []struct {
		script     string
		status     int
		kind       string
		retryAfter string
	}{
		{script: "429", status: http.StatusTooManyRequests, kind: "rate_limit_error", retryAfter: "5"},
		{script: "503", status: statusOverloaded, kind: "overloaded_error", retryAfter: "5"},
		{script: "fatal", status: http.StatusBadGateway, kind: "api_error"},
	}

	// Pings fall due well before the banner: a stream must still not commit
	// to 200 until a real event exists.
	setVar(t, &pingInterval, fakeBannerDelay/10)

	for _, tt := range tests {
		for _, stream := range []bool{false, true} {
			t.Run(tt.script+"/stream="+strconv.FormatBool(stream), func(t *testing.T) {
				b := newBridge(t, nil)

				body := map[string]any{"model": "x", "stream": stream, "messages": userText("SCRIPT:" + tt.script)}
				resp := b.post(t, body)
				expectStatus(t, resp, tt.status)

				if got := resp.Header.Get("Retry-After"); got != tt.retryAfter {
					t.Fatalf("retry-after %q, want %q", got, tt.retryAfter)
				}

				got := decodeJSON[errorEnvelope](t, resp)

				if got.Type != "error" || got.Error.Type != tt.kind || !strings.HasPrefix(got.Error.Message, "⚠") {
					t.Fatalf("error body = %+v", got)
				}

				// The fake hangs after the banner like agy retrying; only a kill ends it.
				b.waitIdle(t)
			})
		}
	}
}

// The gateway's first-byte timer would cut a stream whose first token is slow:
// past firstByteAfter the head goes out early, pings follow, and the turn ends
// as usual.
func TestSlowFirstTokenStreamsItsHeadEarly(t *testing.T) {
	setVar(t, &firstByteAfter, fakeBannerDelay/6)
	setVar(t, &pingInterval, fakeBannerDelay/6)

	b := newBridge(t, nil)

	resp := b.post(t, map[string]any{"model": "x", "stream": true, "messages": userText("SCRIPT:slow")})
	expectStatus(t, resp, http.StatusOK)

	reader := bufio.NewReader(resp.Body)

	head, err := nextSSE(reader)
	if err != nil || head.name != "message_start" {
		t.Fatalf("first event %v, %v", head.name, err)
	}

	events := append([]sseEvent{head}, readSSE(t, reader)...)
	names := eventNames(events)

	if names[1] != "ping" || streamedText(events) != "late" || names[len(names)-1] != "message_stop" {
		t.Fatalf("stream %v, text %q", names, streamedText(events))
	}

	text, _, stop, _ := flattenStream(t, events)
	if text != "late" || stop != "end_turn" {
		t.Fatalf("text %q, stop %q", text, stop)
	}

	b.waitIdle(t)
}

// After the early head a failure can no longer be a status: it ends the stream
// as an error event, and the request no longer moves to another account. A
// buffered request has nothing to send early and keeps its status and retry.
func TestFailureAfterTheEarlyHeadEndsTheStream(t *testing.T) {
	setVar(t, &firstByteAfter, fakeBannerDelay/6)

	t.Run("stream", func(t *testing.T) {
		b := newBridgeWith(t, nil, newTestSource(testAccount("a"), testAccount("b")))

		resp := b.post(t, map[string]any{"model": "x", "stream": true, "messages": userText("SCRIPT:429")})
		expectStatus(t, resp, http.StatusOK)

		events := readSSE(t, resp.Body)
		names := eventNames(events)

		if names[0] != "message_start" || names[len(names)-1] != "error" ||
			events[len(events)-1].data["error"].(map[string]any)["type"] != "rate_limit_error" {
			t.Fatalf("stream %v", names)
		}

		if n := b.spawned(t); n != 1 {
			t.Fatalf("agy started %d times, want no retry after the head", n)
		}

		b.waitIdle(t)
	})

	t.Run("buffered", func(t *testing.T) {
		fakeAccounts(t, `{"tok-a":"ok","tok-b":"ok"}`)
		b := newBridgeWith(t, nil, newTestSource(testAccount("a"), testAccount("b")))

		resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:slow")})
		expectStatus(t, resp, http.StatusOK)

		if got := decodeJSON[messageResponse](t, resp); got.Content[0].Text != "late" {
			t.Fatalf("answered %+v", got)
		}

		resp = b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:429")})
		expectStatus(t, resp, http.StatusTooManyRequests)

		if n := b.spawned(t); n != 3 {
			t.Fatalf("agy started %d times, want the slow turn, the 429 and its retry", n)
		}

		b.waitIdle(t)
	})
}

func TestSystemTextIsScrubbed(t *testing.T) {
	scrubFile := filepath.Join(t.TempDir(), "scrub.json")
	writeFile(t, scrubFile, `[{"pattern": "codename \\w+", "replace": "codename redacted"}]`)

	system := []map[string]string{
		{"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.318.13b; cc_entrypoint=sdk-cli;"},
		{"type": "text", "text": "You are a Claude agent, built on Anthropic's Claude Agent SDK.\nCodename AZURE stays."},
	}
	want := "You are an AI coding agent.\ncodename redacted stays."

	tests := []struct {
		name  string
		clean string
		tools any
		want  string
	}{
		{name: "without tools", clean: "false", want: want},
		{name: "built-in tools hint", clean: "false", tools: readFileTool, want: want + "\n\n" + toolHint},
		{name: "clean without tools", clean: "true", want: want},
		{name: "clean has no hint", clean: "true", tools: readFileTool, want: want},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := newBridge(t, map[string]string{"AGY_BRIDGE_SCRUB_FILE": scrubFile, "AGY_BRIDGE_CLEAN": tt.clean})

			body := map[string]any{"model": "x", "system": system, "messages": userText("SCRIPT:system")}
			if tt.tools != nil {
				body["tools"] = tt.tools
			}

			got := decodeJSON[messageResponse](t, b.post(t, body))

			if got.Content[0].Text != tt.want {
				t.Fatalf("agy saw system\n%q\nwant\n%q", got.Content[0].Text, tt.want)
			}
		})
	}
}

func TestSharedSecret(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_API_KEY": "s3cret"})

	tests := []struct {
		name   string
		header http.Header
		status int
	}{
		{name: "missing", status: http.StatusUnauthorized},
		{name: "wrong x-api-key", header: http.Header{"X-Api-Key": {"nope"}}, status: http.StatusUnauthorized},
		{name: "wrong bearer", header: http.Header{"Authorization": {"Bearer nope"}}, status: http.StatusUnauthorized},
		{name: "x-api-key", header: http.Header{"X-Api-Key": {"s3cret"}}, status: http.StatusOK},
		{name: "bearer", header: http.Header{"Authorization": {"Bearer s3cret"}}, status: http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := b.do(t, context.Background(), http.MethodGet, "/v1/models", nil, tt.header)
			expectStatus(t, resp, tt.status)

			if tt.status == http.StatusUnauthorized {
				if got := decodeJSON[errorEnvelope](t, resp); got.Error.Type != "authentication_error" {
					t.Fatalf("error body = %+v", got)
				}
			}
		})
	}

	// The stub's callbacks moved to a unix socket: the public mux no longer
	// knows the old path, even with a valid key.
	header := http.Header{"X-Api-Key": {"s3cret"}}
	resp := b.do(t, context.Background(), http.MethodPost, "/_bridge/unknown/call", map[string]any{}, header)
	expectStatus(t, resp, http.StatusNotFound)

	if got := decodeJSON[errorEnvelope](t, resp); got.Error.Type != "not_found_error" {
		t.Fatalf("error body = %+v", got)
	}
}

func TestCountTokensModelsAndProbes(t *testing.T) {
	b := newBridge(t, nil)
	ctx := context.Background()

	// 1. count_tokens estimates a quarter token per body byte, at least one.
	resp := b.do(t, ctx, http.MethodPost, "/v1/messages/count_tokens", map[string]string{"pad": strings.Repeat("a", 391)}, nil)
	if got := decodeJSON[map[string]int](t, resp); got["input_tokens"] != 100 {
		t.Fatalf("count_tokens = %v", got)
	}

	resp = b.do(t, ctx, http.MethodPost, "/v1/messages/count_tokens", map[string]string{}, nil)
	if got := decodeJSON[map[string]int](t, resp); got["input_tokens"] != 1 {
		t.Fatalf("count_tokens of a tiny body = %v", got)
	}

	// 2. models lists the allowed set in Anthropic's list shape.
	resp = b.do(t, ctx, http.MethodGet, "/v1/models", nil, nil)

	type listed struct {
		Type        string `json:"type"`
		ID          string `json:"id"`
		DisplayName string `json:"display_name"`
	}

	models := decodeJSON[struct {
		Data    []listed `json:"data"`
		HasMore bool     `json:"has_more"`
		FirstID string   `json:"first_id"`
		LastID  string   `json:"last_id"`
	}](t, resp)

	want := []listed{
		{Type: "model", ID: "claude-opus-5-5", DisplayName: "Claude Opus 5.5"},
		{Type: "model", ID: "claude-sonnet-5-5", DisplayName: "Claude Sonnet 5.5"},
		{Type: "model", ID: "gemini-3.8-flash", DisplayName: "Gemini 3.8 Flash"},
		{Type: "model", ID: "gemini-4-argon", DisplayName: "Gemini 4 Argon"},
		{Type: "model", ID: "gpt-oss-120b", DisplayName: "GPT-OSS 120B"},
	}

	if !slices.Equal(models.Data, want) || models.HasMore ||
		models.FirstID != "claude-opus-5-5" || models.LastID != "gpt-oss-120b" {
		t.Fatalf("models = %+v", models)
	}

	// 3. HEAD answers 404 with no body; unknown routes get a not_found_error.
	resp = b.do(t, ctx, http.MethodHead, "/v1/messages", nil, nil)
	expectStatus(t, resp, http.StatusNotFound)

	if body, _ := io.ReadAll(resp.Body); len(body) != 0 {
		t.Fatalf("HEAD body %q", body)
	}

	resp = b.do(t, ctx, http.MethodGet, "/v1/nope", nil, nil)
	expectStatus(t, resp, http.StatusNotFound)

	if got := decodeJSON[errorEnvelope](t, resp); got.Error.Type != "not_found_error" {
		t.Fatalf("error body = %+v", got)
	}
}

func TestInvalidRequests(t *testing.T) {
	b := newBridge(t, nil)

	for name, body := range map[string]any{
		"no messages":     map[string]any{"model": "x"},
		"numeric content": map[string]any{"model": "x", "messages": []map[string]any{{"role": "user", "content": 1}}},
		"not an object":   "text",
	} {
		t.Run(name, func(t *testing.T) {
			resp := b.post(t, body)
			expectStatus(t, resp, http.StatusBadRequest)

			if got := decodeJSON[errorEnvelope](t, resp); got.Error.Type != "invalid_request_error" {
				t.Fatalf("error body = %+v", got)
			}
		})
	}

	if n := b.spawned(t); n != 0 {
		t.Fatalf("an invalid request started agy %d times", n)
	}
}

func TestPendingSessionExpires(t *testing.T) {
	b := newBridge(t, nil, func(cfg *Config) { cfg.PendingTTL = 300 * time.Millisecond })
	first := userText("SCRIPT:tool")

	_, calls, _, _ := b.turn(t, false, first)

	// 1. Nobody returns the result: the session and its agy go away.
	b.waitIdle(t)

	// 2. A late result no longer resumes anything and falls back to a replay.
	messages := append(first, assistantCalls("", calls), toolResultMessage(map[string]string{calls[0].id: "late"}))
	_, again, stop, _ := b.turn(t, false, messages)

	if stop != "tool_use" || len(again) != 1 || again[0].id == calls[0].id {
		t.Fatalf("replay gave stop %q, calls %+v", stop, again)
	}

	if n := b.spawned(t); n != 2 {
		t.Fatalf("agy started %d times, want 2", n)
	}
}

func TestHeldToolCallTimesOut(t *testing.T) {
	setVar(t, &toolResultTimeout, 200*time.Millisecond)

	b := newBridge(t, nil)

	b.turn(t, false, userText("SCRIPT:tool"))

	// agy gets the timeout as a failed tool result, finishes its turn and exits.
	b.waitIdle(t)
}

func TestClientDisconnectKillsTheSession(t *testing.T) {
	b := newBridge(t, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	body := map[string]any{"model": "x", "stream": true, "messages": userText("SCRIPT:hang")}
	resp := b.do(t, ctx, http.MethodPost, "/v1/messages", body, nil)
	expectStatus(t, resp, http.StatusOK)

	waitForEvent(t, bufio.NewReader(resp.Body), "content_block_delta")
	cancel()

	b.waitIdle(t)
}

func TestCloseKillsLiveSessions(t *testing.T) {
	b := newBridge(t, nil)

	body := map[string]any{"model": "x", "stream": true, "messages": userText("SCRIPT:hang")}
	resp := b.post(t, body)
	reader := bufio.NewReader(resp.Body)

	waitForEvent(t, reader, "content_block_delta")
	b.srv.Close()

	// The stream ends with an error event once the session is killed.
	waitForEvent(t, reader, "error")
	assertProcessesGone(t)
}

func TestConcurrencyLimitQueuesSessions(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_MAX_CONCURRENT": "1"})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 1. A hanging stream takes the only slot.
	hanging := b.do(t, ctx, http.MethodPost, "/v1/messages", map[string]any{
		"model":    "x",
		"stream":   true,
		"messages": userText("SCRIPT:hang"),
	}, nil)
	waitForEvent(t, bufio.NewReader(hanging.Body), "content_block_delta")

	// 2. A second request waits for it.
	done := make(chan *http.Response, 1)

	go func() {
		done <- b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	}()

	select {
	case <-done:
		t.Fatal("second session ran while the only slot was taken")
	case <-time.After(300 * time.Millisecond):
	}

	// 3. Releasing the slot lets it run.
	cancel()

	resp := <-done
	if got := decodeJSON[messageResponse](t, resp); got.Content[0].Text != "Hello, world" {
		t.Fatalf("queued session answered %+v", got)
	}
}

func waitForEvent(t *testing.T, reader *bufio.Reader, name string) {
	t.Helper()

	for {
		event, err := nextSSE(reader)
		if err != nil {
			t.Fatalf("stream ended before %s: %v", name, err)
		}

		if event.name == name {
			return
		}
	}
}

func setVar[T any](t *testing.T, v *T, value T) {
	t.Helper()

	old := *v
	*v = value

	t.Cleanup(func() { *v = old })
}

// turn sends one Messages request with the read_file tool and returns the
// bodyTokens is the estimate the bridge reports for the request turn sends: a
// quarter of its body bytes plus agy's own prompt.
func bodyTokens(t *testing.T, stream bool, messages []map[string]any) int {
	t.Helper()

	raw, err := json.Marshal(map[string]any{"model": "x", "stream": stream, "tools": readFileTool, "messages": messages})
	if err != nil {
		t.Fatal(err)
	}

	return len(raw)/4 + agyOverheadTokens
}

// response flattened to its text, tool calls, stop reason and usage.
func (b *bridge) turn(t *testing.T, stream bool, messages []map[string]any) (string, []toolUse, string, usage) {
	t.Helper()

	resp := b.post(t, map[string]any{"model": "x", "stream": stream, "tools": readFileTool, "messages": messages})
	expectStatus(t, resp, http.StatusOK)

	if !stream {
		got := decodeJSON[messageResponse](t, resp)

		var (
			text  string
			calls []toolUse
		)

		for _, block := range got.Content {
			switch block.Type {
			case "text":
				text += block.Text
			case "tool_use":
				calls = append(calls, toolUse{id: block.ID, name: block.Name, input: string(block.Input)})
			}
		}

		return text, calls, got.StopReason, got.Usage
	}

	return flattenStream(t, readSSE(t, resp.Body))
}

// flattenStream checks block indexes run 0, 1, 2... and rebuilds the message.
func flattenStream(t *testing.T, events []sseEvent) (string, []toolUse, string, usage) {
	t.Helper()

	var (
		calls []toolUse
		stop  string
		use   usage
		next  float64
	)

	for _, e := range events {
		switch e.name {
		case "content_block_start":
			if e.data["index"] != next {
				t.Fatalf("block index %v, want %v", e.data["index"], next)
			}

			if block := e.data["content_block"].(map[string]any); block["type"] == "tool_use" {
				calls = append(calls, toolUse{id: block["id"].(string), name: block["name"].(string)})
			}
		case "content_block_delta":
			if delta := e.data["delta"].(map[string]any); delta["type"] == "input_json_delta" {
				calls[len(calls)-1].input += delta["partial_json"].(string)
			}
		case "content_block_stop":
			next++
		case "message_delta":
			stop = e.data["delta"].(map[string]any)["stop_reason"].(string)
			u := e.data["usage"].(map[string]any)
			use = usage{
				InputTokens:              int(u["input_tokens"].(float64)),
				CacheCreationInputTokens: int(u["cache_creation_input_tokens"].(float64)),
				CacheReadInputTokens:     int(u["cache_read_input_tokens"].(float64)),
				OutputTokens:             int(u["output_tokens"].(float64)),
			}
		}
	}

	if names := eventNames(events); names[0] != "message_start" || names[len(names)-1] != "message_stop" {
		t.Fatalf("stream framing %v", names)
	}

	return streamedText(events), calls, stop, use
}

func assistantCalls(text string, calls []toolUse) map[string]any {
	content := []map[string]any{}
	if text != "" {
		content = append(content, map[string]any{"type": "text", "text": text})
	}

	for _, call := range calls {
		content = append(content, map[string]any{
			"type":  "tool_use",
			"id":    call.id,
			"name":  call.name,
			"input": json.RawMessage(call.input),
		})
	}

	return map[string]any{"role": "assistant", "content": content}
}

func toolResultMessage(results map[string]string) map[string]any {
	content := []map[string]any{}

	for _, id := range slices.Sorted(maps.Keys(results)) {
		content = append(content, map[string]any{
			"type":        "tool_result",
			"tool_use_id": id,
			"content":     []map[string]string{{"type": "text", "text": results[id]}},
		})
	}

	return map[string]any{"role": "user", "content": content}
}
