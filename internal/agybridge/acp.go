package agybridge

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

const (
	// statusOverloaded is Anthropic's overloaded status; clients retry it.
	statusOverloaded = 529

	failureMessageRunes = 300
)

// agentWaitDelay bounds how long Wait lingers once agy has been killed.
var agentWaitDelay = 5 * time.Second

// apiError is a failure the client sees as an Anthropic error body.
type apiError struct {
	status  int
	kind    string
	message string
	// retryAfter, when set, is the Retry-After the client is told instead of
	// the fixed one.
	retryAfter time.Duration
	// reason names why the bridge itself refused a request, for the request
	// log only; the client never sees it.
	reason string
}

func (e *apiError) Error() string {
	return fmt.Sprintf("%d %s: %s", e.status, e.kind, e.message)
}

// failureKind is what an upstream failure says about the account that ran it.
type failureKind string

const (
	kindOK           failureKind = "ok"
	kindQuota        failureKind = "quota"
	kindRate         failureKind = "rate"
	kindCapacity     failureKind = "capacity"
	kindAuth         failureKind = "auth"
	kindVerification failureKind = "verification"
	kindOther        failureKind = "other"
	// kindExpired is an auth failure on an access-only token that simply ran
	// out: the bridge's own doing, never reported to the source.
	kindExpired failureKind = "expired"
)

// failure is a classified upstream failure. Status is what the client sees;
// RetryAfter is the delay the text named, zero when it named none.
type failure struct {
	Status     int
	Kind       failureKind
	RetryAfter time.Duration
}

// maxRetryAfter caps a parsed delay; agy's longest quota window is a week.
const maxRetryAfter = 7 * 24 * time.Hour

var (
	// bannerPrefixes start the text agy streams in place of an answer. A notice
	// missing here reaches the client as an ordinary reply, so the gateway sees
	// a 200 and cannot fail over from it.
	bannerPrefixes = []string{"⚠ API error", "⚠ Agent execution", "⚠ Individual quota", "Authentication required"}

	// retryAfterPattern finds a delay after a word such as retry, reset, delay
	// or after: `retry in 12.5s`, `retryDelay: "30s"`, `quotaResetDelay`,
	// `resets in 1h2m3s`, `after 40` (a bare count is seconds).
	retryAfterPattern = regexp.MustCompile(
		`(?:retry|reset|delay|after)[a-z_]*["']?\s*[:=]?\s*(?:(?:in|after)\s+)?["']?` +
			`((?:\d+(?:\.\d+)?(?:ms|h|m|s))+|\d+(?:\.\d+)?)`,
	)

	statusCodes = map[string]*regexp.Regexp{
		"401": regexp.MustCompile(`\b401\b`),
		"403": regexp.MustCompile(`\b403\b`),
		"429": regexp.MustCompile(`\b429\b`),
		"503": regexp.MustCompile(`\b503\b`),
		"529": regexp.MustCompile(`\b529\b`),
	}

	// emailPattern also catches an address URL-encoded into a query string.
	emailPattern = regexp.MustCompile(`[A-Za-z0-9._%+-]+(?:@|%40)[A-Za-z0-9.-]+\.[A-Za-z]{2,}`)

	// tokenPattern finds a run long enough to be an access token or a key, which
	// agy's stderr must never carry into the bridge's log.
	tokenPattern = regexp.MustCompile(`[A-Za-z0-9_.\-+=]{32,}`)
)

// classifyBanner recognises the text agy streams in place of an answer when
// the upstream call failed, and nil for anything else. agy then retries for
// minutes, so the caller kills it at once rather than waiting the retries out.
func classifyBanner(chunk string) *failure {
	text := strings.TrimSpace(chunk)

	if !slices.ContainsFunc(bannerPrefixes, func(prefix string) bool { return strings.HasPrefix(text, prefix) }) {
		return nil
	}

	f := classifyText(text)

	return &f
}

// classifyText classifies a failure by its words alone, the whole text: the
// cut to failureMessageRunes is for the message only. Google's wording is not
// pinned down, so every match is tolerant; the order decides an overlap.
func classifyText(text string) failure {
	lower := strings.ToLower(text)
	f := failure{Status: http.StatusBadGateway, Kind: kindOther, RetryAfter: retryAfterIn(lower)}

	switch {
	case containsAny(lower, "validation_required", "verify your account") ||
		(hasStatus(lower, "403") && strings.Contains(lower, "validat")):
		f.Kind = kindVerification
	case containsAny(lower, "authentication required", "unauthenticated", "invalid_grant") || hasStatus(lower, "401"):
		f.Kind = kindAuth
	case containsAny(lower, "resource_exhausted", "quota"):
		f.Status, f.Kind = http.StatusTooManyRequests, kindQuota
	case hasStatus(lower, "429"):
		f.Status, f.Kind = http.StatusTooManyRequests, kindRate
	case containsAny(lower, "unavailable", "overloaded") || hasStatus(lower, "503") || hasStatus(lower, "529"):
		f.Status, f.Kind = statusOverloaded, kindCapacity
	}

	return f
}

// apiError is the failure as the client sees it: 429, 529 or 502.
func (f failure) apiError(message string) *apiError {
	switch f.Status {
	case http.StatusTooManyRequests:
		return &apiError{status: f.Status, kind: "rate_limit_error", message: message}
	case statusOverloaded:
		return &apiError{status: f.Status, kind: "overloaded_error", message: message}
	}

	return &apiError{status: http.StatusBadGateway, kind: "api_error", message: message}
}

// bannerMessage is the failure text fit for the client and the log: trimmed,
// cut, and without an email address.
func bannerMessage(text string) string {
	return truncateRunes(emailPattern.ReplaceAllString(strings.TrimSpace(text), "<email>"), failureMessageRunes)
}

func retryAfterIn(lower string) time.Duration {
	match := retryAfterPattern.FindStringSubmatch(lower)
	if match == nil {
		return 0
	}

	d, err := time.ParseDuration(match[1])
	if err != nil {
		seconds, err := strconv.ParseFloat(match[1], 64)
		if err != nil {
			return 0
		}

		d = time.Duration(seconds * float64(time.Second))
	}

	return min(max(d, 0), maxRetryAfter)
}

func containsAny(s string, words ...string) bool {
	return slices.ContainsFunc(words, func(word string) bool { return strings.Contains(s, word) })
}

func hasStatus(lower, code string) bool {
	return statusCodes[code].MatchString(lower)
}

// agent is one `agy acp` process spoken to in newline-delimited JSON-RPC 2.0.
// agy runs in its own process group so that killing it also takes down the
// helpers it starts, the MCP stub among them.
type agent struct {
	cmd      *exec.Cmd
	stdin    io.WriteCloser
	onChunk  func(text string)
	onDenied func(method string)
	// onToolCall runs for every tool call agy starts itself (its own web
	// search among them) or hands to the client.
	onToolCall func()

	// prompts is what agy takes in a prompt, read by initialize.
	prompts promptCapabilities

	// activity is when agy last showed a sign of life, as unix nanoseconds: a
	// message or thought chunk, a tool call or an update to one, or a request
	// of its own. A command list or a plan says nothing about the turn.
	activity atomic.Int64

	killOnce sync.Once
	killErr  error

	writeMu sync.Mutex
	nextID  atomic.Int64

	mu      sync.Mutex
	waiting map[int64]chan rpcMessage

	readDone chan struct{}
}

type rpcMessage struct {
	ID     json.RawMessage `json:"id"`
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
	Result json.RawMessage `json:"result"`
	Error  *rpcError       `json:"error"`
}

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      int64  `json:"id"`
	Method  string `json:"method"`
	Params  any    `json:"params"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *rpcError) Error() string {
	return fmt.Sprintf("agy error %d: %s", e.Code, truncateRunes(e.Message, failureMessageRunes))
}

// errAgyStart marks a failure to launch agy at all.
var errAgyStart = errors.New("start agy")

// agentEnv is the environment agy runs with: the bridge's own without the
// bridge API key, which agy and the model behind it have no use for, with
// agy's self-updater off so a running bridge never changes the agy it pinned,
// and with home as HOME. exec takes the last of duplicate keys, so HOME goes
// last and wins over the bridge's own.
func agentEnv(base []string, home string) []string {
	env := make([]string, 0, len(base)+2)

	for _, entry := range base {
		if strings.HasPrefix(entry, "AGY_BRIDGE_API_KEY=") {
			continue
		}

		env = append(env, entry)
	}

	return append(env, "AGY_CLI_DISABLE_AUTO_UPDATE=true", "HOME="+home)
}

// stderrTailBytes bounds the tail of agy's stderr kept for the failure log.
const stderrTailBytes = 2048

// tailBuffer keeps the last stderrTailBytes bytes written to it: agy's stderr
// is the only place a crash explains itself, and a chatty agy must not grow
// the bridge's memory.
type tailBuffer struct {
	mu  sync.Mutex
	buf []byte
}

func (t *tailBuffer) Write(p []byte) (int, error) {
	n := len(p)
	p = p[max(0, n-stderrTailBytes):]

	t.mu.Lock()
	defer t.mu.Unlock()

	t.buf = append(t.buf, p...)
	t.buf = t.buf[max(0, len(t.buf)-stderrTailBytes):]

	return n, nil
}

// masked is the tail fit for the log: trimmed, without an email address and
// without a token-like run.
func (t *tailBuffer) masked() string {
	t.mu.Lock()
	defer t.mu.Unlock()

	text := emailPattern.ReplaceAllString(string(t.buf), "<email>")

	return strings.TrimSpace(tokenPattern.ReplaceAllString(text, "<redacted>"))
}

// startAgent spawns agy with home as its HOME; cancelling ctx kills it.
// onChunk receives every agent_message_chunk text, onDenied the method of
// every agent request answered cancelled and onToolCall every tool_call
// update, all on the reader goroutine; none may block. stderr receives agy's
// stderr until the process is reaped.
// extraArgs go between `acp` and `--disable-telemetry`.
func startAgent(
	ctx context.Context,
	bin string,
	extraArgs []string,
	home string,
	stderr io.Writer,
	onChunk, onDenied func(string),
	onToolCall func(),
) (*agent, error) {
	args := append(append([]string{"acp"}, extraArgs...), "--disable-telemetry")

	cmd := exec.CommandContext(ctx, bin, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.WaitDelay = agentWaitDelay
	cmd.Env = agentEnv(os.Environ(), home)
	cmd.Stderr = stderr

	a := &agent{
		cmd:        cmd,
		onChunk:    onChunk,
		onDenied:   onDenied,
		onToolCall: onToolCall,
		waiting:    map[int64]chan rpcMessage{},
		readDone:   make(chan struct{}),
	}
	cmd.Cancel = a.kill

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("%w: stdin: %w", errAgyStart, err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("%w: stdout: %w", errAgyStart, err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("%w: %w", errAgyStart, err)
	}

	a.stdin = stdin

	go a.readLoop(stdout)

	return a, nil
}

// kill ends the whole process group agy leads, once: the context and close
// both reach here, and a repeat would hit a group of zombies.
func (a *agent) kill() error {
	a.killOnce.Do(func() {
		a.killErr = syscall.Kill(-a.cmd.Process.Pid, syscall.SIGKILL)
	})

	return a.killErr
}

// close kills agy, reaps it and waits for the reader to stop. Wait closes the
// stdout pipe, which also ends the reader when a stray helper still holds it.
func (a *agent) close() {
	// A group that already exited answers ESRCH, or EPERM on macOS while its
	// members are unreaped zombies.
	err := a.kill()
	if err != nil && !errors.Is(err, syscall.ESRCH) && !errors.Is(err, syscall.EPERM) {
		log.Printf("kill agy: %v", err)
	}

	// The exit status of a process killed on purpose carries nothing.
	_ = a.cmd.Wait()

	<-a.readDone
}

// initialize opens the ACP connection and keeps the prompt content agy
// advertised, which is all a prompt may carry besides text.
func (a *agent) initialize(ctx context.Context) error {
	params := map[string]any{"protocolVersion": 1, "clientCapabilities": struct{}{}}

	// A reply with no result advertises nothing; it is not a failure.
	result, err := a.request(ctx, "initialize", params)
	if err != nil {
		return err
	}

	a.prompts = parsePromptCapabilities(result)

	return nil
}

// call sends a request and waits for its response, decoding the result into
// result when it is not nil.
func (a *agent) call(ctx context.Context, method string, params, result any) error {
	raw, err := a.request(ctx, method, params)
	if err != nil {
		return err
	}

	if result == nil {
		return nil
	}

	if err := json.Unmarshal(raw, result); err != nil {
		return fmt.Errorf("%s result: %w", method, err)
	}

	return nil
}

// request sends a request and returns the raw result of its response, empty
// when the response carried none.
func (a *agent) request(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := a.nextID.Add(1)
	reply := make(chan rpcMessage, 1)

	a.mu.Lock()
	a.waiting[id] = reply
	a.mu.Unlock()

	defer func() {
		a.mu.Lock()
		delete(a.waiting, id)
		a.mu.Unlock()
	}()

	if err := a.send(rpcRequest{JSONRPC: "2.0", ID: id, Method: method, Params: params}); err != nil {
		return nil, fmt.Errorf("%s: %w", method, err)
	}

	var m rpcMessage

	select {
	case m = <-reply:
	case <-a.readDone:
		select {
		case m = <-reply:
		default:
			return nil, fmt.Errorf("%s: agy exited", method)
		}
	case <-ctx.Done():
		return nil, fmt.Errorf("%s: %w", method, ctx.Err())
	}

	if m.Error != nil {
		return nil, fmt.Errorf("%s: %w", method, m.Error)
	}

	return m.Result, nil
}

func (a *agent) send(v any) error {
	line, err := json.Marshal(v)
	if err != nil {
		return err
	}

	a.writeMu.Lock()
	defer a.writeMu.Unlock()

	_, err = a.stdin.Write(append(line, '\n'))

	return err
}

func (a *agent) readLoop(stdout io.Reader) {
	defer close(a.readDone)

	r := bufio.NewReader(stdout)

	for {
		line, err := r.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			a.dispatch(line)
		}

		if err != nil {
			return
		}
	}
}

func (a *agent) dispatch(line []byte) {
	var m rpcMessage
	if err := json.Unmarshal(line, &m); err != nil {
		log.Printf("agy wrote a non JSON-RPC line (%d bytes)", len(line))

		return
	}

	switch {
	case m.Method != "" && hasID(m.ID):
		a.deny(m)
	case m.Method == "session/update":
		a.update(m.Params)
	case m.Method == "":
		a.deliver(m)
	}
}

// deny answers every agent request, permission prompts above all: the client
// performs each action itself, and an unanswered request stalls the turn.
func (a *agent) deny(m rpcMessage) {
	answer := rpcResponse{
		JSONRPC: "2.0",
		ID:      m.ID,
		Result:  map[string]any{"outcome": map[string]string{"outcome": "cancelled"}},
	}

	if err := a.send(answer); err != nil {
		log.Printf("answer agy %s: %v", m.Method, err)

		return
	}

	a.touch()
	a.onDenied(m.Method)
}

// touch notes a sign of life.
func (a *agent) touch() {
	a.activity.Store(time.Now().UnixNano())
}

func (a *agent) update(params json.RawMessage) {
	var p struct {
		Update struct {
			SessionUpdate string `json:"sessionUpdate"`
			Content       struct {
				Text string `json:"text"`
			} `json:"content"`
		} `json:"update"`
	}

	if err := json.Unmarshal(params, &p); err != nil {
		log.Printf("agy session/update: %v", err)

		return
	}

	switch p.Update.SessionUpdate {
	case "agent_message_chunk", "agent_thought_chunk", "tool_call", "tool_call_update":
		a.touch()
	}

	switch {
	case p.Update.SessionUpdate == "agent_message_chunk" && p.Update.Content.Text != "":
		a.onChunk(p.Update.Content.Text)
	case p.Update.SessionUpdate == "tool_call":
		a.onToolCall()
	}
}

func (a *agent) deliver(m rpcMessage) {
	// The bridge numbers its requests; any other id is not an answer to one.
	var id int64
	if err := json.Unmarshal(m.ID, &id); err != nil {
		return
	}

	a.mu.Lock()
	reply, ok := a.waiting[id]
	a.mu.Unlock()

	if !ok {
		return
	}

	// A duplicate answer must not stall the reader.
	select {
	case reply <- m:
	default:
	}
}

func hasID(id json.RawMessage) bool {
	return !isAbsent(id)
}

func truncateRunes(s string, n int) string {
	runes := []rune(s)
	if len(runes) <= n {
		return s
	}

	return string(runes[:n])
}
