// Package agybridge serves the Anthropic Messages API over Google
// Antigravity's `agy acp`, so Anthropic clients such as Claude Code can use
// Antigravity models. Client tools reach agy through a stdio MCP stub that
// holds each call inside the agy turn until the client returns its result.
package agybridge

import (
	"cmp"
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log"
	"maps"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const (
	maxBodyBytes    = 64 << 20
	shutdownTimeout = 10 * time.Second
)

// baseModel is a model agy serves at one or more thinking levels.
type baseModel struct {
	id          string
	displayName string
	levels      []string
}

var allLevels = []string{levelLow, levelMedium, levelHigh}

// catalogue is what agy 1.5.2 offers in session/new.
var catalogue = []baseModel{
	{id: "gemini-4-argon", displayName: "Gemini 4 Argon", levels: allLevels},
	{id: "gemini-3.8-flash", displayName: "Gemini 3.8 Flash", levels: allLevels},
	{id: "claude-opus-5-5", displayName: "Claude Opus 5.5", levels: allLevels},
	{id: "claude-sonnet-5-5", displayName: "Claude Sonnet 5.5", levels: allLevels},
	{id: "gpt-oss-120b", displayName: "GPT-OSS 120B", levels: []string{levelMedium}},
}

// lookupModel finds a base model in the catalogue; a configured id agy may
// know beyond it is named by its id and taken at every level.
func lookupModel(id string) baseModel {
	for _, model := range catalogue {
		if model.id == id {
			return model
		}
	}

	return baseModel{id: id, displayName: id, levels: allLevels}
}

// agyID returns the model id agy takes for the level and the level it runs
// at: a level the model lacks falls to its first, for a single-level model
// its only one.
func (m baseModel) agyID(level string) (id, effective string) {
	if !slices.Contains(m.levels, level) {
		level = m.levels[0]
	}

	return m.id + "-" + level, level
}

// Config is the bridge's runtime configuration.
type Config struct {
	Listen string
	AgyBin string
	// Models are the base model ids clients may name, AGY_BRIDGE_MODELS as a
	// comma list; the request picks the thinking level agy runs them at.
	Models []string
	// DefaultModel is the base model an unlisted model name runs on.
	DefaultModel string
	// MaxConcurrent bounds agy processes at work; a parked turn holds none.
	MaxConcurrent int
	// PerAccount bounds the agy processes at work under one account.
	PerAccount int
	// AffinityTTL is how long a conversation sticks to its account after its
	// last turn, so Google's prompt cache for it is reused.
	AffinityTTL time.Duration
	// MaxParked bounds turns parked on tool results; at the cap a new turn
	// evicts the oldest parked one.
	MaxParked int
	// PendingTTL is how long a parked turn waits for its tool results.
	PendingTTL time.Duration
	// SlotWait bounds the wait for a free slot before answering 529.
	SlotWait time.Duration
	// APIKey, when set, is required on every route.
	APIKey string
	// ScrubFile names an optional JSON array of extra scrub rules.
	ScrubFile string
	// Clean runs agy as a plain model: no built-in agent harness, only the
	// client's tools and, with Web, agy's web search and fetch.
	Clean bool
	Web   bool
	// TokenFile names an agy token file whose one account the bridge serves,
	// for development. Without it Kodizm pushes the accounts to the admin
	// routes and polls back the outcomes and the quota readings.
	TokenFile string
	// UsageInterval spaces the `agy -p /usage` polls of the accounts; zero
	// turns them off.
	UsageInterval time.Duration
	// Homes is the absolute directory every agy process gets a throwaway HOME
	// below, as <Homes>/<account id>/<random hex>.
	Homes string
}

// checkBaseModel refuses an agy composite id such as gemini-3.8-flash-low: the
// bridge appends the level itself, so it would reach agy as
// gemini-3.8-flash-low-medium and every request would fail.
func checkBaseModel(key, id string) error {
	for _, level := range []string{levelLow, levelMedium, levelHigh} {
		if strings.HasSuffix(id, "-"+level) {
			return fmt.Errorf("%s: %q names a level; list the base model and let the request choose it", key, id)
		}
	}

	return nil
}

// LoadConfig reads the AGY_BRIDGE_* variables and AGY_BIN.
func LoadConfig(getenv func(string) string) (Config, error) {
	cfg := Config{
		Listen:        envOr(getenv, "AGY_BRIDGE_LISTEN", "127.0.0.1:8095"),
		AgyBin:        envOr(getenv, "AGY_BIN", "agy"),
		DefaultModel:  envOr(getenv, "AGY_BRIDGE_DEFAULT_MODEL", "gemini-3.8-flash"),
		APIKey:        getenv("AGY_BRIDGE_API_KEY"),
		ScrubFile:     getenv("AGY_BRIDGE_SCRUB_FILE"),
		TokenFile:     getenv("AGY_BRIDGE_TOKEN_FILE"),
		Homes:         filepath.Clean(envOr(getenv, "AGY_BRIDGE_HOMES", filepath.Join(os.TempDir(), "agy-bridge"))),
		UsageInterval: defaultUsageInterval,
	}

	if !filepath.IsAbs(cfg.Homes) {
		return Config{}, fmt.Errorf("AGY_BRIDGE_HOMES must be an absolute path, got %q", cfg.Homes)
	}

	if raw := getenv("AGY_BRIDGE_USAGE_INTERVAL"); raw != "" {
		interval, err := time.ParseDuration(raw)
		if err != nil {
			return Config{}, fmt.Errorf("AGY_BRIDGE_USAGE_INTERVAL must be a duration such as 5m, got %q", raw)
		}

		cfg.UsageInterval = interval
	}

	if err := checkUsageInterval(cfg.UsageInterval); err != nil {
		return Config{}, err
	}

	flags := []struct {
		key    string
		target *bool
	}{
		{key: "AGY_BRIDGE_CLEAN", target: &cfg.Clean},
		{key: "AGY_BRIDGE_WEB", target: &cfg.Web},
	}

	for _, v := range flags {
		on, err := boolOr(getenv, v.key, true)
		if err != nil {
			return Config{}, err
		}

		*v.target = on
	}

	for _, model := range catalogue {
		cfg.Models = append(cfg.Models, model.id)
	}

	if raw := getenv("AGY_BRIDGE_MODELS"); raw != "" {
		cfg.Models = nil

		for model := range strings.SplitSeq(raw, ",") {
			if model = strings.TrimSpace(model); model != "" {
				cfg.Models = append(cfg.Models, model)
			}
		}
	}

	for _, id := range cfg.Models {
		if err := checkBaseModel("AGY_BRIDGE_MODELS", id); err != nil {
			return Config{}, err
		}
	}

	if err := checkBaseModel("AGY_BRIDGE_DEFAULT_MODEL", cfg.DefaultModel); err != nil {
		return Config{}, err
	}

	ints := []struct {
		key      string
		fallback int
		target   *int
	}{
		{key: "AGY_BRIDGE_MAX_CONCURRENT", fallback: 8, target: &cfg.MaxConcurrent},
		{key: "AGY_BRIDGE_PER_ACCOUNT", fallback: 2, target: &cfg.PerAccount},
		{key: "AGY_BRIDGE_MAX_PARKED", fallback: 6, target: &cfg.MaxParked},
	}

	for _, v := range ints {
		n, err := positiveInt(getenv, v.key, v.fallback)
		if err != nil {
			return Config{}, err
		}

		*v.target = n
	}

	durations := []struct {
		key      string
		fallback int
		target   *time.Duration
	}{
		{key: "AGY_BRIDGE_PENDING_TTL", fallback: 300, target: &cfg.PendingTTL},
		{key: "AGY_BRIDGE_SLOT_WAIT", fallback: 30, target: &cfg.SlotWait},
		{key: "AGY_BRIDGE_AFFINITY_TTL", fallback: 3600, target: &cfg.AffinityTTL},
	}

	for _, v := range durations {
		seconds, err := positiveInt(getenv, v.key, v.fallback)
		if err != nil {
			return Config{}, err
		}

		*v.target = time.Duration(seconds) * time.Second
	}

	return cfg, nil
}

func envOr(getenv func(string) string, key, fallback string) string {
	if v := getenv(key); v != "" {
		return v
	}

	return fallback
}

func boolOr(getenv func(string) string, key string, fallback bool) (bool, error) {
	raw := getenv(key)
	if raw == "" {
		return fallback, nil
	}

	on, err := strconv.ParseBool(raw)
	if err != nil {
		return false, fmt.Errorf("%s must be a boolean, got %q", key, raw)
	}

	return on, nil
}

func positiveInt(getenv func(string) string, key string, fallback int) (int, error) {
	raw := getenv(key)
	if raw == "" {
		return fallback, nil
	}

	n, err := strconv.Atoi(raw)
	if err != nil || n < 1 {
		return 0, fmt.Errorf("%s must be a positive integer, got %q", key, raw)
	}

	return n, nil
}

// Server is the HTTP front. Sessions outlive the requests that start them,
// so the server, not a request, owns them.
type Server struct {
	cfg     Config
	models  map[string]baseModel
	keyHash [sha256.Size]byte
	scrub   *scrubber
	reg     *registry
	slots   chan struct{}
	pool    *pool
	homes   *spawnHomes
	logins  *loginManager
	// pushed is the account source the admin routes feed; nil when the
	// bridge serves a token file.
	pushed *pushedSource

	root      string
	workspace string
	socket    string
	selfExe   string

	callbacks     *http.Server
	callbacksDone chan struct{}

	// usage polls agy for each account's quota; nil when turned off.
	usage *usagePoller

	ctx    context.Context
	cancel context.CancelFunc

	lifeMu   sync.Mutex
	closed   bool
	sessions sync.WaitGroup
}

// New prepares a server running agy under the accounts source supplies, and
// starts serving the MCP stub's callbacks on a unix socket in its private
// work dir.
func New(cfg Config, source AccountSource) (*Server, error) {
	if cfg.MaxConcurrent < 1 || cfg.MaxParked < 1 || cfg.PerAccount < 1 {
		return nil, fmt.Errorf(
			"concurrency limits must be at least 1, got %d, %d and %d",
			cfg.MaxConcurrent,
			cfg.MaxParked,
			cfg.PerAccount,
		)
	}

	if source == nil {
		return nil, errors.New("no account source")
	}

	if err := checkUsageInterval(cfg.UsageInterval); err != nil {
		return nil, err
	}

	self, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("locate own binary: %w", err)
	}

	homes, err := newSpawnHomes(cfg.Homes)
	if err != nil {
		return nil, err
	}

	// MkdirTemp makes the dir 0700: only this user reaches the workspace,
	// the tool lists and the callback socket.
	root, err := os.MkdirTemp("", "agy-bridge-")
	if err != nil {
		return nil, fmt.Errorf("work dir: %w", err)
	}

	pushed, _ := source.(*pushedSource)

	ctx, cancel := context.WithCancel(context.Background())
	srv := &Server{
		pushed:    pushed,
		cfg:       cfg,
		models:    map[string]baseModel{},
		keyHash:   sha256.Sum256([]byte(cfg.APIKey)),
		scrub:     newScrubber(cfg.ScrubFile),
		reg:       newRegistry(),
		slots:     make(chan struct{}, cfg.MaxConcurrent),
		pool:      newPool(ctx, source, cfg.PerAccount, cfg.AffinityTTL),
		homes:     homes,
		logins:    newLoginManager(ctx, cfg.AgyBin, homes),
		root:      root,
		workspace: filepath.Join(root, "workspace"),
		selfExe:   self,
		ctx:       ctx,
		cancel:    cancel,
	}

	for _, id := range cfg.Models {
		srv.models[id] = lookupModel(id)
	}

	// agy runs in an empty workspace: the client performs every action.
	if err := os.Mkdir(srv.workspace, 0o700); err != nil {
		cancel()
		srv.pool.close()

		return nil, errors.Join(fmt.Errorf("workspace: %w", err), os.RemoveAll(root))
	}

	if err := srv.serveCallbacks(); err != nil {
		cancel()
		srv.pool.close()

		return nil, errors.Join(err, os.RemoveAll(root))
	}

	if cfg.UsageInterval > 0 {
		sink, _ := source.(UsageSink)
		srv.usage = startUsagePoller(ctx, cfg.AgyBin, cfg.UsageInterval, homes, srv.pool, sink)
	}

	return srv, nil
}

// Close kills every session, waits for them to finish, stops the callback
// socket and removes the work dir. It unblocks handlers waiting on a
// session, so call it before draining the HTTP server.
func (srv *Server) Close() {
	srv.lifeMu.Lock()
	if srv.closed {
		srv.lifeMu.Unlock()

		return
	}

	srv.closed = true
	srv.lifeMu.Unlock()

	srv.cancel()
	srv.sessions.Wait()
	srv.logins.close()

	if srv.usage != nil {
		<-srv.usage.done
	}

	srv.pool.close()

	if err := srv.callbacks.Close(); err != nil {
		log.Printf("close callback socket: %v", err)
	}

	<-srv.callbacksDone

	if err := os.RemoveAll(srv.root); err != nil {
		log.Printf("remove work dir: %v", err)
	}
}

// Run serves on cfg.Listen until ctx ends, under the account source the
// configuration names.
func Run(ctx context.Context, cfg Config) error {
	// The pool hands out live Google logins, the login routes mint new ones
	// and the admin routes take them in: with pushed accounts the listener is
	// never left open.
	if cfg.TokenFile == "" && cfg.APIKey == "" {
		return errors.New("AGY_BRIDGE_API_KEY is empty: the account push, the account pool and the login routes " +
			"would be open to anyone reaching the listen address; set AGY_BRIDGE_API_KEY, " +
			"or AGY_BRIDGE_TOKEN_FILE for development")
	}

	source, err := newSource(cfg)
	if err != nil {
		return err
	}

	srv, err := New(cfg, source)
	if err != nil {
		return err
	}

	ln, err := net.Listen("tcp", cfg.Listen)
	if err != nil {
		srv.Close()

		return err
	}

	httpServer := &http.Server{
		Handler:           srv,
		ReadHeaderTimeout: 10 * time.Second,
	}

	served := make(chan error, 1)

	go func() {
		served <- httpServer.Serve(ln)
	}()

	log.Printf("agy-bridge listening on %s, models %s", ln.Addr(), strings.Join(cfg.Models, ","))

	select {
	case err := <-served:
		srv.Close()

		return err
	case <-ctx.Done():
	}

	srv.Close()

	drain, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()

	if err := httpServer.Shutdown(drain); err != nil {
		return err
	}

	if err := <-served; !errors.Is(err, http.ErrServerClosed) {
		return err
	}

	return nil
}

func (srv *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodHead {
		w.WriteHeader(http.StatusNotFound)

		return
	}

	if !srv.authorised(r) {
		writeError(w, &apiError{
			status:  http.StatusUnauthorized,
			kind:    "authentication_error",
			message: "invalid x-api-key",
		})

		return
	}

	switch {
	case r.Method == http.MethodPost && r.URL.Path == "/v1/messages":
		srv.handleMessages(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/v1/messages/count_tokens":
		srv.handleCountTokens(w, r)
	case r.Method == http.MethodGet && r.URL.Path == "/v1/models":
		srv.handleModels(w)
	case r.Method == http.MethodPost && r.URL.Path == loginPath:
		srv.handleLoginStart(w, r)
	case strings.HasPrefix(r.URL.Path, loginPathPrefix):
		srv.handleLogin(w, r)
	case r.Method == http.MethodPut && r.URL.Path == accountsPath:
		srv.handlePushAccounts(w, r)
	case r.Method == http.MethodGet && r.URL.Path == outcomesPath:
		srv.handleOutcomes(w, r)
	case r.Method == http.MethodGet && r.URL.Path == usagePath:
		srv.handleUsage(w)
	default:
		writeError(w, &apiError{status: http.StatusNotFound, kind: "not_found_error", message: r.URL.Path})
	}
}

func (srv *Server) authorised(r *http.Request) bool {
	if srv.cfg.APIKey == "" {
		return true
	}

	bearer, _ := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")

	return srv.matchesKey(r.Header.Get("x-api-key")) || srv.matchesKey(bearer)
}

// matchesKey compares digests so neither content nor length leaks by timing.
func (srv *Server) matchesKey(candidate string) bool {
	if candidate == "" {
		return false
	}

	sum := sha256.Sum256([]byte(candidate))

	return subtle.ConstantTimeCompare(sum[:], srv.keyHash[:]) == 1
}

// requestLog is the one line logged per Messages request. It carries counts,
// outcomes and the session's short hash only, never prompt, system or tool
// text; client strings are quoted so they cannot forge a line.
type requestLog struct {
	id        string
	model     string
	agyModel  string
	level     string
	messages  int
	tools     int
	scrubHits int
	session   string
	// account is the id of the account the response came from, attempts the
	// agy sessions the request ran on, affinity whether the account followed
	// the conversation (hit) or was picked anew (miss).
	account   string
	attempts  int
	affinity  string
	continued bool
	// toolsGrew marks a held turn ended because the request brought a tool
	// agy never listed for it.
	toolsGrew bool
	// tokenNearExpiry marks a held turn ended because the token its agy was
	// spawned with was about to run out.
	tokenNearExpiry bool
	orphans         int
	denied          int64
	clean           bool
	// cleanupFailed counts removals of agy's home that failed.
	cleanupFailed int64
	// agyUsage is agy's raw usage; reported is what the client was told, and
	// agyToolCalls how many tool calls agy made in the session so far.
	agyUsage     agyUsage
	reported     agyUsage
	agyToolCalls int64
	outcome      string
	// reason is why the bridge itself refused the request, "-" for any other
	// outcome.
	reason string
	// bodyBytes and systemBytes size the request, toolChoice and maxTokens echo
	// two of its knobs: enough to tell a heavy request from a refused one.
	bodyBytes   int
	systemBytes int
	toolChoice  string
	maxTokens   int
	started     time.Time
}

// useSession notes the session the response comes from; a continuation keeps
// the affinity its turn opened with.
func (l *requestLog) useSession(s *session) {
	l.session = s.tag
	l.account = s.lease.Account.ID
	l.attempts++
	l.affinity = "miss"

	if s.lease.AffinityHit {
		l.affinity = "hit"
	}
}

func (l *requestLog) write() {
	u := l.agyUsage
	reported := l.reported.messages()

	account, affinity := cmp.Or(l.account, "-"), cmp.Or(l.affinity, "-")

	log.Printf(
		"%s model=%q agy_model=%q level=%s messages=%d tools=%d scrub_hits=%d session=%s "+
			"account=%s attempts=%d affinity=%s continued=%t "+
			"tools_grew=%t token_near_expiry=%t orphan_results=%d denied=%d clean=%t cleanup_failed=%d "+
			"agy_tool_calls=%d agy_usage=in:%d,out:%d,cr:%d,cw:%d,th:%d,total:%d reported=in:%d,out:%d outcome=%q "+
			"reason=%s body_bytes=%d system_bytes=%d tool_choice=%q max_tokens=%d seconds=%.2f",
		l.id, l.model, l.agyModel, l.level, l.messages, l.tools, l.scrubHits, l.session,
		account, l.attempts, affinity, l.continued,
		l.toolsGrew, l.tokenNearExpiry, l.orphans, l.denied, l.clean, l.cleanupFailed,
		l.agyToolCalls,
		u.InputTokens, u.OutputTokens, u.CachedReadTokens, u.CachedWriteTokens, u.ThoughtTokens, u.TotalTokens,
		reported.InputTokens, reported.OutputTokens,
		l.outcome,
		cmp.Or(l.reason, "-"), l.bodyBytes, l.systemBytes, cmp.Or(l.toolChoice, "-"), l.maxTokens,
		time.Since(l.started).Seconds(),
	)
}

// countingReader counts the bytes read through it: the size of a request body.
type countingReader struct {
	r io.Reader
	n int
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += n

	return n, err
}

// answerProgressLabel ends a progress label request with the label of the agent's last tool call. No agy ran, so
// the request line names no session and no account. The usage is the label's own, never zero: a gateway reads a
// reply without usage as a lost one and books the whole history as its input estimate.
func answerProgressLabel(out responder, req *messagesRequest, rl *requestLog) string {
	rl.session = "local"
	label := progressLabel(req.Messages)

	err := out.text(label)
	if err == nil {
		err = out.finish(turnEnd{
			stopReason: "end_turn",
			usage:      agyUsage{InputTokens: 1, OutputTokens: max(1, utf8.RuneCountInString(label)/4)},
		})
	}

	if err != nil {
		return "end_turn local, client gone"
	}

	return "end_turn local"
}

func (srv *Server) handleMessages(w http.ResponseWriter, r *http.Request) {
	rl := &requestLog{id: "msg_" + randomHex(12), clean: srv.cfg.Clean, started: time.Now()}
	defer rl.write()

	var req messagesRequest

	body := &countingReader{r: r.Body}
	r.Body = io.NopCloser(body)

	err := decodeBody(w, r, &req)
	rl.bodyBytes = body.n

	if err != nil {
		rl.outcome = reject(w, invalidRequest(err.Error()))

		return
	}

	rl.model = req.Model
	rl.messages = len(req.Messages)
	rl.systemBytes = len(req.System)
	rl.toolChoice = toolChoiceType(&req)
	rl.maxTokens = maxTokens(&req)

	if len(req.Messages) == 0 {
		rl.outcome = reject(w, invalidRequest("messages: at least one message is required"))

		return
	}

	// 1. A progress label request never reaches agy: it opens no session and touches no held turn, even when
	// its tool results carry the id of one. The fork carries the agent's tools, so a request without any is
	// someone's own prompt and goes on to agy.
	if len(req.Tools) > 0 && asksProgressLabel(req.Messages) {
		rl.outcome = answerProgressLabel(newResponder(w, &req, rl.id), &req, rl)

		return
	}

	// 2. Every refusal (bad input, no slot, parked cap) leaves before any
	// response byte, so the client sees a real status it can act on.
	s, err := srv.openTurn(r.Context(), &req, rl)
	if errors.Is(err, errClientGone) {
		rl.outcome = "client gone"

		return
	}

	if err != nil {
		refusal := asAPIError(err)
		rl.reason = refusal.reason
		rl.outcome = reject(w, refusal)

		return
	}

	rl.useSession(s)
	markAccount(w, s)
	deniedBefore := s.denied.Load()

	out := newResponder(w, &req, rl.id)

	// 3. Relay the turn until it ends, hands tool calls back or fails. The
	// Model's timeout on the gateway is a stream's first-byte timer, and a
	// thinking-high first call can stay silent longer than it: past
	// firstByteAfter a stream sends its head anyway, and pings follow. From
	// then on the response is not pristine, so a later failure ends the
	// stream as an error event rather than a status, and the turn no longer
	// moves to another account; a failure before it keeps both. A buffered
	// response has nothing to send early. The timer spans the retry too.
	early := time.NewTimer(firstByteAfter)
	defer early.Stop()

	end, err := s.drive(r.Context(), out, early.C)

	// 4. An account failure before any output may move a fresh turn to another
	// account, once.
	if next := srv.retry(r.Context(), s, err, out, rl.continued); next != nil {
		rl.denied += s.denied.Load() - deniedBefore
		rl.cleanupFailed += s.cleanupFailed.Load()
		s, deniedBefore = next, 0
		rl.useSession(s)
		markAccount(w, s)

		end, err = s.drive(r.Context(), out, early.C)
	}

	rl.denied += s.denied.Load() - deniedBefore
	rl.cleanupFailed += s.cleanupFailed.Load()

	switch {
	case errors.Is(err, errClientGone):
		rl.outcome = "client gone"
	case err != nil:
		failure := asAPIError(err)
		out.fail(failure)
		rl.outcome = outcomeOf(failure)
	default:
		rl.outcome = end.stopReason
		rl.agyUsage = end.usage
		rl.agyToolCalls = s.agyToolCalls.Load()
		exact := !rl.continued && rl.agyToolCalls == 0
		end.usage = reportedUsage(end, exact, body.n-imageBytes(req.Messages), int(s.reportedOut.Load()), int(rl.agyToolCalls))
		rl.reported = end.usage

		if end.stopReason == "tool_use" {
			s.reportedOut.Add(int64(end.usage.OutputTokens))
		}

		if err := out.finish(end); err != nil {
			rl.outcome = end.stopReason + ", client gone"

			// The client never got the tool calls, so nobody can resume.
			if end.stopReason == "tool_use" {
				s.cancel()
			}
		}
	}
}

// openTurn resumes the agy turn the request's tool results belong to, or
// starts a fresh one that replays the conversation as text.
func (srv *Server) openTurn(ctx context.Context, req *messagesRequest, rl *requestLog) (*session, error) {
	tools := clientTools(req.Tools)

	// A request that says no tool may be called runs without the client's tools, so the model can only answer
	// in text.
	if forbidsTools(req.Messages) {
		tools = nil
	}

	rl.tools = len(tools)

	results, extra, err := toolResults(req.Messages)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	if len(results) > 0 {
		// A note beside the results is something the held turn can never
		// read, and a tool agy did not list when the turn opened (one Claude
		// Code loaded through ToolSearch) is one it can never call, and an agy
		// whose access-only token is about to run out cannot refresh it: end
		// that turn and replay with the current tools, note included, in a
		// session that takes a fresh token.
		owners := srv.reg.owners(results)
		rl.toolsGrew = !extra && slices.ContainsFunc(owners, func(s *session) bool { return s.lacksAnyOf(tools) })
		rl.tokenNearExpiry = !extra && !rl.toolsGrew &&
			slices.ContainsFunc(owners, func(s *session) bool { return s.lease.nearExpiry() })

		if extra || rl.toolsGrew || rl.tokenNearExpiry {
			if err := endTurns(ctx, owners); err != nil {
				return nil, err
			}
		} else if s, err := srv.resume(ctx, results); s != nil || err != nil {
			if s != nil {
				rl.continued = true
				rl.agyModel, rl.level = s.agyModel, s.level
			}

			return s, err
		}

		rl.orphans = len(results)
	}

	system, err := systemText(req.System)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	rendered, err := renderPrompt(req.Messages)
	if err != nil {
		return nil, invalidRequest(err.Error())
	}

	model := srv.modelFor(req.Model)
	spec := sessionSpec{
		prompt:    rendered,
		tools:     tools,
		groups:    quotaGroups(model.id),
		affinity:  affinityKey(system, tools, req.Messages),
		retryable: len(results) == 0,
	}
	spec.model, spec.level = model.agyID(requestLevel(req))
	spec.system, rl.scrubHits = srv.scrub.apply(system)
	rl.agyModel, rl.level = spec.model, spec.level

	return srv.startSession(ctx, spec, "", srv.cfg.SlotWait)
}

// affinityKey names a conversation by its stable prefix: the system text,
// the sorted tool names and the first user message. The same prefix runs on
// the same account, which is where Google's prompt cache for it lives. Every
// part is length-prefixed, so no two prefixes hash alike by shifting text.
func affinityKey(system string, tools []mcpTool, messages []message) string {
	h := sha256.New()

	write := func(s string) {
		h.Write(binary.BigEndian.AppendUint64(nil, uint64(len(s))))
		h.Write([]byte(s))
	}

	write(system)

	for _, name := range toolNames(tools) {
		write(name)
	}

	if i := slices.IndexFunc(messages, func(m message) bool { return m.Role == "user" }); i >= 0 {
		// A message the render refused never reaches here.
		text, _ := contentText(messages[i].Content)
		write("first user message")
		write(text)
	}

	return hex.EncodeToString(h.Sum(nil))
}

// endTurns ends held turns the request cannot continue and waits until each
// is gone, so its parked place is free before the replay asks for one. A held
// call answers its stub "session closed", as on any other end.
func endTurns(ctx context.Context, sessions []*session) error {
	for _, s := range sessions {
		s.cancel()
	}

	for _, s := range sessions {
		select {
		case <-s.done:
		case <-ctx.Done():
			return errClientGone
		}
	}

	return nil
}

// resume hands the results to the parked turn they answer once it has a
// slot again. Without a slot the turn stays parked and the client gets 529;
// a nil session without error means the calls are gone and the caller
// replays.
func (srv *Server) resume(ctx context.Context, results []deliveredResult) (*session, error) {
	s, calls, ok := srv.reg.reserve(results)
	if !ok {
		return nil, nil
	}

	if err := s.acquireSlot(ctx, srv.cfg.SlotWait); err != nil {
		srv.reg.restore(results, calls)

		if s.ctx.Err() != nil {
			return nil, nil
		}

		return nil, slotFailure(ctx, reasonResumeSlot)
	}

	// The turn may have expired or been evicted while the request waited for
	// the slot.
	if !s.resumeFromPark() {
		s.releaseSlot()

		return nil, nil
	}

	for i, res := range results {
		calls[i].result <- res.result
	}

	return s, nil
}

// modelFor is the listed model the request names, or the default one.
func (srv *Server) modelFor(requested string) baseModel {
	if model, ok := srv.models[requested]; ok {
		return model
	}

	return lookupModel(srv.cfg.DefaultModel)
}

// evictAtParkedCap makes room for a fresh turn by ending the oldest parked
// turns until the count is under the cap. Clients abandon tool_use turns, so
// refusing the request that arrives now would keep it out for the whole
// PendingTTL; the evicted turn's late result replays on a fresh session. A turn
// agy is working on is never touched.
func (srv *Server) evictAtParkedCap(ctx context.Context) error {
	for srv.reg.parked() >= srv.cfg.MaxParked {
		oldest := srv.reg.oldestParked()
		if oldest == nil {
			return nil
		}

		parked, cancelled := oldest.evictIfParked()
		if !parked {
			continue
		}

		if cancelled {
			log.Printf("session %s: evicted at the parked cap", oldest.tag)
		}

		if err := endTurns(ctx, []*session{oldest}); err != nil {
			return err
		}
	}

	return nil
}

// startSession admits a fresh turn: with an account other than exclude and with
// a server slot, evicting the oldest parked turn when the parked cap is full,
// or not at all. accountWait bounds the wait for an account slot; the server
// slot waits what is left of the configured wait. The eviction comes last: a
// parked turn holds no slot, so ending one for a request that is then refused
// for want of a slot would cost it its life for nothing.
func (srv *Server) startSession(
	ctx context.Context,
	spec sessionSpec,
	exclude string,
	accountWait time.Duration,
) (*session, error) {
	started := time.Now()

	lease, err := srv.pool.acquire(ctx, spec.affinity, spec.groups, exclude, accountWait)
	if err != nil {
		return nil, acquireFailure(ctx, err)
	}

	s := newSession(srv.ctx, srv.reg, srv.slots, srv.cfg.PendingTTL)
	s.agyModel, s.level, s.tools = spec.model, spec.level, toolNames(spec.tools)
	s.lease, s.spec = lease, spec

	if err := s.acquireSlot(ctx, srv.cfg.SlotWait-time.Since(started)); err != nil {
		s.cancel()

		return nil, slotFailure(ctx, reasonSlotWait)
	}

	if err := srv.evictAtParkedCap(ctx); err != nil {
		s.releaseSlot()
		s.cancel()

		return nil, err
	}

	srv.lifeMu.Lock()
	defer srv.lifeMu.Unlock()

	if srv.closed {
		s.releaseSlot()
		s.cancel()

		return nil, refused(reasonShutdown, "the bridge is shutting down")
	}

	srv.reg.add(s)
	srv.sessions.Go(func() {
		srv.runSession(s, spec)
	})

	return s, nil
}

// retry starts the request's turn again on another account, or returns nil.
// Only a turn this request opened fresh moves (never a continuation, never a
// replay), only on an account failure (quota, rate, auth, verification, a
// token that ran out), only while the client has seen nothing of it, and only
// to an account free right now. The failed agy is gone before the new one
// starts.
func (srv *Server) retry(ctx context.Context, s *session, err error, out responder, continued bool) *session {
	if err == nil || errors.Is(err, errClientGone) || continued || !s.spec.retryable || !out.pristine() {
		return nil
	}

	f := s.recordedFailure()
	if f == nil || !slices.Contains([]failureKind{kindQuota, kindRate, kindAuth, kindVerification, kindExpired}, f.Kind) {
		return nil
	}

	select {
	case <-s.done:
	case <-ctx.Done():
		return nil
	}

	next, err := srv.startSession(ctx, s.spec, s.lease.Account.ID, 0)
	if err != nil {
		return nil
	}

	return next
}

// Why the bridge refuses a request with 529, as the request log names it.
const (
	reasonSlotWait   = "slot_wait"
	reasonResumeSlot = "resume_slot"
	reasonShutdown   = "shutdown"
)

// refused is a 529 the bridge sends itself, tagged with its cause for the log.
func refused(reason, message string) *apiError {
	failure := overloaded(message)
	failure.reason = reason

	return failure
}

func slotFailure(ctx context.Context, reason string) error {
	if ctx.Err() != nil {
		return errClientGone
	}

	return refused(reason, "no free agy slot")
}

// acquireFailure maps a failed account lease to what the client sees.
func acquireFailure(ctx context.Context, err error) error {
	var none *errNoAccount
	if errors.As(err, &none) {
		return none.apiError()
	}

	if errors.Is(err, errNoSlot) || ctx.Err() != nil {
		return slotFailure(ctx, reasonSlotWait)
	}

	return refused(reasonShutdown, "the bridge is shutting down")
}

func (srv *Server) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	if err != nil {
		writeError(w, invalidRequest(err.Error()))

		return
	}

	writeJSON(w, http.StatusOK, map[string]int{"input_tokens": max(1, len(body)/4)})
}

func (srv *Server) handleModels(w http.ResponseWriter) {
	type model struct {
		Type        string `json:"type"`
		ID          string `json:"id"`
		DisplayName string `json:"display_name"`
	}

	ids := slices.Sorted(maps.Keys(srv.models))

	data := make([]model, 0, len(ids))
	for _, id := range ids {
		data = append(data, model{Type: "model", ID: id, DisplayName: srv.models[id].displayName})
	}

	list := map[string]any{"data": data, "has_more": false, "first_id": nil, "last_id": nil}
	if len(ids) > 0 {
		list["first_id"] = ids[0]
		list["last_id"] = ids[len(ids)-1]
	}

	writeJSON(w, http.StatusOK, list)
}
