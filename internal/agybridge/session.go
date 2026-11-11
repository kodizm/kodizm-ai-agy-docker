package agybridge

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Timings are variables so tests can shorten them.
var (
	// batchWindow gathers parallel tool calls into one assistant message.
	batchWindow = 400 * time.Millisecond

	// toolResultTimeout bounds one held tool call.
	toolResultTimeout = 900 * time.Second

	// pingInterval keeps a started stream alive while the turn is quiet.
	pingInterval = 10 * time.Second

	// setupTimeout bounds each ACP call before the prompt.
	setupTimeout = 60 * time.Second

	// firstByteAfter is how long a stream may stay silent before its head
	// goes out anyway.
	firstByteAfter = 12 * time.Second

	// minWatchTick floors how often the silence watchdog looks.
	minWatchTick = 10 * time.Millisecond
)

var (
	errClientGone = errors.New("client went away")
	errNoSlot     = errors.New("no free agy slot")
)

type eventKind int

const (
	eventText eventKind = iota
	eventTool
	eventDone
	eventFailed
)

// event is what a session reports to the request driving it.
type event struct {
	kind eventKind
	text string
	call toolCall
	end  turnEnd
	err  error
}

type toolCall struct {
	id    string
	name  string
	input json.RawMessage
}

// usage is the Anthropic Messages usage block.
type usage struct {
	InputTokens              int `json:"input_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	OutputTokens             int `json:"output_tokens"`
}

// agyUsage is the usage agy reports with a finished session/prompt.
type agyUsage struct {
	InputTokens       int `json:"inputTokens"`
	OutputTokens      int `json:"outputTokens"`
	CachedReadTokens  int `json:"cachedReadTokens"`
	CachedWriteTokens int `json:"cachedWriteTokens"`
	ThoughtTokens     int `json:"thoughtTokens"`
	TotalTokens       int `json:"totalTokens"`
}

// messages maps agy's usage to Anthropic's. agy's output tokens already
// include its thinking (a call read in:450 out:18 th:17 total:468), and
// Anthropic counts thinking as output too, so ThoughtTokens is not added.
func (u agyUsage) messages() usage {
	return usage{
		InputTokens:              u.InputTokens,
		CacheCreationInputTokens: u.CachedWriteTokens,
		CacheReadInputTokens:     u.CachedReadTokens,
		OutputTokens:             u.OutputTokens,
	}
}

// agyOverheadTokens is what agy reads on every model call besides the request
// itself: its own base prompt and agent definition in clean mode. A plain reply
// to a 100 byte request reads 458 input tokens from agy, a quarter of the body
// is 25. Without it the first tool_use end of a small conversation reports
// about 80 and the end reports about 760, so the input would jump by 680 inside
// one turn. About 410 to 450 was measured, 400 errs low: Claude Code only uses
// input_tokens as a size, so low is the safe side.
const agyOverheadTokens = 400

// reportedUsage is the usage the client is told for one response.
//
// agy reports usage once, when a session/prompt returns, and it adds up every
// model call of that prompt: a prompt with one tool round trip reads about
// twice the context (15.9k, 32.0k and 64.4k tokens were seen for zero, one and
// three tool round trips of the same conversation), and a tool_use end has no
// usage yet. Claude Code takes input_tokens as the size of the context, so
// those numbers would jump and double. Only a fresh session that ends without
// any tool call (its own web search counts) made exactly one model call and
// keeps agy's numbers (exact). Every other response reports an estimate of
// the context: a quarter of the request body, in bytes (the gateway's
// count_tokens way; imageBytes is taken off by the caller) plus agy's own
// prompt (agyOverheadTokens), which no context is smaller than, and at an end
// also agy's total over its model calls (one
// more than its tool calls), which carries what agy read on its own, such as
// search results, and when every call read about the same context equals it
// (64.4k over four calls of a 16.1k context). Parallel tool calls make fewer
// model calls than tool calls, which only shrinks that second figure. The
// output is agy's (thinking included) less what the tool_use ends of the turn
// already reported (priorOut), and an estimate of the calls at a tool_use.
func reportedUsage(end turnEnd, exact bool, bodyBytes, priorOut, agyToolCalls int) agyUsage {
	if exact && end.stopReason != "tool_use" {
		return end.usage
	}

	in := bodyBytes/4 + agyOverheadTokens
	out := end.usage.OutputTokens

	if end.stopReason == "tool_use" {
		for _, call := range end.calls {
			out += max(1, (len(call.name)+len(call.input))/4)
		}
	} else {
		total := end.usage.InputTokens + end.usage.CachedReadTokens + end.usage.CachedWriteTokens
		in = max(in, total/(agyToolCalls+1))
		out = max(0, out-priorOut)
	}

	return agyUsage{InputTokens: max(1, in), OutputTokens: out}
}

// turnEnd is how one HTTP response ends: the agy turn finished, or it waits
// on the listed tool calls; usage is agy's raw numbers, which are zero at a
// tool_use end.
type turnEnd struct {
	stopReason string
	usage      agyUsage
	calls      []toolCall
}

// sessionSpec is everything a fresh agy turn starts from; model is agy's id,
// the base model with the level appended.
type sessionSpec struct {
	model  string
	level  string
	system string
	prompt prompt
	tools  []mcpTool
	// groups are the quota groups of the model; affinity keys the
	// conversation to the account it ran on before.
	groups   []string
	affinity string
	// retryable marks a request that carried no tool result: only such a
	// fresh turn may move to another account after an early failure.
	retryable bool
}

// eventQueue is unbounded so agy's reader never blocks on a client that is
// between requests.
type eventQueue struct {
	mu    sync.Mutex
	items []event
	ready chan struct{}
}

func (q *eventQueue) push(e event) {
	q.mu.Lock()
	q.items = append(q.items, e)
	q.mu.Unlock()

	select {
	case q.ready <- struct{}{}:
	default:
	}
}

// pop takes the oldest event; with onlyTools it takes it only when it is a
// tool call.
func (q *eventQueue) pop(onlyTools bool) (event, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.items) == 0 || (onlyTools && q.items[0].kind != eventTool) {
		return event{}, false
	}

	e := q.items[0]
	q.items = q.items[1:]

	return e, true
}

// session is one agy process running one conversation turn. The turn may
// span several HTTP requests: a tool call ends the response and parks the
// turn, and the request carrying the result picks the same turn up.
//
// A session holds one of the server's slots and one slot of its account while
// agy works. A parked turn gives both back, since agy idles until the client
// answers, and takes both again to resume, on the same account.
type session struct {
	id     string
	tag    string
	ctx    context.Context
	cancel context.CancelFunc
	reg    *registry
	slots  chan struct{}
	ttl    time.Duration
	events eventQueue
	done   chan struct{}

	// agyModel and level are fixed when agy opens the session; tools are the
	// sorted names of the client tools agy listed then, which it never lists
	// again. These, the lease and the spec are set before the session is
	// registered.
	agyModel string
	level    string
	tools    []string
	lease    *Lease
	spec     sessionSpec

	// denied counts agy permission requests answered cancelled.
	denied atomic.Int64

	// agyToolCalls counts the tool_call updates agy sent: a turn that made
	// any made more than one model call, so its usage total is not one call's.
	agyToolCalls atomic.Int64

	// reportedOut is the output the turn's tool_use ends already reported, so
	// the final response does not count it twice.
	reportedOut atomic.Int64

	// cleanupFailed counts removals of agy's home that failed.
	cleanupFailed atomic.Int64

	// awake is when the turn last began working, as unix nanoseconds: the prompt
	// going out, or a parked turn getting its tool results. agy owes no sign of
	// life for the time before it, since it was waiting for the client.
	awake atomic.Int64

	// stderr holds the tail of agy's stderr, logged once agy is gone.
	stderr tailBuffer

	mu      sync.Mutex
	hasSlot bool
	parked  bool
	// parkedAt is when the turn last parked; it orders the eviction at the cap.
	parkedAt time.Time
	expiry   *time.Timer
	// failure is the first upstream failure of the session, already settled
	// with the pool.
	failure *failure
}

// newSession prepares a session; ttl is how long it may stay parked.
func newSession(parent context.Context, reg *registry, slots chan struct{}, ttl time.Duration) *session {
	ctx, cancel := context.WithCancel(parent)
	id := randomHex(16)

	return &session{
		id:     id,
		tag:    sessionTag(id),
		ctx:    ctx,
		cancel: cancel,
		reg:    reg,
		slots:  slots,
		ttl:    ttl,
		events: eventQueue{ready: make(chan struct{}, 1)},
		done:   make(chan struct{}),
	}
}

// sessionTag is the only form of a session id that reaches the log: the id
// is the capability the MCP stub calls back with.
func sessionTag(id string) string {
	sum := sha256.Sum256([]byte(id))

	return hex.EncodeToString(sum[:4])
}

// acquireSlot waits up to wait for the session's account slot, unless the
// lease still holds it, and then for a server slot, failing with errNoSlot.
// On any failure it holds neither. Every path takes the account slot first,
// so two waiters never hold one each of what the other needs.
func (s *session) acquireSlot(ctx context.Context, wait time.Duration) error {
	timer := time.NewTimer(wait)
	defer timer.Stop()

	if err := s.lease.retake(ctx, s.ctx, timer.C); err != nil {
		return err
	}

	// A free slot is taken even when the wait is already used up.
	select {
	case s.slots <- struct{}{}:
	default:
		select {
		case s.slots <- struct{}{}:
		case <-timer.C:
			s.lease.release()

			return errNoSlot
		case <-ctx.Done():
			s.lease.release()

			return ctx.Err()
		case <-s.ctx.Done():
			s.lease.release()

			return s.ctx.Err()
		}
	}

	s.mu.Lock()
	s.hasSlot = true
	s.mu.Unlock()

	return nil
}

func (s *session) releaseSlot() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.releaseSlotLocked()
}

func (s *session) releaseSlotLocked() {
	if s.hasSlot {
		s.hasSlot = false
		<-s.slots
	}

	s.lease.release()
}

// park leaves the turn waiting for tool results: both slots go back and the
// expiry starts.
func (s *session) park() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.releaseSlotLocked()
	s.stopExpiryLocked()
	s.parked = true
	s.parkedAt = time.Now()
	s.expiry = time.AfterFunc(s.ttl, s.cancel)
}

// unpark ends the parked state; the caller already holds a slot for the turn.
func (s *session) unpark() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.parked = false
	s.stopExpiryLocked()
}

// resumeFromPark unparks a turn for its tool results, or reports false when it
// was cancelled first (expired, or evicted at the parked cap). Checking and
// unparking under one lock is what settles a race with evictIfParked: exactly
// one of them wins.
func (s *session) resumeFromPark() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.ctx.Err() != nil {
		return false
	}

	s.parked = false
	s.awake.Store(time.Now().UnixNano())
	s.stopExpiryLocked()

	return true
}

// quiet is how long agy has shown no sign of life in the turn's working time.
func (s *session) quiet(a *agent) time.Duration {
	return time.Since(time.Unix(0, max(a.activity.Load(), s.awake.Load())))
}

// reasonAgySilent is the request log's reason for a turn the silence watchdog ended.
const reasonAgySilent = "agy_silent"

// watch ends a turn agy has gone silent on. agy retries a dead upstream for minutes without a word and, until
// then, the request holds its slots; when agy shows no sign of life for silence while the turn is working (a
// parked turn waits for the client, not for agy), agy is killed and the request fails with a 529: a stream whose
// head has gone out already gets it as an error event, and the gateway cannot fail over from there. The account
// is neither penalised nor credited, since a hang follows the model and not the login. The returned func stops
// the watch and waits for it.
func (s *session) watch(a *agent, silence time.Duration) (stop func()) {
	done := make(chan struct{})
	finished := make(chan struct{})

	go func() {
		defer close(finished)

		tick := time.NewTicker(max(silence/4, minWatchTick))
		defer tick.Stop()

		for {
			select {
			case <-tick.C:
			case <-done:
				return
			case <-s.ctx.Done():
				return
			}

			if s.isParked() || s.quiet(a) < silence {
				continue
			}

			log.Printf("session %s: agy showed no sign of life for %s; killed", s.tag, silence.Round(time.Second))
			s.events.push(event{
				kind: eventFailed,
				err:  refused(reasonAgySilent, "agy produced nothing for "+silence.Round(time.Second).String()),
			})
			s.cancel()

			return
		}
	}()

	return func() {
		close(done)
		<-finished
	}
}

// evictIfParked cancels the turn only while it is parked: a turn agy is working
// on keeps its slot, and one a resume has just unparked is not touched. parked
// tells whether the turn was parked, so the caller may wait for it to end;
// cancelled is true only for the call that ended it, since an expired turn is
// still flagged parked until it has torn down.
func (s *session) evictIfParked() (parked, cancelled bool) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.parked {
		return false, false
	}

	cancelled = s.ctx.Err() == nil

	s.cancel()

	return true, cancelled
}

// lacksAnyOf reports whether a tool is not among those agy listed for the session.
func (s *session) lacksAnyOf(tools []mcpTool) bool {
	return slices.ContainsFunc(tools, func(tool mcpTool) bool {
		_, found := slices.BinarySearch(s.tools, tool.Name)

		return !found
	})
}

func (s *session) isParked() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.parked
}

func (s *session) stopExpiryLocked() {
	if s.expiry != nil {
		s.expiry.Stop()
		s.expiry = nil
	}
}

// onChunk checks every chunk for agy's failure banner: agy would retry for
// minutes, so the process dies here and the failure goes out now, whether
// it comes first or after a tool result.
func (s *session) onChunk(text string) {
	if f := classifyBanner(text); f != nil {
		message := bannerMessage(text)
		failed := s.fail(*f, message)
		s.events.push(event{kind: eventFailed, err: failed.apiError(message)})
		s.cancel()

		return
	}

	s.events.push(event{kind: eventText, text: text})
}

// fail records the session's first upstream failure and settles it with the
// pool at once, so the next request already avoids the account. The raw text
// is logged once, cut and without an email, to tighten the patterns later.
// It returns f as it bears on the account: an auth failure on a token that
// ran out comes back expired.
func (s *session) fail(f failure, detail string) failure {
	f = s.lease.classify(f)

	s.mu.Lock()
	first := s.failure == nil

	if first {
		s.failure = &f
	}

	s.mu.Unlock()

	if !first {
		return f
	}

	log.Printf("warn: session %s account=%s agy failure kind=%s: %q", s.tag, s.lease.Account.ID, f.Kind, detail)
	s.lease.settle(s.spec.groups, &f, detail)

	return f
}

func (s *session) recordedFailure() *failure {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.failure
}

func (s *session) onDenied(method string) {
	if method == "session/request_permission" {
		s.denied.Add(1)
	}
}

func (s *session) onToolCall() {
	s.agyToolCalls.Add(1)
}

// drive relays the session's events to out until the turn ends, hands tool
// calls back, fails, or the client goes away. When early fires while the
// turn is quiet, out commits to its response head.
func (s *session) drive(ctx context.Context, out responder, early <-chan time.Time) (turnEnd, error) {
	ping := time.NewTicker(pingInterval)
	defer ping.Stop()

	for {
		e, ok := s.events.pop(false)
		if !ok {
			select {
			case <-s.events.ready:
			case <-early:
				if err := out.commit(); err != nil {
					return turnEnd{}, s.abandon(err)
				}
			case <-ping.C:
				if err := out.ping(); err != nil {
					return turnEnd{}, s.abandon(err)
				}
			case <-ctx.Done():
				return turnEnd{}, s.abandon(ctx.Err())
			}

			continue
		}

		switch e.kind {
		case eventText:
			if err := out.text(e.text); err != nil {
				return turnEnd{}, s.abandon(err)
			}
		case eventTool:
			calls, err := s.batch(ctx, e.call)
			if err != nil {
				return turnEnd{}, s.abandon(err)
			}

			// Park before the response goes out, so the client's next
			// request always finds the turn parked and its calls shown.
			s.reg.show(calls)
			s.park()

			return turnEnd{stopReason: "tool_use", calls: calls}, nil
		case eventDone:
			return e.end, nil
		case eventFailed:
			return turnEnd{}, e.err
		}
	}
}

// batch waits briefly for parallel calls so they leave as one message.
func (s *session) batch(ctx context.Context, first toolCall) ([]toolCall, error) {
	timer := time.NewTimer(batchWindow)
	defer timer.Stop()

	select {
	case <-timer.C:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	calls := []toolCall{first}

	for {
		e, ok := s.events.pop(true)
		if !ok {
			return calls, nil
		}

		calls = append(calls, e.call)
	}
}

// abandon handles a client that went away mid-response. Only a call the
// client has already seen can bring it back, so without one agy dies; with
// one, the turn parks under the expiry.
func (s *session) abandon(cause error) error {
	if s.reg.shownBy(s) == 0 {
		s.cancel()
	} else {
		s.park()
	}

	return fmt.Errorf("%w: %w", errClientGone, cause)
}

// registry indexes live sessions and the tool calls they hold.
type registry struct {
	mu       sync.Mutex
	sessions map[string]*session
	calls    map[string]*heldCall
}

// heldCall is a tool call agy is blocked on until the client's result lands.
// shown marks a call already handed to the client in a response.
type heldCall struct {
	session *session
	result  chan toolResult
	shown   bool
}

func newRegistry() *registry {
	return &registry{
		sessions: map[string]*session{},
		calls:    map[string]*heldCall{},
	}
}

func (r *registry) add(s *session) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.sessions[s.id] = s
}

// remove forgets the session and every call it holds.
func (r *registry) remove(s *session) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.sessions, s.id)

	for id, call := range r.calls {
		if call.session == s {
			delete(r.calls, id)
		}
	}
}

func (r *registry) session(id string) *session {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.sessions[id]
}

// hold registers a new call for a live session.
func (r *registry) hold(s *session) (string, *heldCall, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.sessions[s.id] != s {
		return "", nil, false
	}

	id := "toolu_" + randomHex(12)
	call := &heldCall{session: s, result: make(chan toolResult, 1)}
	r.calls[id] = call

	return id, call, true
}

func (r *registry) drop(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	delete(r.calls, id)
}

func (r *registry) show(calls []toolCall) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, c := range calls {
		if call, ok := r.calls[c.id]; ok {
			call.shown = true
		}
	}
}

// reserve takes the held calls the results answer, when all of them belong
// to one live session, out of the index so no other request can resume the
// same calls. A session already cancelled, by its expiry for one, counts as
// gone: the caller replays instead.
func (r *registry) reserve(results []deliveredResult) (*session, []*heldCall, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	var owner *session

	calls := make([]*heldCall, 0, len(results))
	seen := map[string]bool{}

	for _, res := range results {
		call, ok := r.calls[res.id]
		if !ok || seen[res.id] || (owner != nil && call.session != owner) || call.session.ctx.Err() != nil {
			return nil, nil, false
		}

		owner = call.session
		seen[res.id] = true
		calls = append(calls, call)
	}

	if owner == nil {
		return nil, nil, false
	}

	for _, res := range results {
		delete(r.calls, res.id)
	}

	return owner, calls, true
}

// restore puts reserved calls back while their session is still registered.
func (r *registry) restore(results []deliveredResult, calls []*heldCall) {
	r.mu.Lock()
	defer r.mu.Unlock()

	for i, res := range results {
		if r.sessions[calls[i].session.id] == calls[i].session {
			r.calls[res.id] = calls[i]
		}
	}
}

// owners returns the live sessions holding any of the given calls.
func (r *registry) owners(results []deliveredResult) []*session {
	r.mu.Lock()
	defer r.mu.Unlock()

	var owners []*session

	for _, res := range results {
		if call, ok := r.calls[res.id]; ok && !containsSession(owners, call.session) {
			owners = append(owners, call.session)
		}
	}

	return owners
}

func containsSession(sessions []*session, s *session) bool {
	for _, candidate := range sessions {
		if candidate == s {
			return true
		}
	}

	return false
}

func (r *registry) shownBy(s *session) int {
	r.mu.Lock()
	defer r.mu.Unlock()

	n := 0

	for _, call := range r.calls {
		if call.session == s && call.shown {
			n++
		}
	}

	return n
}

func (r *registry) parked() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	n := 0

	for _, s := range r.sessions {
		if s.isParked() {
			n++
		}
	}

	return n
}

// oldestParked is the session that has waited longest on tool results, or nil
// when none is parked.
func (r *registry) oldestParked() *session {
	r.mu.Lock()
	defer r.mu.Unlock()

	var (
		oldest   *session
		oldestAt time.Time
	)

	for _, s := range r.sessions {
		s.mu.Lock()
		parked, at := s.parked, s.parkedAt
		s.mu.Unlock()

		if parked && (oldest == nil || at.Before(oldestAt)) {
			oldest, oldestAt = s, at
		}
	}

	return oldest
}

func (r *registry) size() (sessions, calls int) {
	r.mu.Lock()
	defer r.mu.Unlock()

	return len(r.sessions), len(r.calls)
}

// ACP payloads the bridge sends.
type (
	newSessionParams struct {
		Cwd        string       `json:"cwd"`
		MCPServers []mcpServer  `json:"mcpServers"`
		Meta       *sessionMeta `json:"_meta,omitempty"`
	}

	mcpServer struct {
		Name    string   `json:"name"`
		Command string   `json:"command"`
		Args    []string `json:"args"`
		Env     []envVar `json:"env"`
	}

	envVar struct {
		Name  string `json:"name"`
		Value string `json:"value"`
	}

	sessionMeta struct {
		Antigravity struct {
			SystemPrompt struct {
				Append string `json:"append"`
			} `json:"systemPrompt"`
		} `json:"antigravity"`
	}

	promptResult struct {
		StopReason string   `json:"stopReason"`
		Usage      agyUsage `json:"usage"`
	}
)

// runSession owns the session for its whole life and always leaves agy
// killed, its slots released, its directories removed and its held calls
// forgotten. The session arrives holding both slots.
func (srv *Server) runSession(s *session, spec sessionSpec) {
	defer close(s.done)
	defer srv.reg.remove(s)
	defer s.releaseSlot()
	defer s.unpark()
	defer s.cancel()

	end, err := srv.converse(s, spec)

	s.logStderrTail(err)

	if err != nil {
		s.events.push(event{kind: eventFailed, err: s.failureFor(err)})

		return
	}

	if s.recordedFailure() == nil {
		s.lease.settle(spec.groups, nil, "")
	}

	s.events.push(event{kind: eventDone, end: end})
}

// logStderrTail logs what agy wrote to stderr when its turn failed. agy is
// reaped by now, so the tail is complete; a turn cancelled on purpose (evicted,
// client gone, shutdown) says nothing worth logging.
func (s *session) logStderrTail(err error) {
	failed := s.recordedFailure() != nil || (err != nil && s.ctx.Err() == nil)

	if tail := s.stderr.masked(); failed && tail != "" {
		log.Printf("session %s: agy stderr_tail=%q", s.tag, tail)
	}
}

func (srv *Server) converse(s *session, spec sessionSpec) (turnEnd, error) {
	// 1. Hand the client's tools to agy through the MCP stub.
	params, err := srv.sessionParams(s, spec)
	defer srv.removeSessionDir(s)

	if err != nil {
		return turnEnd{}, err
	}

	// 2. agy keeps every conversation, log and login under HOME, so it gets a
	// throwaway one holding only the leased account's token. Deferred before
	// a.close, the removal runs once agy is dead and cannot write again.
	home, err := srv.homes.prepare(s.lease.Account)
	if err != nil {
		return turnEnd{}, err
	}

	defer srv.removeHome(s, home)

	// 3. Start agy and open the ACP session.
	var extraArgs []string
	if srv.cfg.Clean {
		extraArgs = cleanArgs()
	}

	a, err := startAgent(s.ctx, srv.cfg.AgyBin, extraArgs, home, &s.stderr, s.onChunk, s.onDenied, s.onToolCall)
	if err != nil {
		return turnEnd{}, err
	}

	defer a.close()

	var opened struct {
		SessionID string `json:"sessionId"`
	}

	setup, cancel := context.WithTimeout(s.ctx, setupTimeout)
	defer cancel()

	if err := a.initialize(setup); err != nil {
		return turnEnd{}, err
	}

	if err := a.call(setup, "session/new", params, &opened); err != nil {
		return turnEnd{}, err
	}

	model := map[string]string{"sessionId": opened.SessionID, "modelId": spec.model}
	if err := a.call(setup, "session/set_model", model, nil); err != nil {
		return turnEnd{}, err
	}

	// 4. Run the turn; text and tool calls reach the client while it runs. The
	// prompt carries only the attachments this agy advertised.
	turn := map[string]any{
		"sessionId": opened.SessionID,
		"prompt":    spec.prompt.forAgent(a.prompts),
	}

	s.awake.Store(time.Now().UnixNano())

	if srv.cfg.SilenceTimeout > 0 {
		defer s.watch(a, srv.cfg.SilenceTimeout)()
	}

	var result promptResult
	if err := a.call(s.ctx, "session/prompt", turn, &result); err != nil {
		return turnEnd{}, err
	}

	// stopReason folds every reason but max_tokens into end_turn, so a refusal
	// or a cancel agy reported would otherwise leave no trace.
	mapped := stopReason(result.StopReason)
	if mapped != result.StopReason {
		log.Printf("session %s: agy stopReason=%q reported as %s", s.tag, result.StopReason, mapped)
	}

	return turnEnd{stopReason: mapped, usage: result.Usage}, nil
}

func (srv *Server) sessionParams(s *session, spec sessionSpec) (newSessionParams, error) {
	params := newSessionParams{Cwd: srv.workspace, MCPServers: []mcpServer{}}
	system := spec.system

	// The agent file must exist before agy starts: agy finds it through the
	// cwd of session/new. A clean agy has no built-in tools to warn about.
	if srv.cfg.Clean {
		params.Cwd = filepath.Join(srv.sessionDir(s), "workspace")

		if err := writeCleanAgent(params.Cwd, s.tools, srv.cfg.Web); err != nil {
			return params, err
		}
	}

	if len(spec.tools) > 0 {
		server, err := srv.toolServer(s, spec.tools)
		if err != nil {
			return params, err
		}

		params.MCPServers = append(params.MCPServers, server)

		if !srv.cfg.Clean {
			system = strings.TrimSpace(system + "\n\n" + toolHint)
		}
	}

	if system != "" {
		params.Meta = &sessionMeta{}
		params.Meta.Antigravity.SystemPrompt.Append = system
	}

	return params, nil
}

// toolServer writes the tool list and points agy at this binary's MCP stub.
func (srv *Server) toolServer(s *session, tools []mcpTool) (mcpServer, error) {
	dir := srv.sessionDir(s)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return mcpServer{}, fmt.Errorf("session dir: %w", err)
	}

	raw, err := json.Marshal(tools)
	if err != nil {
		return mcpServer{}, err
	}

	file := filepath.Join(dir, "tools.json")
	if err := os.WriteFile(file, raw, 0o600); err != nil {
		return mcpServer{}, fmt.Errorf("tools file: %w", err)
	}

	return mcpServer{
		Name:    "client",
		Command: srv.selfExe,
		Args:    []string{"mcp-stub"},
		Env: []envVar{
			{Name: "TOOLS_FILE", Value: file},
			{Name: "BRIDGE_SOCKET", Value: srv.socket},
			{Name: "BRIDGE_SESSION", Value: s.id},
		},
	}, nil
}

func (srv *Server) sessionDir(s *session) string {
	return filepath.Join(srv.root, "sessions", s.id)
}

func (srv *Server) removeSessionDir(s *session) {
	if err := os.RemoveAll(srv.sessionDir(s)); err != nil {
		log.Printf("session %s: remove dir failed", s.tag)
	}
}

// removeHome deletes the session's agy home and counts a failure. The request
// line may already be written when the client left, so a failure is logged
// here too, with the cause but never the path.
func (srv *Server) removeHome(s *session, home string) {
	err := srv.homes.remove(home)
	if err == nil {
		return
	}

	s.cleanupFailed.Add(1)

	var pathErr *fs.PathError
	if errors.As(err, &pathErr) {
		err = pathErr.Err
	}

	log.Printf("session %s: removing agy's home failed: %v", s.tag, err)
}

func stopReason(agy string) string {
	if agy == "max_tokens" {
		return "max_tokens"
	}

	return "end_turn"
}

// failureFor maps a session failure to what the client sees. A JSON-RPC error
// from agy (a refused login at initialize, say) is classified like a banner
// and settled with the pool; its text may carry local paths, so the client
// gets a message naming only the kind. Any other failure's detail goes to the
// log and the client gets a generic message.
func (s *session) failureFor(err error) error {
	var rpc *rpcError

	switch {
	case errors.Is(err, errAgyStart):
		log.Printf("session %s: %v", s.tag, err)

		return &apiError{status: http.StatusBadGateway, kind: "api_error", message: "agy failed to start"}
	case s.ctx.Err() != nil:
		// Killed on purpose (client gone, expiry, shutdown, a banner): nothing
		// to report.
		return &apiError{status: http.StatusBadGateway, kind: "api_error", message: "agy session ended"}
	case errors.As(err, &rpc):
		f := s.fail(classifyText(rpc.Message), bannerMessage(rpc.Message))

		return f.apiError("agy session failed (" + string(f.Kind) + ")")
	}

	log.Printf("session %s: %v", s.tag, err)

	return &apiError{status: http.StatusBadGateway, kind: "api_error", message: "agy session failed"}
}

func randomHex(n int) string {
	b := make([]byte, n)
	// crypto/rand.Read never returns an error.
	rand.Read(b)

	return hex.EncodeToString(b)
}
