package agybridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"
)

const (
	loginPath       = "/v1/admin/login"
	loginPathPrefix = loginPath + "/"

	// loginHomesDir is the account directory login homes live under, so the
	// guarded removal and the startup sweep cover them.
	loginHomesDir = "login"

	maxPendingLogins  = 2
	maxLoginCodeRunes = 4096

	// minRedactedCode is the shortest pasted code taken out of a failure
	// message; a shorter one would blank ordinary words.
	minRedactedCode = 8

	// loginTailBytes is how much of agy's output a login keeps for its
	// failure message; loginLineBytes bounds the line the URL is sought in.
	loginTailBytes = 2 << 10
	loginLineBytes = 16 << 10
)

var (
	// loginTTL is how long a login waits for its code; loginStartTimeout how
	// long agy has to print the URL; loginCodeTimeout how long it has to
	// exchange the code once it has it.
	loginTTL          = 10 * time.Minute
	loginStartTimeout = 15 * time.Second
	loginCodeTimeout  = 60 * time.Second

	loginURLPattern = regexp.MustCompile(`https://accounts\.google\.com/\S+`)
	anyURLPattern   = regexp.MustCompile(`[A-Za-z][A-Za-z0-9+.-]*://\S+`)

	// codeParamPattern finds the code parameter of a pasted landing address.
	codeParamPattern = regexp.MustCompile(`(?:^|[?&#])code=([^&#\s]+)`)
)

// login is one `agy acp --login` waiting for, or exchanging, its code. agy
// holds the PKCE verifier, so the process lives from the URL to the code.
type login struct {
	id      string
	home    string
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	out     *loginOutput
	started time.Time

	// cancel kills agy's process group; exited closes once Wait returned,
	// which is when waitErr is set.
	cancel  context.CancelFunc
	exited  chan struct{}
	waitErr error
}

// loginManager runs the owner's agy logins. A login is pending from its URL
// until a code, a delete, its expiry, an eviction or Close takes it; whoever
// takes it out of pending ends it, so every login ends exactly once.
type loginManager struct {
	bin    string
	homes  *spawnHomes
	ctx    context.Context
	cancel context.CancelFunc

	mu      sync.Mutex
	closed  bool
	pending []*login // oldest first
	wg      sync.WaitGroup
}

func newLoginManager(parent context.Context, bin string, homes *spawnHomes) *loginManager {
	ctx, cancel := context.WithCancel(parent)

	return &loginManager{bin: bin, homes: homes, ctx: ctx, cancel: cancel}
}

// enter admits a call while the manager is open; the caller runs done after.
func (m *loginManager) enter() bool {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.closed {
		return false
	}

	m.wg.Add(1)

	return true
}

// close ends every login and waits until nothing of the manager runs.
func (m *loginManager) close() {
	m.mu.Lock()
	m.closed = true
	pending := m.pending
	m.pending = nil
	m.mu.Unlock()

	m.cancel()

	for _, l := range pending {
		m.end(l, "cancelled")
	}

	m.wg.Wait()
}

// start runs agy's login and returns it once agy printed the Google URL. A
// third login evicts the oldest pending one: an abandoned wizard never locks
// the owner out.
func (m *loginManager) start(ctx context.Context) (*login, string, error) {
	if !m.enter() {
		return nil, "", overloaded("the bridge is shutting down")
	}
	defer m.wg.Done()

	// 1. agy signs in by itself in an empty throwaway home.
	l, err := m.spawn()
	if err != nil {
		log.Printf("login start: %v", err)

		return nil, "", loginBadGateway("agy could not start a login")
	}

	// 2. Wait for the URL, never past the start timeout.
	timer := time.NewTimer(loginStartTimeout)
	defer timer.Stop()

	select {
	case <-l.out.found:
	case <-l.exited:
	case <-timer.C:
	case <-ctx.Done():
		m.end(l, "cancelled")

		return nil, "", errClientGone
	}

	url := l.out.loginURL()
	if url == "" {
		m.end(l, "failed")

		return nil, "", loginBadGateway("agy printed no login URL")
	}

	// 3. Admit it, evicting the oldest past the cap.
	m.mu.Lock()

	if m.closed {
		m.mu.Unlock()
		m.end(l, "cancelled")

		return nil, "", overloaded("the bridge is shutting down")
	}

	var evicted *login
	if len(m.pending) >= maxPendingLogins {
		evicted, m.pending = m.pending[0], m.pending[1:]
	}

	m.pending = append(m.pending, l)
	m.wg.Go(func() { m.watch(l) })
	m.mu.Unlock()

	if evicted != nil {
		m.end(evicted, "evicted")
	}

	m.logLine(l, "started")

	return l, url, nil
}

// spawn starts agy in its own process group in a fresh empty home.
func (m *loginManager) spawn() (*login, error) {
	home, err := m.homes.prepareEmpty(loginHomesDir)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithCancel(m.ctx)

	cmd := exec.CommandContext(ctx, m.bin, "acp", "--login")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.WaitDelay = agentWaitDelay
	cmd.Env = agentEnv(os.Environ(), home)
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}

	l := &login{
		id:      randomHex(16),
		home:    home,
		cmd:     cmd,
		out:     newLoginOutput(),
		started: time.Now(),
		cancel:  cancel,
		exited:  make(chan struct{}),
	}

	// One writer for both streams: exec then gives them a single pipe.
	cmd.Stdout = l.out
	cmd.Stderr = l.out

	l.stdin, err = cmd.StdinPipe()
	if err == nil {
		err = cmd.Start()
	}

	if err != nil {
		cancel()

		return nil, errors.Join(fmt.Errorf("start agy login: %w", err), m.homes.remove(home))
	}

	m.wg.Go(func() {
		l.waitErr = l.cmd.Wait()
		close(l.exited)
	})

	return l, nil
}

// watch ends a pending login when it expires or when agy exits by itself;
// a login another path took first is left to it.
func (m *loginManager) watch(l *login) {
	timer := time.NewTimer(loginTTL)
	defer timer.Stop()

	outcome := "expired"

	select {
	case <-timer.C:
	case <-l.exited:
		outcome = "failed"
	}

	if m.take(l.id) == l {
		m.end(l, outcome)
	}
}

// take removes a pending login and hands it to the caller, or returns nil.
func (m *loginManager) take(id string) *login {
	m.mu.Lock()
	defer m.mu.Unlock()

	for i, l := range m.pending {
		if l.id == id {
			m.pending = append(m.pending[:i:i], m.pending[i+1:]...)

			return l
		}
	}

	return nil
}

// end kills agy's process group, reaps it and removes its home.
func (m *loginManager) end(l *login, outcome string) {
	l.cancel()
	<-l.exited

	if err := m.homes.remove(l.home); err != nil {
		log.Printf("login id=%s: its home was not removed", l.id[:8])
	}

	m.logLine(l, outcome)
}

// logLine never carries the URL, the code, the token file or an email.
func (m *loginManager) logLine(l *login, outcome string) {
	log.Printf("login id=%s outcome=%s seconds=%d", l.id[:8], outcome, int(time.Since(l.started).Seconds()))
}

// submit hands the code to the pending login and returns the token file agy
// wrote. The login is over after this one attempt, whatever its result.
func (m *loginManager) submit(id, code string) (json.RawMessage, error) {
	if !m.enter() {
		return nil, overloaded("the bridge is shutting down")
	}
	defer m.wg.Done()

	l := m.take(id)
	if l == nil {
		return nil, loginNotFound()
	}

	// 1. agy reads one line; closing stdin after it means it never waits for
	// a second. A failed write shows as agy's own failure below.
	_, err := io.WriteString(l.stdin, code+"\n")
	err = errors.Join(err, l.stdin.Close())

	if err != nil {
		l.cancel()
	}

	// 2. The exchange runs under its own cap, not the request's: the login
	// ends here either way, so the client leaving changes nothing.
	timer := time.NewTimer(loginCodeTimeout)
	defer timer.Stop()

	select {
	case <-l.exited:
	case <-timer.C:
		l.cancel()
		<-l.exited
	}

	// 3. Trust only exit 0 and the file agy wrote, read before its home goes.
	var tokenFile json.RawMessage
	if l.waitErr == nil {
		tokenFile = readLoginTokenFile(l.home)
	}

	if tokenFile == nil {
		m.end(l, "failed")

		return nil, &apiError{
			status:  http.StatusUnprocessableEntity,
			kind:    "login_failed",
			message: l.out.failureMessage(code),
		}
	}

	m.end(l, "ok")

	return tokenFile, nil
}

// readLoginTokenFile returns agy's token file unchanged when it holds both
// tokens, or nil.
func readLoginTokenFile(home string) json.RawMessage {
	raw, err := os.ReadFile(filepath.Join(home, agyTokenPath))
	if err != nil {
		return nil
	}

	var file agyTokenFile
	if json.Unmarshal(raw, &file) != nil || file.Token.AccessToken == "" || file.Token.RefreshToken == "" {
		return nil
	}

	return raw
}

// cancelLogin ends a pending login; an unknown one is already gone.
func (m *loginManager) cancelLogin(id string) {
	if !m.enter() {
		return
	}
	defer m.wg.Done()

	if l := m.take(id); l != nil {
		m.end(l, "cancelled")
	}
}

func loginNotFound() *apiError {
	return &apiError{status: http.StatusNotFound, kind: "not_found_error", message: "no such pending login"}
}

func loginBadGateway(message string) *apiError {
	return &apiError{status: http.StatusBadGateway, kind: "api_error", message: message}
}

// loginOutput takes agy's merged stdout and stderr: it finds the URL line and
// keeps the tail for a failure message.
type loginOutput struct {
	mu        sync.Mutex
	tail      []byte
	truncated bool
	line      []byte
	url       string
	found     chan struct{}
}

func newLoginOutput() *loginOutput {
	return &loginOutput{found: make(chan struct{})}
}

func (o *loginOutput) Write(p []byte) (int, error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	o.tail = append(o.tail, p...)
	if over := len(o.tail) - loginTailBytes; over > 0 {
		o.tail = o.tail[over:]
		o.truncated = true
	}

	if o.url != "" {
		return len(p), nil
	}

	for _, b := range p {
		if b != '\n' {
			if len(o.line) < loginLineBytes {
				o.line = append(o.line, b)
			}

			continue
		}

		if match := loginURLPattern.Find(o.line); match != nil {
			o.url = string(match)
			close(o.found)

			break
		}

		o.line = o.line[:0]
	}

	return len(p), nil
}

func (o *loginOutput) loginURL() string {
	o.mu.Lock()
	defer o.mu.Unlock()

	return o.url
}

// failureMessage is the end of agy's output without the pasted code, an email
// or a URL. A cut tail loses its first, partial line, where half an address
// could hide.
func (o *loginOutput) failureMessage(code string) string {
	o.mu.Lock()
	text := string(o.tail)
	if o.truncated {
		if _, rest, ok := strings.Cut(text, "\n"); ok {
			text = rest
		}
	}
	o.mu.Unlock()

	for _, secret := range pastedSecrets(code) {
		text = strings.ReplaceAll(text, secret, "<code>")
	}

	redacted := emailPattern.ReplaceAllString(anyURLPattern.ReplaceAllString(text, "<url>"), "<email>")
	redacted = strings.ToValidUTF8(strings.TrimSpace(redacted), "")

	if runes := []rune(redacted); len(runes) > failureMessageRunes {
		redacted = string(runes[len(runes)-failureMessageRunes:])
	}

	if redacted == "" {
		return "agy did not finish the login"
	}

	return redacted
}

// pastedSecrets is every form of the pasted text agy may echo: as pasted and
// URL-decoded, and the code parameter of a pasted landing address in both
// forms. A form shorter than minRedactedCode is left alone; the longest go
// first, so a whole form is replaced before a part of it.
func pastedSecrets(code string) []string {
	code = strings.TrimSpace(code)
	whole := []string{code}

	if decoded, err := url.QueryUnescape(code); err == nil {
		whole = append(whole, decoded)
	}

	secrets := slices.Clone(whole)

	for _, form := range whole {
		for _, match := range codeParamPattern.FindAllStringSubmatch(form, -1) {
			secrets = append(secrets, match[1])
		}
	}

	secrets = slices.DeleteFunc(secrets, func(s string) bool { return len(s) < minRedactedCode })
	slices.SortFunc(secrets, func(a, b string) int { return len(b) - len(a) })

	return secrets
}

func (srv *Server) handleLoginStart(w http.ResponseWriter, r *http.Request) {
	l, url, err := srv.logins.start(r.Context())
	if errors.Is(err, errClientGone) {
		return
	}

	if err != nil {
		writeError(w, asAPIError(err))

		return
	}

	writeJSON(w, http.StatusCreated, map[string]any{
		"login_id":   l.id,
		"url":        url,
		"expires_in": int(loginTTL / time.Second),
	})
}

// handleLogin serves POST <prefix><id>/code and DELETE <prefix><id>.
func (srv *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	rest := strings.TrimPrefix(r.URL.Path, loginPathPrefix)
	id, action, nested := strings.Cut(rest, "/")

	switch {
	case r.Method == http.MethodPost && nested && action == "code" && id != "":
		srv.handleLoginCode(w, r, id)
	case r.Method == http.MethodDelete && !nested && id != "":
		srv.logins.cancelLogin(id)
		w.WriteHeader(http.StatusNoContent)
	default:
		writeError(w, &apiError{status: http.StatusNotFound, kind: "not_found_error", message: r.URL.Path})
	}
}

func (srv *Server) handleLoginCode(w http.ResponseWriter, r *http.Request, id string) {
	var body struct {
		Code string `json:"code"`
	}

	if err := decodeBody(w, r, &body); err != nil {
		writeError(w, invalidRequest(err.Error()))

		return
	}

	switch {
	case strings.TrimSpace(body.Code) == "":
		writeError(w, invalidRequest("code: required"))

		return
	case utf8.RuneCountInString(body.Code) > maxLoginCodeRunes:
		writeError(w, invalidRequest(fmt.Sprintf("code: at most %d characters", maxLoginCodeRunes)))

		return
	case strings.ContainsAny(body.Code, "\r\n"):
		writeError(w, invalidRequest("code: must be one line"))

		return
	}

	tokenFile, err := srv.logins.submit(id, body.Code)
	if err != nil {
		writeError(w, asAPIError(err))

		return
	}

	writeJSON(w, http.StatusOK, map[string]json.RawMessage{"token_file": tokenFile})
}
