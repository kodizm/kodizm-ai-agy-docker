package agybridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"maps"
	"math"
	"net/http"
	"slices"
	"strconv"
	"sync"
	"time"
)

const (
	accountsPath = "/v1/admin/accounts"
	outcomesPath = "/v1/admin/outcomes"
	usagePath    = "/v1/admin/usage"

	// maxPushBytes bounds the body of an account push.
	maxPushBytes = 1 << 20

	// outcomeBufferSize bounds the outcomes Kodizm has not acknowledged; past
	// it the oldest goes and counts as dropped.
	outcomeBufferSize = 1024

	// outcomePageSize bounds the outcomes one poll returns.
	outcomePageSize = 256

	// okEvery is how often one account's success is recorded: Kodizm only
	// stamps the time of it.
	okEvery = 60 * time.Second
)

// outcomeKinds names every failureKind on the wire, as Kodizm validates it.
var outcomeKinds = map[failureKind]string{
	kindOK:           "ok",
	kindQuota:        "quota",
	kindRate:         "rate",
	kindCapacity:     "capacity",
	kindAuth:         "auth",
	kindVerification: "verification",
	kindOther:        "other",
}

// pushedSource holds the accounts Kodizm pushes over the admin routes and
// keeps what the bridge learns of them until Kodizm polls it: Kodizm connects
// to the bridge, never the other way. Only access tokens arrive, never a
// refresh token. Everything is in memory: a restart loses it, a new boot id
// tells Kodizm so, and its next push restores the accounts.
type pushedSource struct {
	// boot names this process; the outcome sequence starts over with it.
	boot string
	now  func() time.Time

	mu       sync.Mutex
	accounts []Account
	// outcomes are the outcomes not yet acknowledged, ascending by seq.
	outcomes []wireOutcome
	lastSeq  uint64
	// dropped counts the outcomes the full buffer pushed out since boot.
	dropped uint64
	// okSent is when each account's last success was recorded.
	okSent map[string]time.Time
	// usage is the latest quota reading per account.
	usage map[string]wireUsage
}

func newPushedSource() *pushedSource {
	return &pushedSource{
		boot:   randomHex(16),
		now:    time.Now,
		okSent: map[string]time.Time{},
		usage:  map[string]wireUsage{},
	}
}

// Accounts returns the last pushed list; before the first push it is empty.
func (s *pushedSource) Accounts(context.Context) ([]Account, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	return slices.Clone(s.accounts), nil
}

func (s *pushedSource) store(list []Account) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.accounts = list
}

// wireAccount is one entry of PUT /v1/admin/accounts.
type wireAccount struct {
	ID          string `json:"id"`
	Email       string `json:"email"`
	AccessToken string `json:"access_token"`
	ExpiresAt   string `json:"expires_at"`
	State       struct {
		Cooldowns map[string]string  `json:"cooldowns"`
		Quota     map[string]float64 `json:"quota"`
	} `json:"state"`
}

// decodeAccounts reads a push body as a whole: one entry that is not a usable
// account refuses all of it. Its errors name the entry and the rule, never a
// value, since a decoder error may quote the body and the body carries tokens.
func decodeAccounts(body []byte) ([]Account, error) {
	var doc struct {
		Accounts []json.RawMessage `json:"accounts"`
	}

	if json.Unmarshal(body, &doc) != nil || doc.Accounts == nil {
		return nil, errors.New(`request body: want {"accounts":[...]}`)
	}

	accounts := make([]Account, 0, len(doc.Accounts))
	listed := map[string]bool{}

	for i, raw := range doc.Accounts {
		acct, err := decodeAccount(raw)
		if err != nil {
			return nil, fmt.Errorf("accounts[%d]: %w", i, err)
		}

		if listed[acct.ID] {
			return nil, fmt.Errorf("accounts[%d]: the id is listed twice", i)
		}

		listed[acct.ID] = true
		accounts = append(accounts, acct)
	}

	return accounts, nil
}

// decodeAccount turns one entry into an Account; its errors carry no value
// of the entry.
func decodeAccount(raw json.RawMessage) (Account, error) {
	var wire wireAccount
	if json.Unmarshal(raw, &wire) != nil {
		return Account{}, errors.New("not an account object")
	}

	switch {
	case !validAccountID(wire.ID):
		return Account{}, errors.New("id is not [A-Za-z0-9_.-]")
	case wire.ID == loginHomesDir:
		return Account{}, errors.New(`id "` + loginHomesDir + `" is reserved for the login homes`)
	case wire.AccessToken == "":
		return Account{}, errors.New("no access_token")
	}

	expires, err := time.Parse(time.RFC3339Nano, wire.ExpiresAt)
	if err != nil {
		return Account{}, errors.New("expires_at is not an RFC 3339 time")
	}

	acct := Account{
		ID:          wire.ID,
		Email:       wire.Email,
		AccessToken: wire.AccessToken,
		ExpiresAt:   expires,
		Cooldowns:   make(map[string]time.Time, len(wire.State.Cooldowns)),
		Quota:       make(map[string]float64, len(wire.State.Quota)),
	}

	for group, until := range wire.State.Cooldowns {
		if acct.Cooldowns[group], err = time.Parse(time.RFC3339Nano, until); err != nil {
			return Account{}, errors.New("a cooldown is not an RFC 3339 time")
		}
	}

	for group, left := range wire.State.Quota {
		if left < 0 || left > 1 {
			return Account{}, errors.New("a quota is outside 0 to 1")
		}

		acct.Quota[group] = left
	}

	return acct, nil
}

// wireOutcome is one entry of GET /v1/admin/outcomes. An outcome about the
// whole account carries no group.
type wireOutcome struct {
	Seq         uint64 `json:"seq"`
	AccountID   string `json:"account_id"`
	Group       string `json:"group,omitempty"`
	Kind        string `json:"kind"`
	RetryAfterS int64  `json:"retry_after_s"`
	Detail      string `json:"detail"`
	At          string `json:"at"`
}

// Report buffers one outcome for Kodizm's next poll. Successes are kept at
// most once per account per okEvery, except the first after a failure, which
// clears it in Kodizm.
func (s *pushedSource) Report(_ context.Context, o Outcome) {
	kind, ok := outcomeKinds[o.Kind]
	if !ok {
		log.Printf("agy outcomes: no wire name for kind %q; dropped", o.Kind)

		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if !s.dueLocked(o.AccountID, o.Kind, s.now()) {
		return
	}

	s.lastSeq++
	s.outcomes = append(s.outcomes, wireOutcome{
		Seq:         s.lastSeq,
		AccountID:   o.AccountID,
		Group:       o.Group,
		Kind:        kind,
		RetryAfterS: int64(math.Ceil(o.RetryAfter.Seconds())),
		Detail:      o.Detail,
		// The failure's own time: a poll that comes later must not read it
		// newer than a login the owner made meanwhile.
		At: o.At.UTC().Format(time.RFC3339),
	})

	if over := len(s.outcomes) - outcomeBufferSize; over > 0 {
		s.outcomes = slices.Delete(s.outcomes, 0, over)
		s.dropped += uint64(over)
	}
}

// dueLocked decides whether an outcome is kept, and notes a success that is.
func (s *pushedSource) dueLocked(id string, kind failureKind, now time.Time) bool {
	if kind != kindOK {
		delete(s.okSent, id)

		return true
	}

	if last, sent := s.okSent[id]; sent && now.Sub(last) < okEvery {
		return false
	}

	s.okSent[id] = now

	return true
}

// take acknowledges every outcome up to after, the cursor of Kodizm's last
// poll, and returns the oldest page of the rest with the drop count.
func (s *pushedSource) take(after uint64) ([]wireOutcome, uint64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.outcomes = slices.DeleteFunc(s.outcomes, func(o wireOutcome) bool { return o.Seq <= after })

	page := make([]wireOutcome, min(len(s.outcomes), outcomePageSize))
	copy(page, s.outcomes)

	return page, s.dropped
}

// wireUsage is one entry of GET /v1/admin/usage. At is when the poller
// handed the reading over, so Kodizm can skip one it already stored.
type wireUsage struct {
	AccountID string             `json:"account_id"`
	Quota     map[string]float64 `json:"quota"`
	Windows   []wireWindow       `json:"windows"`
	At        string             `json:"at"`
}

type wireWindow struct {
	Group    string  `json:"group"`
	Label    string  `json:"label"`
	Percent  float64 `json:"percent"`
	ResetsAt *string `json:"resets_at"`
}

// ReportUsage keeps the reading as the account's latest.
func (s *pushedSource) ReportUsage(_ context.Context, r UsageReading) {
	entry := wireUsage{AccountID: r.AccountID, Quota: maps.Clone(r.Quota), Windows: []wireWindow{}}
	if entry.Quota == nil {
		entry.Quota = map[string]float64{}
	}

	for _, w := range r.Windows {
		window := wireWindow{Group: w.Group, Label: w.Label, Percent: w.Percent}

		if !w.ResetsAt.IsZero() {
			resets := w.ResetsAt.UTC().Format(time.RFC3339)
			window.ResetsAt = &resets
		}

		entry.Windows = append(entry.Windows, window)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	entry.At = s.now().UTC().Format(time.RFC3339)
	s.usage[r.AccountID] = entry
}

// latestUsage is every account's latest reading, ordered by account id.
func (s *pushedSource) latestUsage() []wireUsage {
	s.mu.Lock()
	defer s.mu.Unlock()

	usage := make([]wireUsage, 0, len(s.usage))
	for _, id := range slices.Sorted(maps.Keys(s.usage)) {
		usage = append(usage, s.usage[id])
	}

	return usage
}

// pushedOnly returns the pushed source, or answers 409 when the bridge
// serves a token file and returns nil.
func (srv *Server) pushedOnly(w http.ResponseWriter) *pushedSource {
	if srv.pushed == nil {
		writeError(w, &apiError{
			status:  http.StatusConflict,
			kind:    "invalid_request_error",
			message: "the bridge serves AGY_BRIDGE_TOKEN_FILE; its accounts are not pushed",
		})
	}

	return srv.pushed
}

// handlePushAccounts serves PUT /v1/admin/accounts: the list replaces the
// pool's whole, through the path a refresh takes, or nothing changes.
func (srv *Server) handlePushAccounts(w http.ResponseWriter, r *http.Request) {
	src := srv.pushedOnly(w)
	if src == nil {
		return
	}

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxPushBytes))

	var tooLarge *http.MaxBytesError

	switch {
	case errors.As(err, &tooLarge):
		writeError(w, invalidRequest(fmt.Sprintf("request body: larger than %d bytes", maxPushBytes)))

		return
	case err != nil:
		writeError(w, invalidRequest("request body: unreadable"))

		return
	}

	list, err := decodeAccounts(body)
	if err != nil {
		writeError(w, invalidRequest(err.Error()))

		return
	}

	srv.pool.replace(list, func() { src.store(list) })

	writeJSON(w, http.StatusOK, struct {
		Boot    string `json:"boot"`
		Applied int    `json:"applied"`
	}{
		Boot:    src.boot,
		Applied: len(list),
	})
}

// handleOutcomes serves GET /v1/admin/outcomes?after=<seq>&boot=<id>. The
// cursor acknowledges only when boot names this process: a cursor from an
// earlier one would otherwise discard outcomes it never saw.
func (srv *Server) handleOutcomes(w http.ResponseWriter, r *http.Request) {
	src := srv.pushedOnly(w)
	if src == nil {
		return
	}

	query := r.URL.Query()
	after := uint64(0)

	if raw := query.Get("after"); raw != "" {
		parsed, err := strconv.ParseUint(raw, 10, 64)
		if err != nil {
			writeError(w, invalidRequest("after: want a sequence number"))

			return
		}

		after = parsed
	}

	if query.Get("boot") != src.boot {
		after = 0
	}

	outcomes, dropped := src.take(after)

	writeJSON(w, http.StatusOK, struct {
		Boot     string        `json:"boot"`
		Dropped  uint64        `json:"dropped"`
		Outcomes []wireOutcome `json:"outcomes"`
	}{
		Boot:     src.boot,
		Dropped:  dropped,
		Outcomes: outcomes,
	})
}

// handleUsage serves GET /v1/admin/usage.
func (srv *Server) handleUsage(w http.ResponseWriter) {
	src := srv.pushedOnly(w)
	if src == nil {
		return
	}

	writeJSON(w, http.StatusOK, struct {
		Boot  string      `json:"boot"`
		Usage []wireUsage `json:"usage"`
	}{
		Boot:  src.boot,
		Usage: src.latestUsage(),
	})
}
