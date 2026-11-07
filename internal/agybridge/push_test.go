package agybridge

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"net/http"
	"net/url"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

const testAPIKey = "bridge-key-value"

// newPushedBridge is a bridge on a pushed source, cold until a test pushes.
func newPushedBridge(t *testing.T, env map[string]string, configure ...func(*Config)) (*bridge, *pushedSource) {
	t.Helper()

	src := newPushedSource()

	return newBridgeWith(t, env, src, configure...), src
}

// admin sends a raw body to an admin route, with key as x-api-key when set.
func (b *bridge) admin(t *testing.T, method, path, body, key string) *http.Response {
	t.Helper()

	var reader io.Reader
	if body != "" {
		reader = strings.NewReader(body)
	}

	req, err := http.NewRequestWithContext(context.Background(), method, b.ts.URL+path, reader)
	if err != nil {
		t.Fatal(err)
	}

	if key != "" {
		req.Header.Set("x-api-key", key)
	}

	resp, err := b.ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = resp.Body.Close() })

	return resp
}

type pushAnswer struct {
	Boot    string `json:"boot"`
	Applied int    `json:"applied"`
}

type outcomesAnswer struct {
	Boot     string            `json:"boot"`
	Dropped  uint64            `json:"dropped"`
	Outcomes []json.RawMessage `json:"outcomes"`
}

// push PUTs the accounts and expects them applied.
func (b *bridge) push(t *testing.T, accounts ...string) pushAnswer {
	t.Helper()

	resp := b.admin(t, http.MethodPut, accountsPath, `{"accounts":[`+strings.Join(accounts, ",")+`]}`, testAPIKey)
	expectStatus(t, resp, http.StatusOK)

	answer := decodeJSON[pushAnswer](t, resp)
	if answer.Applied != len(accounts) {
		t.Fatalf("applied %d, want %d", answer.Applied, len(accounts))
	}

	return answer
}

// pollOutcomes polls with Kodizm's cursor and the boot it was taken under.
func (b *bridge) pollOutcomes(t *testing.T, after uint64, boot string) outcomesAnswer {
	t.Helper()

	query := url.Values{"after": {strconv.FormatUint(after, 10)}}
	if boot != "" {
		query.Set("boot", boot)
	}

	resp := b.admin(t, http.MethodGet, outcomesPath+"?"+query.Encode(), "", testAPIKey)
	expectStatus(t, resp, http.StatusOK)

	return decodeJSON[outcomesAnswer](t, resp)
}

// sameJSON fails unless got and want are the same JSON value.
func sameJSON(t *testing.T, got, want string) {
	t.Helper()

	var g, w any
	if err := json.Unmarshal([]byte(got), &g); err != nil {
		t.Fatalf("body %s: %v", got, err)
	}

	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("want %s: %v", want, err)
	}

	gotRaw, _ := json.Marshal(g)
	wantRaw, _ := json.Marshal(w)

	if !bytes.Equal(gotRaw, wantRaw) {
		t.Fatalf("body\n%s\nwant\n%s", gotRaw, wantRaw)
	}
}

func accountJSON(id, token string) string {
	return `{"id":"` + id + `","email":"` + id + `@example.test","access_token":"` + token + `",` +
		`"expires_at":"` + time.Now().Add(time.Hour).UTC().Format(time.RFC3339) + `",` +
		`"state":{"cooldowns":{},"quota":{}}}`
}

// present lists the ids the pool holds as listed, sorted.
func present(p *pool) []string {
	p.mu.Lock()
	defer p.mu.Unlock()

	var ids []string

	for id, st := range p.accounts {
		if st.present {
			ids = append(ids, id)
		}
	}

	slices.Sort(ids)

	return ids
}

func TestAdminRoutesNeedTheKey(t *testing.T) {
	b, _ := newPushedBridge(t, map[string]string{"AGY_BRIDGE_API_KEY": testAPIKey})
	b.push(t, accountJSON("acct-a", "ya29.a"))

	routes := []struct {
		method string
		path   string
		body   string
	}{
		{method: http.MethodPut, path: accountsPath, body: `{"accounts":[]}`},
		{method: http.MethodGet, path: outcomesPath},
		{method: http.MethodGet, path: usagePath},
	}

	// 1. Without the key, or with a wrong one, every route is refused.
	for _, route := range routes {
		for _, key := range []string{"", "wrong"} {
			resp := b.admin(t, route.method, route.path, route.body, key)
			expectStatus(t, resp, http.StatusUnauthorized)
		}
	}

	// 2. The refused push cleared nothing.
	if got := present(b.srv.pool); !slices.Equal(got, []string{"acct-a"}) {
		t.Fatalf("present %v after refused pushes", got)
	}

	// 3. With the key, every route answers; a bearer token works as well.
	for _, route := range routes {
		expectStatus(t, b.admin(t, route.method, route.path, route.body, testAPIKey), http.StatusOK)
	}

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, b.ts.URL+usagePath, nil)
	if err != nil {
		t.Fatal(err)
	}

	req.Header.Set("Authorization", "Bearer "+testAPIKey)

	resp, err := b.ts.Client().Do(req)
	if err != nil {
		t.Fatal(err)
	}

	defer resp.Body.Close()

	expectStatus(t, resp, http.StatusOK)
}

func TestPushDecodesTheAccounts(t *testing.T) {
	b, src := newPushedBridge(t, nil)
	resets := time.Now().Add(time.Hour).UTC().Truncate(time.Second)

	resp := b.admin(t, http.MethodPut, accountsPath, `{"accounts":[
		{"id":"acct-1","email":"one@example.test","access_token":"ya29.one","expires_at":"2026-11-07T12:00:00Z",
		 "refresh_token":"1//never","extra":true,
		 "state":{"cooldowns":{"gemini":"`+resets.Format(time.RFC3339)+`"},"quota":{"gemini":0.82,"argon":1}}},
		{"id":"acct-2","email":null,"access_token":"ya29.two","expires_at":"2026-11-07T12:00:00Z",
		 "state":{"cooldowns":{},"quota":{}}}
	]}`, testAPIKey)
	expectStatus(t, resp, http.StatusOK)

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}

	// 1. The answer names the boot and the count, never a token.
	var answer pushAnswer
	if err := json.Unmarshal(raw, &answer); err != nil || answer.Applied != 2 ||
		!regexp.MustCompile(`^[0-9a-f]{32}$`).MatchString(answer.Boot) || bytes.Contains(raw, []byte("ya29")) {
		t.Fatalf("answer %s", raw)
	}

	// 2. The source holds the list as decoded; a refresh token never arrives.
	accounts, err := src.Accounts(context.Background())
	if err != nil || len(accounts) != 2 {
		t.Fatalf("accounts %d, %v", len(accounts), err)
	}

	one, two := accounts[0], accounts[1]

	if one.ID != "acct-1" || one.Email != "one@example.test" || one.AccessToken != "ya29.one" || one.RefreshToken != "" ||
		!one.ExpiresAt.Equal(time.Date(2026, 11, 7, 12, 0, 0, 0, time.UTC)) ||
		!one.Cooldowns["gemini"].Equal(resets) || one.Quota["gemini"] != 0.82 || one.Quota["argon"] != 1 {
		t.Fatalf("first account %+v", one)
	}

	if two.ID != "acct-2" || two.Email != "" || len(two.Cooldowns) != 0 || len(two.Quota) != 0 {
		t.Fatalf("second account %+v", two)
	}

	// 3. The pool took it at once, without waiting for a refresh.
	if got := present(b.srv.pool); !slices.Equal(got, []string{"acct-1", "acct-2"}) {
		t.Fatalf("present %v", got)
	}
}

// "login" names the directory the login homes live under: an account of that
// id would share it.
func TestDecodeAccountRefusesTheReservedID(t *testing.T) {
	if _, err := decodeAccount(json.RawMessage(accountJSON("login", "ya29.leak"))); err == nil ||
		strings.Contains(err.Error(), "ya29") {
		t.Fatalf("decodeAccount = %v, want a refusal without the token", err)
	}

	if _, err := decodeAccount(json.RawMessage(accountJSON("login-2", "ya29.x"))); err != nil {
		t.Fatalf("decodeAccount(login-2) = %v", err)
	}
}

func TestPushReplacesAndClearsThePool(t *testing.T) {
	b, src := newPushedBridge(t, nil)
	p := b.srv.pool
	ctx := context.Background()

	// 1. Two accounts, both leasable.
	first := b.push(t, accountJSON("acct-a", "ya29.a"), accountJSON("acct-b", "ya29.b"))

	if got := present(p); !slices.Equal(got, []string{"acct-a", "acct-b"}) {
		t.Fatalf("present %v", got)
	}

	// 2. A list without a leaves only b, even past a refresh.
	if second := b.push(t, accountJSON("acct-b", "ya29.b")); second.Boot != first.Boot {
		t.Fatalf("boot moved from %s to %s", first.Boot, second.Boot)
	}

	p.refresh(ctx, true)

	if got := present(p); !slices.Equal(got, []string{"acct-b"}) {
		t.Fatalf("present %v", got)
	}

	if _, err := p.acquire(ctx, "k", flash, "acct-b", 0); err == nil {
		t.Fatal("an account the push left out was leased")
	}

	// 3. An empty list is valid and clears the pool.
	b.push(t)

	if got := present(p); len(got) != 0 {
		t.Fatalf("present %v after an empty push", got)
	}

	if accounts, _ := src.Accounts(ctx); len(accounts) != 0 {
		t.Fatalf("the source still holds %d", len(accounts))
	}

	var none *errNoAccount
	if _, err := p.acquire(ctx, "k", flash, "", 0); !errors.As(err, &none) || none.RetryAfter != 0 {
		t.Fatalf("acquire on a cleared pool = %v", err)
	}
}

func TestPushRefusesAnInvalidListWhole(t *testing.T) {
	logs := captureLog(t)
	b, src := newPushedBridge(t, nil)
	b.push(t, accountJSON("acct-a", "ya29.a"))

	valid := accountJSON("acct-b", "ya29.b")
	huge := `{"accounts":[{"id":"acct-b","access_token":"ya29.b","email":"` + strings.Repeat("x", maxPushBytes) + `"}]}`

	bodies := map[string]string{
		"not json":           `ya29.leak`,
		"no accounts key":    `{}`,
		"null accounts":      `{"accounts":null}`,
		"accounts not list":  `{"accounts":"ya29.leak"}`,
		"bad id":             `{"accounts":[` + valid + `,` + accountJSON("../evil", "ya29.leak") + `]}`,
		"reserved id":        `{"accounts":[` + valid + `,` + accountJSON("login", "ya29.leak") + `]}`,
		"no access token":    `{"accounts":[` + valid + `,{"id":"acct-3","access_token":"","expires_at":"2026-11-07T12:00:00Z"}]}`,
		"bad expiry":         `{"accounts":[` + valid + `,{"id":"acct-4","access_token":"ya29.leak","expires_at":"tomorrow"}]}`,
		"quota out of range": `{"accounts":[` + valid + `,{"id":"acct-5","access_token":"ya29.leak","expires_at":"2026-11-07T12:00:00Z","state":{"quota":{"gemini":1.5}}}]}`,
		"bad cooldown":       `{"accounts":[` + valid + `,{"id":"acct-6","access_token":"ya29.leak","expires_at":"2026-11-07T12:00:00Z","state":{"cooldowns":{"gemini":"soon"}}}]}`,
		"not an object":      `{"accounts":[` + valid + `,"ya29.leak"]}`,
		"duplicate id":       `{"accounts":[` + valid + `,` + valid + `]}`,
		"over the cap":       huge,
	}

	for name, body := range bodies {
		t.Run(name, func(t *testing.T) {
			resp := b.admin(t, http.MethodPut, accountsPath, body, testAPIKey)

			raw, _ := io.ReadAll(resp.Body)
			if resp.StatusCode != http.StatusBadRequest || bytes.Contains(raw, []byte("ya29")) {
				t.Fatalf("status %d: %s", resp.StatusCode, raw)
			}

			// Nothing of the list applied: a stays, b never arrives.
			if got := present(b.srv.pool); !slices.Equal(got, []string{"acct-a"}) {
				t.Fatalf("present %v", got)
			}

			if accounts, _ := src.Accounts(context.Background()); len(accounts) != 1 || accounts[0].ID != "acct-a" {
				t.Fatalf("source holds %+v", accounts)
			}
		})
	}

	if got := logs.String(); strings.Contains(got, "ya29") {
		t.Fatalf("log leaks %q", got)
	}
}

// A login an auth failure condemned stays out while the same credentials are
// pushed again; a new access token brings the account back.
func TestPushKeepsACondemnedLoginOut(t *testing.T) {
	b, _ := newPushedBridge(t, nil)
	p := b.srv.pool
	ctx := context.Background()

	b.push(t, accountJSON("acct-a", "ya29.a"))

	l, err := p.acquire(ctx, "k", flash, "", 0)
	if err != nil {
		t.Fatal(err)
	}

	l.settle(flash, &failure{Kind: kindAuth}, "signed out")
	l.release()

	b.push(t, accountJSON("acct-a", "ya29.a"))

	if _, err := p.acquire(ctx, "k", flash, "", 0); err == nil {
		t.Fatal("the condemned credentials were leased again")
	}

	b.push(t, accountJSON("acct-a", "ya29.fresh"))

	if got := lease(t, p, "k", flash); got != "acct-a" {
		t.Fatalf("picked %s", got)
	}
}

func TestAdminRoutesConflictOnATokenFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "token")
	writeLoginFile(t, path, "ya29.x")

	src, err := newSource(Config{TokenFile: path})
	if err != nil {
		t.Fatal(err)
	}

	b := newBridgeWith(t, nil, src)

	for _, route := range [][2]string{
		{http.MethodPut, accountsPath},
		{http.MethodGet, outcomesPath},
		{http.MethodGet, usagePath},
	} {
		expectStatus(t, b.admin(t, route[0], route[1], `{"accounts":[]}`, ""), http.StatusConflict)
	}

	if accounts, _ := src.Accounts(context.Background()); len(accounts) != 1 {
		t.Fatalf("the token file source changed: %d accounts", len(accounts))
	}
}

// With pushed accounts the pool and the login routes are never left open.
func TestRunRefusesPushedAccountsWithoutAnAPIKey(t *testing.T) {
	err := Run(context.Background(), Config{Listen: "127.0.0.1:0"})
	if err == nil || !strings.Contains(err.Error(), "AGY_BRIDGE_API_KEY") {
		t.Fatalf("Run = %v, want a refusal naming AGY_BRIDGE_API_KEY", err)
	}
}

func TestColdStartAnswersNoAccount(t *testing.T) {
	b, _ := newPushedBridge(t, nil)

	resp := b.post(t, map[string]any{"model": "gemini-3.8-flash", "messages": userText("SCRIPT:plain")})
	expectStatus(t, resp, http.StatusServiceUnavailable)

	if body, _ := io.ReadAll(resp.Body); !strings.Contains(string(body), "no agy account available") {
		t.Fatalf("body %s", body)
	}

	if b.spawned(t) != 0 {
		t.Fatal("an agy started without an account")
	}
}

// TestOutcomeKindsCoverEveryFailureKind reads every failureKind constant of
// the package's source, so a kind added later without a wire name fails here.
func TestOutcomeKindsCoverEveryFailureKind(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}

	var declared []string

	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue
		}

		parsed, err := parser.ParseFile(token.NewFileSet(), file, nil, 0)
		if err != nil {
			t.Fatal(err)
		}

		ast.Inspect(parsed, func(n ast.Node) bool {
			spec, ok := n.(*ast.ValueSpec)
			if !ok {
				return true
			}

			if ident, ok := spec.Type.(*ast.Ident); !ok || ident.Name != "failureKind" {
				return true
			}

			for _, value := range spec.Values {
				if lit, ok := value.(*ast.BasicLit); ok {
					kind, _ := strconv.Unquote(lit.Value)
					declared = append(declared, kind)
				}
			}

			return true
		})
	}

	// kindExpired is the bridge's own and never reported: the pool drops it.
	declared = slices.DeleteFunc(declared, func(kind string) bool { return kind == string(kindExpired) })

	// Kodizm's AgyPoolService::KINDS.
	kodizm := []string{"ok", "quota", "rate", "capacity", "auth", "verification", "other"}

	if len(declared) != len(kodizm) || len(outcomeKinds) != len(declared) {
		t.Fatalf("declared %v, mapped %v", declared, outcomeKinds)
	}

	for _, kind := range declared {
		if wire := outcomeKinds[failureKind(kind)]; !slices.Contains(kodizm, wire) || wire != kind {
			t.Fatalf("kind %q maps to %q", kind, wire)
		}
	}
}

func TestOutcomeShapeAndOKCoalescing(t *testing.T) {
	src := newPushedSource()
	c := &clock{t: time.Now().Truncate(time.Second)}
	src.now = c.now
	ctx := context.Background()

	at := func() string { return c.now().UTC().Format(time.RFC3339) }

	var want []string

	expect := func(item string) {
		want = append(want, `{"seq":`+strconv.Itoa(len(want)+1)+`,`+item+`,"at":"`+at()+`"}`)
	}

	// The pool stamps each outcome when it settles it.
	report := func(o Outcome) {
		o.At = c.now()
		src.Report(ctx, o)
	}

	// 1. Every kind goes in by its wire name, retry_after_s rounded up.
	for _, kind := range []failureKind{kindQuota, kindRate, kindCapacity, kindAuth, kindVerification, kindOther} {
		report(Outcome{AccountID: "acct-1", Group: "gemini", Kind: kind, RetryAfter: 1200 * time.Millisecond, Detail: "d"})
		expect(`"account_id":"acct-1","group":"gemini","kind":"` + string(kind) + `","retry_after_s":2,"detail":"d"`)
	}

	// 2. An account-wide outcome carries no group.
	report(Outcome{AccountID: "acct-1", Kind: kindAuth, Detail: "agy /usage"})
	expect(`"account_id":"acct-1","kind":"auth","retry_after_s":0,"detail":"agy /usage"`)

	// 3. One ok per account per minute; the first after a failure always.
	report(Outcome{AccountID: "acct-1", Group: "gemini", Kind: kindOK})
	expect(`"account_id":"acct-1","group":"gemini","kind":"ok","retry_after_s":0,"detail":""`)

	report(Outcome{AccountID: "acct-1", Group: "argon", Kind: kindOK})
	c.advance(59 * time.Second)
	report(Outcome{AccountID: "acct-1", Group: "gemini", Kind: kindOK})

	report(Outcome{AccountID: "acct-2", Group: "gemini", Kind: kindOK})
	expect(`"account_id":"acct-2","group":"gemini","kind":"ok","retry_after_s":0,"detail":""`)

	c.advance(time.Second)
	report(Outcome{AccountID: "acct-1", Group: "gemini", Kind: kindOK})
	expect(`"account_id":"acct-1","group":"gemini","kind":"ok","retry_after_s":0,"detail":""`)

	report(Outcome{AccountID: "acct-1", Group: "gemini", Kind: kindRate, RetryAfter: 30 * time.Second})
	expect(`"account_id":"acct-1","group":"gemini","kind":"rate","retry_after_s":30,"detail":""`)

	report(Outcome{AccountID: "acct-1", Group: "gemini", Kind: kindOK})
	expect(`"account_id":"acct-1","group":"gemini","kind":"ok","retry_after_s":0,"detail":""`)

	// 4. An outcome polled late carries the time of the failure, not of the poll.
	failed := c.now()
	src.Report(ctx, Outcome{AccountID: "acct-1", Group: "argon", Kind: kindQuota, At: failed})
	expect(`"account_id":"acct-1","group":"argon","kind":"quota","retry_after_s":0,"detail":""`)
	c.advance(3 * time.Minute)

	page, dropped := src.take(0)

	got, err := json.Marshal(page)
	if err != nil {
		t.Fatal(err)
	}

	if dropped != 0 {
		t.Fatalf("dropped %d", dropped)
	}

	sameJSON(t, string(got), "["+strings.Join(want, ",")+"]")
}

func TestOutcomeCursorAcknowledges(t *testing.T) {
	b, src := newPushedBridge(t, nil)

	for _, id := range []string{"a", "b", "c"} {
		src.Report(context.Background(), Outcome{AccountID: id, Kind: kindRate, At: time.Now()})
	}

	seqs := func(answer outcomesAnswer) []uint64 {
		var got []uint64

		for _, raw := range answer.Outcomes {
			var item struct {
				Seq uint64 `json:"seq"`
			}

			if err := json.Unmarshal(raw, &item); err != nil {
				t.Fatal(err)
			}

			got = append(got, item.Seq)
		}

		return got
	}

	// 1. Sequence numbers start at 1; a poll alone acknowledges nothing.
	first := b.pollOutcomes(t, 0, "")
	if got := seqs(first); !slices.Equal(got, []uint64{1, 2, 3}) || first.Dropped != 0 || first.Boot != src.boot {
		t.Fatalf("first poll %v, %+v", got, first)
	}

	boot := first.Boot

	if got := seqs(b.pollOutcomes(t, 0, boot)); !slices.Equal(got, []uint64{1, 2, 3}) {
		t.Fatalf("repeated poll %v", got)
	}

	// 2. A cursor under a missing or stale boot is from another process: it
	// acknowledges and drops nothing, and the answer starts at seq 1.
	for _, stale := range []string{"", "0123456789abcdef0123456789abcdef"} {
		answer := b.pollOutcomes(t, 5000, stale)
		if got := seqs(answer); !slices.Equal(got, []uint64{1, 2, 3}) || answer.Boot != boot || answer.Dropped != 0 {
			t.Fatalf("boot %q, after 5000: %v, %+v", stale, got, answer)
		}
	}

	// 3. Under this boot the cursor acknowledges everything up to it.
	if got := seqs(b.pollOutcomes(t, 2, boot)); !slices.Equal(got, []uint64{3}) {
		t.Fatalf("after 2: %v", got)
	}

	if got := seqs(b.pollOutcomes(t, 1, boot)); !slices.Equal(got, []uint64{3}) {
		t.Fatalf("an older cursor brought back %v", got)
	}

	// 4. An empty buffer answers an empty list, and the sequence goes on.
	resp := b.admin(t, http.MethodGet, outcomesPath+"?after=3&boot="+boot, "", "")
	expectStatus(t, resp, http.StatusOK)

	if raw, _ := io.ReadAll(resp.Body); !bytes.Contains(raw, []byte(`"outcomes":[]`)) {
		t.Fatalf("empty answer %s", raw)
	}

	src.Report(context.Background(), Outcome{AccountID: "a", Kind: kindQuota, At: time.Now()})

	last := b.pollOutcomes(t, 3, boot)
	if got := seqs(last); !slices.Equal(got, []uint64{4}) || last.Boot != boot {
		t.Fatalf("after 3: %v, boot %s then %s", got, boot, last.Boot)
	}

	// 5. A missing cursor is 0; a cursor that is not a sequence number is refused.
	expectStatus(t, b.admin(t, http.MethodGet, outcomesPath, "", ""), http.StatusOK)

	for _, bad := range []string{"x", "-1", "1.5"} {
		expectStatus(t, b.admin(t, http.MethodGet, outcomesPath+"?after="+bad, "", ""), http.StatusBadRequest)
	}
}

func TestOutcomeBufferDropsTheOldestAndPages(t *testing.T) {
	src := newPushedSource()
	extra := 6

	for i := range outcomeBufferSize + extra {
		src.Report(context.Background(), Outcome{AccountID: fmt.Sprintf("acct-%d", i), Kind: kindRate})
	}

	// 1. The buffer kept the newest, counted the rest, and pages ascending.
	page, dropped := src.take(0)
	if len(page) != outcomePageSize || dropped != uint64(extra) ||
		page[0].Seq != uint64(extra+1) || page[len(page)-1].Seq != uint64(extra+outcomePageSize) {
		t.Fatalf("page of %d from %d to %d, dropped %d", len(page), page[0].Seq, page[len(page)-1].Seq, dropped)
	}

	// 2. The next cursor brings the next page; dropped stays cumulative.
	next, dropped := src.take(page[len(page)-1].Seq)
	if len(next) != outcomePageSize || next[0].Seq != page[len(page)-1].Seq+1 || dropped != uint64(extra) {
		t.Fatalf("next page of %d from %d, dropped %d", len(next), next[0].Seq, dropped)
	}

	// 3. A drop after an acknowledgement adds to the count.
	src.take(uint64(outcomeBufferSize + extra))

	for range outcomeBufferSize + 1 {
		src.Report(context.Background(), Outcome{AccountID: "acct", Kind: kindRate})
	}

	if _, dropped := src.take(0); dropped != uint64(extra+1) {
		t.Fatalf("dropped %d", dropped)
	}
}

func TestUsageKeepsTheLatestPerAccount(t *testing.T) {
	b, src := newPushedBridge(t, nil)
	resets := time.Date(2026, 11, 13, 23, 48, 1, 0, time.UTC)
	ctx := context.Background()

	c := &clock{t: time.Date(2026, 11, 7, 10, 0, 0, 0, time.FixedZone("CET", 3600))}
	src.mu.Lock()
	src.now = c.now
	src.mu.Unlock()

	quota := map[string]float64{"gemini": 0.5}
	src.ReportUsage(ctx, UsageReading{AccountID: "acct-1", Quota: quota})
	src.ReportUsage(ctx, UsageReading{AccountID: "acct-2"})

	// The newer reading of acct-1 replaces the older one, its time with it.
	c.advance(5 * time.Minute)
	src.ReportUsage(ctx, UsageReading{
		AccountID: "acct-1",
		Quota:     map[string]float64{"gemini": 0.97, "argon": 1, "thirdparty": 0.99},
		Windows: []UsageWindow{
			{Group: "Gemini Models", Label: "Weekly Limit Remaining", Percent: 97, ResetsAt: resets},
			{Group: "Gemini 4 Argon", Label: "Gemini 4 Argon Weekly Limit", Percent: 100},
		},
	})

	// A reading is a copy: the poller's map changing later changes nothing.
	quota["gemini"] = 0

	resp := b.admin(t, http.MethodGet, usagePath, "", "")
	expectStatus(t, resp, http.StatusOK)

	answer := decodeJSON[struct {
		Boot  string          `json:"boot"`
		Usage json.RawMessage `json:"usage"`
	}](t, resp)

	if answer.Boot != src.boot {
		t.Fatalf("boot %q, want %q", answer.Boot, src.boot)
	}

	sameJSON(t, string(answer.Usage), `[
		{"account_id":"acct-1","quota":{"gemini":0.97,"argon":1.0,"thirdparty":0.99},"windows":[
			{"group":"Gemini Models","label":"Weekly Limit Remaining","percent":97,"resets_at":"2026-11-13T23:48:01Z"},
			{"group":"Gemini 4 Argon","label":"Gemini 4 Argon Weekly Limit","percent":100,"resets_at":null}],
		 "at":"2026-11-07T09:05:00Z"},
		{"account_id":"acct-2","quota":{},"windows":[],"at":"2026-11-07T09:00:00Z"}]`)
}

// stalledSource reads its list, then holds the first Accounts call until the
// gate opens: a refresh that read an older list than a push applied meanwhile.
type stalledSource struct {
	*pushedSource
	once    sync.Once
	entered chan struct{}
	gate    chan struct{}
}

func (s *stalledSource) Accounts(ctx context.Context) ([]Account, error) {
	list, err := s.pushedSource.Accounts(ctx)

	s.once.Do(func() {
		close(s.entered)
		<-s.gate
	})

	return list, err
}

func TestARefreshNeverUndoesAPush(t *testing.T) {
	src := &stalledSource{pushedSource: newPushedSource(), entered: make(chan struct{}), gate: make(chan struct{})}
	src.store([]Account{testAccount("a")})

	p, _ := newTestPool(t, src, 2)
	refreshed := make(chan struct{})

	go func() {
		defer close(refreshed)

		p.refresh(context.Background(), false)
	}()

	<-src.entered

	pushed := []Account{testAccount("b")}
	p.replace(pushed, func() { src.store(pushed) })

	close(src.gate)
	<-refreshed

	if got := present(p); !slices.Equal(got, []string{"b"}) {
		t.Fatalf("present %v, want the pushed list", got)
	}
}

// The bridge end to end on pushed accounts: a conversation runs on a pushed
// token, and the polled quota and the outcome wait for Kodizm's poll.
func TestBridgeOnPushedAccounts(t *testing.T) {
	setVar(t, &minUsageInterval, time.Millisecond)
	t.Setenv(fakeUsageEnv, `{"tok-a":"Gemini Models\tWeekly Limit Remaining\t10%\t2026-11-13T23:48:01Z\n",`+
		`"tok-b":"Gemini Models\tWeekly Limit Remaining\t90%\t2026-11-13T23:48:01Z\n"}`)

	b, _ := newPushedBridge(t, nil, func(cfg *Config) { cfg.UsageInterval = 20 * time.Millisecond })
	b.push(t, accountJSON("acct-a", "tok-a"), accountJSON("acct-b", "tok-b"))

	usage := func() []json.RawMessage {
		resp := b.admin(t, http.MethodGet, usagePath, "", testAPIKey)
		expectStatus(t, resp, http.StatusOK)

		return decodeJSON[struct {
			Usage []json.RawMessage `json:"usage"`
		}](t, resp).Usage
	}

	// 1. The polled quota waits for Kodizm, one reading per account.
	eventually(t, "a usage reading per account", func() bool { return len(usage()) == 2 })

	var reading map[string]any
	if err := json.Unmarshal(usage()[0], &reading); err != nil {
		t.Fatal(err)
	}

	at, _ := reading["at"].(string)
	if _, err := time.Parse(time.RFC3339, at); err != nil || !strings.HasSuffix(at, "Z") {
		t.Fatalf("reading time %q", at)
	}

	delete(reading, "at")

	raw, err := json.Marshal(reading)
	if err != nil {
		t.Fatal(err)
	}

	sameJSON(t, string(raw), `{"account_id":"acct-a","quota":{"gemini":0.1},"windows":[
		{"group":"Gemini Models","label":"Weekly Limit Remaining","percent":10,"resets_at":"2026-11-13T23:48:01Z"}]}`)

	// 2. A new conversation goes to the account with the most quota left, on
	// the token Kodizm pushed.
	if got := b.askAccount(t, false, conversation("SCRIPT:account")); got != "tok-b" {
		t.Fatalf("answered by %q", got)
	}

	eventually(t, "the ok outcome", func() bool {
		return slices.ContainsFunc(b.pollOutcomes(t, 0, "").Outcomes, func(raw json.RawMessage) bool {
			return bytes.Contains(raw, []byte(`"account_id":"acct-b"`)) && bytes.Contains(raw, []byte(`"kind":"ok"`))
		})
	})
}
