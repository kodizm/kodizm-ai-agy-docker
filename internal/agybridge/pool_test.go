package agybridge

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// testSource is an AccountSource a test steers: the accounts it hands out,
// an error to fail with, and the outcomes it heard.
type testSource struct {
	mu       sync.Mutex
	accounts []Account
	err      error
	calls    int
	reports  []Outcome
	usage    []UsageReading
}

func (s *testSource) ReportUsage(_ context.Context, r UsageReading) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.usage = append(s.usage, r)
}

func (s *testSource) readings() []UsageReading {
	s.mu.Lock()
	defer s.mu.Unlock()

	return slices.Clone(s.usage)
}

func newTestSource(accounts ...Account) *testSource {
	return &testSource{accounts: accounts}
}

func (s *testSource) Accounts(context.Context) ([]Account, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.calls++

	if s.err != nil {
		return nil, s.err
	}

	return slices.Clone(s.accounts), nil
}

func (s *testSource) Report(_ context.Context, o Outcome) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.reports = append(s.reports, o)
}

func (s *testSource) set(err error, accounts ...Account) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.err, s.accounts = err, accounts
}

func (s *testSource) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.calls
}

func (s *testSource) outcomes() []Outcome {
	s.mu.Lock()
	defer s.mu.Unlock()

	return slices.Clone(s.reports)
}

// testAccount is an access-only account, as Kodizm pushes them,
// whose token the fake agy knows as "tok-<id>".
func testAccount(id string) Account {
	return Account{
		ID:          id,
		Email:       id + "@example.test",
		AccessToken: "tok-" + id,
		ExpiresAt:   time.Now().Add(time.Hour),
	}
}

// clock is a settable time for a pool.
type clock struct {
	mu sync.Mutex
	t  time.Time
}

func (c *clock) now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()

	return c.t
}

func (c *clock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.t = c.t.Add(d)
}

func newTestPool(t *testing.T, src AccountSource, perAccount int) (*pool, *clock) {
	t.Helper()

	c := &clock{t: time.Now()}
	p := newPool(context.Background(), src, perAccount, time.Hour)
	p.now = c.now

	t.Cleanup(p.close)

	return p, c
}

var flash = quotaGroups("gemini-3.8-flash")

// lease acquires without waiting and gives the slot straight back, naming
// the account picked.
func lease(t *testing.T, p *pool, key string, groups []string) string {
	t.Helper()

	l, err := p.acquire(context.Background(), key, groups, "", 0)
	if err != nil {
		t.Fatalf("acquire %q: %v", key, err)
	}

	l.release()

	return l.Account.ID
}

func TestQuotaGroups(t *testing.T) {
	tests := map[string][]string{
		"gemini-3.8-flash":  {"gemini"},
		"gemini-4-argon":    {"gemini", "argon"},
		"claude-opus-5-5":   {"thirdparty"},
		"claude-sonnet-5-5": {"thirdparty"},
		"gpt-oss-120b":      {"thirdparty"},
	}

	for base, want := range tests {
		if got := quotaGroups(base); !slices.Equal(got, want) {
			t.Fatalf("quotaGroups(%q) = %v, want %v", base, got, want)
		}
	}
}

func TestPoolPicksTheLeastRecentlyUsed(t *testing.T) {
	p, c := newTestPool(t, newTestSource(testAccount("a"), testAccount("b")), 2)

	var got []string

	for _, key := range []string{"k1", "k2", "k3"} {
		got = append(got, lease(t, p, key, flash))
		c.advance(time.Second)
	}

	if !slices.Equal(got, []string{"a", "b", "a"}) {
		t.Fatalf("picked %v", got)
	}
}

func TestPoolAffinity(t *testing.T) {
	// The clock runs for hours: the tokens must outlive it.
	a, b := testAccount("a"), testAccount("b")
	a.ExpiresAt, b.ExpiresAt = time.Now().Add(24*time.Hour), time.Now().Add(24*time.Hour)

	p, c := newTestPool(t, newTestSource(a, b), 2)

	// 1. A conversation sticks to its account, though another is less used.
	if lease(t, p, "conv", flash) != "a" {
		t.Fatal("first pick")
	}

	c.advance(time.Second)

	l, err := p.acquire(context.Background(), "conv", flash, "", 0)
	if err != nil || l.Account.ID != "a" || !l.AffinityHit {
		t.Fatalf("lease %+v, %v; want a hit on a", l, err)
	}

	l.release()

	// 2. Every use refreshes the TTL.
	c.advance(50 * time.Minute)

	if lease(t, p, "conv", flash) != "a" {
		t.Fatal("affinity lost inside the TTL")
	}

	c.advance(50 * time.Minute)

	if lease(t, p, "conv", flash) != "a" {
		t.Fatal("affinity lost although the last use refreshed it")
	}

	// 3. Past the TTL it is a miss, and the least used account is picked.
	c.advance(61 * time.Minute)

	l, err = p.acquire(context.Background(), "conv", flash, "", 0)
	if err != nil || l.Account.ID != "b" || l.AffinityHit {
		t.Fatalf("lease %+v, %v; want a miss on b", l, err)
	}

	l.release()
}

func TestPoolAffinityCapDropsTheOldest(t *testing.T) {
	setVar(t, &affinityCap, 2)

	p, c := newTestPool(t, newTestSource(testAccount("a")), 2)

	for _, key := range []string{"k1", "k2", "k3"} {
		lease(t, p, key, flash)
		c.advance(time.Second)
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if _, ok := p.affinity["k1"]; ok || len(p.affinity) != 2 {
		t.Fatalf("affinity %v, want k2 and k3", p.affinity)
	}
}

func TestPoolPrefersTheMostQuotaLeft(t *testing.T) {
	a, b := testAccount("a"), testAccount("b")
	a.Quota = map[string]float64{"gemini": 0.1, "argon": 0.9}
	b.Quota = map[string]float64{"argon": 0.3}

	p, _ := newTestPool(t, newTestSource(a, b), 2)

	tests := []struct {
		base string
		want string
	}{
		// a has 0.1 of gemini left, b an unknown amount, which counts as all.
		{base: "gemini-3.8-flash", want: "b"},
		// The smallest of a model's groups counts: a 0.1, b 0.3.
		{base: "gemini-4-argon", want: "b"},
		// Neither names the group: the least recently used wins.
		{base: "claude-opus-5-5", want: "a"},
	}

	for _, tt := range tests {
		if got := lease(t, p, tt.base, quotaGroups(tt.base)); got != tt.want {
			t.Fatalf("%s: picked %s, want %s", tt.base, got, tt.want)
		}
	}
}

func TestPoolCooldowns(t *testing.T) {
	src := newTestSource(testAccount("a"), testAccount("b"))
	p, c := newTestPool(t, src, 2)
	ctx := context.Background()

	settle := func(id string, f *failure) {
		t.Helper()

		l, err := p.acquire(ctx, "settle-"+id, flash, map[string]string{"a": "b", "b": "a"}[id], 0)
		if err != nil || l.Account.ID != id {
			t.Fatalf("lease for %s: %+v, %v", id, l, err)
		}

		l.release()
		l.settle(flash, f, "detail")
	}

	// 1. A quota failure without a delay cools the group for the ladder's
	// first step; another group stays free.
	settle("a", &failure{Kind: kindQuota})

	if got := lease(t, p, "x", flash); got != "b" {
		t.Fatalf("picked %s during a's cooldown", got)
	}

	if got := lease(t, p, "y", quotaGroups("claude-opus-5-5")); got != "a" {
		t.Fatalf("thirdparty picked %s, want a: only gemini cools", got)
	}

	// 2. With both cooling the pool names the earliest end.
	settle("b", &failure{Kind: kindRate, RetryAfter: 10 * time.Second})

	_, err := p.acquire(ctx, "z", flash, "", 0)

	var none *errNoAccount
	if !errors.As(err, &none) || none.RetryAfter != 10*time.Second {
		t.Fatalf("acquire = %v, want every account cooling for 10s", err)
	}

	// 3. A cooldown ends.
	c.advance(11 * time.Second)

	if got := lease(t, p, "z", flash); got != "b" {
		t.Fatalf("picked %s after b's cooldown", got)
	}

	// 4. Consecutive quota failures climb the ladder; a success resets it.
	c.advance(time.Minute)
	settle("a", &failure{Kind: kindQuota})

	if until := p.accounts["a"].cooldowns["gemini"]; until.Sub(c.now()) != 5*time.Minute {
		t.Fatalf("second quota failure cools for %s", until.Sub(c.now()))
	}

	c.advance(5 * time.Minute)
	settle("a", nil)
	settle("a", &failure{Kind: kindQuota})

	if until := p.accounts["a"].cooldowns["gemini"]; until.Sub(c.now()) != time.Minute {
		t.Fatalf("a quota failure after a success cools for %s", until.Sub(c.now()))
	}

	// 5. Capacity has its own fallback; a source's cooldown counts too.
	c.advance(time.Minute)
	settle("a", &failure{Kind: kindCapacity})

	if until := p.accounts["a"].cooldowns["gemini"]; until.Sub(c.now()) != capacityCooldown {
		t.Fatalf("capacity cools for %s", until.Sub(c.now()))
	}

	c.advance(capacityCooldown)

	b := testAccount("b")
	b.Cooldowns = map[string]time.Time{"gemini": c.now().Add(time.Hour)}
	src.set(nil, testAccount("a"), b)
	c.advance(sourceRefreshEvery)

	if got := lease(t, p, "w", flash); got != "a" {
		t.Fatalf("picked %s while the source cools b", got)
	}
}

func TestPoolDisablesADeadAccountUntilItsCredentialsChange(t *testing.T) {
	for _, kind := range []failureKind{kindAuth, kindVerification} {
		t.Run(string(kind), func(t *testing.T) {
			src := newTestSource(testAccount("a"))
			p, c := newTestPool(t, src, 2)

			l, err := p.acquire(context.Background(), "k", flash, "", 0)
			if err != nil {
				t.Fatal(err)
			}

			l.release()
			l.settle(flash, &failure{Kind: kind}, "")

			// 1. Gone, through any number of refreshes with the same token.
			for range 3 {
				c.advance(sourceRefreshEvery)

				_, err := p.acquire(context.Background(), "k", flash, "", 0)

				var none *errNoAccount
				if !errors.As(err, &none) || none.RetryAfter != 0 || none.apiError().status != http.StatusServiceUnavailable {
					t.Fatalf("acquire = %v, want no account", err)
				}
			}

			// 2. New credentials bring it back.
			fresh := testAccount("a")
			fresh.AccessToken = "tok-a2"
			src.set(nil, fresh)
			c.advance(sourceRefreshEvery)

			if got := lease(t, p, "k", flash); got != "a" {
				t.Fatalf("picked %s", got)
			}
		})
	}
}

func TestPoolTokenExpiry(t *testing.T) {
	short := testAccount("short")
	short.ExpiresAt = time.Now().Add(30 * time.Second)

	refreshable := testAccount("refreshable")
	refreshable.ExpiresAt = time.Now().Add(-time.Hour)
	refreshable.RefreshToken = "1//r"

	p, _ := newTestPool(t, newTestSource(short, refreshable), 2)

	for range 2 {
		if got := lease(t, p, "k", flash); got != "refreshable" {
			t.Fatalf("picked %s, want only the refreshable account", got)
		}
	}
}

func TestPoolWithoutAccounts(t *testing.T) {
	invalid := testAccount("../evil")

	p, _ := newTestPool(t, newTestSource(invalid), 2)

	_, err := p.acquire(context.Background(), "k", flash, "", 0)

	var none *errNoAccount
	if !errors.As(err, &none) || none.RetryAfter != 0 {
		t.Fatalf("acquire = %v, want no account", err)
	}
}

func TestPoolWaitsForABusyAccount(t *testing.T) {
	p, _ := newTestPool(t, newTestSource(testAccount("a")), 1)
	ctx := context.Background()

	held, err := p.acquire(ctx, "k", flash, "", 0)
	if err != nil {
		t.Fatal(err)
	}

	// 1. The only slot is taken: the wait runs out.
	if _, err := p.acquire(ctx, "k", flash, "", 50*time.Millisecond); !errors.Is(err, errNoSlot) {
		t.Fatalf("acquire = %v, want errNoSlot", err)
	}

	// 2. A release inside the wait hands the slot over.
	go func() {
		time.Sleep(20 * time.Millisecond)
		held.release()
	}()

	l, err := p.acquire(ctx, "k", flash, "", time.Second)
	if err != nil {
		t.Fatal(err)
	}

	l.release()
}

func TestPoolRefreshesTheList(t *testing.T) {
	src := newTestSource(testAccount("a"))
	p, c := newTestPool(t, src, 2)

	// 1. Within the refresh interval the list is not fetched again.
	lease(t, p, "k", flash)
	lease(t, p, "k", flash)

	if n := src.callCount(); n != 1 {
		t.Fatalf("%d source calls, want 1", n)
	}

	// 2. A failing source keeps the last good list.
	src.set(errors.New("down"))
	c.advance(sourceRefreshEvery)

	if got := lease(t, p, "k", flash); got != "a" || src.callCount() != 2 {
		t.Fatalf("picked %s after a failed refresh, %d calls", got, src.callCount())
	}

	// 3. A new account shows up after the next interval.
	src.set(nil, testAccount("b"))
	c.advance(sourceRefreshEvery)

	if got := lease(t, p, "other", flash); got != "b" {
		t.Fatalf("picked %s, want the only listed account", got)
	}

	// 4. With nothing eligible the pool asks again at once, at most every
	// sourceRetryGap.
	src.set(nil)
	c.advance(sourceRefreshEvery)

	if _, err := p.acquire(context.Background(), "k", flash, "", 0); err == nil {
		t.Fatal("acquired from an empty list")
	}

	src.set(nil, testAccount("c"))
	c.advance(sourceRetryGap)

	if got := lease(t, p, "k", flash); got != "c" {
		t.Fatalf("picked %s, want the account the on-demand refresh found", got)
	}
}

func TestPoolReportsOutcomes(t *testing.T) {
	src := newTestSource(testAccount("a"))
	p, c := newTestPool(t, src, 2)
	argon := quotaGroups("gemini-4-argon")

	l, err := p.acquire(context.Background(), "k", argon, "", 0)
	if err != nil {
		t.Fatal(err)
	}

	l.release()

	failed := c.now()
	l.settle(argon, &failure{Kind: kindQuota, RetryAfter: 30 * time.Second}, "quota banner")
	c.advance(time.Minute)
	l.settle(argon, nil, "")

	// A cooling failure names only the narrowest group; each outcome carries
	// the time it was settled, not the time it is sent.
	want := []Outcome{
		{AccountID: "a", Group: "argon", Kind: kindQuota, RetryAfter: 30 * time.Second, Detail: "quota banner", At: failed},
		{AccountID: "a", Group: "gemini", Kind: kindOK, At: failed.Add(time.Minute)},
		{AccountID: "a", Group: "argon", Kind: kindOK, At: failed.Add(time.Minute)},
	}

	eventually(t, "the outcomes to reach the source", func() bool {
		return slices.Equal(src.outcomes(), want)
	})
}

// A weekly argon limit must not lock flash on the account: a cooling failure
// of a model on several groups cools only the narrowest, while a pick still
// needs every group of the model free.
func TestCoolingFailureCoolsOnlyTheNarrowestGroup(t *testing.T) {
	argon := quotaGroups("gemini-4-argon")

	for _, kind := range []failureKind{kindQuota, kindRate, kindCapacity} {
		t.Run(string(kind), func(t *testing.T) {
			src := newTestSource(testAccount("a"))
			p, _ := newTestPool(t, src, 2)

			l, err := p.acquire(context.Background(), "k", argon, "", 0)
			if err != nil {
				t.Fatal(err)
			}

			l.release()
			l.settle(argon, &failure{Kind: kind, RetryAfter: 120 * time.Hour}, "")

			p.mu.Lock()
			cooldowns := maps.Clone(p.accounts["a"].cooldowns)
			p.mu.Unlock()

			if _, gemini := cooldowns["gemini"]; gemini || cooldowns["argon"].IsZero() {
				t.Fatalf("cooldowns %v, want argon only", cooldowns)
			}

			if got := lease(t, p, "flash", flash); got != "a" {
				t.Fatalf("flash picked %s", got)
			}

			if _, err := p.acquire(context.Background(), "argon", argon, "", 0); err == nil {
				t.Fatal("argon leased while its group cools")
			}

			eventually(t, "the one outcome", func() bool {
				got := src.outcomes()

				return len(got) == 1 && got[0].Group == "argon" && got[0].Kind == kind
			})
		})
	}
}

// An access-only token that ran out inside a long turn says nothing about the
// account: the auth failure becomes an expired one, which neither cools, nor
// disables, nor reaches the source.
func TestAuthFailureOnARunOutTokenIsExpired(t *testing.T) {
	refreshable := testAccount("r")
	refreshable.RefreshToken = "1//r"

	tests := []struct {
		name    string
		account Account
		left    time.Duration
		want    failureKind
	}{
		{name: "token ran out", account: testAccount("a"), left: 30 * time.Second, want: kindExpired},
		{name: "token past its expiry", account: testAccount("a"), left: -time.Minute, want: kindExpired},
		{name: "token well ahead", account: testAccount("a"), left: 10 * time.Minute, want: kindAuth},
		{name: "refresh token", account: refreshable, left: -time.Minute, want: kindAuth},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			src := newTestSource()
			p, c := newTestPool(t, src, 2)

			acct := tt.account
			acct.ExpiresAt = c.now().Add(2 * time.Hour)
			src.set(nil, acct)

			l, err := p.acquire(context.Background(), "k", flash, "", 0)
			if err != nil {
				t.Fatal(err)
			}

			l.release()
			c.advance(2*time.Hour - tt.left)

			got := l.classify(failure{Status: http.StatusBadGateway, Kind: kindAuth})
			if got.Kind != tt.want {
				t.Fatalf("kind %s, want %s", got.Kind, tt.want)
			}

			if tt.want != kindExpired {
				return
			}

			if got.Status != statusOverloaded {
				t.Fatalf("status %d, want 529", got.Status)
			}

			l.settle(flash, &got, "Authentication required")
			l.settle(quotaGroups("claude-opus-5-5"), nil, "")

			p.mu.Lock()
			st := p.accounts[acct.ID]
			dead, cooling := st.dead != nil, len(st.cooldowns) > 0
			p.mu.Unlock()

			if dead || cooling {
				t.Fatalf("dead %t, cooling %t: an expired token must leave the account alone", dead, cooling)
			}

			// The ok outcome settled after it is the only one the source hears.
			eventually(t, "the ok outcome", func() bool {
				got := src.outcomes()

				return len(got) == 1 && got[0].Kind == kindOK
			})
		})
	}
}

// End to end: agy refusing a run-out token moves the request to another
// account with a 200, and the account keeps taking work.
func TestExpiredTokenRetriesWithoutDisablingTheAccount(t *testing.T) {
	setVar(t, &expirySlack, 2*time.Hour)
	fakeAccounts(t, `{"tok-a":"auth","tok-b":"ok"}`)

	src := newTestSource(testAccount("a"), testAccount("b"))
	b := newBridgeWith(t, nil, src)

	for i := range 2 {
		if got := b.askAccount(t, false, conversation("SCRIPT:account "+strings.Repeat("x", i))); got != "tok-b" {
			t.Fatalf("request %d answered by %q", i, got)
		}
	}

	b.waitIdle(t)

	if n := b.spawnedBy(t, "tok-a"); n != 2 {
		t.Fatalf("the expired account ran %d times, want it tried again", n)
	}

	eventually(t, "b's ok outcome", func() bool {
		return slices.ContainsFunc(src.outcomes(), func(o Outcome) bool { return o.AccountID == "b" })
	})

	if slices.ContainsFunc(src.outcomes(), func(o Outcome) bool { return o.AccountID == "a" }) {
		t.Fatalf("outcomes %+v: an expired token reached the source", src.outcomes())
	}
}

func TestExpiredTokenOnTheOnlyAccountAnswers529(t *testing.T) {
	setVar(t, &expirySlack, 2*time.Hour)
	fakeAccounts(t, `{"tok-a":"auth"}`)

	b := newBridge(t, nil)

	for range 2 {
		resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
		expectOverloaded(t, resp)
	}

	if n := b.spawned(t); n != 2 {
		t.Fatalf("agy started %d times, want the account kept", n)
	}

	b.waitIdle(t)
}

func TestPolledQuotaOutlivesASourceRefresh(t *testing.T) {
	a, b := testAccount("a"), testAccount("b")
	a.Quota = map[string]float64{"gemini": 0.9, "thirdparty": 0.4}
	b.Quota = map[string]float64{"gemini": 0.5}

	src := newTestSource(a, b)
	p, c := newTestPool(t, src, 2)

	// 1. The source's quota seeds the pool: a has more gemini left.
	if got := lease(t, p, "k1", flash); got != "a" {
		t.Fatalf("picked %s, want a by the source's quota", got)
	}

	// 2. A poll finds a nearly drained; a refresh carrying Kodizm's older
	// stored reading must not undo it, per group.
	p.setQuota("a", map[string]float64{"gemini": 0.1}, nil)
	c.advance(sourceRefreshEvery)

	if got := lease(t, p, "k2", flash); got != "b" || src.callCount() != 2 {
		t.Fatalf("picked %s after %d source calls, want b by the polled quota", got, src.callCount())
	}

	p.mu.Lock()
	quota := p.accounts["a"].account.Quota
	p.mu.Unlock()

	if quota["gemini"] != 0.1 || quota["thirdparty"] != 0.4 {
		t.Fatalf("quota %v, want the polled gemini and the source's thirdparty", quota)
	}

	// 3. A polled zero cools the group until the latest reset; without a
	// reset the quota ladder decides.
	resets := c.now().Add(3 * time.Hour)
	p.setQuota("b", map[string]float64{"gemini": 0}, map[string]time.Time{"gemini": resets})

	if until := p.accounts["b"].cooldowns["gemini"]; !until.Equal(resets) {
		t.Fatalf("b cools until %s, want %s", until, resets)
	}

	p.setQuota("a", map[string]float64{"gemini": 0}, map[string]time.Time{"gemini": {}})

	if until := p.accounts["a"].cooldowns["gemini"]; until.Sub(c.now()) != quotaLadder[0] {
		t.Fatalf("a cools for %s, want the ladder's first step", until.Sub(c.now()))
	}

	if _, err := p.acquire(context.Background(), "k3", flash, "", 0); err == nil {
		t.Fatal("a drained group was leased")
	}

	// 4. An unknown account is ignored.
	p.setQuota("ghost", map[string]float64{"gemini": 1}, nil)
}

// The account scenarios below run requests through the bridge and the fake
// agy, which acts under each access token as FAKE_AGY_ACCOUNTS says.

func fakeAccounts(t *testing.T, accounts string) {
	t.Helper()

	t.Setenv(fakeAccountsEnv, accounts)
}

func (b *bridge) askAccount(t *testing.T, stream bool, messages []map[string]any) string {
	t.Helper()

	resp := b.post(t, map[string]any{"model": "gemini-3.8-flash", "stream": stream, "messages": messages})
	expectStatus(t, resp, http.StatusOK)

	if stream {
		return streamedText(readSSE(t, resp.Body))
	}

	return decodeJSON[messageResponse](t, resp).Content[0].Text
}

func conversation(first string, turns ...string) []map[string]any {
	messages := userText(first)

	for _, turn := range turns {
		messages = append(messages,
			map[string]any{"role": "assistant", "content": "ok"},
			map[string]any{"role": "user", "content": turn},
		)
	}

	return messages
}

func TestConversationSticksToOneAccount(t *testing.T) {
	logs := captureLog(t)
	b := newBridgeWith(t, nil, newTestSource(testAccount("a"), testAccount("b")))

	first := b.askAccount(t, false, conversation("SCRIPT:account one"))
	again := b.askAccount(t, false, conversation("SCRIPT:account one", "SCRIPT:account"))
	other := b.askAccount(t, false, conversation("SCRIPT:account two"))

	if first != "tok-a" || again != first || other != "tok-b" {
		t.Fatalf("accounts %q, %q, %q; want the conversation on a and the next one on b", first, again, other)
	}

	b.waitIdle(t)

	if line := logs.String(); strings.Count(line, "affinity=hit") != 1 || strings.Count(line, "affinity=miss") != 2 {
		t.Fatalf("want one hit and two misses:\n%s", line)
	}
}

func TestQuotaFailureMovesToAnotherAccount(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "buffered", true: "stream"}[stream], func(t *testing.T) {
			logs := captureLog(t)
			fakeAccounts(t, `{"tok-a":"quota","tok-b":"ok"}`)
			b := newBridgeWith(t, nil, newTestSource(testAccount("a"), testAccount("b")))

			// 1. a fails on quota before any output: the request survives on b.
			if got := b.askAccount(t, stream, conversation("SCRIPT:account")); got != "tok-b" {
				t.Fatalf("answered by %q", got)
			}

			// 2. New work goes to b while a cools down, and so does the
			// conversation, which moved with the retry.
			if got := b.askAccount(t, stream, conversation("SCRIPT:account other")); got != "tok-b" {
				t.Fatalf("new conversation answered by %q", got)
			}

			if got := b.askAccount(t, stream, conversation("SCRIPT:account", "SCRIPT:account")); got != "tok-b" {
				t.Fatalf("the moved conversation answered by %q", got)
			}

			b.waitIdle(t)

			if a, bb := b.spawnedBy(t, "tok-a"), b.spawnedBy(t, "tok-b"); a != 1 || bb != 3 {
				t.Fatalf("agy started %d times on a, %d on b", a, bb)
			}

			if line := logs.String(); !strings.Contains(line, " account=b attempts=2 ") ||
				!strings.Contains(line, "agy failure kind=quota") {
				t.Fatalf("want the retry and the warned banner in the log:\n%s", line)
			}
		})
	}
}

func TestResponsesNameTheAccountThatRan(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "buffered", true: "stream"}[stream], func(t *testing.T) {
			body := map[string]any{"model": "gemini-3.8-flash", "stream": stream, "messages": userText("SCRIPT:account")}

			b := newBridgeWith(t, nil, newTestSource(testAccount("a")))

			resp := b.post(t, body)
			expectStatus(t, resp, http.StatusOK)

			if got := resp.Header.Get(accountHeader); got != "a" {
				t.Fatalf("%s = %q, want a", accountHeader, got)
			}

			b.waitIdle(t)
		})
	}
}

func TestRetriedTurnNamesTheAccountThatAnswered(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "buffered", true: "stream"}[stream], func(t *testing.T) {
			// a fails on quota before any output, so b answers.
			fakeAccounts(t, `{"tok-a":"quota","tok-b":"ok"}`)
			b := newBridgeWith(t, nil, newTestSource(testAccount("a"), testAccount("b")))

			resp := b.post(t, map[string]any{
				"model":    "gemini-3.8-flash",
				"stream":   stream,
				"messages": userText("SCRIPT:account"),
			})
			expectStatus(t, resp, http.StatusOK)

			if got := resp.Header.Get(accountHeader); got != "b" {
				t.Fatalf("%s = %q, want b", accountHeader, got)
			}

			b.waitIdle(t)

			if n := b.spawnedBy(t, "tok-a"); n != 1 {
				t.Fatalf("a ran %d times, want the failed attempt", n)
			}
		})
	}
}

func TestNoAccountLeasedNoAccountHeader(t *testing.T) {
	fakeAccounts(t, `{"tok-a":"verification"}`)
	b := newBridge(t, nil)

	// 1. The first request ran on a, which failed.
	resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	expectStatus(t, resp, http.StatusBadGateway)

	if got := resp.Header.Get(accountHeader); got != "a" {
		t.Fatalf("%s = %q, want the account that failed", accountHeader, got)
	}

	// 2. Then none is left: no session ran, so nothing names an account.
	resp = b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	expectStatus(t, resp, http.StatusServiceUnavailable)

	if got := resp.Header.Values(accountHeader); len(got) != 0 {
		t.Fatalf("%s = %q on a response without a session", accountHeader, got)
	}
}

func TestDeadAccountsAreNeverRetried(t *testing.T) {
	for _, kind := range []string{"auth", "verification"} {
		t.Run(kind, func(t *testing.T) {
			fakeAccounts(t, `{"tok-a":"`+kind+`","tok-b":"ok"}`)
			b := newBridgeWith(t, nil, newTestSource(testAccount("a"), testAccount("b")))

			for i := range 3 {
				if got := b.askAccount(t, false, conversation("SCRIPT:account "+strings.Repeat("x", i))); got != "tok-b" {
					t.Fatalf("request %d answered by %q", i, got)
				}
			}

			b.waitIdle(t)

			if n := b.spawnedBy(t, "tok-a"); n != 1 {
				t.Fatalf("the dead account ran %d times", n)
			}
		})
	}
}

func TestADeadOnlyAccountLeavesNoneAvailable(t *testing.T) {
	fakeAccounts(t, `{"tok-a":"verification"}`)
	b := newBridge(t, nil)

	resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	expectStatus(t, resp, http.StatusBadGateway)

	resp = b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	expectStatus(t, resp, http.StatusServiceUnavailable)

	if got := decodeJSON[errorEnvelope](t, resp); got.Error.Message != "no agy account available" {
		t.Fatalf("error body %+v", got)
	}

	if n := b.spawned(t); n != 1 {
		t.Fatalf("agy started %d times", n)
	}
}

func TestAllAccountsCoolingAnswers429WithThePoolsRetryAfter(t *testing.T) {
	fakeAccounts(t, `{"tok-a":"quota"}`)
	b := newBridge(t, nil)

	// 1. The banner itself is the cause: the fixed Retry-After.
	resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	expectStatus(t, resp, http.StatusTooManyRequests)

	if got := resp.Header.Get("Retry-After"); got != "5" {
		t.Fatalf("retry-after %q", got)
	}

	// 2. Then the pool is: its cooldown is the Retry-After.
	for _, stream := range []bool{false, true} {
		resp = b.post(t, map[string]any{"model": "x", "stream": stream, "messages": userText("SCRIPT:plain")})
		expectStatus(t, resp, http.StatusTooManyRequests)

		if got := resp.Header.Get("Retry-After"); got != "60" {
			t.Fatalf("retry-after %q, want the ladder's first step", got)
		}

		if got := decodeJSON[errorEnvelope](t, resp); got.Error.Type != "rate_limit_error" {
			t.Fatalf("error body %+v", got)
		}
	}

	if n := b.spawned(t); n != 1 {
		t.Fatalf("agy started %d times", n)
	}
}

func TestNoAccountsAnswers503(t *testing.T) {
	b := newBridgeWith(t, nil, newTestSource())

	resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	expectStatus(t, resp, http.StatusServiceUnavailable)

	if n := b.spawned(t); n != 0 {
		t.Fatalf("agy started %d times", n)
	}
}

func TestPerAccountCapQueues(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_PER_ACCOUNT": "1"}, func(cfg *Config) {
		cfg.SlotWait = time.Second
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 1. A hanging stream takes the account's only slot; the server has more.
	hanging := b.do(t, ctx, http.MethodPost, "/v1/messages", map[string]any{
		"model":    "x",
		"stream":   true,
		"messages": userText("SCRIPT:hang"),
	}, nil)
	waitForEvent(t, bufio.NewReader(hanging.Body), "content_block_delta")

	// 2. A second request waits the slot wait out and is told 529.
	started := time.Now()
	expectOverloaded(t, b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")}))

	if waited := time.Since(started); waited < time.Second || waited > 3*time.Second {
		t.Fatalf("account wait took %s", waited)
	}

	// 3. One waiting while the slot frees up runs.
	done := make(chan *http.Response, 1)

	go func() {
		done <- b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	}()

	time.Sleep(50 * time.Millisecond)
	cancel()

	if got := decodeJSON[messageResponse](t, <-done); got.Content[0].Text != "Hello, world" {
		t.Fatalf("queued request answered %+v", got)
	}
}

// A parked turn gives its account slot back and resumes on the same account,
// waiting for that account's slot as for a server slot.
func TestParkedTurnKeepsItsAccount(t *testing.T) {
	b := newBridgeWith(t, map[string]string{"AGY_BRIDGE_PER_ACCOUNT": "1"},
		newTestSource(testAccount("a"), testAccount("b")),
		func(cfg *Config) { cfg.SlotWait = 200 * time.Millisecond },
	)
	first := userText("SCRIPT:tool")

	_, calls, _, _ := b.turn(t, false, first)

	// 1. The same conversation, so the same account, hangs while the turn is
	// parked: the parked turn gave the account's only slot back.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	hanging := b.do(t, ctx, http.MethodPost, "/v1/messages", map[string]any{
		"model":    "x",
		"stream":   true,
		"tools":    readFileTool,
		"messages": append(slices.Clone(first), map[string]any{"role": "assistant", "content": "x"}, userText("SCRIPT:hang")[0]),
	}, nil)
	waitForEvent(t, bufio.NewReader(hanging.Body), "content_block_delta")

	if a := b.spawnedBy(t, "tok-a"); a != 2 {
		t.Fatalf("agy started %d times on a, want the turn and the hang", a)
	}

	// 2. b is free, but the continuation needs a's slot: 529, still parked.
	messages := append(first, assistantCalls("Let me look.", calls), toolResultMessage(map[string]string{calls[0].id: "ok"}))
	body := map[string]any{"model": "x", "tools": readFileTool, "messages": messages}

	expectOverloaded(t, b.post(t, body))

	if _, held := b.srv.reg.size(); held != 1 {
		t.Fatalf("held calls after a failed resume: %d, want 1", held)
	}

	// 3. Once a is free the turn resumes on it.
	cancel()

	if text, _, _, _ := b.turn(t, false, messages); text != "Tool said: ok" {
		t.Fatalf("continuation answered %q", text)
	}

	if a, bb := b.spawnedBy(t, "tok-a"), b.spawnedBy(t, "tok-b"); a != 2 || bb != 0 {
		t.Fatalf("agy started %d times on a, %d on b", a, bb)
	}
}

func TestNoRetryForAContinuation(t *testing.T) {
	b := newBridgeWith(t, nil, newTestSource(testAccount("a"), testAccount("b")))
	first := userText("SCRIPT:tool429")

	_, calls, _, _ := b.turn(t, false, first)
	messages := append(first, assistantCalls("Let me look.", calls), toolResultMessage(map[string]string{calls[0].id: "ok"}))

	resp := b.post(t, map[string]any{"model": "x", "tools": readFileTool, "messages": messages})
	expectStatus(t, resp, http.StatusTooManyRequests)

	if n := b.spawned(t); n != 1 {
		t.Fatalf("agy started %d times, want no retry", n)
	}

	b.waitIdle(t)
}

func TestNoRetryAfterOutput(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(map[bool]string{false: "buffered", true: "stream"}[stream], func(t *testing.T) {
			b := newBridgeWith(t, nil, newTestSource(testAccount("a"), testAccount("b")))

			resp := b.post(t, map[string]any{"model": "x", "stream": stream, "messages": userText("SCRIPT:text429")})

			if stream {
				expectStatus(t, resp, http.StatusOK)

				events := readSSE(t, resp.Body)
				if last := events[len(events)-1]; last.name != "error" || streamedText(events) != "partial" {
					t.Fatalf("stream %v, text %q", eventNames(events), streamedText(events))
				}
			} else {
				expectStatus(t, resp, http.StatusTooManyRequests)
			}

			if n := b.spawned(t); n != 1 {
				t.Fatalf("agy started %d times, want no retry", n)
			}

			b.waitIdle(t)
		})
	}
}

// A capacity failure says nothing about the account: no other is tried.
func TestNoRetryOnCapacity(t *testing.T) {
	fakeAccounts(t, `{"tok-a":"capacity","tok-b":"ok"}`)
	b := newBridgeWith(t, nil, newTestSource(testAccount("a"), testAccount("b")))

	resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	expectStatus(t, resp, statusOverloaded)

	if n := b.spawned(t); n != 1 {
		t.Fatalf("agy started %d times", n)
	}

	b.waitIdle(t)
}

func TestSpawnHomeOfARequest(t *testing.T) {
	b := newBridge(t, nil)

	var first, second fakeHome

	for _, home := range []*fakeHome{&first, &second} {
		raw := b.ask(t, "home", nil)
		if err := json.Unmarshal([]byte(raw), home); err != nil {
			t.Fatal(err)
		}
	}

	if first.Home == second.Home || first.TokenMode != "-rw-------" || first.HasRefresh ||
		first.AuthMethod != "consumer" || first.Email != "a@example.test" ||
		first.InstallationID == "" || first.InstallationID != second.InstallationID {
		t.Fatalf("homes %+v and %+v", first, second)
	}

	b.waitIdle(t)
	assertHomesGone(t, b.homes)
}
