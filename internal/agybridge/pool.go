package agybridge

import (
	"context"
	"crypto/sha256"
	"fmt"
	"log"
	"maps"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
)

// Pool timings and limits are variables so tests can shorten them.
var (
	// sourceRefreshEvery is how old the account list may get.
	sourceRefreshEvery = 30 * time.Second

	// sourceRetryGap spaces source calls when no account is eligible or the
	// source keeps failing.
	sourceRetryGap = time.Second

	// tokenMargin is how far ahead an access token without a refresh token
	// must expire to be used.
	tokenMargin = 60 * time.Second

	// expirySlack is how close to its expiry an access-only token may be when
	// an auth failure still counts against the account rather than the clock.
	expirySlack = 60 * time.Second

	// resumeMargin is how far ahead the token of a parked turn's agy must
	// expire for the turn to resume; nearer, the turn replays on a fresh one.
	resumeMargin = 2 * time.Minute

	// quotaLadder is the cooldown of a quota failure that named no delay, by
	// the consecutive count of such failures of the account and group.
	quotaLadder = []time.Duration{time.Minute, 5 * time.Minute, 30 * time.Minute, 2 * time.Hour}

	rateCooldown     = 30 * time.Second
	capacityCooldown = 15 * time.Second

	affinityCap = 4096
)

// quotaGroups are the quota buckets a model draws on, as agy reports them:
// Gemini Models, the weekly Gemini 4 Argon limit, Claude and GPT models.
func quotaGroups(base string) []string {
	switch {
	case base == "gemini-4-argon":
		return []string{"gemini", "argon"}
	case strings.HasPrefix(base, "gemini-"):
		return []string{"gemini"}
	}

	return []string{"thirdparty"}
}

// errNoAccount means no account can run the model. With RetryAfter every
// usable one is cooling down until then; without, there is none at all.
type errNoAccount struct {
	RetryAfter time.Duration
}

func (e *errNoAccount) Error() string {
	if e.RetryAfter > 0 {
		return fmt.Sprintf("every agy account is cooling down, for %s", e.RetryAfter)
	}

	return "no agy account available"
}

func (e *errNoAccount) apiError() *apiError {
	if e.RetryAfter > 0 {
		return &apiError{
			status:     http.StatusTooManyRequests,
			kind:       "rate_limit_error",
			message:    "every agy account is cooling down",
			retryAfter: e.RetryAfter,
		}
	}

	return &apiError{status: http.StatusServiceUnavailable, kind: "api_error", message: e.Error()}
}

// accountState is what the pool knows of one account; it outlives the
// account's absence from a refresh, so a lease and a disable stay valid.
type accountState struct {
	account     Account
	fingerprint [sha256.Size]byte
	present     bool
	inflight    int
	lastUsed    time.Time
	cooldowns   map[string]time.Time
	strikes     map[string]int
	// polled is the quota the bridge read from agy itself, per group; it
	// wins over the source's stored reading, which is older.
	polled map[string]float64
	// dead holds the credentials an auth or verification failure condemned;
	// the account stays out until the source hands it different ones.
	dead *[sha256.Size]byte
}

func credentialsOf(a Account) [sha256.Size]byte {
	return sha256.Sum256([]byte(a.AccessToken + "\x00" + a.RefreshToken))
}

type affinityEntry struct {
	id      string
	expires time.Time
}

// availability is why an account can or cannot take a lease right now.
type availability int

const (
	unusable availability = iota
	cooling
	busy
	usable
)

// pool leases accounts to agy sessions: one per-account slot per lease, the
// conversation's account when it can, else the one with the most quota left.
type pool struct {
	source      AccountSource
	perAccount  int
	affinityTTL time.Duration
	now         func() time.Time

	ctx    context.Context
	cancel context.CancelFunc

	mu       sync.Mutex
	accounts map[string]*accountState
	affinity map[string]affinityEntry
	// changed is closed and replaced whenever a slot frees up or the list
	// changes, waking every waiter.
	changed    chan struct{}
	loadedAt   time.Time
	triedAt    time.Time
	refreshing chan struct{}
	// generation counts the pushed lists applied; a refresh that saw another
	// one before it read the source drops what it read.
	generation uint64
}

// newPool builds a pool; its waits end when parent ends or close is called.
func newPool(parent context.Context, source AccountSource, perAccount int, affinityTTL time.Duration) *pool {
	ctx, cancel := context.WithCancel(parent)

	return &pool{
		source:      source,
		perAccount:  perAccount,
		affinityTTL: affinityTTL,
		now:         time.Now,
		ctx:         ctx,
		cancel:      cancel,
		accounts:    map[string]*accountState{},
		affinity:    map[string]affinityEntry{},
		changed:     make(chan struct{}),
	}
}

// close ends every wait for an account slot.
func (p *pool) close() {
	p.cancel()
}

// Lease is one account slot taken for one agy session. A parked session
// gives the slot back and takes it again, always of the same account.
type Lease struct {
	// Account is the account as it was when leased; its token goes to agy.
	Account Account
	// AffinityHit marks a lease that followed the conversation's account.
	AffinityHit bool

	pool  *pool
	state *accountState
	// held is guarded by pool.mu.
	held bool
}

// acquire leases an account for a model drawing on groups. exclude names an
// account not to use (the one that just failed). When every candidate is only
// busy it waits up to wait for a slot and fails with errNoSlot; when none is
// usable it fails with an *errNoAccount.
func (p *pool) acquire(
	ctx context.Context,
	key string,
	groups []string,
	exclude string,
	wait time.Duration,
) (*Lease, error) {
	timer := time.NewTimer(wait)
	defer timer.Stop()

	p.refresh(ctx, false)

	forced := false

	for {
		p.mu.Lock()
		lease, state, retryAfter := p.pickLocked(key, groups, exclude)
		changed := p.changed
		p.mu.Unlock()

		switch {
		case lease != nil:
			return lease, nil
		case state != busy && !forced:
			// The list may be behind: an account may have come back.
			forced = true
			p.refresh(ctx, true)

			continue
		case state != busy:
			return nil, &errNoAccount{RetryAfter: retryAfter}
		}

		select {
		case <-changed:
		case <-timer.C:
			return nil, errNoSlot
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-p.ctx.Done():
			return nil, p.ctx.Err()
		}
	}
}

// pickLocked leases the best usable account, or reports the best state seen:
// busy when some account only lacks a free slot, cooling with the earliest
// time one cools off, else unusable.
func (p *pool) pickLocked(key string, groups []string, exclude string) (*Lease, availability, time.Duration) {
	now := p.now()

	// 1. The conversation's account, while it can take the lease.
	if entry, ok := p.affinity[key]; ok && entry.expires.After(now) && entry.id != exclude {
		if st := p.accounts[entry.id]; st != nil {
			if state, _ := p.availabilityLocked(st, groups, now); state == usable {
				return p.leaseLocked(st, key, true, now), usable, 0
			}
		}
	}

	// 2. Else the most quota left; ties go to the least recently used.
	var (
		best      *accountState
		bestScore float64
		seen      = unusable
		earliest  time.Time
	)

	for _, id := range slices.Sorted(maps.Keys(p.accounts)) {
		st := p.accounts[id]
		if id == exclude {
			continue
		}

		state, until := p.availabilityLocked(st, groups, now)
		seen = max(seen, state)

		switch state {
		case usable:
			score := quotaScore(st.account, groups)
			if best == nil || score > bestScore || (score == bestScore && st.lastUsed.Before(best.lastUsed)) {
				best, bestScore = st, score
			}
		case cooling:
			if earliest.IsZero() || until.Before(earliest) {
				earliest = until
			}
		}
	}

	if best != nil {
		return p.leaseLocked(best, key, false, now), usable, 0
	}

	if seen == cooling {
		return nil, cooling, max(earliest.Sub(now), time.Second)
	}

	return nil, seen, 0
}

// liveLocked reports whether the account is listed, not condemned and holds
// a token agy can still sign in with.
func liveLocked(st *accountState, now time.Time) bool {
	acct := st.account

	switch {
	case !st.present, st.dead != nil && *st.dead == st.fingerprint:
		return false
	case expiresWithin(acct, now, tokenMargin):
		return false
	}

	return true
}

func (p *pool) availabilityLocked(st *accountState, groups []string, now time.Time) (availability, time.Time) {
	acct := st.account

	if !liveLocked(st, now) {
		return unusable, time.Time{}
	}

	var until time.Time

	for _, group := range groups {
		for _, end := range []time.Time{st.cooldowns[group], acct.Cooldowns[group]} {
			if end.After(now) && end.After(until) {
				until = end
			}
		}
	}

	switch {
	case !until.IsZero():
		return cooling, until
	case st.inflight >= p.perAccount:
		return busy, time.Time{}
	}

	return usable, time.Time{}
}

// quotaScore is the smallest known fraction left over the groups.
func quotaScore(acct Account, groups []string) float64 {
	score := 1.0

	for _, group := range groups {
		if left, ok := acct.Quota[group]; ok {
			score = min(score, left)
		}
	}

	return score
}

func (p *pool) leaseLocked(st *accountState, key string, hit bool, now time.Time) *Lease {
	st.inflight++
	st.lastUsed = now
	p.rememberLocked(key, st.account.ID, now)

	return &Lease{Account: st.account, AffinityHit: hit, pool: p, state: st, held: true}
}

// rememberLocked sticks the conversation to the account for another TTL; at
// the cap the entry closest to expiry goes.
func (p *pool) rememberLocked(key, id string, now time.Time) {
	p.affinity[key] = affinityEntry{id: id, expires: now.Add(p.affinityTTL)}

	if len(p.affinity) <= affinityCap {
		return
	}

	maps.DeleteFunc(p.affinity, func(_ string, e affinityEntry) bool { return !e.expires.After(now) })

	for len(p.affinity) > affinityCap {
		oldest := ""

		for k, e := range p.affinity {
			if oldest == "" || e.expires.Before(p.affinity[oldest].expires) {
				oldest = k
			}
		}

		delete(p.affinity, oldest)
	}
}

// signalLocked wakes every waiter.
func (p *pool) signalLocked() {
	close(p.changed)
	p.changed = make(chan struct{})
}

// release gives the account slot back; a lease not holding one is a no-op.
func (l *Lease) release() {
	if l == nil {
		return
	}

	p := l.pool

	p.mu.Lock()
	defer p.mu.Unlock()

	if l.held {
		l.held = false
		l.state.inflight--
		p.signalLocked()
	}
}

// retake takes the same account's slot again, whatever became of the account
// meanwhile: the agy process it serves is already signed in. It waits until
// timeout fires, failing with errNoSlot.
func (l *Lease) retake(ctx, sessionCtx context.Context, timeout <-chan time.Time) error {
	if l == nil {
		return nil
	}

	p := l.pool

	for {
		p.mu.Lock()

		if l.held {
			p.mu.Unlock()

			return nil
		}

		if l.state.inflight < p.perAccount {
			l.state.inflight++
			l.state.lastUsed = p.now()
			l.held = true
			p.mu.Unlock()

			return nil
		}

		changed := p.changed
		p.mu.Unlock()

		select {
		case <-changed:
		case <-timeout:
			return errNoSlot
		case <-ctx.Done():
			return ctx.Err()
		case <-sessionCtx.Done():
			return sessionCtx.Err()
		}
	}
}

// settle records how the leased account fared on a model drawing on groups:
// f nil is a finished session. A cooling failure cools the narrowest group
// down, an auth or verification failure takes the account out for good, an
// expired token changes nothing, a success resets the quota ladder. The
// source hears of it after the pool's lock is released.
func (l *Lease) settle(groups []string, f *failure, detail string) {
	l.pool.settle(l.state, l.Account, groups, f, detail)
}

// classify is f as it bears on the leased account; see pool.classify.
func (l *Lease) classify(f failure) failure {
	return l.pool.classify(l.Account, f)
}

// nearExpiry reports whether the token the lease's agy was spawned with runs
// out within resumeMargin and cannot be refreshed by agy.
func (l *Lease) nearExpiry() bool {
	return expiresWithin(l.Account, l.pool.now(), resumeMargin)
}

// classify turns an auth failure under an access-only token that runs out
// within expirySlack into kindExpired: agy cannot refresh the token it was
// spawned with, so the clock failed, not the login. The client is told 529
// and the next lease takes a fresh token.
func (p *pool) classify(acct Account, f failure) failure {
	if f.Kind == kindAuth && expiresWithin(acct, p.now(), expirySlack) {
		f.Status, f.Kind = statusOverloaded, kindExpired
	}

	return f
}

// expiresWithin reports whether acct holds an access-only token that expires
// within d of now.
func expiresWithin(acct Account, now time.Time, d time.Duration) bool {
	return acct.RefreshToken == "" && !acct.ExpiresAt.After(now.Add(d))
}

// settleAccount settles a verdict found outside a session, such as a dead
// login the usage poll met, as a session under acct would have. Without
// groups the source hears one outcome for the whole account.
func (p *pool) settleAccount(acct Account, groups []string, f *failure, detail string) {
	p.mu.Lock()
	st := p.accounts[acct.ID]
	p.mu.Unlock()

	if st == nil {
		return
	}

	if f != nil {
		classified := p.classify(acct, *f)
		f = &classified
	}

	p.settle(st, acct, groups, f, detail)
}

// settle condemns acct, not whatever the state holds now: a refresh may have
// brought new credentials since.
func (p *pool) settle(st *accountState, acct Account, groups []string, f *failure, detail string) {
	// A token that ran out says nothing of the account: nothing to record or
	// report.
	if f != nil && f.Kind == kindExpired {
		return
	}

	kind := kindOK
	applied := time.Duration(0)

	p.mu.Lock()

	now := p.now()

	if f != nil {
		kind = f.Kind
	}

	switch kind {
	case kindOK:
		for _, group := range groups {
			delete(st.strikes, group)
		}
	case kindQuota, kindRate, kindCapacity:
		// The narrowest group is the one a multi-group model ran out of: a
		// weekly argon limit must not cool, nor report, the gemini group
		// flash shares. A pick still needs every group of the model free.
		if n := len(groups); n > 1 {
			groups = groups[n-1:]
		}

		for _, group := range groups {
			cooldown := f.RetryAfter

			if kind == kindQuota {
				st.strikes[group]++
			}

			if cooldown <= 0 {
				cooldown = fallbackCooldown(kind, st.strikes[group])
			}

			if end := now.Add(cooldown); end.After(st.cooldowns[group]) {
				st.cooldowns[group] = end
			}

			applied = max(applied, cooldown)
		}
	case kindAuth, kindVerification:
		dead := credentialsOf(acct)
		st.dead = &dead
	}

	p.mu.Unlock()

	if f != nil && applied == 0 {
		applied = f.RetryAfter
	}

	if len(groups) == 0 {
		groups = []string{""}
	}

	for _, group := range groups {
		p.source.Report(p.ctx, Outcome{
			AccountID:  acct.ID,
			Group:      group,
			Kind:       kind,
			RetryAfter: applied,
			Detail:     detail,
			At:         now,
		})
	}
}

// pollable lists the accounts worth a usage poll: listed, not condemned, with
// a token agy can sign in with. A cooling account is polled too, to learn
// when its quota is back.
func (p *pool) pollable() []Account {
	p.mu.Lock()
	defer p.mu.Unlock()

	now := p.now()

	var list []Account

	for _, id := range slices.Sorted(maps.Keys(p.accounts)) {
		if st := p.accounts[id]; liveLocked(st, now) {
			list = append(list, st.account)
		}
	}

	return list
}

// setQuota takes a reading agy gave for the account: quota per group, and the
// groups it found exhausted with the latest time one of their windows resets,
// zero when none named one. An exhausted group cools until then, or by the
// quota ladder, so the pool skips it before agy ever meets a 429.
func (p *pool) setQuota(id string, quota map[string]float64, exhausted map[string]time.Time) {
	p.mu.Lock()
	defer p.mu.Unlock()

	st := p.accounts[id]
	if st == nil {
		return
	}

	now := p.now()

	if st.polled == nil {
		st.polled = map[string]float64{}
	}

	maps.Copy(st.polled, quota)
	st.account.Quota = mergedQuota(st.account.Quota, st.polled)

	for group, resets := range exhausted {
		end := resets

		if !end.After(now) {
			st.strikes[group]++
			end = now.Add(fallbackCooldown(kindQuota, st.strikes[group]))
		}

		if end.After(st.cooldowns[group]) {
			st.cooldowns[group] = end
		}
	}
}

// mergedQuota is the source's quota with the polled groups laid over it.
func mergedQuota(source, polled map[string]float64) map[string]float64 {
	merged := maps.Clone(source)
	if merged == nil {
		merged = map[string]float64{}
	}

	maps.Copy(merged, polled)

	return merged
}

func fallbackCooldown(kind failureKind, strikes int) time.Duration {
	switch kind {
	case kindRate:
		return rateCooldown
	case kindCapacity:
		return capacityCooldown
	}

	return quotaLadder[min(max(strikes, 1), len(quotaLadder))-1]
}

// refresh reloads the account list when it is older than sourceRefreshEvery,
// or with force whenever sourceRetryGap has passed since the last try. One
// refresh runs at a time; a caller arriving meanwhile waits for it. A failed
// refresh keeps the last good list.
func (p *pool) refresh(ctx context.Context, force bool) {
	p.mu.Lock()

	now := p.now()
	stale := p.loadedAt.IsZero() || now.Sub(p.loadedAt) >= sourceRefreshEvery

	if running := p.refreshing; running != nil {
		p.mu.Unlock()

		select {
		case <-running:
		case <-ctx.Done():
		}

		return
	}

	if (!stale && !force) || (!p.triedAt.IsZero() && now.Sub(p.triedAt) < sourceRetryGap) {
		p.mu.Unlock()

		return
	}

	running := make(chan struct{})
	p.refreshing, p.triedAt = running, now
	generation := p.generation
	p.mu.Unlock()

	list, err := p.source.Accounts(ctx)

	p.mu.Lock()
	defer p.mu.Unlock()

	p.refreshing = nil
	close(running)

	if err != nil {
		log.Printf("account source: %v; keeping %d known account(s)", err, len(p.accounts))

		return
	}

	// A list pushed meanwhile is newer than the one read here.
	if p.generation != generation {
		return
	}

	p.applyLocked(list)
	p.loadedAt = now
	p.signalLocked()
}

// replace applies a pushed list at once, as a refresh applies a fetched one.
// store runs under the pool's lock, so the source and the pool hold the same
// list even when two pushes race.
func (p *pool) replace(list []Account, store func()) {
	p.mu.Lock()
	defer p.mu.Unlock()

	store()
	p.applyLocked(list)
	p.generation++
	p.loadedAt = p.now()
	p.signalLocked()
}

// applyLocked takes a fresh list: new credentials lift a dead mark, and an
// account missing from the list stays known but out of use.
func (p *pool) applyLocked(list []Account) {
	listed := map[string]bool{}

	for _, acct := range list {
		if !validAccountID(acct.ID) {
			log.Printf("account source: refused an account whose id is not [A-Za-z0-9_.-]: %q", acct.ID)

			continue
		}

		if listed[acct.ID] {
			continue
		}

		listed[acct.ID] = true

		st := p.accounts[acct.ID]
		if st == nil {
			st = &accountState{cooldowns: map[string]time.Time{}, strikes: map[string]int{}}
			p.accounts[acct.ID] = st
		}

		st.account = acct
		st.fingerprint = credentialsOf(acct)

		// The source's quota only seeds a group the bridge has not polled.
		if st.polled != nil {
			st.account.Quota = mergedQuota(acct.Quota, st.polled)
		}
		st.present = true

		if st.dead != nil && *st.dead != st.fingerprint {
			st.dead = nil
		}
	}

	for id, st := range p.accounts {
		st.present = listed[id]
	}
}
