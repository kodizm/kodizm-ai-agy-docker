package agybridge

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"log"
	"math/rand/v2"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Usage poll timings are variables so tests can shorten them.
var (
	defaultUsageInterval = 5 * time.Minute
	minUsageInterval     = 30 * time.Second

	// usageTimeout bounds one `agy -p /usage`. Without a login agy waits up
	// to a minute for a pasted code, so the bound is what ends it.
	usageTimeout = 30 * time.Second
)

const (
	// maxUsageWindows and maxUsageText are what Kodizm accepts per account.
	maxUsageWindows = 20
	maxUsageText    = 120

	usageAuthMarker = "Authentication required"
)

// UsageReading is one account's quota as `agy -p /usage` printed it: the
// fraction left per pool group, the smallest over the group's windows, and
// the windows themselves.
type UsageReading struct {
	AccountID string
	Quota     map[string]float64
	Windows   []UsageWindow
}

// UsageWindow is one line of the output. Group is agy's name for it, such
// as "Gemini Models"; ResetsAt is zero when the line named no time.
type UsageWindow struct {
	Group    string
	Label    string
	Percent  float64
	ResetsAt time.Time
}

func checkUsageInterval(d time.Duration) error {
	if d != 0 && d < minUsageInterval {
		return fmt.Errorf("AGY_BRIDGE_USAGE_INTERVAL must be 0 (off) or at least %s, got %s", minUsageInterval, d)
	}

	return nil
}

// usageOutput is what one output says: the windows, the fraction left per
// pool group, and the groups with a drained window, each with the latest
// time one of those resets (zero when none named one).
type usageOutput struct {
	windows   []UsageWindow
	quota     map[string]float64
	exhausted map[string]time.Time
}

// unknownUsageGroups holds the group names already logged, once per process.
var unknownUsageGroups sync.Map

// parseUsage reads agy's tab separated lines, `group, label, percent[, reset]`,
// leniently: anything else is skipped, an unknown group logged once.
func parseUsage(out string) usageOutput {
	parsed := usageOutput{quota: map[string]float64{}, exhausted: map[string]time.Time{}}

	for line := range strings.Lines(out) {
		fields := strings.Split(strings.TrimSpace(line), "\t")
		if len(fields) < 3 {
			continue
		}

		for i := range fields {
			fields[i] = strings.TrimSpace(fields[i])
		}

		group, known := poolGroup(fields[0])
		if !known {
			noteUnknownGroup(fields[0])

			continue
		}

		percent, err := strconv.ParseFloat(strings.TrimSpace(strings.TrimSuffix(fields[2], "%")), 64)
		if err != nil || !(percent >= 0 && percent <= 100) || len(parsed.windows) == maxUsageWindows {
			continue
		}

		var resets time.Time
		if len(fields) > 3 {
			if t, err := time.Parse(time.RFC3339, fields[3]); err == nil {
				resets = t.UTC()
			}
		}

		parsed.windows = append(parsed.windows, UsageWindow{
			Group:    truncateRunes(fields[0], maxUsageText),
			Label:    truncateRunes(fields[1], maxUsageText),
			Percent:  percent,
			ResetsAt: resets,
		})

		if left, seen := parsed.quota[group]; !seen || percent/100 < left {
			parsed.quota[group] = percent / 100
		}

		if percent == 0 {
			if latest, seen := parsed.exhausted[group]; !seen || resets.After(latest) {
				parsed.exhausted[group] = resets
			}
		}
	}

	return parsed
}

// UsageSink hears the quota readings the bridge takes from agy itself.
type UsageSink interface {
	ReportUsage(ctx context.Context, r UsageReading)
}

// poolGroup maps agy's group name to the pool's quota group.
func poolGroup(name string) (string, bool) {
	switch {
	case strings.Contains(strings.ToLower(name), "argon"):
		return "argon", true
	case strings.EqualFold(name, "Gemini Models"):
		return "gemini", true
	case strings.EqualFold(name, "Claude and GPT models"):
		return "thirdparty", true
	}

	return "", false
}

func noteUnknownGroup(name string) {
	if _, seen := unknownUsageGroups.LoadOrStore(name, true); !seen {
		log.Printf("usage poll: ignored the unknown quota group %q", truncateRunes(name, maxUsageText))
	}
}

// usagePoller runs `agy -p /usage` for every account the pool can use, one
// after the other on its own goroutine: a poll never takes a request slot.
type usagePoller struct {
	bin      string
	interval time.Duration
	homes    *spawnHomes
	pool     *pool
	// sink, when the source is one, hears every reading.
	sink UsageSink

	// cleanupFailed counts removals of a poll's home that failed.
	cleanupFailed int

	done chan struct{}
}

// startUsagePoller polls every interval, plus up to a fifth of it, until ctx
// ends; done closes then.
func startUsagePoller(
	ctx context.Context,
	bin string,
	interval time.Duration,
	homes *spawnHomes,
	p *pool,
	sink UsageSink,
) *usagePoller {
	u := &usagePoller{bin: bin, interval: interval, homes: homes, pool: p, sink: sink, done: make(chan struct{})}

	go u.run(ctx)

	return u
}

func (u *usagePoller) run(ctx context.Context) {
	defer close(u.done)

	for {
		// The jitter keeps a fleet of bridges from polling Google in step.
		timer := time.NewTimer(u.interval + rand.N(u.interval/5+1))

		select {
		case <-ctx.Done():
			timer.Stop()

			return
		case <-timer.C:
		}

		u.pollAll(ctx)
	}
}

func (u *usagePoller) pollAll(ctx context.Context) {
	u.pool.refresh(ctx, false)

	for _, acct := range u.pool.pollable() {
		if ctx.Err() != nil {
			return
		}

		u.poll(ctx, acct)
	}
}

// poll reads one account's quota. A dead login disables the account as a
// failed session would; any other failure keeps the last reading.
func (u *usagePoller) poll(ctx context.Context, acct Account) {
	// 1. agy reads the login from a throwaway home, gone again right after.
	home, err := u.homes.prepare(acct)
	if err != nil {
		log.Printf("usage poll account=%s: %v", acct.ID, err)

		return
	}

	out, runErr := runUsage(ctx, u.bin, home)

	if err := u.homes.remove(home); err != nil {
		u.cleanupFailed++
		log.Printf("usage poll account=%s: remove home (%d failed so far): %v", acct.ID, u.cleanupFailed, err)
	}

	if ctx.Err() != nil {
		return
	}

	// 2. agy asks for a login even when it then times out waiting for one.
	if strings.Contains(out, usageAuthMarker) {
		log.Printf("warn: usage poll account=%s: agy has no valid login; the account is disabled", acct.ID)
		u.pool.settleAccount(acct, nil, &failure{Kind: kindAuth}, "agy /usage: "+usageAuthMarker)

		return
	}

	if runErr != nil {
		log.Printf("usage poll account=%s: %v; keeping the last reading", acct.ID, runErr)

		return
	}

	// 3. The reading is the pool's truth before it is Kodizm's.
	parsed := parseUsage(out)
	if len(parsed.windows) == 0 {
		log.Printf("usage poll account=%s: no quota line in %d bytes; keeping the last reading", acct.ID, len(out))

		return
	}

	u.pool.setQuota(acct.ID, parsed.quota, parsed.exhausted)

	if u.sink != nil {
		u.sink.ReportUsage(ctx, UsageReading{AccountID: acct.ID, Quota: parsed.quota, Windows: parsed.windows})
	}
}

// runUsage runs `agy -p /usage` under home and returns what it printed. agy
// runs in its own process group, so a timeout kills whatever it started too.
func runUsage(ctx context.Context, bin, home string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, usageTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, bin, "-p", "/usage")
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.WaitDelay = agentWaitDelay
	cmd.Env = agentEnv(os.Environ(), home)
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}

	// A nil Stdin is the null device: agy's login prompt reads EOF, never a
	// code. Its output is the reading, whichever stream it uses.
	var out bytes.Buffer

	cmd.Stdout = &out
	cmd.Stderr = &out

	err := cmd.Run()
	if errors.Is(ctx.Err(), context.DeadlineExceeded) {
		err = fmt.Errorf("agy /usage timed out after %s", usageTimeout)
	}

	return out.String(), err
}
