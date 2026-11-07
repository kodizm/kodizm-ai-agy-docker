package agybridge

import (
	"context"
	"encoding/json"
	"maps"
	"os"
	"strings"
	"testing"
	"time"
)

// realUsage is `agy -p /usage` as agy 1.5.2 printed it.
const realUsage = "Gemini Models\tWeekly Limit Remaining\t100%\t2026-11-13T23:48:01Z\n" +
	"Gemini Models\tFive Hour Limit Remaining\t99%\t2026-11-07T04:48:01Z\n" +
	"Claude and GPT models\tWeekly Limit Remaining\t99%\t2026-11-13T23:48:01Z\n" +
	"Claude and GPT models\tFive Hour Limit Remaining\t98%\t2026-11-07T04:48:01Z\n" +
	"Gemini 4 Argon\tGemini 4 Argon Weekly Limit\t100%\t2026-11-13T23:48:01Z\n"

// weeklyReset is the reset time of the weekly windows in realUsage.
var weeklyReset = time.Date(2026, 11, 13, 23, 48, 1, 0, time.UTC)

func TestParseUsage(t *testing.T) {
	tests := []struct {
		name      string
		out       string
		windows   int
		quota     map[string]float64
		exhausted map[string]time.Time
	}{
		{
			name:      "the real sample",
			out:       realUsage,
			windows:   5,
			quota:     map[string]float64{"gemini": 0.99, "thirdparty": 0.98, "argon": 1},
			exhausted: map[string]time.Time{},
		},
		{
			name: "blank and odd lines",
			out: "\n  \nYour quota:\n" +
				"  Gemini Models \t Weekly Limit Remaining \t 50 % \n" +
				"Gemini Models\tbroken\n" +
				"Gemini Models\tFive Hour Limit Remaining\tabc%\t2026-11-07T04:48:01Z\n" +
				"Gemini Models\tFive Hour Limit Remaining\t150%\n" +
				"Gemini Models\tFive Hour Limit Remaining\t-1\n" +
				"Gemini Models\tFive Hour Limit Remaining\t70\tsoon\n",
			windows:   2,
			quota:     map[string]float64{"gemini": 0.5},
			exhausted: map[string]time.Time{},
		},
		{
			name:      "an exhausted window without a reset",
			out:       "Claude and GPT models\tWeekly Limit Remaining\t0%\n",
			windows:   1,
			quota:     map[string]float64{"thirdparty": 0},
			exhausted: map[string]time.Time{"thirdparty": {}},
		},
		{
			name: "the latest reset among the exhausted windows",
			out: "Gemini Models\tWeekly Limit Remaining\t0%\t2026-11-13T23:48:01Z\n" +
				"Gemini Models\tFive Hour Limit Remaining\t0%\t2026-11-07T04:48:01Z\n" +
				"Gemini Models\tDaily Limit Remaining\t0%\n" +
				"Gemini Models\tMonthly Limit Remaining\t40%\t2026-12-01T00:00:00Z\n",
			windows:   4,
			quota:     map[string]float64{"gemini": 0},
			exhausted: map[string]time.Time{"gemini": weeklyReset},
		},
		{
			name: "the smallest window of a group counts",
			out: "Gemini Models\tWeekly Limit Remaining\t30%\n" +
				"Gemini Models\tFive Hour Limit Remaining\t80%\n" +
				"gemini 4 argon (preview)\tWeekly\t60%\n",
			windows:   3,
			quota:     map[string]float64{"gemini": 0.3, "argon": 0.6},
			exhausted: map[string]time.Time{},
		},
		{
			name:      "an unknown group",
			out:       "Imagen Models\tWeekly Limit Remaining\t10%\n",
			quota:     map[string]float64{},
			exhausted: map[string]time.Time{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseUsage(tt.out)

			if len(got.windows) != tt.windows || !maps.Equal(got.quota, tt.quota) ||
				!maps.EqualFunc(got.exhausted, tt.exhausted, time.Time.Equal) {
				t.Fatalf("parsed %+v", got)
			}
		})
	}
}

func TestParseUsageKeepsTheWindowText(t *testing.T) {
	got := parseUsage(realUsage)
	first := got.windows[0]

	if first.Group != "Gemini Models" || first.Label != "Weekly Limit Remaining" || first.Percent != 100 ||
		!first.ResetsAt.Equal(weeklyReset) {
		t.Fatalf("window %+v", first)
	}
}

func TestParseUsageLogsAnUnknownGroupOnce(t *testing.T) {
	logs := captureLog(t)
	// The name is new to this process, however often the test runs in it.
	out := "Mystery Models " + randomHex(4) + "\tWeekly Limit Remaining\t10%\n"

	parseUsage(out)
	parseUsage(out)

	if n := strings.Count(logs.String(), "unknown quota group"); n != 1 {
		t.Fatalf("logged %d times:\n%s", n, logs.String())
	}
}

// fakeUsage sets what the fake prints for each access token.
func fakeUsage(t *testing.T, outputs map[string]string) {
	t.Helper()

	raw, err := json.Marshal(outputs)
	if err != nil {
		t.Fatal(err)
	}

	t.Setenv(fakeUsageEnv, string(raw))
}

func newTestPoller(t *testing.T, src *testSource) (*usagePoller, *pool, *clock) {
	t.Helper()

	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	p, c := newTestPool(t, src, 2)
	homes := newTestHomes(t)

	t.Cleanup(func() {
		assertProcessesGone(t)
		assertHomesGone(t, homes.base)
	})

	return &usagePoller{bin: self, homes: homes, pool: p, sink: src, done: make(chan struct{})}, p, c
}

func TestUsagePollRanksAndCools(t *testing.T) {
	resets := time.Now().Add(3 * time.Hour).UTC().Truncate(time.Second)

	fakeUsage(t, map[string]string{
		"tok-a": "Gemini Models\tWeekly Limit Remaining\t10%\n",
		"tok-b": "Gemini Models\tWeekly Limit Remaining\t0%\t" + resets.Format(time.RFC3339) + "\n" +
			"Claude and GPT models\tWeekly Limit Remaining\t80%\n",
	})

	src := newTestSource(testAccount("a"), testAccount("b"))
	u, p, _ := newTestPoller(t, src)

	u.pollAll(context.Background())

	// 1. b is drained: skipped before any 429, until its reset.
	if got := lease(t, p, "k", flash); got != "a" {
		t.Fatalf("picked %s", got)
	}

	if until := p.accounts["b"].cooldowns["gemini"]; !until.Equal(resets) {
		t.Fatalf("b cools until %s, want %s", until, resets)
	}

	// 2. Each reading reached the sink.
	readings := src.readings()
	if len(readings) != 2 || readings[0].AccountID != "a" || readings[0].Quota["gemini"] != 0.1 ||
		readings[1].AccountID != "b" || readings[1].Quota["thirdparty"] != 0.8 || len(readings[1].Windows) != 2 {
		t.Fatalf("readings %+v", readings)
	}

	if outcomes := src.outcomes(); len(outcomes) != 0 {
		t.Fatalf("a reading reported outcomes %+v", outcomes)
	}
}

func TestUsagePollDisablesADeadLogin(t *testing.T) {
	logs := captureLog(t)
	fakeUsage(t, map[string]string{"tok-b": realUsage})

	src := newTestSource(testAccount("a"), testAccount("b"))
	u, p, _ := newTestPoller(t, src)

	u.pollAll(context.Background())

	// 1. a's login is dead: only b is left.
	for range 2 {
		if got := lease(t, p, "k", flash); got != "b" {
			t.Fatalf("picked %s", got)
		}
	}

	// 2. The source hears one auth outcome for the whole account.
	eventually(t, "the auth outcome", func() bool {
		outcomes := src.outcomes()

		return len(outcomes) == 1 && outcomes[0].AccountID == "a" && outcomes[0].Kind == kindAuth && outcomes[0].Group == ""
	})

	if got := logs.String(); strings.Contains(got, "tok-") || strings.Contains(got, "@") {
		t.Fatalf("log leaks %q", got)
	}
}

func TestFailedUsagePollKeepsTheLastReading(t *testing.T) {
	src := newTestSource(testAccount("a"))
	u, p, _ := newTestPoller(t, src)
	ctx := context.Background()

	fakeUsage(t, map[string]string{"tok-a": "Gemini Models\tWeekly Limit Remaining\t40%\n"})
	u.pollAll(ctx)

	for _, output := range []string{"fail", "hang", "nothing to see\n"} {
		// Only the hang needs the bound short: a fake under -race starts slowly.
		if output == "hang" {
			setVar(t, &usageTimeout, 2*time.Second)
		}

		fakeUsage(t, map[string]string{"tok-a": output})
		u.pollAll(ctx)

		p.mu.Lock()
		quota := p.accounts["a"].account.Quota["gemini"]
		p.mu.Unlock()

		if quota != 0.4 {
			t.Fatalf("%s: quota %v, want the last reading", output, quota)
		}
	}

	if readings, outcomes := src.readings(), src.outcomes(); len(readings) != 1 || len(outcomes) != 0 {
		t.Fatalf("readings %+v, outcomes %+v", readings, outcomes)
	}

	if got := lease(t, p, "k", flash); got != "a" {
		t.Fatalf("picked %s", got)
	}
}

func TestUsagePollerRunsUntilClose(t *testing.T) {
	setVar(t, &minUsageInterval, time.Millisecond)
	fakeUsage(t, map[string]string{"tok-a": realUsage})

	src := newTestSource(testAccount("a"))
	b := newBridgeWith(t, nil, src, func(cfg *Config) { cfg.UsageInterval = 20 * time.Millisecond })

	eventually(t, "two polls", func() bool { return len(src.readings()) >= 2 })

	b.srv.Close()
	assertNoGoroutines(t)

	polls := len(fakePIDs(t, "usage"))
	time.Sleep(100 * time.Millisecond)

	if after := len(fakePIDs(t, "usage")); after != polls {
		t.Fatalf("%d polls after Close", after-polls)
	}
}

func TestUsageIntervalZeroDisablesThePoller(t *testing.T) {
	setVar(t, &minUsageInterval, time.Millisecond)

	before := len(fakePIDs(t, "usage"))
	b := newBridge(t, map[string]string{"AGY_BRIDGE_USAGE_INTERVAL": "0"})

	if b.srv.usage != nil {
		t.Fatal("a poller runs with the interval at 0")
	}

	b.ask(t, "plain", nil)

	if after := len(fakePIDs(t, "usage")); after != before {
		t.Fatalf("%d polls with the poller off", after-before)
	}
}

func TestNewRefusesAShortUsageInterval(t *testing.T) {
	cfg := Config{MaxConcurrent: 1, MaxParked: 1, PerAccount: 1, UsageInterval: time.Second, Homes: t.TempDir()}

	if _, err := New(cfg, newTestSource()); err == nil || !strings.Contains(err.Error(), "AGY_BRIDGE_USAGE_INTERVAL") {
		t.Fatalf("New = %v", err)
	}

	if err := checkUsageInterval(0); err != nil {
		t.Fatalf("interval 0 must turn the poller off: %v", err)
	}
}
