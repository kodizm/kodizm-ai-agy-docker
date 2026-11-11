package agybridge

import (
	"net/http"
	"strings"
	"testing"
	"time"
)

// silence is short enough for a test; the watchdog polls at a quarter of it.
const testSilence = 300 * time.Millisecond

func watched(cfg *Config) { cfg.SilenceTimeout = testSilence }

func TestSilentAgyIsKilledAndTheRequestGetsA529(t *testing.T) {
	logs := captureLog(t)
	b := newBridge(t, nil, watched)

	started := time.Now()
	resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:silent")})
	expectStatus(t, resp, statusOverloaded)

	if got := decodeJSON[errorEnvelope](t, resp); got.Error.Type != "overloaded_error" {
		t.Fatalf("error body = %+v", got)
	}

	if took := time.Since(started); took < testSilence || took > 10*testSilence {
		t.Fatalf("answered after %s, want about the silence of %s", took, testSilence)
	}

	// The process and both slots are gone: a hung agy no longer waits for the client to leave.
	b.waitIdle(t)

	// No second agy: the kill is neither an account failure nor a reason to try another account.
	if n := b.spawned(t); n != 1 {
		t.Fatalf("agy started %d times, want 1", n)
	}

	eventually(t, "the request line to carry the reason", func() bool {
		return strings.Contains(logs.String(), " reason=agy_silent ")
	})
}

// Past the early head a status can no longer be sent, so a stream learns of the kill as an error event.
func TestSilentAgyOnAStreamEndsItAsAnErrorEvent(t *testing.T) {
	setVar(t, &firstByteAfter, testSilence/6)

	logs := captureLog(t)
	b := newBridge(t, nil, watched)

	resp := b.post(t, map[string]any{"model": "x", "stream": true, "messages": userText("SCRIPT:silent")})
	expectStatus(t, resp, http.StatusOK)

	events := readSSE(t, resp.Body)
	names := eventNames(events)

	if names[0] != "message_start" || names[len(names)-1] != "error" ||
		events[len(events)-1].data["error"].(map[string]any)["type"] != "overloaded_error" {
		t.Fatalf("stream %v", names)
	}

	b.waitIdle(t)

	if n := b.spawned(t); n != 1 {
		t.Fatalf("agy started %d times, want 1", n)
	}

	eventually(t, "the request line to carry the reason", func() bool {
		return strings.Contains(logs.String(), " reason=agy_silent ")
	})
}

// The clock starts at the resume: a turn parked for longer than the silence is not killed on arrival, and an agy
// that goes quiet after it is.
func TestSilentResumedTurnIsKilled(t *testing.T) {
	setVar(t, &batchWindow, 20*time.Millisecond)

	b := newBridge(t, nil, watched)
	first := userText("SCRIPT:toolsilent")

	_, calls, stop, _ := b.turn(t, false, first)
	if stop != "tool_use" || len(calls) != 1 {
		t.Fatalf("stop %q, calls %+v", stop, calls)
	}

	time.Sleep(2 * testSilence)

	messages := append(first, assistantCalls("", calls), toolResultMessage(map[string]string{calls[0].id: "file body"}))

	started := time.Now()
	resp := b.post(t, map[string]any{"model": "x", "tools": readFileTool, "messages": messages})
	expectStatus(t, resp, statusOverloaded)

	if took := time.Since(started); took < testSilence || took > 10*testSilence {
		t.Fatalf("killed %s after the resume, want about the silence of %s", took, testSilence)
	}

	b.waitIdle(t)
}

func TestAgyThatKeepsTalkingIsNotKilled(t *testing.T) {
	b := newBridge(t, nil, watched)

	// The turn runs for 800 ms, well past the silence, but never goes quiet for that long.
	text, _, stop, _ := b.turn(t, false, userText("SCRIPT:drip"))

	if stop != "end_turn" || text != "........" {
		t.Fatalf("stop %q, text %q", stop, text)
	}
}

// A parked turn waits for the client, not for agy, and the clock starts over when the results arrive.
func TestParkedTurnIsNotWatched(t *testing.T) {
	// The batch window is part of the turn before it parks; the setting's floor of a second clears it, a
	// test's 300 ms does not.
	setVar(t, &batchWindow, 20*time.Millisecond)

	b := newBridge(t, nil, watched)
	first := userText("SCRIPT:tool")

	_, calls, stop, _ := b.turn(t, false, first)
	if stop != "tool_use" || len(calls) != 1 {
		t.Fatalf("stop %q, calls %+v", stop, calls)
	}

	time.Sleep(3 * testSilence)

	messages := append(first, assistantCalls("", calls), toolResultMessage(map[string]string{calls[0].id: "file body"}))
	text, _, stop, _ := b.turn(t, false, messages)

	if stop != "end_turn" || !strings.Contains(text, "Tool said: file body") {
		t.Fatalf("stop %q, text %q", stop, text)
	}

	if n := b.spawned(t); n != 1 {
		t.Fatalf("agy started %d times, want the one held turn", n)
	}
}

func TestResumeRestartsTheSilenceClock(t *testing.T) {
	s := newSession(t.Context(), newRegistry(), make(chan struct{}, 1), time.Minute)
	a := &agent{}

	a.touch()

	time.Sleep(2 * testSilence)

	if quiet := s.quiet(a); quiet < testSilence {
		t.Fatalf("quiet for %s with no sign of life, want it counted", quiet)
	}

	if !s.resumeFromPark() {
		t.Fatal("the turn could not resume")
	}

	if quiet := s.quiet(a); quiet > testSilence/2 {
		t.Fatalf("quiet for %s right after the resume, want the clock restarted", quiet)
	}
}

func TestLoadConfigSilenceTimeout(t *testing.T) {
	cfg, err := LoadConfig(func(string) string { return "" })
	if err != nil || cfg.SilenceTimeout != 180*time.Second {
		t.Fatalf("default: %v, %s", err, cfg.SilenceTimeout)
	}

	cfg, err = LoadConfig(func(k string) string { return map[string]string{"AGY_BRIDGE_SILENCE_TIMEOUT": "45"}[k] })
	if err != nil || cfg.SilenceTimeout != 45*time.Second {
		t.Fatalf("override: %v, %s", err, cfg.SilenceTimeout)
	}

	for _, bad := range []string{"0", "-1", "soon"} {
		if _, err := LoadConfig(func(k string) string { return map[string]string{"AGY_BRIDGE_SILENCE_TIMEOUT": bad}[k] }); err == nil {
			t.Fatalf("AGY_BRIDGE_SILENCE_TIMEOUT=%q must be refused", bad)
		}
	}
}
