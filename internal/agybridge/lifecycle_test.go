package agybridge

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestContinuationWithExtraContentReplays(t *testing.T) {
	b := newBridge(t, nil)
	first := userText("SCRIPT:tool")

	_, calls, _, _ := b.turn(t, false, first)
	parked := livePIDs(t)

	// The user interrupts: the last message carries a new instruction beside
	// the result, which the held turn could never see.
	last := toolResultMessage(map[string]string{calls[0].id: "file body"})
	last["content"] = append(last["content"].([]map[string]any), map[string]any{
		"type": "text",
		"text": "Stop. SCRIPT:prompt new instruction",
	})

	text, _, stop, _ := b.turn(t, false, append(first, assistantCalls("Let me look.", calls), last))

	for _, want := range []string{"<conversation_history>", "[tool_result " + calls[0].id + "]\nfile body", "new instruction"} {
		if !strings.Contains(text, want) {
			t.Fatalf("replayed prompt lacks %q:\n%s", want, text)
		}
	}

	if stop != "end_turn" || b.spawned(t) != 2 {
		t.Fatalf("stop %q, agy started %d times", stop, b.spawned(t))
	}

	eventually(t, "the interrupted agy to exit", func() bool {
		for _, pid := range livePIDs(t) {
			for _, old := range parked {
				if pid == old {
					return false
				}
			}
		}

		return true
	})

	b.waitIdle(t)
}

var webFetchTool = map[string]any{"name": "web_fetch", "input_schema": map[string]any{"type": "object"}}

// Claude Code loads a deferred tool through ToolSearch, so the request after a
// tool call can carry a tool the held agy turn never listed. That turn cannot
// call it: the bridge ends it and replays with the current tools. The limits
// are one slot and one parked turn, so the old turn must be gone before the
// new one is admitted.
func TestToolSetGrowthReplaysInAFreshSession(t *testing.T) {
	logs := captureLog(t)
	limits := map[string]string{"AGY_BRIDGE_MAX_CONCURRENT": "1", "AGY_BRIDGE_MAX_PARKED": "1"}
	b := newBridge(t, limits, func(cfg *Config) {
		cfg.SlotWait = 200 * time.Millisecond
	})
	first := userText("SCRIPT:tool")

	_, calls, _, _ := b.turn(t, false, first)
	parked := livePIDs(t)

	// The result names the script the replayed session runs: it reports the
	// tools agy listed.
	messages := append(first,
		assistantCalls("Let me look.", calls),
		toolResultMessage(map[string]string{calls[0].id: "SCRIPT:tools"}),
	)
	grown := append(slices.Clone(readFileTool), webFetchTool)

	resp := b.post(t, map[string]any{"model": "x", "tools": grown, "messages": messages})
	expectStatus(t, resp, http.StatusOK)

	got := decodeJSON[messageResponse](t, resp)
	if got.Content[0].Text != "read_file,web_fetch" || got.StopReason != "end_turn" {
		t.Fatalf("grown request answered %+v", got)
	}

	if n := b.spawned(t); n != 2 {
		t.Fatalf("agy started %d times, want 2 (the held turn and the replay)", n)
	}

	eventually(t, "the held agy to exit", func() bool {
		return !slices.ContainsFunc(livePIDs(t), func(pid int) bool { return slices.Contains(parked, pid) })
	})

	b.waitIdle(t)

	if line := logs.String(); !strings.Contains(line, "tools=2 ") || !strings.Contains(line, "continued=false ") ||
		!strings.Contains(line, "tools_grew=true ") {
		t.Fatalf("want the growth replay in the request line:\n%s", line)
	}
}

// Claude Code's compaction sends the conversation with every tool and a reminder that none may be called. A model
// that is offered the tools calls one anyway, the response ends as tool_use and the compaction reads that as an
// empty summary, so a request carrying the reminder runs without the client's tools.
func TestNoToolsReminderRunsWithoutClientTools(t *testing.T) {
	logs := captureLog(t)
	b := newBridge(t, nil)
	reminder := "\n\nREMINDER: Do NOT call any tools. Respond with plain text only. Tool calls will be rejected."

	// The same request without the reminder lists the tool.
	resp := b.post(t, map[string]any{"model": "x", "tools": readFileTool, "messages": userText("SCRIPT:tools")})
	expectStatus(t, resp, http.StatusOK)

	if got := decodeJSON[messageResponse](t, resp); got.Content[0].Text != "read_file" {
		t.Fatalf("control request listed %+v, want read_file", got.Content)
	}

	// Claude Code appends a system message after the last user message on every step.
	withNote := append(userText("SCRIPT:tools"+reminder), map[string]any{"role": "system", "content": "a note from the harness"})

	resp = b.post(t, map[string]any{"model": "x", "tools": readFileTool, "messages": withNote})
	expectStatus(t, resp, http.StatusOK)

	got := decodeJSON[messageResponse](t, resp)
	for _, block := range got.Content {
		if strings.Contains(block.Text, "read_file") {
			t.Fatalf("a request with the reminder still listed the tool: %+v", got.Content)
		}
	}

	if got.StopReason != "end_turn" {
		t.Fatalf("stop_reason = %q, want end_turn", got.StopReason)
	}

	b.waitIdle(t)

	if line := logs.String(); !strings.Contains(line, "tools=0 ") {
		t.Fatalf("want tools=0 in the request line of the reminder request:\n%s", line)
	}
}

// Every half minute Claude Code forks a running background agent for a progress label: the whole history, the same
// tools (denied on its own side) and a request for three to five words. Through agy that replays the history into a
// new session, which the model answers with a tool call that nobody takes, so the session parks until the cap
// evicts it. The label comes from the agent's last tool call instead: no session starts, and the live turn the
// fork's tool result shares an id with stays parked and resumes.
func TestProgressLabelRequestStartsNoAgy(t *testing.T) {
	logs := captureLog(t)
	b := newBridge(t, nil)

	first := userText("SCRIPT:tool label")

	_, held, _, _ := b.turn(t, false, first)
	if len(held) != 1 {
		t.Fatalf("the first turn held %d calls, want 1", len(held))
	}

	// The fork's history ends in the live turn's tool result; the prompt joins that user message.
	read := []toolUse{{id: held[0].id, name: "Read", input: `{"file_path":"/repo/internal/render.go"}`}}
	fork := lateResult(first, read, "ok")
	last := fork[len(fork)-1]
	last["content"] = append(last["content"].([]map[string]any), map[string]any{"type": "text", "text": progressPrompt})
	fork = append(fork, map[string]any{"role": "system", "content": "a note from the harness"})

	for _, stream := range []bool{false, true} {
		text, calls, stop, use := b.turn(t, stream, fork)

		if text != "Reading render.go" || len(calls) != 0 || stop != "end_turn" {
			t.Fatalf("stream=%t: label request answered %q %+v stop %q, want the label and end_turn", stream, text, calls, stop)
		}

		// A gateway reads a reply without usage as a lost one and books the whole history as its input.
		if use.InputTokens < 1 || use.OutputTokens < 1 {
			t.Fatalf("stream=%t: usage %+v, want the label's own tokens", stream, use)
		}
	}

	if n := b.spawned(t); n != 1 {
		t.Fatalf("agy started %d times, want only the live turn's", n)
	}

	if sessions, calls := b.srv.reg.size(); sessions != 1 || calls != 1 {
		t.Fatalf("registry holds %d sessions and %d calls, want the live turn's one each", sessions, calls)
	}

	// The same prompt in a request that carries no tools is someone's own prompt: it goes on to agy.
	resp := b.post(t, map[string]any{"model": "x", "messages": []map[string]any{{"role": "user", "content": []map[string]any{
		{"type": "text", "text": "SCRIPT:plain"},
		{"type": "text", "text": progressPrompt},
	}}}})
	expectStatus(t, resp, http.StatusOK)

	if got := decodeJSON[messageResponse](t, resp); got.Content[0].Text != "Hello, world" {
		t.Fatalf("a toolless request with the prompt answered %+v, want agy's reply", got.Content)
	}

	if n := b.spawned(t); n != 2 {
		t.Fatalf("agy started %d times, want the live turn's and the toolless request's", n)
	}

	// The live turn still resumes with its result.
	if text, _, _, _ := b.turn(t, false, lateResult(first, held, "ok")); text != "Tool said: ok" {
		t.Fatalf("the live turn answered %q after the label requests", text)
	}

	if n := b.spawned(t); n != 2 {
		t.Fatalf("the continuation started agy: %d spawns, want the same two", n)
	}

	b.waitIdle(t)

	if line := logs.String(); !strings.Contains(line, `outcome="end_turn local"`) {
		t.Fatalf("want the local answer in the request lines:\n%s", line)
	}
}

// agy cannot refresh the access-only token it was spawned with: a parked turn
// whose token is about to run out is not resumed but replayed into a fresh
// session, which takes a fresh token. A login that refreshes itself resumes.
func TestResumeOnANearlyExpiredTokenReplays(t *testing.T) {
	refreshable := testAccount("a")
	refreshable.RefreshToken = "1//r"

	tests := []struct {
		name     string
		account  Account
		spawned  int
		replayed bool
	}{
		{name: "access only", account: testAccount("a"), spawned: 2, replayed: true},
		{name: "refresh token", account: refreshable, spawned: 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Every token of the test is inside the margin.
			setVar(t, &resumeMargin, 2*time.Hour)

			logs := captureLog(t)
			b := newBridgeWith(t, nil, newTestSource(tt.account))
			first := userText("SCRIPT:tool")

			_, calls, _, _ := b.turn(t, false, first)

			messages := append(first,
				assistantCalls("Let me look.", calls),
				toolResultMessage(map[string]string{calls[0].id: "SCRIPT:tools"}),
			)

			text, _, stop, _ := b.turn(t, false, messages)

			want := "Tool said: SCRIPT:tools"
			if tt.replayed {
				want = "read_file"
			}

			if text != want || stop != "end_turn" {
				t.Fatalf("continuation answered %q, stop %q; want %q", text, stop, want)
			}

			if n := b.spawned(t); n != tt.spawned {
				t.Fatalf("agy started %d times, want %d", n, tt.spawned)
			}

			b.waitIdle(t)

			line := logs.String()
			if strings.Contains(line, "token_near_expiry=true ") != tt.replayed ||
				strings.Contains(line, "continued=true ") == tt.replayed {
				t.Fatalf("request lines:\n%s", line)
			}
		})
	}
}

// Only growth reopens the session: the same tools, or fewer, resume the held turn.
func TestUnchangedOrShrunkToolSetResumes(t *testing.T) {
	tests := []struct {
		name  string
		tools any
	}{
		{name: "equal", tools: readFileTool},
		{name: "shrunk", tools: []map[string]any{}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logs := captureLog(t)
			b := newBridge(t, nil)
			first := userText("SCRIPT:tool")

			_, calls, _, _ := b.turn(t, false, first)

			messages := append(first,
				assistantCalls("Let me look.", calls),
				toolResultMessage(map[string]string{calls[0].id: "ok"}),
			)
			resp := b.post(t, map[string]any{"model": "x", "tools": tt.tools, "messages": messages})
			expectStatus(t, resp, http.StatusOK)

			if got := decodeJSON[messageResponse](t, resp); got.Content[0].Text != "Tool said: ok" {
				t.Fatalf("continuation answered %+v", got)
			}

			if n := b.spawned(t); n != 1 {
				t.Fatalf("agy started %d times, want once for the whole turn", n)
			}

			b.waitIdle(t)

			if strings.Contains(logs.String(), "tools_grew=true") || !strings.Contains(logs.String(), "continued=true ") {
				t.Fatalf("want a plain continuation in the request line:\n%s", logs.String())
			}
		})
	}
}

// lateResult is the request a client sends when it finally answers a call.
func lateResult(first []map[string]any, calls []toolUse, result string) []map[string]any {
	return append(slices.Clone(first),
		assistantCalls("Let me look.", calls),
		toolResultMessage(map[string]string{calls[0].id: result}),
	)
}

// Abandoned tool_use turns must not lock fresh requests out: at the cap the
// oldest parked turn goes and the new request is served.
func TestParkedCapEvictsTheOldestTurn(t *testing.T) {
	b := newBridge(t, nil)

	// 1. The cap fills with parked turns, the first the oldest.
	firsts := make([][]map[string]any, b.srv.cfg.MaxParked)
	held := make([][]toolUse, b.srv.cfg.MaxParked)

	for i := range firsts {
		firsts[i] = userText("SCRIPT:tool turn " + strconv.Itoa(i))
		_, held[i], _, _ = b.turn(t, false, firsts[i])
	}

	// 2. A fresh request at the cap is answered, not refused.
	for _, stream := range []bool{false, true} {
		if text, _, stop, _ := b.turn(t, stream, userText("SCRIPT:plain")); text != "Hello, world" || stop != "end_turn" {
			t.Fatalf("fresh turn at the cap answered %q, stop %q", text, stop)
		}
	}

	if n := b.srv.reg.parked(); n != b.srv.cfg.MaxParked-1 {
		t.Fatalf("%d turns parked after the eviction, want %d", n, b.srv.cfg.MaxParked-1)
	}

	// 3. The evicted turn's late result replays on a fresh session.
	spawned := b.spawned(t)

	if text, _, stop, _ := b.turn(t, false, lateResult(firsts[0], held[0], "SCRIPT:plain")); text != "Hello, world" ||
		stop != "end_turn" {
		t.Fatalf("late result of the evicted turn answered %q, stop %q", text, stop)
	}

	if n := b.spawned(t); n != spawned+1 {
		t.Fatalf("agy started %d times for the replay, want 1", n-spawned)
	}

	// 4. The younger turns still resume.
	if text, _, _, _ := b.turn(t, false, lateResult(firsts[1], held[1], "ok")); text != "Tool said: ok" {
		t.Fatalf("a younger parked turn answered %q", text)
	}

	if n := b.spawned(t); n != spawned+1 {
		t.Fatalf("the continuation started agy: %d spawns after the replay", n-spawned-1)
	}
}

func TestEvictedTurnReplays(t *testing.T) {
	logs := captureLog(t)
	limits := map[string]string{"AGY_BRIDGE_MAX_CONCURRENT": "1", "AGY_BRIDGE_MAX_PARKED": "1"}
	b := newBridge(t, limits, func(cfg *Config) {
		cfg.SlotWait = 200 * time.Millisecond
	})

	// 1. The second turn evicts the first at the cap of one.
	firstA, firstB := userText("SCRIPT:tool a"), userText("SCRIPT:tool b")
	_, callsA, _, _ := b.turn(t, false, firstA)
	_, callsB, _, _ := b.turn(t, false, firstB)

	if n := b.srv.reg.parked(); n != 1 {
		t.Fatalf("%d turns parked, want only the second", n)
	}

	// 2. The first one's late result replays, evicting the second in turn.
	if text, _, stop, _ := b.turn(t, false, lateResult(firstA, callsA, "SCRIPT:plain")); text != "Hello, world" ||
		stop != "end_turn" {
		t.Fatalf("replay answered %q, stop %q", text, stop)
	}

	// 3. The second one replays too, and nothing holds a slot or a call.
	if text, _, _, _ := b.turn(t, false, lateResult(firstB, callsB, "SCRIPT:plain")); text != "Hello, world" {
		t.Fatalf("second replay answered %q", text)
	}

	b.waitIdle(t)

	if n := b.spawned(t); n != 4 {
		t.Fatalf("agy started %d times, want 4 (two turns, two replays)", n)
	}

	if held := len(b.srv.slots); held != 0 {
		t.Fatalf("%d server slots leaked", held)
	}

	if line := logs.String(); strings.Count(line, "orphan_results=1 ") != 2 || strings.Contains(line, "continued=true ") {
		t.Fatalf("want two replays and no continuation in the request lines:\n%s", line)
	}
}

// A result and an eviction for the same parked turn arrive together: the turn
// either resumes or is evicted and replayed, never both and never neither.
func TestEvictionRacingResume(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_MAX_PARKED": "1"})

	for i := range 8 {
		spawned := b.spawned(t)
		first := userText("SCRIPT:tool race " + strconv.Itoa(i))
		_, calls, _, _ := b.turn(t, false, first)

		resumed := b.postAsync(map[string]any{
			"model":    "x",
			"tools":    readFileTool,
			"messages": lateResult(first, calls, "SCRIPT:plain"),
		})
		fresh := b.postAsync(map[string]any{"model": "x", "tools": readFileTool, "messages": userText("SCRIPT:plain")})

		late, evictor := <-resumed, <-fresh

		if evictor.err != nil || evictor.status != http.StatusOK || evictor.text != "Hello, world" {
			t.Fatalf("round %d: fresh request got %+v", i, evictor)
		}

		// One spawn for the turn itself and one for the fresh request; a
		// replay is a third.
		wantSpawns := map[string]int{"Tool said: SCRIPT:plain": 2, "Hello, world": 3}[late.text]
		if late.err != nil || late.status != http.StatusOK || wantSpawns == 0 {
			t.Fatalf("round %d: late result got %+v", i, late)
		}

		if n := b.spawned(t) - spawned; n != wantSpawns {
			t.Fatalf("round %d: late result answered %q with %d spawns, want %d", i, late.text, n, wantSpawns)
		}

		b.waitIdle(t)

		if parked, held := b.srv.reg.parked(), len(b.srv.slots); parked != 0 || held != 0 {
			t.Fatalf("round %d: %d parked turns and %d slots left", i, parked, held)
		}
	}
}

type asyncReply struct {
	status int
	text   string
	err    error
}

// postAsync sends a Messages request from another goroutine, so two can race.
func (b *bridge) postAsync(body any) <-chan asyncReply {
	reply := make(chan asyncReply, 1)

	go func() {
		raw, err := json.Marshal(body)
		if err != nil {
			reply <- asyncReply{err: err}

			return
		}

		resp, err := b.ts.Client().Post(b.ts.URL+"/v1/messages", "application/json", bytes.NewReader(raw))
		if err != nil {
			reply <- asyncReply{err: err}

			return
		}

		defer resp.Body.Close()

		var got messageResponse

		out := asyncReply{status: resp.StatusCode, err: json.NewDecoder(resp.Body).Decode(&got)}
		if len(got.Content) > 0 {
			out.text = got.Content[0].Text
		}

		reply <- out
	}()

	return reply
}

func TestSlotWaitAnswers529BeforeAnyByte(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_MAX_CONCURRENT": "1"}, func(cfg *Config) {
		cfg.SlotWait = 200 * time.Millisecond
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	hanging := b.do(t, ctx, http.MethodPost, "/v1/messages", map[string]any{
		"model":    "x",
		"stream":   true,
		"messages": userText("SCRIPT:hang"),
	}, nil)
	waitForEvent(t, bufio.NewReader(hanging.Body), "content_block_delta")

	started := time.Now()
	resp := b.post(t, map[string]any{"model": "x", "stream": true, "messages": userText("SCRIPT:plain")})
	expectOverloaded(t, resp)

	if waited := time.Since(started); waited < 200*time.Millisecond || waited > 2*time.Second {
		t.Fatalf("slot wait took %s", waited)
	}
}

// A fresh request that cannot get a server slot must not cost a parked turn its
// life: a parked turn holds no slot, so ending one frees nothing the request can
// use, and a client retrying its 529 would empty the parked turns for nothing.
func TestSlotWaitDoesNotEvictAParkedTurn(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_MAX_PARKED": "1"}, func(cfg *Config) {
		cfg.SlotWait = 200 * time.Millisecond
	})

	first := userText("SCRIPT:tool parked")
	_, held, _, _ := b.turn(t, false, first)

	// Every server slot is busy: the fresh request waits and is refused.
	for range cap(b.srv.slots) {
		b.srv.slots <- struct{}{}
	}

	expectOverloaded(t, b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")}))

	for range cap(b.srv.slots) {
		<-b.srv.slots
	}

	if n := b.srv.reg.parked(); n != 1 {
		t.Fatalf("%d turns parked after the refused request, want the one it must not evict", n)
	}

	if text, _, _, _ := b.turn(t, false, lateResult(first, held, "ok")); text != "Tool said: ok" {
		t.Fatalf("the parked turn answered %q after the refusal", text)
	}
}

func TestParkedTurnReleasesItsSlot(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_MAX_CONCURRENT": "1"}, func(cfg *Config) {
		cfg.SlotWait = 200 * time.Millisecond
	})
	first := userText("SCRIPT:tool")

	_, calls, _, _ := b.turn(t, false, first)

	// 1. The parked turn leaves the only slot to a new one.
	if text, _, _, _ := b.turn(t, false, userText("SCRIPT:plain")); text != "Hello, world" {
		t.Fatalf("new turn beside a parked one answered %q", text)
	}

	// 2. The parked turn takes the slot back to finish.
	messages := append(first, assistantCalls("Let me look.", calls), toolResultMessage(map[string]string{calls[0].id: "ok"}))
	if text, _, _, _ := b.turn(t, false, messages); text != "Tool said: ok" {
		t.Fatalf("continuation answered %q", text)
	}
}

func TestContinuationWithoutASlotStaysParked(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_MAX_CONCURRENT": "1"}, func(cfg *Config) {
		cfg.SlotWait = 200 * time.Millisecond
	})
	first := userText("SCRIPT:tool")

	_, calls, _, _ := b.turn(t, false, first)

	ctx, cancel := context.WithCancel(context.Background())
	hanging := b.do(t, ctx, http.MethodPost, "/v1/messages", map[string]any{
		"model":    "x",
		"stream":   true,
		"messages": userText("SCRIPT:hang"),
	}, nil)
	waitForEvent(t, bufio.NewReader(hanging.Body), "content_block_delta")

	messages := append(first, assistantCalls("Let me look.", calls), toolResultMessage(map[string]string{calls[0].id: "ok"}))
	body := map[string]any{"model": "x", "stream": true, "tools": readFileTool, "messages": messages}

	// 1. No slot: 529, and the held call is still there to resume.
	expectOverloaded(t, b.post(t, body))

	if _, held := b.srv.reg.size(); held != 1 {
		t.Fatalf("held calls after a failed resume: %d, want 1", held)
	}

	// 2. Once the slot frees up, the same request resumes the same turn.
	cancel()

	if text, _, _, _ := b.turn(t, true, messages); text != "Tool said: ok" {
		t.Fatalf("continuation answered %q", text)
	}

	if n := b.spawned(t); n != 2 {
		t.Fatalf("agy started %d times, want 2 (the turn and the hang)", n)
	}
}

func TestBannerAfterContinuation(t *testing.T) {
	t.Run("before any byte", func(t *testing.T) {
		b := newBridge(t, nil)
		first := userText("SCRIPT:tool429")

		_, calls, _, _ := b.turn(t, true, first)
		messages := append(first, assistantCalls("Let me look.", calls), toolResultMessage(map[string]string{calls[0].id: "ok"}))

		resp := b.post(t, map[string]any{"model": "x", "stream": true, "tools": readFileTool, "messages": messages})
		expectStatus(t, resp, http.StatusTooManyRequests)

		if resp.Header.Get("Retry-After") != "5" {
			t.Fatalf("retry-after %q", resp.Header.Get("Retry-After"))
		}

		b.waitIdle(t)
	})

	t.Run("mid stream", func(t *testing.T) {
		b := newBridge(t, nil)
		first := userText("SCRIPT:tooltext429")

		_, calls, _, _ := b.turn(t, true, first)
		messages := append(first, assistantCalls("Let me look.", calls), toolResultMessage(map[string]string{calls[0].id: "ok"}))

		resp := b.post(t, map[string]any{"model": "x", "stream": true, "tools": readFileTool, "messages": messages})
		expectStatus(t, resp, http.StatusOK)

		events := readSSE(t, resp.Body)
		last := events[len(events)-1]

		if last.name != "error" || last.data["error"].(map[string]any)["type"] != "rate_limit_error" {
			t.Fatalf("stream ended with %v", eventNames(events))
		}

		if text := streamedText(events); strings.Contains(text, "⚠") || text != "Working." {
			t.Fatalf("banner leaked as text: %q", text)
		}

		b.waitIdle(t)
	})
}

func TestCallbacksUseAUnixSocket(t *testing.T) {
	b := newBridge(t, nil)

	// 1. The socket sits in a directory only this user can enter.
	info, err := os.Stat(filepath.Dir(b.srv.socket))
	if err != nil || info.Mode().Perm() != 0o700 {
		t.Fatalf("socket dir: %v, %v", info, err)
	}

	if info, err := os.Stat(b.srv.socket); err != nil || info.Mode().Type() != os.ModeSocket {
		t.Fatalf("socket: %v, %v", info, err)
	}

	// 2. A tool round trip works with no TCP listener at all and a client
	// address that is not loopback: the callback never depends on either.
	first := userText("SCRIPT:tool")
	got := serveDirect(t, b.srv, map[string]any{"model": "x", "tools": readFileTool, "messages": first})

	if got.StopReason != "tool_use" {
		t.Fatalf("first answer %+v", got)
	}

	calls := []toolUse{{id: got.Content[1].ID, name: got.Content[1].Name, input: string(got.Content[1].Input)}}
	messages := append(first, assistantCalls("Let me look.", calls), toolResultMessage(map[string]string{calls[0].id: "via socket"}))

	got = serveDirect(t, b.srv, map[string]any{"model": "x", "tools": readFileTool, "messages": messages})

	if got.Content[0].Text != "Tool said: via socket" {
		t.Fatalf("continuation answered %+v", got)
	}
}

func serveDirect(t *testing.T, srv *Server, body any) messageResponse {
	t.Helper()

	raw, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(raw))
	req.RemoteAddr = "203.0.113.7:4711"
	rec := httptest.NewRecorder()

	srv.ServeHTTP(rec, req)

	var got messageResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil || rec.Code != http.StatusOK {
		t.Fatalf("status %d: %s", rec.Code, rec.Body)
	}

	return got
}

func TestDisconnectInsideTheBatchWindow(t *testing.T) {
	setVar(t, &batchWindow, time.Second)

	b := newBridge(t, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	resp := b.do(t, ctx, http.MethodPost, "/v1/messages", map[string]any{
		"model":    "x",
		"stream":   true,
		"tools":    readFileTool,
		"messages": userText("SCRIPT:tool"),
	}, nil)
	waitForEvent(t, bufio.NewReader(resp.Body), "content_block_delta")

	// The call is held but still inside the window: the client never sees it.
	eventually(t, "the tool call to be held", func() bool {
		_, held := b.srv.reg.size()

		return held == 1
	})
	cancel()

	b.waitIdle(t)
}

func TestReserveTreatsACancelledSessionAsAnOrphan(t *testing.T) {
	reg := newRegistry()
	s := newSession(context.Background(), reg, make(chan struct{}, 1), time.Minute)
	reg.add(s)

	id, _, ok := reg.hold(s)
	if !ok {
		t.Fatal("hold failed")
	}

	// Expiry cancelled the session; its cleanup has not removed it yet.
	s.cancel()

	if _, _, ok := reg.reserve([]deliveredResult{{id: id}}); ok {
		t.Fatal("reserved a call of a cancelled session")
	}
}

// syncBuffer collects log output written from several goroutines.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()

	return b.buf.String()
}

func captureLog(t *testing.T) *syncBuffer {
	t.Helper()

	buf := &syncBuffer{}
	log.SetOutput(io.MultiWriter(buf, os.Stderr))
	t.Cleanup(func() { log.SetOutput(os.Stderr) })

	return buf
}

func TestRequestLog(t *testing.T) {
	logs := captureLog(t)
	b := newBridge(t, nil)

	resp := b.post(t, map[string]any{"model": "evil\nmodel=forged", "messages": userText("SCRIPT:plain")})
	expectStatus(t, resp, http.StatusOK)

	b.waitIdle(t)

	line := logs.String()

	if !strings.Contains(line, `model="evil\nmodel=forged"`) {
		t.Fatalf("model not quoted:\n%s", line)
	}

	if !regexp.MustCompile(`session=[0-9a-f]{8} `).MatchString(line) || regexp.MustCompile(`[0-9a-f]{32}`).MatchString(line) {
		t.Fatalf("want a short session hash and no raw id:\n%s", line)
	}

	if !strings.Contains(line, "denied=1 ") {
		t.Fatalf("want the denied permission request counted:\n%s", line)
	}

	if !strings.Contains(line, " clean=true cleanup_failed=0 ") {
		t.Fatalf("want the clean mode and a clean removal in the line:\n%s", line)
	}

	if !strings.Contains(line, " account=a attempts=1 affinity=miss ") {
		t.Fatalf("want the account, the attempts and the affinity in the line:\n%s", line)
	}
}

// A client that leaves while agy is still opening its session leaves state
// agy never named; the throwaway home takes it along all the same.
func TestInterruptedSessionNewLeavesNoState(t *testing.T) {
	logs := captureLog(t)

	t.Setenv("FAKE_AGY_HANG_NEW", "1")

	b := newBridge(t, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 1500*time.Millisecond)
	defer cancel()

	raw, err := json.Marshal(map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.ts.URL+"/v1/messages", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}

	// The client gives up while agy has not answered session/new.
	if resp, err := b.ts.Client().Do(req); err == nil {
		_ = resp.Body.Close()
	}

	b.waitIdle(t)
	assertHomesGone(t, b.homes)

	if strings.Contains(logs.String(), "removing agy's home failed") {
		t.Fatalf("an interrupted session/new failed its removal:\n%s", logs.String())
	}
}

func TestRefusedSessionNewLeavesNoState(t *testing.T) {
	logs := captureLog(t)

	t.Setenv("FAKE_AGY_FAIL_NEW", "1")

	b := newBridge(t, nil)

	resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	if resp.StatusCode == http.StatusOK {
		t.Fatal("a refused session answered 200")
	}

	b.waitIdle(t)
	assertHomesGone(t, b.homes)

	if strings.Contains(logs.String(), "removing agy's home failed") {
		t.Fatalf("a refused session/new failed its removal:\n%s", logs.String())
	}
}

func TestRequestLogNamesTheBuiltInMode(t *testing.T) {
	logs := captureLog(t)
	b := newBridge(t, map[string]string{"AGY_BRIDGE_CLEAN": "false"})

	expectStatus(t, b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")}), http.StatusOK)
	b.waitIdle(t)

	if !strings.Contains(logs.String(), " clean=false cleanup_failed=0 ") {
		t.Fatalf("want clean=false in the line:\n%s", logs.String())
	}
}

func TestCleanupFailuresAreCountedWithoutPaths(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root ignores directory permissions, so the removal cannot be made to fail")
	}

	logs := captureLog(t)
	b := newBridge(t, nil)
	first := userText("SCRIPT:tool")

	// 1. The turn parks with its home in place; the account dir then turns
	// read only, so the home can no longer be unlinked from it.
	_, calls, _, _ := b.turn(t, false, first)

	accountDir := filepath.Join(b.homes, "a")
	if err := os.Chmod(accountDir, 0o500); err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() {
		_ = os.Chmod(accountDir, 0o700)

		entries, _ := os.ReadDir(accountDir)
		for _, entry := range entries {
			if entry.IsDir() {
				_ = os.RemoveAll(filepath.Join(accountDir, entry.Name()))
			}
		}
	})

	// 2. The turn finishes and its removal fails, counted in its line.
	messages := append(first, assistantCalls("Let me look.", calls), toolResultMessage(map[string]string{calls[0].id: "ok"}))
	if text, _, _, _ := b.turn(t, false, messages); text != "Tool said: ok" {
		t.Fatalf("continuation answered %q", text)
	}

	b.waitIdle(t)

	line := logs.String()

	if !strings.Contains(line, " cleanup_failed=1 ") || !strings.Contains(line, "removing agy's home failed: permission denied") {
		t.Fatalf("want one failed removal counted with its cause:\n%s", line)
	}

	for _, secret := range []string{b.homes, b.home, b.srv.root, fakeSessionID()} {
		if strings.Contains(line, secret) {
			t.Fatalf("log carries %q:\n%s", secret, line)
		}
	}
}

func TestLogsCarryNoPromptToolArgumentOrPath(t *testing.T) {
	logs := captureLog(t)
	b := newBridge(t, map[string]string{"AGY_BRIDGE_API_KEY": "key-marker-4711"})

	body := map[string]any{
		"model":    "x",
		"system":   "system-marker-4711",
		"tools":    readFileTool,
		"messages": userText("SCRIPT:tool prompt-marker-4711"),
	}
	header := http.Header{"X-Api-Key": {"key-marker-4711"}}

	resp := b.do(t, context.Background(), http.MethodPost, "/v1/messages", body, header)
	expectStatus(t, resp, http.StatusOK)

	got := decodeJSON[messageResponse](t, resp)

	messages := append(userText("SCRIPT:tool prompt-marker-4711"),
		assistantCalls("", []toolUse{{id: got.Content[1].ID, name: "read_file", input: string(got.Content[1].Input)}}),
		toolResultMessage(map[string]string{got.Content[1].ID: "result-marker-4711"}),
	)
	body["messages"] = messages
	expectStatus(t, b.do(t, context.Background(), http.MethodPost, "/v1/messages", body, header), http.StatusOK)
	b.waitIdle(t)

	line := logs.String()

	acct := testAccount("a")

	for _, secret := range []string{
		"marker-4711", "a.txt", b.home, b.homes, b.srv.root, b.srv.workspace, b.srv.socket, fakeSessionID(), "clean-chat",
		".agents", acct.AccessToken, acct.Email,
	} {
		if strings.Contains(line, secret) {
			t.Fatalf("log carries %q:\n%s", secret, line)
		}
	}
}

func TestHungSessionStateIsRemovedWhenTheClientLeaves(t *testing.T) {
	b := newBridge(t, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	resp := b.do(t, ctx, http.MethodPost, "/v1/messages", map[string]any{
		"model":    "x",
		"stream":   true,
		"messages": userText("SCRIPT:hang"),
	}, nil)
	waitForEvent(t, bufio.NewReader(resp.Body), "content_block_delta")
	cancel()

	b.waitIdle(t)
	assertHomesGone(t, b.homes)
}

func TestSessionFailureKeepsLocalDetailInTheLog(t *testing.T) {
	logs := captureLog(t)
	missing := filepath.Join(t.TempDir(), "no-such-agy")
	b := newBridge(t, map[string]string{"AGY_BIN": missing})

	resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:plain")})
	expectStatus(t, resp, http.StatusBadGateway)

	got := decodeJSON[errorEnvelope](t, resp)

	if got.Error.Message != "agy failed to start" {
		t.Fatalf("client saw %q", got.Error.Message)
	}

	if !strings.Contains(logs.String(), missing) {
		t.Fatalf("log lacks the detail:\n%s", logs.String())
	}
}

func expectOverloaded(t *testing.T, resp *http.Response) {
	t.Helper()

	expectStatus(t, resp, statusOverloaded)

	if resp.Header.Get("Retry-After") != "5" || resp.Header.Get("Content-Type") != "application/json" {
		t.Fatalf("headers %v", resp.Header)
	}

	if got := decodeJSON[errorEnvelope](t, resp); got.Error.Type != "overloaded_error" {
		t.Fatalf("error body = %+v", got)
	}
}
