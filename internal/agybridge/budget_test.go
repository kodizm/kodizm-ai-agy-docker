package agybridge

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"unicode/utf8"
)

func budgetText(t *testing.T, role, text string) message {
	t.Helper()

	raw, err := json.Marshal(text)
	if err != nil {
		t.Fatal(err)
	}

	return message{Role: role, Content: raw}
}

func budgetBlocks(t *testing.T, role string, blocks ...map[string]any) message {
	t.Helper()

	raw, err := json.Marshal(blocks)
	if err != nil {
		t.Fatal(err)
	}

	return message{Role: role, Content: raw}
}

func budgetToolUse(id, name string, input map[string]any) map[string]any {
	return map[string]any{"type": "tool_use", "id": id, "name": name, "input": input}
}

func budgetToolResult(id, text string) map[string]any {
	return map[string]any{"type": "tool_result", "tool_use_id": id, "content": text}
}

// budgetLoop is a typed request answered through two tool calls with large results, then a new typed request.
func budgetLoop(t *testing.T, first, second int) []message {
	t.Helper()

	return []message{
		budgetText(t, "user", "start the job"),
		budgetBlocks(t, "assistant", budgetToolUse("toolu_1", "read_file", map[string]any{"path": "a.txt"})),
		budgetBlocks(t, "user", budgetToolResult("toolu_1", "A1"+strings.Repeat("a", first)+"A2")),
		budgetBlocks(t, "assistant", budgetToolUse("toolu_2", "read_file", map[string]any{"path": "b.txt"})),
		budgetBlocks(t, "user", budgetToolResult("toolu_2", "B1"+strings.Repeat("b", second)+"B2")),
		budgetText(t, "assistant", "both files read"),
		budgetText(t, "user", "now summarise them"),
	}
}

func baselineBytes(t *testing.T, messages []message) int {
	t.Helper()

	base, err := renderPrompt(messages)
	if err != nil {
		t.Fatal(err)
	}

	return textBytes(base)
}

func TestCutApply(t *testing.T) {
	long := strings.Repeat("a", 5000)

	tests := []struct {
		name string
		cut  *cut
		text string
		want string
	}{
		{name: "no cut leaves the text", cut: nil, text: long, want: long},
		{name: "a text within min stays whole", cut: &cut{keep: 300, min: 5000}, text: long, want: long},
		{
			name: "the middle goes, both ends stay",
			cut:  &cut{keep: 4, min: 10},
			text: "head" + strings.Repeat("M", 200) + "tail",
			want: "head" + omissionMarker(200) + "tail",
		},
		{name: "a cut that does not shorten leaves the text", cut: &cut{keep: 40, min: 50}, text: strings.Repeat("a", 120), want: strings.Repeat("a", 120)},
		{name: "keep zero drops the whole text", cut: &cut{keep: 0, min: 10}, text: strings.Repeat("a", 200), want: omissionMarker(200)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.cut.apply(tt.text); got != tt.want {
				t.Fatalf("apply = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCutApplyKeepsRunesWhole(t *testing.T) {
	text := "x" + strings.Repeat("ğ世", 400)

	for keep := 1; keep < 40; keep++ {
		got := (&cut{keep: keep, min: 10}).apply(text)
		if !utf8.ValidString(got) {
			t.Fatalf("keep %d split a rune: %q", keep, got)
		}
	}
}

func TestRenderPromptWithinLeavesAFittingPromptWhole(t *testing.T) {
	messages := budgetLoop(t, 100, 100)

	want, err := renderPrompt(messages)
	if err != nil {
		t.Fatal(err)
	}

	for _, limit := range []int{0, textBytes(want), 1 << 20} {
		got, shrunk, err := renderPromptWithin(messages, limit)
		if err != nil {
			t.Fatal(err)
		}

		if shrunk || got.text() != want.text() {
			t.Fatalf("limit %d: shrunk %t, prompt changed:\n%s", limit, shrunk, got.text())
		}
	}
}

func TestRenderPromptWithinShortensTheOldestToolOutputFirst(t *testing.T) {
	messages := budgetLoop(t, 30000, 30000)
	limit := 40000

	got, shrunk, err := renderPromptWithin(messages, limit)
	if err != nil {
		t.Fatal(err)
	}

	text := got.text()

	if !shrunk || textBytes(got) > limit {
		t.Fatalf("shrunk %t, %d bytes, want a prompt within %d", shrunk, textBytes(got), limit)
	}

	if !strings.HasPrefix(text, omissionNote) {
		t.Fatalf("the prompt does not open with the note:\n%.200s", text)
	}

	if !strings.Contains(text, "A1a") || !strings.Contains(text, "aA2") || !strings.Contains(text, " bytes omitted by the bridge") {
		t.Fatalf("the oldest result lost its ends or its marker:\n%.600s", text)
	}

	if strings.Contains(text, strings.Repeat("a", 1000)) {
		t.Fatal("the oldest result was not shortened")
	}

	if !strings.Contains(text, "B1"+strings.Repeat("b", 30000)+"B2") {
		t.Fatal("the newer result was cut although the oldest one was enough")
	}

	if !strings.HasSuffix(text, "now summarise them") {
		t.Fatalf("the request's own text is not last:\n%s", text[len(text)-200:])
	}
}

func TestRenderPromptWithinCutsHarderWhenItMustNot(t *testing.T) {
	messages := budgetLoop(t, 30000, 30000)
	limit := 1200

	got, shrunk, err := renderPromptWithin(messages, limit)
	if err != nil {
		t.Fatal(err)
	}

	text := got.text()

	if !shrunk || textBytes(got) > limit {
		t.Fatalf("shrunk %t, %d bytes, want a prompt within %d", shrunk, textBytes(got), limit)
	}

	if strings.Contains(text, "aaaa") || strings.Contains(text, "bbbb") {
		t.Fatalf("a result kept a run of its body:\n%s", text)
	}

	if !strings.HasSuffix(text, "now summarise them") || !strings.Contains(text, "start the job") {
		t.Fatalf("a typed request was cut:\n%s", text)
	}
}

func TestRenderPromptWithinNeverCutsTheCurrentRequest(t *testing.T) {
	messages := append(budgetLoop(t, 100, 100)[:6], budgetText(t, "user", strings.Repeat("z", 50000)))
	limit := 10000

	_, _, err := renderPromptWithin(messages, limit)

	var failure *apiError
	if !errors.As(err, &failure) {
		t.Fatalf("error = %v, want an apiError", err)
	}

	want := fmt.Sprintf("prompt is too long: %d bytes > %d maximum", baselineBytes(t, messages), limit)
	if failure.status != http.StatusBadRequest || failure.kind != "invalid_request_error" ||
		failure.reason != reasonPromptTooLong || !strings.HasPrefix(failure.message, want) {
		t.Fatalf("failure = %+v, want a 400 starting %q", failure, want)
	}
}

func TestRenderPromptWithinShortensTheRoundNewestToolOutputLast(t *testing.T) {
	messages := []message{
		budgetText(t, "user", "go"),
		budgetBlocks(t, "assistant", budgetToolUse("toolu_1", "read_file", map[string]any{"path": "a.txt"})),
		budgetBlocks(t, "user", budgetToolResult("toolu_1", "N1"+strings.Repeat("n", 50000)+"N2")),
	}
	limit := 20000

	got, shrunk, err := renderPromptWithin(messages, limit)
	if err != nil {
		t.Fatal(err)
	}

	text := got.text()

	if !shrunk || textBytes(got) > limit || !strings.Contains(text, "\ngo\n") && !strings.Contains(text, "[user]\ngo") {
		t.Fatalf("shrunk %t, %d bytes, prompt:\n%.300s", shrunk, textBytes(got), text)
	}

	if !strings.Contains(text, "N1n") || !strings.Contains(text, "nN2") {
		t.Fatalf("the newest result lost its ends:\n%.300s", text)
	}
}

func TestRenderPromptWithinShortensToolArguments(t *testing.T) {
	messages := []message{
		budgetText(t, "user", "write it"),
		budgetBlocks(t, "assistant", budgetToolUse("toolu_1", "write_file", map[string]any{
			"content": "C1" + strings.Repeat("c", 30000) + "C2",
		})),
		budgetBlocks(t, "user", budgetToolResult("toolu_1", "written")),
		budgetText(t, "assistant", "done"),
		budgetText(t, "user", "next please"),
	}
	limit := 5000

	got, shrunk, err := renderPromptWithin(messages, limit)
	if err != nil {
		t.Fatal(err)
	}

	text := got.text()

	if !shrunk || textBytes(got) > limit || !strings.Contains(text, "[tool_use write_file ") ||
		!strings.Contains(text, " bytes omitted by the bridge") || strings.Contains(text, strings.Repeat("c", 1000)) {
		t.Fatalf("shrunk %t, %d bytes, prompt:\n%.600s", shrunk, textBytes(got), text)
	}
}

func TestRenderPromptWithinKeepsAttachments(t *testing.T) {
	png := map[string]any{
		"type":   "image",
		"source": map[string]string{"type": "base64", "media_type": "image/png", "data": "AAAA"},
	}

	messages := []message{
		budgetBlocks(t, "user", map[string]any{"type": "text", "text": "see this"}, png),
		budgetBlocks(t, "assistant", budgetToolUse("toolu_1", "read_file", map[string]any{"path": "a.txt"})),
		budgetBlocks(t, "user", budgetToolResult("toolu_1", strings.Repeat("a", 30000))),
		budgetText(t, "assistant", "read"),
		budgetText(t, "user", "and now?"),
	}

	got, shrunk, err := renderPromptWithin(messages, 5000)
	if err != nil {
		t.Fatal(err)
	}

	images := 0

	for _, block := range got {
		if block.Type == "image" {
			images++
		}
	}

	if !shrunk || images != 1 {
		t.Fatalf("shrunk %t, %d image blocks, want 1", shrunk, images)
	}
}

func TestRenderPromptWithinKeepsRunesWhole(t *testing.T) {
	text := "x" + strings.Repeat("ğ世", 10000)
	messages := []message{
		budgetText(t, "user", "start"),
		budgetBlocks(t, "assistant", budgetToolUse("toolu_1", "read_file", map[string]any{"path": "a.txt"})),
		budgetBlocks(t, "user", budgetToolResult("toolu_1", text)),
		budgetText(t, "assistant", "read"),
		budgetText(t, "user", "next"),
	}

	for _, limit := range []int{1500, 3000, 5000, 9000, 20000} {
		got, _, err := renderPromptWithin(messages, limit)
		if err != nil {
			t.Fatalf("limit %d: %v", limit, err)
		}

		for _, block := range got {
			if !utf8.ValidString(block.Text) {
				t.Fatalf("limit %d: a cut split a rune", limit)
			}
		}
	}
}

// The owner's task is worth more than the output of tools long since answered: small results go to their markers
// before any message text is cut.
func TestRenderPromptWithinCutsOldToolOutputBeforeTheTask(t *testing.T) {
	brief := "BRIEF" + strings.Repeat("t", 8400) + "END"
	messages := []message{budgetText(t, "user", brief)}

	for i := range 40 {
		id := fmt.Sprintf("toolu_%d", i)

		messages = append(messages,
			budgetBlocks(t, "assistant", budgetToolUse(id, "read_file", map[string]any{"path": fmt.Sprintf("f%d.txt", i)})),
			budgetBlocks(t, "user", budgetToolResult(id, strings.Repeat("r", 1800))),
		)
	}

	messages = append(messages, budgetText(t, "assistant", "all read"), budgetText(t, "user", "now summarise them"))
	limit := 30000

	got, shrunk, err := renderPromptWithin(messages, limit)
	if err != nil {
		t.Fatal(err)
	}

	text := got.text()

	if !shrunk || textBytes(got) > limit {
		t.Fatalf("shrunk %t, %d bytes, want a prompt within %d", shrunk, textBytes(got), limit)
	}

	if !strings.Contains(text, brief) {
		t.Fatal("the task was cut while old tool output was still whole")
	}

	// The cutting stops at the first cut that fits, so the newest results stay whole and the oldest do not.
	whole := strings.Count(text, strings.Repeat("r", 1800))
	if !strings.Contains(text, " bytes omitted by the bridge") || whole == 0 || whole >= 40 {
		t.Fatalf("%d of 40 results are whole, want the oldest cut and the newest kept", whole)
	}

	if first, last := strings.Index(text, " bytes omitted by the bridge"), strings.Index(text, strings.Repeat("r", 1800)); first > last {
		t.Fatal("a whole result comes before a cut one: the cutting did not start with the oldest")
	}
}

// An old Edit call keeps its file_path when its new_string goes: each string of the arguments is cut on its own.
func TestRenderPromptWithinKeepsToolArgumentKeys(t *testing.T) {
	messages := []message{
		budgetText(t, "user", "edit it"),
		budgetBlocks(t, "assistant", budgetToolUse("toolu_1", "Edit", map[string]any{
			"file_path":  "/src/a.go",
			"new_string": "N1" + strings.Repeat("n", 30000) + "N2",
		})),
		budgetBlocks(t, "user", budgetToolResult("toolu_1", "edited")),
		budgetText(t, "assistant", "done"),
		budgetText(t, "user", "next please"),
	}

	got, shrunk, err := renderPromptWithin(messages, 900)
	if err != nil {
		t.Fatal(err)
	}

	text := got.text()

	if !shrunk || !strings.Contains(text, `"file_path":"/src/a.go"`) ||
		!strings.Contains(text, " bytes omitted by the bridge") || strings.Contains(text, "nnnn") {
		t.Fatalf("shrunk %t, prompt:\n%s", shrunk, text)
	}
}

func TestRenderPromptWithinShortensOldTextDocumentsAndSystemMessages(t *testing.T) {
	document := map[string]any{
		"type": "document",
		"source": map[string]string{
			"type": "text", "media_type": "text/plain", "data": "D1" + strings.Repeat("d", 30000) + "D2",
		},
	}

	messages := []message{
		budgetBlocks(t, "user", map[string]any{"type": "text", "text": "read this"}, document),
		budgetText(t, "system", "S1"+strings.Repeat("s", 20000)+"S2"),
		budgetText(t, "assistant", "read"),
		budgetText(t, "user", "and now?"),
	}

	got, shrunk, err := renderPromptWithin(messages, 5000)
	if err != nil {
		t.Fatal(err)
	}

	text := got.text()

	for _, want := range []string{"D1d", "dD2", "S1s", "sS2", " bytes omitted by the bridge"} {
		if !shrunk || !strings.Contains(text, want) {
			t.Fatalf("shrunk %t, prompt lacks %q:\n%.600s", shrunk, want, text)
		}
	}

	if !strings.HasSuffix(text, "and now?") || textBytes(got) > 5000 {
		t.Fatalf("the request is not last, or %d bytes is over the limit", textBytes(got))
	}
}

func TestRenderPromptWithinKeepsAnImageInsideAToolResult(t *testing.T) {
	png := map[string]any{
		"type":   "image",
		"source": map[string]string{"type": "base64", "media_type": "image/png", "data": "AAAA"},
	}

	messages := []message{
		budgetText(t, "user", "look"),
		budgetBlocks(t, "assistant", budgetToolUse("toolu_1", "screenshot", map[string]any{})),
		budgetBlocks(t, "user", map[string]any{
			"type":        "tool_result",
			"tool_use_id": "toolu_1",
			"content":     []map[string]any{{"type": "text", "text": strings.Repeat("a", 30000)}, png},
		}),
		budgetText(t, "assistant", "seen"),
		budgetText(t, "user", "and now?"),
	}

	got, shrunk, err := renderPromptWithin(messages, 5000)
	if err != nil {
		t.Fatal(err)
	}

	images := 0

	for _, block := range got {
		if block.Type == "image" {
			images++
		}
	}

	if !shrunk || images != 1 {
		t.Fatalf("shrunk %t, %d image blocks, want 1", shrunk, images)
	}
}

// Bytes below the limit and a conversation of one message are the replay of a first request: nothing to age out.
func TestRenderPromptWithinOneMessage(t *testing.T) {
	messages := []message{budgetText(t, "user", strings.Repeat("q", 500))}

	got, shrunk, err := renderPromptWithin(messages, 1000)
	if err != nil || shrunk || got.text() != strings.Repeat("q", 500) {
		t.Fatalf("shrunk %t, err %v, prompt %q", shrunk, err, got.text())
	}

	if _, _, err := renderPromptWithin(messages, 100); err == nil {
		t.Fatal("a lone request over the limit must be refused")
	}
}

func TestOversizedReplayIsShortenedBeforeAgyReadsIt(t *testing.T) {
	logs := captureLog(t)
	b := newBridge(t, map[string]string{"AGY_BRIDGE_MAX_PROMPT_BYTES": "4000"})

	messages := append(userText("start"),
		assistantCalls("", []toolUse{{id: "toolu_old", name: "read_file", input: `{}`}}),
		toolResultMessage(map[string]string{"toolu_old": "HEAD" + strings.Repeat("x", 20000) + "TAIL"}),
		map[string]any{"role": "assistant", "content": "read it"},
		map[string]any{"role": "user", "content": "SCRIPT:prompt final"},
	)

	text, _, stop, _ := b.turn(t, false, messages)

	if stop != "end_turn" || len(text) > 4000 {
		t.Fatalf("stop %q, agy read %d bytes, want a turn within 4000", stop, len(text))
	}

	for _, want := range []string{omissionNote, "HEAD", "TAIL", " bytes omitted by the bridge"} {
		if !strings.Contains(text, want) {
			t.Fatalf("agy's prompt lacks %q:\n%.600s", want, text)
		}
	}

	if !strings.HasSuffix(text, "SCRIPT:prompt final") || strings.Contains(text, strings.Repeat("x", 1000)) {
		t.Fatalf("the request is not last, or the result was not shortened:\n%.600s", text)
	}

	b.waitIdle(t)

	eventually(t, "the request line to carry the budget fields", func() bool {
		line := logs.String()

		return strings.Contains(line, " shrunk=true ") && strings.Contains(line, " prompt_bytes=")
	})
}

func TestOversizedReplayThatCannotShrinkIsRefused(t *testing.T) {
	logs := captureLog(t)
	b := newBridge(t, map[string]string{"AGY_BRIDGE_MAX_PROMPT_BYTES": "1000"})

	resp := b.post(t, map[string]any{"model": "x", "messages": userText("SCRIPT:prompt " + strings.Repeat("z", 5000))})
	expectStatus(t, resp, http.StatusBadRequest)

	got := decodeJSON[errorEnvelope](t, resp)
	if got.Error.Type != "invalid_request_error" || !strings.HasPrefix(got.Error.Message, "prompt is too long: ") ||
		!strings.Contains(got.Error.Message, " bytes > 1000 maximum") {
		t.Fatalf("error body = %+v", got)
	}

	if n := b.spawned(t); n != 0 {
		t.Fatalf("a refused prompt started agy %d times", n)
	}

	eventually(t, "the request line to carry the reason", func() bool {
		return strings.Contains(logs.String(), " reason=prompt_too_long ")
	})
}

// A resumed turn replays nothing, so the budget has no say over it.
func TestResumedTurnIsNotBudgeted(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_MAX_PROMPT_BYTES": "1000"})
	first := userText("SCRIPT:tool")

	_, calls, _, _ := b.turn(t, false, first)

	messages := append(first, assistantCalls("", calls), toolResultMessage(map[string]string{calls[0].id: strings.Repeat("r", 5000)}))
	text, _, stop, _ := b.turn(t, false, messages)

	if stop != "end_turn" || !strings.Contains(text, strings.Repeat("r", 5000)) {
		t.Fatalf("stop %q, text %.200s", stop, text)
	}

	if n := b.spawned(t); n != 1 {
		t.Fatalf("agy started %d times, want the one held turn", n)
	}
}

func TestLoadConfigPromptBudget(t *testing.T) {
	cfg, err := LoadConfig(func(string) string { return "" })
	if err != nil || cfg.MaxPromptBytes != 180000 {
		t.Fatalf("default: %v, %d", err, cfg.MaxPromptBytes)
	}

	cfg, err = LoadConfig(func(k string) string { return map[string]string{"AGY_BRIDGE_MAX_PROMPT_BYTES": "50000"}[k] })
	if err != nil || cfg.MaxPromptBytes != 50000 {
		t.Fatalf("override: %v, %d", err, cfg.MaxPromptBytes)
	}

	for _, bad := range []string{"0", "-5", "lots"} {
		if _, err := LoadConfig(func(k string) string { return map[string]string{"AGY_BRIDGE_MAX_PROMPT_BYTES": bad}[k] }); err == nil {
			t.Fatalf("AGY_BRIDGE_MAX_PROMPT_BYTES=%q must be refused", bad)
		}
	}
}
