package agybridge

import (
	"encoding/json"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestRenderPrompt(t *testing.T) {
	tests := []struct {
		name     string
		messages string
		want     string
	}{
		{
			name:     "single string message",
			messages: `[{"role": "user", "content": "Hello"}]`,
			want:     "Hello",
		},
		{
			name:     "single block message",
			messages: `[{"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]}]`,
			want:     "A\nB",
		},
		{
			name: "history with tools, images, documents and thinking",
			messages: `[
				{"role": "user", "content": [
					{"type": "text", "text": "Read it"},
					{"type": "image", "source": {}},
					{"type": "document", "source": {"type": "url", "url": "https://example.test/a.pdf"}}
				]},
				{"role": "assistant", "content": [
					{"type": "thinking", "thinking": "hmm"},
					{"type": "text", "text": "Sure."},
					{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}}
				]},
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "body"}]}
				]},
				{"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_2", "name": "ls", "input": {}}]},
				{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_2", "content": "x y"}]}
			]`,
			want: "<conversation_history>\n" +
				"[user]\nRead it\n[image omitted]\n[document omitted]\n\n" +
				"[assistant]\nSure.\n[tool_use read_file {\"path\":\"a.txt\"}]\n\n" +
				"[user]\n[tool_result toolu_1]\nbody\n\n" +
				"[assistant]\n[tool_use ls {}]\n" +
				"</conversation_history>\n\n" +
				"[tool_result toolu_2]\nx y",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var messages []message
			if err := json.Unmarshal([]byte(tt.messages), &messages); err != nil {
				t.Fatal(err)
			}

			got, err := renderPrompt(messages)
			if err != nil {
				t.Fatal(err)
			}

			if got.text() != tt.want {
				t.Fatalf("renderPrompt =\n%q\nwant\n%q", got.text(), tt.want)
			}
		})
	}
}

const (
	pngSource = `{"type": "base64", "media_type": "image/png", "data": "AAAA"}`
	pdfSource = `{"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}`
	pngBlock  = `{"type":"image","mimeType":"image/png","data":"AAAA"}`
)

func TestRenderPromptBlocks(t *testing.T) {
	all := promptCapabilities{image: true, embeddedContext: true}
	look := `[{"role": "user", "content": [
		{"type": "text", "text": "Look"},
		{"type": "image", "source": ` + pngSource + `},
		{"type": "text", "text": "there"}
	]}]`
	single := func(blocks ...string) string {
		return `[{"role": "user", "content": [` + strings.Join(blocks, ",") + `]}]`
	}

	tests := []struct {
		name         string
		messages     string
		capabilities promptCapabilities
		want         string
	}{
		{
			name:         "an advertised image sits where it occurred",
			messages:     look,
			capabilities: all,
			want:         `[{"type":"text","text":"Look\n"},` + pngBlock + `,{"type":"text","text":"\nthere"}]`,
		},
		{
			name:     "an image agy did not advertise is a placeholder",
			messages: look,
			want:     `[{"type":"text","text":"Look\n[image omitted]\nthere"}]`,
		},
		{
			name:         "image/jpg is image/jpeg",
			messages:     single(`{"type": "image", "source": {"type": "base64", "media_type": "image/jpg", "data": "AAAA"}}`),
			capabilities: all,
			want:         `[{"type":"image","mimeType":"image/jpeg","data":"AAAA"}]`,
		},
		{
			name:         "a url image is never fetched",
			messages:     single(`{"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}}`),
			capabilities: all,
			want:         `[{"type":"text","text":"[image omitted]"}]`,
		},
		{
			name:         "data that is not base64 is a placeholder",
			messages:     single(`{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "no base64!"}}`),
			capabilities: all,
			want:         `[{"type":"text","text":"[image omitted]"}]`,
		},
		{
			name:         "empty data is a placeholder",
			messages:     single(`{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": ""}}`),
			capabilities: all,
			want:         `[{"type":"text","text":"[image omitted]"}]`,
		},
		{
			name:         "an unsupported image type is a placeholder",
			messages:     single(`{"type": "image", "source": {"type": "base64", "media_type": "image/bmp", "data": "AAAA"}}`),
			capabilities: all,
			want:         `[{"type":"text","text":"[image omitted]"}]`,
		},
		{
			name: "pdfs are resources named by a sanitised title and their number",
			messages: single(
				`{"type": "document", "title": "../My Report (v2)", "source": `+pdfSource+`}`,
				`{"type": "document", "source": `+pdfSource+`}`,
			),
			capabilities: all,
			want: `[{"type":"resource","resource":{"uri":"file:///attachments/.._My_Report__v2_-1.pdf",` +
				`"mimeType":"application/pdf","blob":"JVBERi0="}},{"type":"text","text":"\n"},` +
				`{"type":"resource","resource":{"uri":"file:///attachments/document-2.pdf",` +
				`"mimeType":"application/pdf","blob":"JVBERi0="}}]`,
		},
		{
			name:         "a pdf without embedded context is a placeholder",
			messages:     single(`{"type": "document", "source": ` + pdfSource + `}`),
			capabilities: promptCapabilities{image: true},
			want:         `[{"type":"text","text":"[document omitted]"}]`,
		},
		{
			name: "a text document is text",
			messages: single(
				`{"type": "text", "text": "Read:"}`,
				`{"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "plain body"}}`,
			),
			want: `[{"type":"text","text":"Read:\nplain body"}]`,
		},
		{
			name:         "a url document is a placeholder",
			messages:     single(`{"type": "document", "source": {"type": "url", "url": "https://example.test/a.pdf"}}`),
			capabilities: all,
			want:         `[{"type":"text","text":"[document omitted]"}]`,
		},
		{
			name: "the history keeps each attachment in order",
			messages: `[
				{"role": "user", "content": [{"type": "text", "text": "see"}, {"type": "image", "source": ` + pngSource + `}]},
				{"role": "assistant", "content": "ok"},
				{"role": "user", "content": "next"}
			]`,
			capabilities: all,
			want: `[{"type":"text","text":"<conversation_history>\n[user]\nsee\n"},` + pngBlock + `,` +
				`{"type":"text","text":"\n\n[assistant]\nok\n</conversation_history>\n\nnext"}]`,
		},
		{
			name: "media in a tool result follows its text",
			messages: single(`{"type": "tool_result", "tool_use_id": "t1", "content": [
				{"type": "text", "text": "shot"},
				{"type": "image", "source": ` + pngSource + `}
			]}`),
			capabilities: all,
			want:         `[{"type":"text","text":"[tool_result t1]\nshot\n"},` + pngBlock + `]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var messages []message
			if err := json.Unmarshal([]byte(tt.messages), &messages); err != nil {
				t.Fatal(err)
			}

			rendered, err := renderPrompt(messages)
			if err != nil {
				t.Fatal(err)
			}

			got, err := json.Marshal(rendered.forAgent(tt.capabilities))
			if err != nil {
				t.Fatal(err)
			}

			assertSameJSON(t, string(got), tt.want)
		})
	}
}

// assertSameJSON compares two JSON documents by value, so escaping does not matter.
func assertSameJSON(t *testing.T, got, want string) {
	t.Helper()

	var gotValue, wantValue any

	if err := json.Unmarshal([]byte(got), &gotValue); err != nil {
		t.Fatalf("got is not JSON: %v\n%s", err, got)
	}

	if err := json.Unmarshal([]byte(want), &wantValue); err != nil {
		t.Fatalf("want is not JSON: %v\n%s", err, want)
	}

	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Fatalf("got\n%s\nwant\n%s", got, want)
	}
}

// A ToolSearch result is a list of tool_reference blocks; text order is kept.
func TestToolReferenceBlocksRender(t *testing.T) {
	block := contentBlock{
		Type:      "tool_result",
		ToolUseID: "toolu_1",
		Content: json.RawMessage(`[
			{"type": "text", "text": "Found:"},
			{"type": "tool_reference", "tool_name": "WebFetch"},
			{"type": "image", "source": {}},
			{"type": "tool_reference", "tool_name": "WebSearch"}
		]`),
	}

	var r promptRenderer

	got, err := r.result(block)
	if err != nil {
		t.Fatal(err)
	}

	if want := "Found:\ntool_reference: WebFetch\n[image omitted]\ntool_reference: WebSearch"; got.text() != want {
		t.Fatalf("result = %q, want %q", got.text(), want)
	}

	messages := []message{{Role: "user", Content: json.RawMessage(`[{"type": "tool_result", "tool_use_id": "toolu_1",
		"content": [{"type": "tool_reference", "tool_name": "WebFetch"}]}]`)}}

	results, _, err := toolResults(messages)
	if err != nil || len(results) != 1 || results[0].result.Text != "tool_reference: WebFetch" {
		t.Fatalf("toolResults = %+v, %v", results, err)
	}
}

func TestRenderPromptRejectsMalformedContent(t *testing.T) {
	if _, err := renderPrompt([]message{{Role: "user", Content: json.RawMessage(`42`)}}); err == nil {
		t.Fatal("want an error for numeric content")
	}
}

func TestSystemText(t *testing.T) {
	tests := []struct {
		name   string
		system string
		want   string
	}{
		{name: "absent", system: ``, want: ""},
		{name: "null", system: `null`, want: ""},
		{name: "string", system: `"Be brief."`, want: "Be brief."},
		{
			name:   "blocks",
			system: `[{"type": "text", "text": "One"}, {"type": "image"}, {"type": "text", "text": "Two"}]`,
			want:   "One\n\nTwo",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := systemText(json.RawMessage(tt.system))
			if err != nil {
				t.Fatal(err)
			}

			if got != tt.want {
				t.Fatalf("systemText = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestClientTools(t *testing.T) {
	var defs []toolDef
	raw := `[
		{"name": "read_file", "description": "Reads", "input_schema": {"type": "object", "required": ["path"]}},
		{"type": "custom", "name": "ls"},
		{"type": "web_search_20250305", "name": "web_search"},
		{"description": "nameless"}
	]`

	if err := json.Unmarshal([]byte(raw), &defs); err != nil {
		t.Fatal(err)
	}

	got, err := json.Marshal(clientTools(defs))
	if err != nil {
		t.Fatal(err)
	}

	want := `[{"name":"read_file","description":"Reads","inputSchema":{"type":"object","required":["path"]}},` +
		`{"name":"ls","description":"","inputSchema":{"type":"object"}}]`

	if string(got) != want {
		t.Fatalf("clientTools =\n%s\nwant\n%s", got, want)
	}
}

func TestClientToolsRewritesNonStringEnums(t *testing.T) {
	var defs []toolDef

	raw := `[{"name":"pick","input_schema":{"type":"object","properties":{"n":{"type":"integer","enum":[1,2]}}}}]`

	if err := json.Unmarshal([]byte(raw), &defs); err != nil {
		t.Fatal(err)
	}

	got, err := json.Marshal(clientTools(defs))
	if err != nil {
		t.Fatal(err)
	}

	want := `[{"name":"pick","description":"","inputSchema":` +
		`{"properties":{"n":{"description":"One of: 1, 2","type":"integer"}},"type":"object"}}]`

	if string(got) != want {
		t.Fatalf("clientTools =\n%s\nwant\n%s", got, want)
	}
}

// agy hands a tool schema to Gemini, which rejects an enum on anything but a
// string and ends the whole turn: the enum moves into the description.
// Claude Code's compaction forks the conversation with the same tools and ends it with a no-tools reminder.
// Only that reminder, in the last user message, takes the tools away.
func TestForbidsTools(t *testing.T) {
	const reminder = "Summarise the conversation.\n\nREMINDER: Do NOT call any tools. Respond with plain text only. " +
		"Tool calls will be rejected."

	tests := []struct {
		name     string
		messages []message
		want     bool
	}{
		{name: "string content", messages: []message{{Role: "user", Content: jsonString(reminder)}}, want: true},
		{
			name:     "block content beside a tool result",
			messages: []message{{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"t1","content":"ok"},{"type":"text","text":` + string(jsonString(reminder)) + `}]`)}},
			want:     true,
		},
		{
			// Claude Code appends a system message after the last user message on every step.
			name: "system message after the reminder",
			messages: []message{
				{Role: "user", Content: jsonString(reminder)},
				{Role: "system", Content: jsonString("a note from the harness")},
				{Role: "system", Content: jsonString("another note")},
			},
			want: true,
		},
		{
			name: "assistant message after the reminder",
			messages: []message{
				{Role: "user", Content: jsonString(reminder)},
				{Role: "assistant", Content: jsonString("done")},
			},
		},
		{
			name: "only an earlier message",
			messages: []message{
				{Role: "user", Content: jsonString(reminder)},
				{Role: "assistant", Content: jsonString("done")},
				{Role: "user", Content: jsonString("now edit the file")},
			},
		},
		{
			name:     "half of the reminder",
			messages: []message{{Role: "user", Content: jsonString("Do NOT call any tools in this phase.")}},
		},
		{name: "assistant last", messages: []message{{Role: "assistant", Content: jsonString(reminder)}}},
		{name: "no messages"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := forbidsTools(tt.messages); got != tt.want {
				t.Fatalf("forbidsTools = %t, want %t", got, tt.want)
			}
		})
	}
}

// progressPrompt is what Claude Code appends when it forks a running background agent for a progress label.
const progressPrompt = "Describe your most recent action in 3-5 words using present tense (-ing). Name the file or " +
	"function, not the branch. Do not use tools.\n\nPrevious: \"Reading old.go\" — say something NEW.\n\n" +
	"Good: \"Reading runAgent.ts\"\nGood: \"Running auth module tests\"\n\nBad (past tense): \"Analyzed the branch diff\""

func TestAsksProgressLabel(t *testing.T) {
	beside := func(text string) json.RawMessage {
		return json.RawMessage(`[{"type":"tool_result","tool_use_id":"t1","content":"ok"},{"type":"text","text":` +
			string(jsonString(text)) + `}]`)
	}

	tests := []struct {
		name     string
		messages []message
		want     bool
	}{
		{name: "string content", messages: []message{{Role: "user", Content: jsonString(progressPrompt)}}, want: true},
		{
			// The agent's history ends in tool results, so the prompt joins that user message as a text block.
			name:     "block content beside a tool result",
			messages: []message{{Role: "user", Content: beside(progressPrompt)}},
			want:     true,
		},
		{
			name: "system message after the prompt",
			messages: []message{
				{Role: "user", Content: beside(progressPrompt)},
				{Role: "system", Content: jsonString("a note from the harness")},
			},
			want: true,
		},
		{
			// A subagent brief that quotes the prompt in a sentence must reach the model.
			name: "the prompt quoted inside a sentence",
			messages: []message{{Role: "user", Content: jsonString(
				"Claude Code asks: `" + progressLabelStart + " Name the file. Do not use tools.` Explain why.",
			)}},
		},
		{
			name:     "first phrase alone",
			messages: []message{{Role: "user", Content: jsonString(progressLabelStart)}},
		},
		{
			name:     "second phrase alone",
			messages: []message{{Role: "user", Content: jsonString("Please run it. Do not use tools.")}},
		},
		{
			// Only the last text block counts: the prompt is appended after everything else.
			name: "the prompt in an earlier block",
			messages: []message{{Role: "user", Content: json.RawMessage(
				`[{"type":"text","text":` + string(jsonString(progressPrompt)) + `},{"type":"text","text":"and then?"}]`,
			)}},
		},
		{
			name: "only an earlier message",
			messages: []message{
				{Role: "user", Content: jsonString(progressPrompt)},
				{Role: "assistant", Content: jsonString("Reading render.go")},
				{Role: "user", Content: jsonString("now edit the file")},
			},
		},
		{name: "assistant last", messages: []message{{Role: "assistant", Content: jsonString(progressPrompt)}}},
		{name: "no messages"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := asksProgressLabel(tt.messages); got != tt.want {
				t.Fatalf("asksProgressLabel = %t, want %t", got, tt.want)
			}
		})
	}
}

func toolUseMessage(blocks ...string) message {
	return message{Role: "assistant", Content: json.RawMessage("[" + strings.Join(blocks, ",") + "]")}
}

func toolUseBlockJSON(name, input string) string {
	return `{"type":"tool_use","id":"toolu_1","name":"` + name + `","input":` + input + `}`
}

func TestProgressLabel(t *testing.T) {
	tests := []struct {
		name     string
		messages []message
		want     string
	}{
		{
			name:     "read",
			messages: []message{toolUseMessage(toolUseBlockJSON("Read", `{"file_path":"/repo/internal/render.go"}`))},
			want:     "Reading render.go",
		},
		{
			name:     "edit",
			messages: []message{toolUseMessage(toolUseBlockJSON("Edit", `{"file_path":"/repo/server.go"}`))},
			want:     "Editing server.go",
		},
		{
			name:     "write",
			messages: []message{toolUseMessage(toolUseBlockJSON("Write", `{"file_path":"/repo/NOTES.md"}`))},
			want:     "Writing NOTES.md",
		},
		{
			name:     "bash keeps the first three words",
			messages: []message{toolUseMessage(toolUseBlockJSON("Bash", `{"command":"go test ./... -race -count=1"}`))},
			want:     "Running go test ./...",
		},
		{
			name:     "bash skips a leading cd",
			messages: []message{toolUseMessage(toolUseBlockJSON("Bash", `{"command":"cd /Users/anilcan/Code/kodizm.com && go vet ./..."}`))},
			want:     "Running go vet ./...",
		},
		{
			name:     "grep",
			messages: []message{toolUseMessage(toolUseBlockJSON("Grep", `{"pattern":"forbidsTools"}`))},
			want:     "Searching for forbidsTools",
		},
		{
			name:     "glob",
			messages: []message{toolUseMessage(toolUseBlockJSON("Glob", `{"pattern":"**/*.go"}`))},
			want:     "Finding **/*.go",
		},
		{
			name:     "subagent",
			messages: []message{toolUseMessage(toolUseBlockJSON("Agent", `{"prompt":"review it"}`))},
			want:     "Delegating to a subagent",
		},
		{
			name:     "unknown tool",
			messages: []message{toolUseMessage(toolUseBlockJSON("mcp__kodizm-ops__db_query", `{}`))},
			want:     "Using mcp__kodizm-ops__db_query",
		},
		{
			name:     "read without a path",
			messages: []message{toolUseMessage(toolUseBlockJSON("Read", `{}`))},
			want:     "Reading files",
		},
		{
			name: "the last tool call of the last assistant message wins",
			messages: []message{
				toolUseMessage(toolUseBlockJSON("Read", `{"file_path":"/a/old.go"}`)),
				{Role: "user", Content: jsonString("go on")},
				toolUseMessage(
					toolUseBlockJSON("Read", `{"file_path":"/a/first.go"}`),
					toolUseBlockJSON("Edit", `{"file_path":"/a/second.go"}`),
				),
				{Role: "user", Content: jsonString("done")},
			},
			want: "Editing second.go",
		},
		{
			name: "an assistant message of text only is skipped",
			messages: []message{
				toolUseMessage(toolUseBlockJSON("Read", `{"file_path":"/a/render.go"}`)),
				{Role: "assistant", Content: jsonString("looking")},
			},
			want: "Reading render.go",
		},
		{name: "no tool call yet", messages: []message{{Role: "user", Content: jsonString("hi")}}, want: "Starting work"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := progressLabel(tt.messages); got != tt.want {
				t.Fatalf("progressLabel = %q, want %q", got, tt.want)
			}
		})
	}

	t.Run("a long value is cut to a short label", func(t *testing.T) {
		long := strings.Repeat("ç", 200)
		got := progressLabel([]message{toolUseMessage(toolUseBlockJSON("Grep", `{"pattern":"`+long+`"}`))})

		if n := len([]rune(got)); n > maxProgressLabelRunes || !utf8.ValidString(got) {
			t.Fatalf("label has %d runes (valid utf-8: %t), want at most %d", n, utf8.ValidString(got), maxProgressLabelRunes)
		}
	})
}

func jsonString(s string) json.RawMessage {
	raw, _ := json.Marshal(s)

	return raw
}

func TestStringEnumsOnly(t *testing.T) {
	tests := []struct {
		name   string
		schema string
		want   string
	}{
		{
			name:   "integer enum without description",
			schema: `{"type":"object","properties":{"n":{"type":"integer","enum":[1,2]}}}`,
			want:   `{"properties":{"n":{"description":"One of: 1, 2","type":"integer"}},"type":"object"}`,
		},
		{
			name:   "boolean enum keeps its description",
			schema: `{"type":"object","properties":{"b":{"type":"boolean","enum":[true,false],"description":"Flag"}}}`,
			want:   `{"properties":{"b":{"description":"Flag (one of: true, false)","type":"boolean"}},"type":"object"}`,
		},
		{
			name:   "number enum",
			schema: `{"type":"number","enum":[0.5,1.5]}`,
			want:   `{"description":"One of: 0.5, 1.5","type":"number"}`,
		},
		{
			name:   "mixed values without a type",
			schema: `{"enum":[1,"a"]}`,
			want:   `{"description":"One of: 1, \"a\""}`,
		},
		{
			name:   "nested in items and anyOf",
			schema: `{"type":"array","items":{"anyOf":[{"type":"integer","enum":[3]},{"type":"string","enum":["x"]}]}}`,
			want: `{"items":{"anyOf":[{"description":"One of: 3","type":"integer"},` +
				`{"enum":["x"],"type":"string"}]},"type":"array"}`,
		},
		{
			name:   "a property named enum is not an enum",
			schema: `{"type":"object","properties":{"enum":{"type":"integer","enum":[7]},"type":{"type":"string"}}}`,
			want: `{"properties":{"enum":{"description":"One of: 7","type":"integer"},` +
				`"type":{"type":"string"}},"type":"object"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := stringEnumsOnly(json.RawMessage(tt.schema))

			if string(got) != tt.want {
				t.Fatalf("stringEnumsOnly =\n%s\nwant\n%s", got, tt.want)
			}
		})
	}
}

func TestStringEnumsOnlyLeavesAcceptedSchemasUntouched(t *testing.T) {
	for _, schema := range []string{
		`{"type":"object","required":["path"],"properties":{"mode":{"type":"string","enum":["r","w"]}}}`,
		`{"enum":["a","b"]}`,
		`{"type":["string","null"],"enum":["a",null]}`,
		`{"type":"object"}`,
		`not json`,
	} {
		if got := stringEnumsOnly(json.RawMessage(schema)); string(got) != schema {
			t.Fatalf("stringEnumsOnly(%s) = %s, want it unchanged", schema, got)
		}
	}
}

func TestToolResults(t *testing.T) {
	messages := []message{
		{Role: "user", Content: json.RawMessage(`"start"`)},
		{Role: "user", Content: json.RawMessage(`[
			{"type": "tool_result", "tool_use_id": "toolu_a", "content": "ok"},
			{"type": "text", "text": "note"},
			{"type": "tool_result", "tool_use_id": "toolu_b", "is_error": true, "content": [{"type": "text", "text": "no"}]}
		]`)},
	}

	got, extra, err := toolResults(messages)
	if err != nil {
		t.Fatal(err)
	}

	want := []deliveredResult{
		{id: "toolu_a", result: toolResult{Text: "ok"}},
		{id: "toolu_b", result: toolResult{Text: "no", IsError: true}},
	}

	if len(got) != len(want) || got[0] != want[0] || got[1] != want[1] || !extra {
		t.Fatalf("toolResults = %+v, %t; want %+v, true", got, extra, want)
	}

	assistantLast := append(messages, message{Role: "assistant", Content: json.RawMessage(`"done"`)})

	if got, _, _ := toolResults(assistantLast); len(got) != 0 {
		t.Fatalf("assistant last message: want no results, got %+v", got)
	}
}

func TestToolResultsLookPastTrailingSystemMessages(t *testing.T) {
	user := message{Role: "user", Content: json.RawMessage(`[{"type": "tool_result", "tool_use_id": "toolu_a", "content": "body\n"}]`)}
	reminder := func(text string) message {
		return message{Role: "system", Content: json.RawMessage(strconv.Quote(text))}
	}

	t.Run("notes follow the last result in order", func(t *testing.T) {
		got, extra, err := toolResults([]message{
			{Role: "user", Content: json.RawMessage(`"start"`)},
			user,
			reminder("first note"),
			reminder(" \n"),
			reminder("second note"),
		})
		if err != nil {
			t.Fatal(err)
		}

		want := deliveredResult{id: "toolu_a", result: toolResult{Text: "body\n\nfirst note\n\nsecond note"}}
		if len(got) != 1 || got[0] != want || extra {
			t.Fatalf("toolResults = %+v, %t; want %+v, false", got, extra, want)
		}
	})

	t.Run("blank notes change nothing", func(t *testing.T) {
		got, _, err := toolResults([]message{user, reminder(" ")})
		if err != nil || len(got) != 1 || got[0].result.Text != "body\n" {
			t.Fatalf("toolResults = %+v, %v", got, err)
		}
	})

	t.Run("system messages after a plain prompt are no results", func(t *testing.T) {
		got, extra, err := toolResults([]message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
			reminder("# Environment"),
		})
		if err != nil || len(got) != 0 || extra {
			t.Fatalf("toolResults = %+v, %t, %v; want nothing", got, extra, err)
		}
	})

	t.Run("system messages alone are no results", func(t *testing.T) {
		if got, _, err := toolResults([]message{reminder("only")}); err != nil || len(got) != 0 {
			t.Fatalf("toolResults = %+v, %v", got, err)
		}
	})
}

func TestToolResultsExtraContent(t *testing.T) {
	tests := []struct {
		name    string
		content string
		extra   bool
	}{
		{
			name:    "results only",
			content: `[{"type": "tool_result", "tool_use_id": "a", "content": "ok"}]`,
		},
		{
			name:    "blank text beside results",
			content: `[{"type": "tool_result", "tool_use_id": "a", "content": "ok"}, {"type": "text", "text": " \n"}]`,
		},
		{
			name:    "a note beside results",
			content: `[{"type": "tool_result", "tool_use_id": "a"}, {"type": "text", "text": "Stop, do Y."}]`,
			extra:   true,
		},
		{
			name:    "an image beside results",
			content: `[{"type": "tool_result", "tool_use_id": "a"}, {"type": "image", "source": {}}]`,
			extra:   true,
		},
		{
			// A held turn takes text only, so the image has to ride a replayed prompt.
			name: "an image inside a result",
			content: `[{"type": "tool_result", "tool_use_id": "a", "content": [
				{"type": "text", "text": "shot"}, {"type": "image", "source": ` + pngSource + `}
			]}]`,
			extra: true,
		},
		{
			name:    "a pdf inside a result",
			content: `[{"type": "tool_result", "tool_use_id": "a", "content": [{"type": "document", "source": ` + pdfSource + `}]}]`,
			extra:   true,
		},
		{
			// Nothing a replay could forward: the held turn reads the text and the placeholder.
			name: "a text document and a url image inside a result",
			content: `[{"type": "tool_result", "tool_use_id": "a", "content": [
				{"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "body"}},
				{"type": "image", "source": {"type": "url", "url": "https://example.test/a.png"}}
			]}]`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			messages := []message{{Role: "user", Content: json.RawMessage(tt.content)}}

			_, extra, err := toolResults(messages)
			if err != nil {
				t.Fatal(err)
			}

			if extra != tt.extra {
				t.Fatalf("extra = %t, want %t", extra, tt.extra)
			}
		})
	}
}

func TestConcatMergesNeighbouringTextAndKeepsMedia(t *testing.T) {
	image := promptBlock{Type: "image", MimeType: "image/png", Data: "AAAA"}
	parts := []prompt{
		textPrompt("a"),
		textPrompt("b"),
		{image},
		textPrompt("c"),
		nil,
		textPrompt("d"),
		{image},
	}

	var want prompt
	for _, part := range parts {
		want.add(part)
	}

	got := concat(parts)

	assertSameJSON(t, mustJSON(t, got), mustJSON(t, want))

	if len(got) != 4 || got[0].Text != "ab" || got[2].Text != "cd" {
		t.Fatalf("concat = %+v, want text(ab) image text(cd) image", got)
	}

	if len(concat(nil)) != 0 {
		t.Fatalf("concat(nil) must be empty")
	}
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()

	raw, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}

	return string(raw)
}
