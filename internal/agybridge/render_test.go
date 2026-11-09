package agybridge

import (
	"encoding/json"
	"strconv"
	"testing"
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
			name: "history with tools, images and thinking",
			messages: `[
				{"role": "user", "content": [{"type": "text", "text": "Read it"}, {"type": "image", "source": {}}]},
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
				"[user]\nRead it\n[image omitted]\n\n" +
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

			if got != tt.want {
				t.Fatalf("renderPrompt =\n%q\nwant\n%q", got, tt.want)
			}
		})
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

	got, err := resultText(block)
	if err != nil {
		t.Fatal(err)
	}

	if want := "Found:\ntool_reference: WebFetch\ntool_reference: WebSearch"; got != want {
		t.Fatalf("resultText = %q, want %q", got, want)
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
