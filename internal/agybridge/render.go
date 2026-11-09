package agybridge

import (
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// toolHint keeps the model on the client's tools: agy's built-in tools are
// denied, so a call to one of them only wastes a step.
const toolHint = "Only call tools whose names start with mcp_client_. Never use built-in tools such as " +
	"run_command, view_file, write_to_file or list_dir: the client performs every action itself."

// messagesRequest is the part of an Anthropic Messages request the bridge reads.
type messagesRequest struct {
	Model    string          `json:"model"`
	Stream   bool            `json:"stream"`
	System   json.RawMessage `json:"system"`
	Messages []message       `json:"messages"`
	Tools    []toolDef       `json:"tools"`
	// OutputConfig and Thinking stay raw: a field of an unexpected shape is no
	// effort, and must not make the whole request a 400.
	OutputConfig json.RawMessage `json:"output_config"`
	Thinking     json.RawMessage `json:"thinking"`
	// ToolChoice and MaxTokens are read for the request log only, so they stay
	// raw too.
	ToolChoice json.RawMessage `json:"tool_choice"`
	MaxTokens  json.RawMessage `json:"max_tokens"`
}

// toolChoiceType is the type of the request's tool_choice, "-" when it has none.
func toolChoiceType(req *messagesRequest) string {
	var choice struct {
		Type string `json:"type"`
	}

	if json.Unmarshal(req.ToolChoice, &choice) != nil {
		return "-"
	}

	// A client string, so it is cut before it reaches the log.
	return cmp.Or(choice.Type[:min(len(choice.Type), toolChoiceLogBytes)], "-")
}

// toolChoiceLogBytes bounds the tool_choice type written to the request log.
const toolChoiceLogBytes = 32

// maxTokens is the request's max_tokens, 0 when it names none.
func maxTokens(req *messagesRequest) int {
	var n float64

	if json.Unmarshal(req.MaxTokens, &n) != nil {
		return 0
	}

	return int(n)
}

// agy's thinking levels; a model id agy takes is "<base>-<level>".
const (
	levelLow    = "low"
	levelMedium = "medium"
	levelHigh   = "high"
)

// requestLevel picks agy's thinking level: an effort wins over a thinking
// budget, and agy's own default, medium, covers adaptive or absent thinking.
func requestLevel(req *messagesRequest) string {
	var config struct {
		Effort string `json:"effort"`
	}

	if json.Unmarshal(req.OutputConfig, &config) == nil {
		switch config.Effort {
		case "low":
			return levelLow
		case "medium":
			return levelMedium
		case "high", "xhigh", "max":
			return levelHigh
		}
	}

	// The budget is a float so that a huge or fractional number still decodes.
	var thinking struct {
		Type         string  `json:"type"`
		BudgetTokens float64 `json:"budget_tokens"`
	}

	if json.Unmarshal(req.Thinking, &thinking) != nil {
		return levelMedium
	}

	switch thinking.Type {
	case "disabled":
		return levelLow
	case "enabled":
		// A heuristic: Google's own table maps the Gemini 2.5 budgets 1,024,
		// 8,192 and 24,576 to low, medium and high.
		switch {
		case thinking.BudgetTokens <= 1024:
			return levelLow
		case thinking.BudgetTokens <= 8192:
			return levelMedium
		default:
			return levelHigh
		}
	}

	return levelMedium
}

// imageBytes is the base64 an image block carries, in the messages and inside
// tool results. agy never gets it (an image renders as a placeholder), so it
// is not context. A message that does not decode was refused by the render.
func imageBytes(messages []message) int {
	total := 0

	for _, m := range messages {
		if _, blocks, err := decodeContent(m.Content); err == nil {
			total += blocksImageBytes(blocks)
		}
	}

	return total
}

func blocksImageBytes(blocks []contentBlock) int {
	total := 0

	for _, b := range blocks {
		switch b.Type {
		case "image":
			var source struct {
				Data string `json:"data"`
			}

			if json.Unmarshal(b.Source, &source) == nil {
				total += len(source.Data)
			}
		case "tool_result":
			if _, inner, err := decodeContent(b.Content); err == nil {
				total += blocksImageBytes(inner)
			}
		}
	}

	return total
}

// message holds content that is either a plain string or a block array.
type message struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

type contentBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	ToolUseID string          `json:"tool_use_id"`
	Content   json.RawMessage `json:"content"`
	IsError   bool            `json:"is_error"`
	ToolName  string          `json:"tool_name"`
	Source    json.RawMessage `json:"source"`
}

type toolDef struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// mcpTool is one entry of the stub's tools/list answer.
type mcpTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"`
}

// toolResult is a tool call's answer as the bridge hands it to the MCP stub.
type toolResult struct {
	Text    string `json:"text"`
	IsError bool   `json:"is_error"`
}

// deliveredResult is a tool_result the client sent back for a held call.
type deliveredResult struct {
	id     string
	result toolResult
}

// renderPrompt flattens the conversation into the single text prompt agy
// takes: earlier messages become a history section, the last one follows.
func renderPrompt(messages []message) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}

	last, err := contentText(messages[len(messages)-1].Content)
	if err != nil {
		return "", err
	}

	if len(messages) == 1 {
		return last, nil
	}

	history := make([]string, 0, len(messages)-1)

	for _, m := range messages[:len(messages)-1] {
		text, err := contentText(m.Content)
		if err != nil {
			return "", err
		}

		history = append(history, "["+m.Role+"]\n"+text)
	}

	return "<conversation_history>\n" + strings.Join(history, "\n\n") + "\n</conversation_history>\n\n" + last, nil
}

// systemText joins the text of a system field given as a string or blocks.
func systemText(raw json.RawMessage) (string, error) {
	text, blocks, err := decodeContent(raw)
	if err != nil {
		return "", fmt.Errorf("system: %w", err)
	}

	if blocks == nil {
		return text, nil
	}

	parts := make([]string, 0, len(blocks))

	for _, b := range blocks {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}

	return strings.Join(parts, "\n\n"), nil
}

// noToolsReminder holds the two phrases of the reminder Claude Code ends its compaction request with
// ("REMINDER: Do NOT call any tools. Respond with plain text only ..."). Either one alone also appears in
// ordinary prompts, so a message needs both.
var noToolsReminder = []string{"Do NOT call any tools", "Respond with plain text only"}

// forbidsTools reports whether the conversation ends in a user message carrying the no-tools reminder. Claude
// Code's compaction forks the conversation with the same tools and asks for a text summary; a model offered the
// tools calls one, the response ends as tool_use, and the compaction fails as an empty summary. Claude Code
// appends system messages after the last user message on every step, so those are skipped.
func forbidsTools(messages []message) bool {
	last := len(messages) - 1
	for last >= 0 && messages[last].Role == "system" {
		last--
	}

	if last < 0 || messages[last].Role != "user" {
		return false
	}

	text, blocks, err := decodeContent(messages[last].Content)
	if err != nil {
		return false
	}

	if blocks != nil {
		parts := make([]string, 0, len(blocks))

		for _, b := range blocks {
			if b.Type == "text" {
				parts = append(parts, b.Text)
			}
		}

		text = strings.Join(parts, "\n")
	}

	for _, phrase := range noToolsReminder {
		if !strings.Contains(text, phrase) {
			return false
		}
	}

	return true
}

// clientTools keeps the tools the client runs itself; server tools such as
// web_search carry their own type and are not handed to agy.
func clientTools(defs []toolDef) []mcpTool {
	var tools []mcpTool

	for _, d := range defs {
		if d.Name == "" || (d.Type != "" && d.Type != "custom") {
			continue
		}

		schema := d.InputSchema
		if isAbsent(schema) {
			schema = json.RawMessage(`{"type":"object"}`)
		}

		tools = append(tools, mcpTool{
			Name:        d.Name,
			Description: d.Description,
			InputSchema: stringEnumsOnly(schema),
		})
	}

	return tools
}

// stringEnumsOnly rewrites a tool schema for Gemini, which agy passes it to
// as a function declaration: Gemini rejects an enum on any type but string
// and agy then ends the whole turn ("Agent execution terminated"). Each such
// enum is removed and its values go into the description, so the model still
// sees the allowed values. A schema that needs no rewrite, or that is not
// valid JSON, comes back byte for byte; it is the client's to get right.
func stringEnumsOnly(schema json.RawMessage) json.RawMessage {
	dec := json.NewDecoder(bytes.NewReader(schema))
	dec.UseNumber()

	var root any
	if err := dec.Decode(&root); err != nil {
		return schema
	}

	if !rewriteEnums(root) {
		return schema
	}

	rewritten, err := json.Marshal(root)
	if err != nil {
		return schema
	}

	return rewritten
}

// rewriteEnums walks a decoded schema, rewrites every enum Gemini refuses in
// place and reports whether it changed anything.
func rewriteEnums(node any) bool {
	changed := false

	switch n := node.(type) {
	case map[string]any:
		if values, ok := n["enum"].([]any); ok && !stringEnum(n["type"], values) {
			delete(n, "enum")
			n["description"] = describeAllowed(n["description"], values)
			changed = true
		}

		for _, child := range n {
			changed = rewriteEnums(child) || changed
		}
	case []any:
		for _, child := range n {
			changed = rewriteEnums(child) || changed
		}
	}

	return changed
}

// stringEnum reports whether an enum is one Gemini takes: a declared type
// that is string (or absent, or a list such as ["string", "null"]) and values
// that are strings, null allowed.
func stringEnum(declared any, values []any) bool {
	if name, ok := declared.(string); ok && name != "string" {
		return false
	}

	for _, value := range values {
		switch value.(type) {
		case string, nil:
		default:
			return false
		}
	}

	return true
}

// describeAllowed appends an enum's values to a schema description.
func describeAllowed(existing any, values []any) string {
	rendered := make([]string, 0, len(values))

	for _, value := range values {
		switch v := value.(type) {
		case string:
			rendered = append(rendered, strconv.Quote(v))
		case nil:
			rendered = append(rendered, "null")
		default:
			rendered = append(rendered, fmt.Sprint(v))
		}
	}

	allowed := "one of: " + strings.Join(rendered, ", ")

	if text, ok := existing.(string); ok && strings.TrimSpace(text) != "" {
		return strings.TrimSpace(text) + " (" + allowed + ")"
	}

	return "One of: " + strings.Join(rendered, ", ")
}

// toolNames returns the tools' names sorted, duplicates kept.
func toolNames(tools []mcpTool) []string {
	names := make([]string, 0, len(tools))
	for _, tool := range tools {
		names = append(names, tool.Name)
	}

	slices.Sort(names)

	return names
}

// toolResults returns the tool_result blocks of the last user message, and
// whether that message carries anything else worth reading (a note, an
// image). A held turn can only take the results, so extra content means the
// conversation has to be replayed instead.
//
// Claude Code follows a tool result with system messages of its own (a token
// budget reminder, hook context), so the user message is looked for beneath
// them. Their text goes to the held turn after the last result, where the
// model would have read it, instead of costing the turn a replay.
func toolResults(messages []message) ([]deliveredResult, bool, error) {
	end := len(messages)

	var notes []string

	for end > 0 && messages[end-1].Role == "system" {
		text, err := contentText(messages[end-1].Content)
		if err != nil {
			return nil, false, err
		}

		if strings.TrimSpace(text) != "" {
			notes = append([]string{text}, notes...)
		}

		end--
	}

	if end == 0 || messages[end-1].Role != "user" {
		return nil, false, nil
	}

	_, blocks, err := decodeContent(messages[end-1].Content)
	if err != nil {
		return nil, false, err
	}

	var results []deliveredResult

	extra := false

	for _, b := range blocks {
		if b.Type != "tool_result" {
			extra = extra || b.Type != "text" || strings.TrimSpace(b.Text) != ""

			continue
		}

		text, err := resultText(b)
		if err != nil {
			return nil, false, err
		}

		results = append(results, deliveredResult{
			id: b.ToolUseID,
			result: toolResult{
				Text:    text,
				IsError: b.IsError,
			},
		})
	}

	if len(results) > 0 && len(notes) > 0 {
		last := &results[len(results)-1].result
		last.Text = strings.TrimRight(last.Text, "\n") + "\n\n" + strings.Join(notes, "\n\n")
	}

	return results, extra, nil
}

func contentText(raw json.RawMessage) (string, error) {
	text, blocks, err := decodeContent(raw)
	if err != nil || blocks == nil {
		return text, err
	}

	parts := make([]string, 0, len(blocks))

	for _, b := range blocks {
		part, err := blockText(b)
		if err != nil {
			return "", err
		}

		if part != "" {
			parts = append(parts, part)
		}
	}

	return strings.Join(parts, "\n"), nil
}

// blockText renders one block for the prompt; thinking and unknown blocks
// carry nothing agy can use and render empty.
func blockText(b contentBlock) (string, error) {
	switch b.Type {
	case "text":
		return b.Text, nil
	case "tool_use":
		return "[tool_use " + b.Name + " " + string(compactJSON(b.Input)) + "]", nil
	case "tool_result":
		text, err := resultText(b)
		if err != nil {
			return "", err
		}

		return "[tool_result " + b.ToolUseID + "]\n" + text, nil
	case "image":
		return "[image omitted]", nil
	}

	return "", nil
}

func resultText(b contentBlock) (string, error) {
	text, blocks, err := decodeContent(b.Content)
	if err != nil {
		return "", fmt.Errorf("tool_result %s: %w", b.ToolUseID, err)
	}

	if blocks == nil {
		return text, nil
	}

	parts := make([]string, 0, len(blocks))

	// A ToolSearch result is made of tool_reference blocks only.
	for _, block := range blocks {
		switch block.Type {
		case "text":
			parts = append(parts, block.Text)
		case "tool_reference":
			parts = append(parts, "tool_reference: "+block.ToolName)
		}
	}

	return strings.Join(parts, "\n"), nil
}

// decodeContent splits string content from block content; blocks is nil for
// string or absent content.
func decodeContent(raw json.RawMessage) (string, []contentBlock, error) {
	if isAbsent(raw) {
		return "", nil, nil
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil, nil
	}

	blocks := []contentBlock{}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", nil, errors.New("content must be a string or an array of blocks")
	}

	return "", blocks, nil
}

// compactJSON renders a tool input on one line; absent input is the empty object.
func compactJSON(raw json.RawMessage) json.RawMessage {
	if isAbsent(raw) {
		return json.RawMessage(`{}`)
	}

	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return json.RawMessage(`{}`)
	}

	return buf.Bytes()
}

func isAbsent(raw json.RawMessage) bool {
	trimmed := bytes.TrimSpace(raw)

	return len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null"))
}
