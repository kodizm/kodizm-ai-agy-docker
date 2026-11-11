package agybridge

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"unicode/utf8"
)

// agy reads about 192,000 bytes of one prompt and drops the rest without a word, so in a replayed conversation
// the newest messages and the request itself are what vanish. The default leaves room under that cut for
// whatever agy adds to the prompt.
const defaultMaxPromptBytes = 180_000

// reasonPromptTooLong is the request log's reason for a replay no shortening could fit.
const reasonPromptTooLong = "prompt_too_long"

// omissionNote opens a shortened prompt, so the model knows the markers are the bridge's and not the client's.
const omissionNote = `Note from the bridge: to fit Antigravity's input limit, the middle of some older messages ` +
	`was replaced by markers reading "omitted by the bridge". Re-read anything you still need, in smaller ` +
	`parts if it is large.`

// omissionMarker stands where the middle of a text was cut out.
func omissionMarker(omitted int) string {
	return fmt.Sprintf("\n[... %d bytes omitted by the bridge: Antigravity's input limit ...]\n", omitted)
}

// cut keeps keep bytes at each end of a text longer than min and puts a marker in place of the middle.
type cut struct {
	keep int
	min  int
}

// apply is the text cut, or the text itself when it is within min or the cut would not make it shorter. A nil
// cut leaves every text whole. Both ends are moved inward to a rune start.
func (c *cut) apply(text string) string {
	if c == nil || len(text) <= c.min {
		return text
	}

	head := c.keep
	for head > 0 && !utf8.RuneStart(text[head]) {
		head--
	}

	tail := len(text) - c.keep
	for tail < len(text) && !utf8.RuneStart(text[tail]) {
		tail++
	}

	if tail <= head {
		return text
	}

	out := text[:head] + omissionMarker(tail-head) + text[tail:]
	if len(out) >= len(text) {
		return text
	}

	return out
}

// applyJSON is a tool input as one line of JSON with every string cut on its own, so an old call keeps its keys
// and short values (an Edit still names its file) when only a long one goes. An input that does not decode is cut
// as the one string it is; a nil cut leaves the input as compactJSON has it.
func (c *cut) applyJSON(raw json.RawMessage) string {
	compact := compactJSON(raw)
	if c == nil {
		return string(compact)
	}

	decoder := json.NewDecoder(bytes.NewReader(compact))
	decoder.UseNumber()

	var value any
	if err := decoder.Decode(&value); err != nil {
		return c.apply(string(compact))
	}

	var out bytes.Buffer

	encoder := json.NewEncoder(&out)
	encoder.SetEscapeHTML(false)

	if err := encoder.Encode(c.cutStrings(value)); err != nil {
		return c.apply(string(compact))
	}

	return strings.TrimSuffix(out.String(), "\n")
}

// cutStrings is a decoded JSON value with every string in it cut.
func (c *cut) cutStrings(value any) any {
	switch v := value.(type) {
	case string:
		return c.apply(v)
	case []any:
		items := make([]any, len(v))
		for i, item := range v {
			items[i] = c.cutStrings(item)
		}

		return items
	case map[string]any:
		fields := make(map[string]any, len(v))
		for key, item := range v {
			fields[key] = c.cutStrings(item)
		}

		return fields
	}

	return value
}

// cutKind is the kind of content a cut shortens.
type cutKind int

const (
	kindResult cutKind = iota // the text of a tool_result
	kindUse                   // the arguments of a tool_use
	kindText                  // the text of a message
	kindCount
)

// cuts holds the cut of each kind; a nil entry leaves its kind whole, so the zero value shortens nothing.
type cuts [kindCount]*cut

// shrinkZone is a run of messages with the cuts to try on it, gentlest pass first.
type shrinkZone struct {
	indexes []int
	passes  []cuts
}

var (
	// agedPasses shorten what the model has long since answered. Tool output goes all the way to its marker
	// before any message text is touched: the task the owner typed is worth more than a result read an hour ago.
	agedPasses = []cuts{
		{kindResult: &cut{keep: 300, min: 2000}, kindUse: &cut{keep: 300, min: 2000}},
		{kindResult: &cut{keep: 0, min: 200}, kindUse: &cut{keep: 0, min: 200}},
		{kindText: &cut{keep: 2000, min: 6000}},
		{kindText: &cut{keep: 500, min: 1500}},
	}

	// newestPasses shorten the round the model is answering now: tool output only, and generously.
	newestPasses = []cuts{
		{kindResult: &cut{keep: 4000, min: 10000}, kindUse: &cut{keep: 4000, min: 10000}},
		{kindResult: &cut{keep: 1000, min: 2500}, kindUse: &cut{keep: 1000, min: 2500}},
	}
)

// textBytes is the size of the text agy reads: attachments ride blocks of their own.
func textBytes(p prompt) int {
	total := 0

	for _, b := range p {
		if b.Type == "text" {
			total += len(b.Text)
		}
	}

	return total
}

// promptTooLong is the refusal for a replay that does not fit however it is shortened. The wording is the one
// clients and the gateway match to treat a request as over the window: the gateway then moves on to a rung
// with room, and Claude Code compacts.
func promptTooLong(bytes, limit int) *apiError {
	failure := invalidRequest(fmt.Sprintf(
		"prompt is too long: %d bytes > %d maximum that Antigravity reads without silently dropping the end "+
			"of the conversation",
		bytes, limit,
	))
	failure.reason = reasonPromptTooLong

	return failure
}

// renderPromptWithin renders the conversation like renderPrompt, kept within limit bytes of text.
//
// agy drops what passes its limit from the end, where the newest messages are, so a replay that is too big is
// shortened instead. Before the client's current request, then in the tool work since it, tool output and
// arguments go first, oldest first, to their head and tail around a marker and then to the marker alone, and
// message text only after that, to head and tail; the newest round's tool output comes last. The request's own
// text is never cut. When even that does not fit the result is a 400 `prompt is too long`. A limit of zero or
// less turns the budget off (a Config built in code; the environment refuses it). The bool reports whether
// anything was cut.
func renderPromptWithin(messages []message, limit int) (prompt, bool, error) {
	base, err := renderPrompt(messages)
	if err != nil {
		return nil, false, invalidRequest(err.Error())
	}

	total := textBytes(base)
	if limit <= 0 || total <= limit {
		return base, false, nil
	}

	// 1. Know what each message weighs, so a cut can be priced without rendering the whole conversation again.
	sizes := make([]int, len(messages))

	for i, m := range messages {
		if sizes[i], err = contentBytes(m, cuts{}); err != nil {
			return nil, false, invalidRequest(err.Error())
		}
	}

	// 2. Cut the oldest first and stop at the first cut that makes the prompt fit; the note costs bytes too.
	plan := make([]cuts, len(messages))
	excess := total - (limit - len(omissionNote) - len("\n\n"))

	for _, zone := range shrinkZones(messages) {
		for _, pass := range zone.passes {
			for kind := range kindCount {
				if pass[kind] == nil {
					continue
				}

				for _, i := range zone.indexes {
					next := plan[i]
					next[kind] = pass[kind]

					size, err := contentBytes(messages[i], next)
					if err != nil {
						return nil, false, invalidRequest(err.Error())
					}

					if size >= sizes[i] {
						continue
					}

					excess -= sizes[i] - size
					sizes[i], plan[i] = size, next

					if excess <= 0 {
						return renderShortened(messages, plan, total, limit)
					}
				}
			}
		}
	}

	return nil, false, promptTooLong(total, limit)
}

// renderShortened is the final render of a plan that priced as fitting; the check guards the arithmetic.
func renderShortened(messages []message, plan []cuts, total, limit int) (prompt, bool, error) {
	shortened, err := renderCut(messages, plan)
	if err != nil {
		return nil, false, invalidRequest(err.Error())
	}

	if textBytes(shortened) > limit {
		return nil, false, promptTooLong(total, limit)
	}

	return shortened, true, nil
}

// contentBytes is the text size of one message's content under the cuts.
func contentBytes(m message, c cuts) (int, error) {
	r := promptRenderer{cuts: c}

	content, err := r.content(m.Content)

	return textBytes(content), err
}

// shrinkZones sorts the messages by how safe they are to cut: the ones before the client's current request (the
// newest user message with text of its own), the tool work since it, and the newest round: the last assistant
// and user messages and the request itself, whose text no pass touches.
func shrinkZones(messages []message) []shrinkZone {
	current, lastAssistant, lastUser := -1, -1, -1

	for i, m := range messages {
		switch m.Role {
		case "assistant":
			lastAssistant = i
		case "user":
			lastUser = i

			if isTypedRequest(m) {
				current = i
			}
		}
	}

	newest := map[int]bool{}

	for _, at := range []int{lastAssistant, lastUser, current} {
		if at >= 0 && at >= current {
			newest[at] = true
		}
	}

	var before, since, last []int

	for i := range messages {
		switch {
		case i < current:
			before = append(before, i)
		case newest[i]:
			last = append(last, i)
		case i > current:
			since = append(since, i)
		}
	}

	return []shrinkZone{
		{indexes: before, passes: agedPasses},
		{indexes: since, passes: agedPasses},
		{indexes: last, passes: newestPasses},
	}
}

// isTypedRequest reports whether a user message carries text of its own, as opposed to tool results alone.
func isTypedRequest(m message) bool {
	text, blocks, err := decodeContent(m.Content)
	if err != nil {
		return false
	}

	if blocks == nil {
		return strings.TrimSpace(text) != ""
	}

	return slices.ContainsFunc(blocks, func(b contentBlock) bool {
		return b.Type == "text" && strings.TrimSpace(b.Text) != ""
	})
}
