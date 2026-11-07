package agybridge

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"net/http"
	"strconv"
	"strings"
)

func decodeBody(w http.ResponseWriter, r *http.Request, v any) error {
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, maxBodyBytes)).Decode(v); err != nil {
		return fmt.Errorf("request body: %w", err)
	}

	return nil
}

func invalidRequest(message string) *apiError {
	return &apiError{status: http.StatusBadRequest, kind: "invalid_request_error", message: message}
}

func overloaded(message string) *apiError {
	return &apiError{status: statusOverloaded, kind: "overloaded_error", message: message}
}

func asAPIError(err error) *apiError {
	var failure *apiError
	if errors.As(err, &failure) {
		return failure
	}

	return &apiError{status: http.StatusInternalServerError, kind: "api_error", message: err.Error()}
}

func outcomeOf(failure *apiError) string {
	return strconv.Itoa(failure.status) + " " + failure.kind
}

func reject(w http.ResponseWriter, failure *apiError) string {
	writeError(w, failure)

	return outcomeOf(failure)
}

// errorEnvelope is the Anthropic error body; a struct keeps its field order.
type errorEnvelope struct {
	Type  string      `json:"type"`
	Error errorDetail `json:"error"`
}

type errorDetail struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func envelope(failure *apiError) errorEnvelope {
	return errorEnvelope{Type: "error", Error: errorDetail{Type: failure.kind, Message: failure.message}}
}

func writeError(w http.ResponseWriter, failure *apiError) {
	switch {
	case failure.retryAfter > 0:
		w.Header().Set("Retry-After", strconv.Itoa(int(math.Ceil(failure.retryAfter.Seconds()))))
	case failure.status == http.StatusTooManyRequests || failure.status == statusOverloaded:
		w.Header().Set("Retry-After", "5")
	}

	writeJSON(w, failure.status, envelope(failure))
}

// accountHeader names the account a response's agy session ran under, by id
// (never email or token), so the gateway can tell which credential answered.
const accountHeader = "X-Kodizm-Account"

// markAccount sets the header for the session that will write the response.
// It only fills the header map: a retry on another account replaces the value
// while the client has seen nothing, and the head carries whichever is last.
func markAccount(w http.ResponseWriter, s *session) {
	w.Header().Set(accountHeader, s.lease.Account.ID)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	body, err := json.Marshal(v)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)

		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(status)

	if _, err := w.Write(body); err != nil {
		log.Printf("write response: %v", err)
	}
}

// responder renders a turn as either an SSE stream or one JSON message.
type responder interface {
	text(chunk string) error
	ping() error
	finish(end turnEnd) error
	fail(failure *apiError)
	// commit sends the response head before any content, where the format
	// has one to send.
	commit() error
	// pristine reports whether nothing of the turn has reached it yet, so the
	// turn may still start over on another account.
	pristine() bool
}

// messageBody is an Anthropic message; a struct keeps its field order.
type messageBody struct {
	ID           string       `json:"id"`
	Type         string       `json:"type"`
	Role         string       `json:"role"`
	Model        string       `json:"model"`
	Content      []contentOut `json:"content"`
	StopReason   *string      `json:"stop_reason"`
	StopSequence *string      `json:"stop_sequence"`
	Usage        usage        `json:"usage"`
}

type contentOut struct {
	Type  string          `json:"type"`
	Text  *string         `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

func textBlock(text string) contentOut {
	return contentOut{Type: "text", Text: &text}
}

func toolUseBlock(call toolCall, input json.RawMessage) contentOut {
	return contentOut{Type: "tool_use", ID: call.id, Name: call.name, Input: input}
}

// collector answers a non-streaming request with one message at the end.
type collector struct {
	w     http.ResponseWriter
	id    string
	model string
	buf   strings.Builder
}

func (c *collector) text(chunk string) error {
	c.buf.WriteString(chunk)

	return nil
}

func (c *collector) ping() error {
	return nil
}

// commit sends nothing: one JSON message has no head to send before its end.
func (c *collector) commit() error {
	return nil
}

func (c *collector) finish(end turnEnd) error {
	content := []contentOut{}
	if c.buf.Len() > 0 {
		content = append(content, textBlock(c.buf.String()))
	}

	for _, call := range end.calls {
		content = append(content, toolUseBlock(call, call.input))
	}

	writeJSON(c.w, http.StatusOK, messageBody{
		ID:         c.id,
		Type:       "message",
		Role:       "assistant",
		Model:      c.model,
		Content:    content,
		StopReason: &end.stopReason,
		Usage:      end.usage.messages(),
	})

	return nil
}

func (c *collector) fail(failure *apiError) {
	writeError(c.w, failure)
}

func (c *collector) pristine() bool {
	return c.buf.Len() == 0
}

// sseResponder streams the turn. Nothing, not even a header or a ping, goes
// out before the first real event or a commit, so a failure before either
// still gets a real HTTP status a gateway can fail over on.
type sseResponder struct {
	w         http.ResponseWriter
	rc        *http.ResponseController
	id        string
	model     string
	started   bool
	index     int
	blockOpen bool
}

func (o *sseResponder) start() error {
	if o.started {
		return nil
	}

	o.started = true

	o.w.Header().Set("Content-Type", "text/event-stream")
	o.w.Header().Set("Cache-Control", "no-cache")
	o.w.WriteHeader(http.StatusOK)

	return o.event("message_start", map[string]any{
		"type": "message_start",
		"message": messageBody{
			ID:      o.id,
			Type:    "message",
			Role:    "assistant",
			Model:   o.model,
			Content: []contentOut{},
		},
	})
}

func (o *sseResponder) text(chunk string) error {
	if err := o.start(); err != nil {
		return err
	}

	if !o.blockOpen {
		o.blockOpen = true

		err := o.event("content_block_start", map[string]any{
			"type":          "content_block_start",
			"index":         o.index,
			"content_block": textBlock(""),
		})
		if err != nil {
			return err
		}
	}

	return o.event("content_block_delta", map[string]any{
		"type":  "content_block_delta",
		"index": o.index,
		"delta": map[string]string{"type": "text_delta", "text": chunk},
	})
}

func (o *sseResponder) ping() error {
	if !o.started {
		return nil
	}

	return o.event("ping", map[string]string{"type": "ping"})
}

func (o *sseResponder) finish(end turnEnd) error {
	// 1. Close the open text block, if any.
	if err := o.start(); err != nil {
		return err
	}

	if err := o.closeBlock(); err != nil {
		return err
	}

	// 2. Each tool call is a whole block: its input goes out as one delta.
	for _, call := range end.calls {
		o.blockOpen = true

		err := o.event("content_block_start", map[string]any{
			"type":          "content_block_start",
			"index":         o.index,
			"content_block": toolUseBlock(call, json.RawMessage(`{}`)),
		})
		if err != nil {
			return err
		}

		err = o.event("content_block_delta", map[string]any{
			"type":  "content_block_delta",
			"index": o.index,
			"delta": map[string]string{"type": "input_json_delta", "partial_json": string(call.input)},
		})
		if err != nil {
			return err
		}

		if err := o.closeBlock(); err != nil {
			return err
		}
	}

	// 3. Stop reason and usage close the message.
	err := o.event("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": end.stopReason, "stop_sequence": nil},
		"usage": end.usage.messages(),
	})
	if err != nil {
		return err
	}

	return o.event("message_stop", map[string]string{"type": "message_stop"})
}

// closeBlock ends the open block and moves to the next index.
func (o *sseResponder) closeBlock() error {
	if !o.blockOpen {
		return nil
	}

	o.blockOpen = false
	index := o.index
	o.index++

	return o.event("content_block_stop", map[string]any{"type": "content_block_stop", "index": index})
}

func (o *sseResponder) fail(failure *apiError) {
	if !o.started {
		writeError(o.w, failure)

		return
	}

	if err := o.event("error", envelope(failure)); err != nil {
		log.Printf("%s: write stream error: %v", o.id, err)
	}
}

func (o *sseResponder) commit() error {
	return o.start()
}

func (o *sseResponder) pristine() bool {
	return !o.started
}

func (o *sseResponder) event(name string, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	if _, err := fmt.Fprintf(o.w, "event: %s\ndata: %s\n\n", name, data); err != nil {
		return err
	}

	return o.rc.Flush()
}
