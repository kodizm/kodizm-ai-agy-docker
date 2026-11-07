package agybridge

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"sync"
	"time"
)

// stubCallTimeout outlasts the bridge's own hold on a tool call, so the
// bridge, not the stub, decides when a call has waited too long.
const stubCallTimeout = 30 * time.Minute

// mcpStub is the stdio MCP server agy starts for the client's tools. It
// lists the tools the bridge wrote to TOOLS_FILE and forwards every call to
// the bridge's unix socket BRIDGE_SOCKET for session BRIDGE_SESSION, blocking
// until the bridge has the client's result.
type mcpStub struct {
	tools   json.RawMessage
	callURL string
	client  *http.Client

	writeMu sync.Mutex
	out     io.Writer
}

type textContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type callToolResult struct {
	Content []textContent `json:"content"`
	IsError bool          `json:"isError"`
}

// RunMCPStub serves MCP over in and out until in ends. getenv supplies
// TOOLS_FILE, BRIDGE_SOCKET and BRIDGE_SESSION.
func RunMCPStub(ctx context.Context, in io.Reader, out io.Writer, getenv func(string) string) error {
	tools, err := os.ReadFile(getenv("TOOLS_FILE"))
	if err != nil {
		return fmt.Errorf("mcp stub tools: %w", err)
	}

	if !json.Valid(tools) {
		return errors.New("mcp stub tools: TOOLS_FILE is not valid JSON")
	}

	socket, session := getenv("BRIDGE_SOCKET"), getenv("BRIDGE_SESSION")
	if socket == "" || session == "" {
		return errors.New("mcp stub: BRIDGE_SOCKET and BRIDGE_SESSION must be set")
	}

	transport := &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var dialer net.Dialer

			return dialer.DialContext(ctx, "unix", socket)
		},
	}
	defer transport.CloseIdleConnections()

	stub := &mcpStub{
		tools: tools,
		// The host is a placeholder: every connection dials the socket.
		callURL: "http://agy-bridge/sessions/" + url.PathEscape(session) + "/call",
		client: &http.Client{
			Transport: transport,
			Timeout:   stubCallTimeout,
		},
		out: out,
	}

	ctx, cancel := context.WithCancel(ctx)

	var calls sync.WaitGroup

	defer func() {
		cancel()
		calls.Wait()
	}()

	r := bufio.NewReader(in)

	for {
		line, err := r.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			stub.handle(ctx, &calls, line)
		}

		if errors.Is(err, io.EOF) {
			return nil
		}

		if err != nil {
			return fmt.Errorf("mcp stub read: %w", err)
		}
	}
}

func (s *mcpStub) handle(ctx context.Context, calls *sync.WaitGroup, line []byte) {
	var m rpcMessage
	if err := json.Unmarshal(line, &m); err != nil {
		log.Printf("mcp stub: skipped a malformed line (%d bytes)", len(line))

		return
	}

	if !hasID(m.ID) {
		return
	}

	switch m.Method {
	case "initialize":
		s.reply(m.ID, s.initializeResult(m.Params))
	case "tools/list":
		s.reply(m.ID, map[string]json.RawMessage{"tools": s.tools})
	case "tools/call":
		// Calls run concurrently: agy issues parallel calls and each blocks
		// until the client has run it.
		calls.Go(func() {
			s.reply(m.ID, s.call(ctx, m.Params))
		})
	default:
		s.reply(m.ID, struct{}{})
	}
}

func (s *mcpStub) initializeResult(params json.RawMessage) map[string]any {
	var p struct {
		ProtocolVersion json.RawMessage `json:"protocolVersion"`
	}

	version := json.RawMessage(`"2025-06-18"`)
	if json.Unmarshal(params, &p) == nil && !isAbsent(p.ProtocolVersion) {
		version = p.ProtocolVersion
	}

	return map[string]any{
		"protocolVersion": version,
		"capabilities":    map[string]any{"tools": struct{}{}},
		"serverInfo":      map[string]string{"name": "client", "version": "0"},
	}
}

func (s *mcpStub) call(ctx context.Context, params json.RawMessage) callToolResult {
	result, err := s.forward(ctx, params)
	if err != nil {
		result = toolResult{Text: "bridge error: " + err.Error(), IsError: true}
	}

	return callToolResult{
		Content: []textContent{{Type: "text", Text: result.Text}},
		IsError: result.IsError,
	}
}

func (s *mcpStub) forward(ctx context.Context, params json.RawMessage) (toolResult, error) {
	var p struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}

	if err := json.Unmarshal(params, &p); err != nil {
		return toolResult{}, fmt.Errorf("tools/call params: %w", err)
	}

	body, err := json.Marshal(map[string]any{
		"name":      p.Name,
		"arguments": compactJSON(p.Arguments),
	})
	if err != nil {
		return toolResult{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, s.callURL, bytes.NewReader(body))
	if err != nil {
		return toolResult{}, err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return toolResult{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return toolResult{}, fmt.Errorf("bridge answered %s", resp.Status)
	}

	var result toolResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return toolResult{}, fmt.Errorf("bridge answer: %w", err)
	}

	return result, nil
}

func (s *mcpStub) reply(id json.RawMessage, result any) {
	line, err := json.Marshal(rpcResponse{JSONRPC: "2.0", ID: id, Result: result})
	if err != nil {
		log.Printf("mcp stub: encode reply: %v", err)

		return
	}

	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	if _, err := s.out.Write(append(line, '\n')); err != nil {
		log.Printf("mcp stub: write reply: %v", err)
	}
}
