package agybridge

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// stubHarness drives RunMCPStub in process the way agy does over stdio.
type stubHarness struct {
	in     *io.PipeWriter
	out    *bufio.Reader
	result chan error
}

// serveSocket serves handler on a unix socket the way the bridge does and
// returns the socket path. The directory is short: macOS caps socket paths
// at 104 bytes.
func serveSocket(t *testing.T, handler http.Handler) string {
	t.Helper()

	dir, err := os.MkdirTemp("", "stub-")
	if err != nil {
		t.Fatal(err)
	}

	socket := filepath.Join(dir, "s")

	ln, err := net.Listen("unix", socket)
	if err != nil {
		t.Fatal(err)
	}

	server := &httptest.Server{Listener: ln, Config: &http.Server{Handler: handler}}
	server.Start()

	t.Cleanup(func() {
		server.Close()
		_ = os.RemoveAll(dir)
	})

	return socket
}

func startStub(t *testing.T, socket string) *stubHarness {
	t.Helper()

	toolsFile := filepath.Join(t.TempDir(), "tools.json")
	writeFile(t, toolsFile, `[{"name":"read_file","description":"Reads","inputSchema":{"type":"object"}}]`)

	env := map[string]string{
		"TOOLS_FILE":     toolsFile,
		"BRIDGE_SOCKET":  socket,
		"BRIDGE_SESSION": "sess-1",
	}

	inR, inW := io.Pipe()
	outR, outW := io.Pipe()
	h := &stubHarness{in: inW, out: bufio.NewReader(outR), result: make(chan error, 1)}

	go func() {
		h.result <- RunMCPStub(context.Background(), inR, outW, func(k string) string { return env[k] })
		_ = outW.Close()
	}()

	t.Cleanup(func() {
		_ = inW.Close()

		select {
		case err := <-h.result:
			if err != nil {
				t.Errorf("RunMCPStub: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Error("RunMCPStub did not return after stdin closed")
		}
	})

	return h
}

func (h *stubHarness) send(t *testing.T, line string) {
	t.Helper()

	if _, err := io.WriteString(h.in, line+"\n"); err != nil {
		t.Fatal(err)
	}
}

func (h *stubHarness) read(t *testing.T) map[string]any {
	t.Helper()

	line, err := h.out.ReadBytes('\n')
	if err != nil {
		t.Fatalf("read stub output: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal(line, &m); err != nil {
		t.Fatalf("stub wrote %q: %v", line, err)
	}

	return m
}

func TestMCPStubProtocol(t *testing.T) {
	release := make(chan struct{})
	socket := serveSocket(t, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost || r.URL.Path != "/sessions/sess-1/call" {
			t.Errorf("stub called %s %s", r.Method, r.URL.Path)
		}

		var body struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}

		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode bridge call: %v", err)
		}

		// The first call blocks until the second has been answered.
		if string(body.Arguments) == `{"n":1}` {
			<-release
		}

		_ = json.NewEncoder(w).Encode(toolResult{Text: body.Name + " " + string(body.Arguments), IsError: true})
	}))

	h := startStub(t, socket)

	// 1. initialize echoes the protocol version.
	h.send(t, `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}`)

	got := h.read(t)
	result := got["result"].(map[string]any)

	if result["protocolVersion"] != "2025-03-26" || result["capabilities"].(map[string]any)["tools"] == nil {
		t.Fatalf("initialize answered %v", got)
	}

	// 2. A notification gets no answer; tools/list answers the file.
	h.send(t, `{"jsonrpc":"2.0","method":"notifications/initialized"}`)
	h.send(t, `{"jsonrpc":"2.0","id":2,"method":"tools/list"}`)

	got = h.read(t)
	tools := got["result"].(map[string]any)["tools"].([]any)

	if got["id"] != float64(2) || len(tools) != 1 || tools[0].(map[string]any)["name"] != "read_file" {
		t.Fatalf("tools/list answered %v", got)
	}

	// 3. Calls run concurrently: the second answers while the first is held.
	h.send(t, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"read_file","arguments":{"n":1}}}`)
	h.send(t, `{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"read_file","arguments":{"n":2}}}`)

	got = h.read(t)
	assertCallResult(t, got, 4, `read_file {"n":2}`, true)

	close(release)

	got = h.read(t)
	assertCallResult(t, got, 3, `read_file {"n":1}`, true)

	// 4. Any other request with an id gets an empty result.
	h.send(t, `{"jsonrpc":"2.0","id":"p","method":"ping"}`)

	got = h.read(t)
	if got["id"] != "p" || len(got["result"].(map[string]any)) != 0 {
		t.Fatalf("ping answered %v", got)
	}
}

func TestMCPStubReportsTransportFailure(t *testing.T) {
	h := startStub(t, filepath.Join(t.TempDir(), "missing.sock"))
	h.send(t, `{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"read_file"}}`)

	got := h.read(t)
	text := callText(t, got)

	if got["result"].(map[string]any)["isError"] != true || !strings.HasPrefix(text, "bridge error: ") {
		t.Fatalf("transport failure answered %v", got)
	}
}

func assertCallResult(t *testing.T, got map[string]any, id float64, text string, isError bool) {
	t.Helper()

	if got["id"] != id || callText(t, got) != text || got["result"].(map[string]any)["isError"] != isError {
		t.Fatalf("tools/call %v answered %v", id, got)
	}
}

func callText(t *testing.T, got map[string]any) string {
	t.Helper()

	content := got["result"].(map[string]any)["content"].([]any)
	if len(content) != 1 {
		t.Fatalf("want one content item, got %v", got)
	}

	return content[0].(map[string]any)["text"].(string)
}
