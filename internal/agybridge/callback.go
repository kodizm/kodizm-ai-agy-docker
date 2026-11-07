package agybridge

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// serveCallbacks listens for the MCP stub on a unix socket inside the 0700
// work dir. The callback never shares the public listener: it works whatever
// AGY_BRIDGE_LISTEN is, and only this user can reach it.
func (srv *Server) serveCallbacks() error {
	dir := filepath.Join(srv.root, "cb")
	if err := os.Mkdir(dir, 0o700); err != nil {
		return fmt.Errorf("callback dir: %w", err)
	}

	srv.socket = filepath.Join(dir, "s")

	ln, err := net.Listen("unix", srv.socket)
	if err != nil {
		return fmt.Errorf("callback socket: %w", err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /sessions/{id}/call", srv.handleCallback)

	srv.callbacks = &http.Server{
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	srv.callbacksDone = make(chan struct{})

	go func() {
		defer close(srv.callbacksDone)

		if err := srv.callbacks.Serve(ln); !errors.Is(err, http.ErrServerClosed) {
			log.Printf("callback socket stopped: %v", err)
		}
	}()

	return nil
}

// handleCallback holds one tool call from the MCP stub until the client's
// result arrives, the call times out, or the session ends.
func (srv *Server) handleCallback(w http.ResponseWriter, r *http.Request) {
	s := srv.reg.session(r.PathValue("id"))
	if s == nil {
		writeJSON(w, http.StatusNotFound, toolResult{Text: "unknown session", IsError: true})

		return
	}

	var body struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}

	if err := decodeBody(w, r, &body); err != nil {
		writeJSON(w, http.StatusBadRequest, toolResult{Text: err.Error(), IsError: true})

		return
	}

	callID, held, ok := srv.reg.hold(s)
	if !ok {
		writeJSON(w, http.StatusNotFound, toolResult{Text: "unknown session", IsError: true})

		return
	}

	s.events.push(event{kind: eventTool, call: toolCall{id: callID, name: body.Name, input: compactJSON(body.Arguments)}})

	timeout := time.NewTimer(toolResultTimeout)
	defer timeout.Stop()

	var result toolResult

	select {
	case result = <-held.result:
	case <-timeout.C:
		srv.reg.drop(callID)
		result = toolResult{Text: "tool result timeout", IsError: true}
	case <-s.done:
		srv.reg.drop(callID)
		result = toolResult{Text: "session closed", IsError: true}
	case <-r.Context().Done():
		srv.reg.drop(callID)

		return
	}

	writeJSON(w, http.StatusOK, result)
}
