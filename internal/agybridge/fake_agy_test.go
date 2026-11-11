package agybridge

import (
	"bufio"
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// The test binary doubles as the fake `agy acp` and as the bridge's own MCP
// stub: the bridge spawns AGY_BIN (this binary) with "acp", and the fake
// spawns the stub command it is given, which is this binary with "mcp-stub".
const (
	fakeAgyEnv  = "AGY_BRIDGE_FAKE"
	fakePIDsEnv = "AGY_BRIDGE_FAKE_PIDS"

	// fakeAccountsEnv maps an access token to how the fake behaves under it:
	// ok, quota, rate, capacity, auth or verification. A token it does not
	// list is refused like an expired login.
	fakeAccountsEnv = "FAKE_AGY_ACCOUNTS"

	// fakeUsageEnv maps an access token to what `agy -p /usage` prints under
	// it; "fail" exits 1 silently and "hang" never ends. A token it does not
	// list gets agy's login prompt, which waits on stdin.
	fakeUsageEnv = "FAKE_AGY_USAGE"

	// fakeLoginSilentEnv makes `agy acp --login` print nothing and hang.
	fakeLoginSilentEnv = "FAKE_AGY_LOGIN_SILENT"

	// fakeCapabilitiesEnv is the agentCapabilities object the fake answers
	// initialize with, {} when unset.
	fakeCapabilitiesEnv = "FAKE_AGY_CAPABILITIES"

	fakeLoginURL   = "https://accounts.google.com/o/oauth2/auth?client_id=fake&state=fake"
	fakeLoginCode  = "good-code"
	fakeLoginEmail = "someone@example.test"
)

// fakeLoginTokenFile is what the fake writes on a good code, with obviously
// fake secrets.
const fakeLoginTokenFile = `{"token":{"access_token":"fake-access-token","token_type":"Bearer",` +
	`"refresh_token":"fake-refresh-token","expiry":"2026-11-07T13:00:00Z","expires_in":3599},` +
	`"auth_method":"consumer","id_token_email":"` + fakeLoginEmail + `"}`

// runFakeLogin plays agy 1.5.2's `acp --login`: the banner with the URL, one
// line of stdin, then a token file and exit 0 on the good code.
func runFakeLogin(in io.Reader) int {
	if os.Getenv(fakeLoginSilentEnv) != "" {
		time.Sleep(time.Hour)
	}

	fmt.Print("Open this link in a browser to log in with Google:\n\n  " + fakeLoginURL + "\n\n" +
		"Then paste the authorization code, or the full URL of the page you land on, and press Enter: ")

	line, err := bufio.NewReader(in).ReadString('\n')
	if err != nil && line == "" {
		fmt.Println("login cancelled: no authorization code was entered")

		return 1
	}

	line = strings.TrimSpace(line)
	if line != fakeLoginCode && !strings.Contains(line, "code="+fakeLoginCode) {
		fmt.Println("login failed: invalid code for " + fakeLoginEmail)

		return 1
	}

	path := filepath.Join(os.Getenv("HOME"), agyTokenPath)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		fmt.Println(err)

		return 1
	}

	if err := os.WriteFile(path, []byte(fakeLoginTokenFile), 0o600); err != nil {
		fmt.Println(err)

		return 1
	}

	fmt.Println("Logged in as " + fakeLoginEmail)

	return 0
}

func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == "mcp-stub" {
		recordPID("stub", "")

		if err := RunMCPStub(context.Background(), os.Stdin, os.Stdout, os.Getenv); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}

		os.Exit(0)
	}

	isAcp := os.Getenv(fakeAgyEnv) == "1" && len(os.Args) > 1 && os.Args[1] == "acp"

	if isAcp && slices.Contains(os.Args[2:], "--login") {
		recordPID("login", "")
		os.Exit(runFakeLogin(os.Stdin))
	}

	if isAcp {
		recordPID("agy", fakeAccessToken())
		os.Exit(runFakeAgy(os.Stdin, os.Stdout))
	}

	if os.Getenv(fakeAgyEnv) == "1" && len(os.Args) > 2 && os.Args[1] == "-p" && os.Args[2] == "/usage" {
		token := fakeAccessToken()
		recordPID("usage", token)
		os.Exit(runFakeUsage(token))
	}

	pids, err := os.MkdirTemp("", "agy-bridge-pids-")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	// The fake writes agy's session state under $HOME, so a test that does not
	// pick its own home must never reach the real one.
	home, err := os.MkdirTemp("", "agy-bridge-home-")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	_ = os.Setenv(fakeAgyEnv, "1")
	_ = os.Setenv(fakePIDsEnv, pids)
	_ = os.Setenv("HOME", home)
	_ = os.Setenv(fakeAccountsEnv, `{"tok-a":"ok","tok-b":"ok","tok-c":"ok"}`)
	_ = os.Unsetenv("AGY_CLI_DISABLE_AUTO_UPDATE")

	code := m.Run()

	_ = os.RemoveAll(pids)
	_ = os.RemoveAll(home)
	os.Exit(code)
}

// fakeSessionID is a UUID of the shape agy names its state files after; the
// pid keeps the ids of concurrent fakes apart.
func fakeSessionID() string {
	return fmt.Sprintf("00000000-0000-4000-8000-%012d", os.Getpid())
}

// fakeAgyHome is where the real agy keeps its state, rooted at $HOME.
func fakeAgyHome() string {
	return filepath.Join(os.Getenv("HOME"), ".gemini", "antigravity-cli")
}

// fakeAccessToken is the access token of the login under $HOME, or "".
func fakeAccessToken() string {
	raw, err := os.ReadFile(filepath.Join(os.Getenv("HOME"), agyTokenPath))
	if err != nil {
		return ""
	}

	var login agyTokenFile
	if json.Unmarshal(raw, &login) != nil {
		return ""
	}

	return login.Token.AccessToken
}

// fakeBehaviour is how the fake acts under token, or "" when it is unknown.
func fakeBehaviour(token string) string {
	var accounts map[string]string
	if json.Unmarshal([]byte(os.Getenv(fakeAccountsEnv)), &accounts) != nil {
		return ""
	}

	return accounts[token]
}

func runFakeUsage(token string) int {
	var outputs map[string]string
	_ = json.Unmarshal([]byte(os.Getenv(fakeUsageEnv)), &outputs)

	text, known := outputs[token]

	switch {
	case !known:
		// The real agy waits up to a minute for a pasted code; a stdin that
		// is the null device ends that wait at once.
		fmt.Println("Authentication required. Please visit the URL to log in:")
		fmt.Println("https://accounts.example.test/auth")
		_, _ = io.Copy(io.Discard, os.Stdin)

		return 1
	case text == "fail":
		return 1
	case text == "hang":
		fmt.Println("Gemini Models\tWeekly Limit Remaining\t1%")
		time.Sleep(time.Hour)
	}

	fmt.Print(text)

	return 0
}

// persistFakeState leaves the files the real agy leaves for a session.
func persistFakeState(sid string) {
	home := fakeAgyHome()

	for _, file := range []string{
		"conversation_summaries.db",
		"annotations/" + sid + ".pbtxt",
		"brain/" + sid + "/plan.md",
		"conversations/" + sid + ".db",
		"conversations/" + sid + ".db-wal",
		"conversations/" + sid + ".db-shm",
		"log/acp-server-20260101T000000-" + strconv.Itoa(os.Getpid()) + ".log",
	} {
		path := filepath.Join(home, file)

		if err := os.MkdirAll(filepath.Dir(path), 0o700); err == nil {
			_ = os.WriteFile(path, []byte("state"), 0o600)
		}
	}
}

// recordPID leaves a file named for the process; it holds marker, the access
// token an agy runs under, so a test can tell which account started it.
func recordPID(kind, marker string) {
	dir := os.Getenv(fakePIDsEnv)
	name := filepath.Join(dir, kind+"-"+strconv.Itoa(os.Getpid()))

	if err := os.WriteFile(name, []byte(marker), 0o600); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// peer is one end of a newline-delimited JSON-RPC link.
type peer struct {
	writeMu sync.Mutex
	w       io.Writer

	next    atomic.Int64
	mu      sync.Mutex
	waiting map[int64]chan rpcMessage
}

func newPeer(w io.Writer) *peer {
	return &peer{w: w, waiting: map[int64]chan rpcMessage{}}
}

func (p *peer) send(v any) {
	line, _ := json.Marshal(v)

	p.writeMu.Lock()
	defer p.writeMu.Unlock()

	_, _ = p.w.Write(append(line, '\n'))
}

func (p *peer) respond(id json.RawMessage, result any) {
	p.send(rpcResponse{JSONRPC: "2.0", ID: id, Result: result})
}

func (p *peer) notify(method string, params any) {
	p.send(map[string]any{"jsonrpc": "2.0", "method": method, "params": params})
}

// request blocks until the other side answers.
func (p *peer) request(method string, params any) rpcMessage {
	id := p.next.Add(1) + 1000
	reply := make(chan rpcMessage, 1)

	p.mu.Lock()
	p.waiting[id] = reply
	p.mu.Unlock()

	p.send(rpcRequest{JSONRPC: "2.0", ID: id, Method: method, Params: params})

	return <-reply
}

// serve reads r until it ends, routing answers to request and everything
// else to handle.
func (p *peer) serve(r io.Reader, handle func(rpcMessage)) {
	reader := bufio.NewReader(r)

	for {
		line, err := reader.ReadBytes('\n')
		if len(strings.TrimSpace(string(line))) > 0 {
			var m rpcMessage
			if json.Unmarshal(line, &m) == nil {
				p.route(m, handle)
			}
		}

		if err != nil {
			return
		}
	}
}

func (p *peer) route(m rpcMessage, handle func(rpcMessage)) {
	if m.Method != "" {
		handle(m)

		return
	}

	var id int64
	if json.Unmarshal(m.ID, &id) != nil {
		return
	}

	p.mu.Lock()
	reply := p.waiting[id]
	delete(p.waiting, id)
	p.mu.Unlock()

	if reply != nil {
		reply <- m
	}
}

// fakeAgy speaks enough ACP to play scripts chosen by the prompt text.
type fakeAgy struct {
	bridge *peer
	system string
	model  string
	cwd    string
	stub   *peer
	tools  []string

	// token and behaviour come from the login under $HOME at initialize.
	token     string
	behaviour string

	// agentFile is what the fake found in the workspace when session/new
	// arrived, which is when the real agy discovers its custom agents.
	agentFile fakeAgentFile

	// blocks is the session/prompt content as the bridge sent it.
	blocks json.RawMessage
}

// fakeAgentFile describes the clean agent file as agy would find it.
type fakeAgentFile struct {
	Exists   bool     `json:"exists"`
	Content  string   `json:"content"`
	FileMode string   `json:"fileMode"`
	DirModes []string `json:"dirModes"`
}

func inspectAgentFile(cwd string) fakeAgentFile {
	dir := filepath.Join(cwd, ".agents", "agents")
	file := filepath.Join(dir, "clean-chat.md")

	content, err := os.ReadFile(file)
	if err != nil {
		return fakeAgentFile{}
	}

	found := fakeAgentFile{Exists: true, Content: string(content)}

	if info, err := os.Stat(file); err == nil {
		found.FileMode = info.Mode().Perm().String()
	}

	for _, d := range []string{cwd, filepath.Join(cwd, ".agents"), dir} {
		if info, err := os.Stat(d); err == nil {
			found.DirModes = append(found.DirModes, info.Mode().Perm().String())
		}
	}

	return found
}

// fakeStateFiles lists every file under the fake's agy home.
func fakeStateFiles() []string {
	home := fakeAgyHome()

	var files []string

	_ = filepath.WalkDir(home, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			rel, _ := filepath.Rel(home, path)
			files = append(files, rel)
		}

		return nil
	})

	slices.Sort(files)

	return files
}

func runFakeAgy(in io.Reader, out io.Writer) int {
	f := &fakeAgy{bridge: newPeer(out)}
	f.bridge.serve(in, f.handle)

	return 0
}

func (f *fakeAgy) handle(m rpcMessage) {
	switch m.Method {
	case "initialize":
		// Without a login it knows, agy cannot reach the model at all.
		f.token = fakeAccessToken()
		f.behaviour = fakeBehaviour(f.token)

		if f.behaviour == "" || f.behaviour == "auth" {
			f.bridge.send(map[string]any{
				"jsonrpc": "2.0",
				"id":      m.ID,
				"error": map[string]any{
					"code":    -32000,
					"message": "Authentication required. Please visit the URL to log in: https://accounts.example.test/auth",
				},
			})

			return
		}

		capabilities := json.RawMessage(cmp.Or(os.Getenv(fakeCapabilitiesEnv), "{}"))
		f.bridge.respond(m.ID, map[string]any{"protocolVersion": 1, "agentCapabilities": capabilities})
	case "session/new":
		var p newSessionParams
		_ = json.Unmarshal(m.Params, &p)

		f.cwd = p.Cwd
		f.agentFile = inspectAgentFile(p.Cwd)

		if p.Meta != nil {
			f.system = p.Meta.Antigravity.SystemPrompt.Append
		}

		if len(p.MCPServers) > 0 {
			f.startStub(p.MCPServers[0])
		}

		// agy refuses the session before storing anything.
		if os.Getenv("FAKE_AGY_FAIL_NEW") != "" {
			f.bridge.send(map[string]any{
				"jsonrpc": "2.0",
				"id":      m.ID,
				"error":   map[string]any{"code": -32000, "message": "refused"},
			})

			return
		}

		persistFakeState(fakeSessionID())

		// A session agy stored but never announced: the reply does not come.
		if os.Getenv("FAKE_AGY_HANG_NEW") != "" {
			return
		}

		f.bridge.respond(m.ID, map[string]string{"sessionId": fakeSessionID()})
	case "session/set_model":
		var p struct {
			ModelID string `json:"modelId"`
		}
		_ = json.Unmarshal(m.Params, &p)

		f.model = p.ModelID
		f.bridge.respond(m.ID, map[string]any{})
	case "session/prompt":
		var p struct {
			Prompt json.RawMessage `json:"prompt"`
		}
		_ = json.Unmarshal(m.Params, &p)

		var blocks []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}
		_ = json.Unmarshal(p.Prompt, &blocks)

		// Scripts read the prompt's text; media blocks only split it.
		var text strings.Builder
		for _, block := range blocks {
			if block.Type == "text" {
				text.WriteString(block.Text)
			}
		}

		f.blocks = p.Prompt

		// The prompt runs beside the read loop so answers to the fake's own
		// requests (permission, nothing else) still arrive.
		go f.prompt(m.ID, text.String())
	default:
		if hasID(m.ID) {
			f.bridge.respond(m.ID, map[string]any{})
		}
	}
}

func (f *fakeAgy) startStub(server mcpServer) {
	cmd := exec.Command(server.Command, server.Args...)
	cmd.Env = os.Environ()

	for _, e := range server.Env {
		cmd.Env = append(cmd.Env, e.Name+"="+e.Value)
	}

	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()

	if err := cmd.Start(); err != nil {
		f.system = "stub start failed: " + err.Error()

		return
	}

	f.stub = newPeer(stdin)

	go f.stub.serve(stdout, func(rpcMessage) {})

	f.stub.request("initialize", map[string]any{"protocolVersion": "2025-06-18"})
	f.stub.notify("notifications/initialized", map[string]any{})

	var listed struct {
		Tools []mcpTool `json:"tools"`
	}

	_ = json.Unmarshal(f.stub.request("tools/list", map[string]any{}).Result, &listed)

	for _, tool := range listed.Tools {
		f.tools = append(f.tools, tool.Name)
	}
}

func (f *fakeAgy) chunk(text string) {
	f.bridge.notify("session/update", map[string]any{
		"sessionId": "fake-session",
		"update": map[string]any{
			"sessionUpdate": "agent_message_chunk",
			"content":       map[string]string{"type": "text", "text": text},
		},
	})
}

func (f *fakeAgy) chunkJSON(v any) {
	raw, _ := json.Marshal(v)
	f.chunk(string(raw))
}

func (f *fakeAgy) end(id json.RawMessage, stopReason string, in, out int) {
	f.bridge.respond(id, map[string]any{
		"stopReason": stopReason,
		"usage": map[string]int{
			"inputTokens":  in,
			"outputTokens": out,
			"totalTokens":  in + out,
		},
	})
}

// callTool runs one MCP tool call through the stub and renders its answer.
func (f *fakeAgy) callTool(args map[string]any) string {
	if len(f.tools) == 0 {
		return "no tools"
	}

	answer := f.stub.request("tools/call", map[string]any{"name": f.tools[0], "arguments": args})

	var result callToolResult
	_ = json.Unmarshal(answer.Result, &result)

	text := result.Content[0].Text
	if result.IsError {
		text += " (error)"
	}

	return text
}

const fakeBannerDelay = 300 * time.Millisecond

var banners = map[string]string{
	"429":          "⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Resource has been exhausted.",
	"503":          "⚠ API error (attempt 1): UNAVAILABLE (code 503): The service is currently unavailable.",
	"fatal":        "⚠ Agent execution terminated due to error.",
	"quota":        "⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Quota exceeded for Gemini Models.",
	"rate":         "⚠ API error (attempt 1): code 429: Too many requests, slow down.",
	"capacity":     "⚠ API error (attempt 1): UNAVAILABLE (code 503): The service is currently unavailable.",
	"verification": "⚠ API error (attempt 1): PERMISSION_DENIED (code 403): validation_required, see https://accounts.example.test/validate",
}

// fakeHome is what the fake found of the home it runs in.
type fakeHome struct {
	Home           string `json:"home"`
	TokenMode      string `json:"tokenMode"`
	HasRefresh     bool   `json:"hasRefresh"`
	AuthMethod     string `json:"authMethod"`
	Email          string `json:"email"`
	InstallationID string `json:"installationId"`
}

func inspectHome() fakeHome {
	home := fakeHome{Home: os.Getenv("HOME")}
	path := filepath.Join(home.Home, agyTokenPath)

	if info, err := os.Stat(path); err == nil {
		home.TokenMode = info.Mode().Perm().String()
	}

	var login struct {
		Token        map[string]any `json:"token"`
		AuthMethod   string         `json:"auth_method"`
		IDTokenEmail string         `json:"id_token_email"`
	}

	if raw, err := os.ReadFile(path); err == nil && json.Unmarshal(raw, &login) == nil {
		_, home.HasRefresh = login.Token["refresh_token"]
		home.AuthMethod, home.Email = login.AuthMethod, login.IDTokenEmail
	}

	if raw, err := os.ReadFile(filepath.Join(fakeAgyHome(), "installation_id")); err == nil {
		home.InstallationID = string(raw)
	}

	return home
}

// hang leaves the turn open; only the bridge's kill ends the process.
func hang() {
	select {}
}

func (f *fakeAgy) prompt(id json.RawMessage, prompt string) {
	// The last script wins, so a replayed history follows its newest instruction.
	script := ""
	if i := strings.LastIndex(prompt, "SCRIPT:"); i >= 0 {
		if fields := strings.Fields(prompt[i+len("SCRIPT:"):]); len(fields) > 0 {
			script = fields[0]

			// "stderr <script>" leaves a line on stderr, then runs the script.
			if script == "stderr" && len(fields) > 1 {
				fmt.Fprintln(os.Stderr, "contact me at a@b.co")

				script = fields[1]
			}
		}
	}

	// An account in trouble answers every prompt with its banner, as Google
	// answers every call of it.
	if f.behaviour != "ok" {
		f.chunk(banners[f.behaviour])
		hang()
	}

	switch script {
	case "account":
		f.chunk(f.token)
		f.end(id, "end_turn", 1, 1)
	case "home":
		f.chunkJSON(inspectHome())
		f.end(id, "end_turn", 1, 1)
	case "text429":
		f.chunk("partial")
		f.chunk(banners["429"])
		hang()
	case "plain":
		f.bridge.notify("session/update", map[string]any{
			"sessionId": "fake-session",
			"update":    map[string]any{"sessionUpdate": "available_commands_update"},
		})

		permission := f.bridge.request("session/request_permission", map[string]any{"sessionId": "fake-session"})
		if !strings.Contains(string(permission.Result), `"cancelled"`) {
			f.chunk("permission was not cancelled")
		}

		f.chunk("Hello")
		f.chunk(", world")
		f.end(id, "end_turn", 11, 7)
	case "model":
		f.chunk(f.model)
		f.end(id, "end_turn", 1, 1)
	case "tools":
		f.chunk(strings.Join(f.tools, ","))
		f.end(id, "end_turn", 1, 1)
	case "usage":
		f.chunk("counted")
		f.bridge.respond(id, map[string]any{
			"stopReason": "end_turn",
			"usage": map[string]int{
				"inputTokens":       6895,
				"outputTokens":      43,
				"cachedReadTokens":  4000,
				"cachedWriteTokens": 120,
				"thoughtTokens":     33,
				"totalTokens":       11058,
			},
		})
	case "system":
		f.chunk(f.system)
		f.end(id, "end_turn", 1, 1)
	case "prompt":
		f.chunk(prompt)
		f.end(id, "end_turn", 1, 1)
	case "blocks":
		f.chunk(string(f.blocks))
		f.end(id, "end_turn", 1, 1)
	case "launch":
		f.chunkJSON(map[string]any{
			"args":       os.Args[1:],
			"autoUpdate": os.Getenv("AGY_CLI_DISABLE_AUTO_UPDATE"),
			"bridgeKey":  os.Getenv("AGY_BRIDGE_API_KEY"),
			"cwd":        f.cwd,
			"home":       os.Getenv("HOME"),
		})
		f.end(id, "end_turn", 1, 1)
	case "agentfile":
		f.chunkJSON(f.agentFile)
		f.end(id, "end_turn", 1, 1)
	case "state":
		f.chunkJSON(fakeStateFiles())
		f.end(id, "end_turn", 1, 1)
	case "cwd":
		entries, err := os.ReadDir(f.cwd)
		f.chunk(fmt.Sprintf("entries=%d err=%v", len(entries), err))
		f.end(id, "end_turn", 1, 1)
	case "maxtokens":
		f.chunk("cut")
		f.end(id, "max_tokens", 3, 4)
	case "tool":
		f.chunk("Let me look.")
		said := f.callTool(map[string]any{"path": "a.txt"})
		f.chunk("Tool said: " + said)
		f.end(id, "end_turn", 20, 9)
	case "tool2":
		var (
			wg      sync.WaitGroup
			answers [2]string
		)

		for i := range answers {
			wg.Go(func() {
				answers[i] = f.callTool(map[string]any{"n": i + 1})
			})
		}

		wg.Wait()
		f.chunk("Results: " + answers[0] + " | " + answers[1])
		f.end(id, "end_turn", 30, 12)
	case "slow":
		// A thinking-high first call: nothing comes for a while.
		time.Sleep(fakeBannerDelay)
		f.chunk("late")
		f.end(id, "end_turn", 1, 1)
	case "429", "503", "fatal":
		// The banner comes late, as after agy's first upstream attempt, so a
		// stream has had time to ping before it.
		time.Sleep(fakeBannerDelay)
		f.chunk(banners[script])
		hang()
	case "tool429":
		f.chunk("Let me look.")
		f.callTool(map[string]any{"path": "a.txt"})
		f.chunk(banners["429"])
		hang()
	case "tooltext429":
		f.chunk("Let me look.")
		f.callTool(map[string]any{"path": "a.txt"})
		f.chunk("Working.")
		f.chunk(banners["429"])
		hang()
	case "hang":
		f.chunk("partial")
		hang()
	default:
		f.chunk("unknown script " + script)
		f.end(id, "end_turn", 0, 0)
	}
}
