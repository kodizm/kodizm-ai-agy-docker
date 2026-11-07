package agybridge

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

type loginStarted struct {
	LoginID   string `json:"login_id"`
	URL       string `json:"url"`
	ExpiresIn int    `json:"expires_in"`
}

// setLoginTiming shortens a login timing var for one test; it is restored
// after the bridge closes, since cleanups run last in first out.
func setLoginTiming(t *testing.T, v *time.Duration, d time.Duration) {
	t.Helper()

	old := *v
	*v = d

	t.Cleanup(func() { *v = old })
}

func (b *bridge) startLogin(t *testing.T) loginStarted {
	t.Helper()

	resp := b.do(t, context.Background(), http.MethodPost, loginPath, nil, nil)
	expectStatus(t, resp, http.StatusCreated)

	return decodeJSON[loginStarted](t, resp)
}

func (b *bridge) sendCode(t *testing.T, id string, body any) *http.Response {
	t.Helper()

	return b.do(t, context.Background(), http.MethodPost, loginPathPrefix+id+"/code", body, nil)
}

// loginHomes lists the homes under the login account dir.
func (b *bridge) loginHomes(t *testing.T) []string {
	t.Helper()

	entries, err := os.ReadDir(filepath.Join(b.homes, loginHomesDir))
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}

	if err != nil {
		t.Fatal(err)
	}

	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		names = append(names, entry.Name())
	}

	return names
}

// liveLogins counts the fake `agy acp --login` processes still running.
func liveLogins(t *testing.T) int {
	t.Helper()

	n := 0

	for _, file := range fakePIDs(t, "login") {
		if deadPIDs[file] {
			continue
		}

		_, raw, _ := strings.Cut(filepath.Base(file), "-")

		pid, err := strconv.Atoi(raw)
		if err != nil {
			t.Fatal(err)
		}

		if errors.Is(syscall.Kill(pid, 0), syscall.ESRCH) {
			deadPIDs[file] = true

			continue
		}

		n++
	}

	return n
}

func expectLoginGone(t *testing.T, b *bridge, id string) {
	t.Helper()

	resp := b.sendCode(t, id, map[string]string{"code": fakeLoginCode})
	expectStatus(t, resp, http.StatusNotFound)

	if got := decodeJSON[errorEnvelope](t, resp); got.Error.Type != "not_found_error" {
		t.Fatalf("error body = %+v", got)
	}
}

func TestLoginStartHandsOutTheGoogleURL(t *testing.T) {
	b := newBridge(t, nil)

	got := b.startLogin(t)

	validID := regexp.MustCompile(`^[0-9a-f]{32}$`).MatchString(got.LoginID)
	if !validID || got.URL != fakeLoginURL || got.ExpiresIn != 600 {
		t.Fatalf("started %+v", got)
	}

	// agy runs in an empty home of its own while it waits for the code.
	if homes := b.loginHomes(t); len(homes) != 1 || !spawnHomePattern.MatchString(homes[0]) {
		t.Fatalf("login homes %v", homes)
	}

	if n := liveLogins(t); n != 1 {
		t.Fatalf("%d login processes", n)
	}
}

func TestLoginGoodCodeReturnsTheTokenFile(t *testing.T) {
	tests := []struct {
		name string
		code string
	}{
		{name: "code", code: fakeLoginCode},
		{
			name: "landing page address",
			code: "https://antigravity.google/oauth-callback?state=fake&code=" + fakeLoginCode,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			b := newBridge(t, nil)
			started := b.startLogin(t)

			resp := b.sendCode(t, started.LoginID, map[string]string{"code": tt.code})
			expectStatus(t, resp, http.StatusOK)

			var got struct {
				TokenFile json.RawMessage `json:"token_file"`
			}

			if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
				t.Fatal(err)
			}

			// The file comes back as agy wrote it, key for key.
			var want, file map[string]any
			_ = json.Unmarshal([]byte(fakeLoginTokenFile), &want)

			if err := json.Unmarshal(got.TokenFile, &file); err != nil {
				t.Fatal(err)
			}

			wantRaw, _ := json.Marshal(want)
			gotRaw, _ := json.Marshal(file)

			refresh := file["token"].(map[string]any)["refresh_token"]
			if string(wantRaw) != string(gotRaw) || refresh != "fake-refresh-token" {
				t.Fatalf("token file %s", got.TokenFile)
			}

			if homes := b.loginHomes(t); len(homes) != 0 {
				t.Fatalf("login homes left %v", homes)
			}

			expectLoginGone(t, b, started.LoginID)
		})
	}
}

func TestLoginBadCodeFailsWithoutLeakingTheEmail(t *testing.T) {
	b := newBridge(t, nil)
	started := b.startLogin(t)

	resp := b.sendCode(t, started.LoginID, map[string]string{"code": "bad-code-123"})
	expectStatus(t, resp, http.StatusUnprocessableEntity)

	got := decodeJSON[errorEnvelope](t, resp)
	message := got.Error.Message

	if got.Type != "error" || got.Error.Type != "login_failed" ||
		!strings.Contains(message, "invalid code for <email>") {
		t.Fatalf("error body = %+v", got)
	}

	for _, secret := range []string{fakeLoginEmail, "accounts.google.com", "bad-code-123"} {
		if strings.Contains(message, secret) {
			t.Fatalf("message %q carries %q", message, secret)
		}
	}

	// One attempt ends the login.
	if homes := b.loginHomes(t); len(homes) != 0 {
		t.Fatalf("login homes left %v", homes)
	}

	expectLoginGone(t, b, started.LoginID)
}

// The wizard invites pasting the whole landing address: agy may echo it
// decoded, or only the code inside it, and neither form may reach the owner's
// screen.
func TestFailureMessageRemovesEveryFormOfThePastedCode(t *testing.T) {
	const (
		value   = "4/0AbCdEfGhIjKlMn"
		encoded = "4%2F0AbCdEfGhIjKlMn"
	)

	pasted := "http://localhost:8085/oauth?state=xyz&code=" + encoded + "&scope=email"

	o := newLoginOutput()
	_, _ = o.Write([]byte("exchange failed for " + value + "\n" +
		"raw " + encoded + "\n" +
		"decoded localhost:8085/oauth?state=xyz&code=" + value + "&scope=email\n" +
		"token endpoint answered invalid_grant\n"))

	message := o.failureMessage(pasted)

	if strings.Contains(message, "0AbCdEfGhIjKlMn") {
		t.Fatalf("message %q carries the code", message)
	}

	if !strings.Contains(message, "invalid_grant") || !strings.Contains(message, "exchange failed for <code>") {
		t.Fatalf("message %q lost the useful part", message)
	}
}

func TestLoginRefusesAMalformedCodeAndStaysPending(t *testing.T) {
	b := newBridge(t, nil)
	started := b.startLogin(t)
	path := loginPathPrefix + started.LoginID + "/code"

	tests := []struct {
		name string
		body any
	}{
		{name: "no body"},
		{name: "not an object", body: "good-code"},
		{name: "code not a string", body: map[string]any{"code": 5}},
		{name: "missing", body: map[string]any{}},
		{name: "empty", body: map[string]string{"code": ""}},
		{name: "blank", body: map[string]string{"code": "  "}},
		{name: "newline", body: map[string]string{"code": fakeLoginCode + "\nmore"}},
		{name: "carriage return", body: map[string]string{"code": fakeLoginCode + "\r"}},
		{name: "oversize", body: map[string]string{"code": strings.Repeat("a", maxLoginCodeRunes+1)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp := b.do(t, context.Background(), http.MethodPost, path, tt.body, nil)
			expectStatus(t, resp, http.StatusBadRequest)

			if got := decodeJSON[errorEnvelope](t, resp); got.Error.Type != "invalid_request_error" {
				t.Fatalf("error body = %+v", got)
			}
		})
	}

	// The refused bodies never reached agy: the login still takes its code.
	expectStatus(t, b.sendCode(t, started.LoginID, map[string]string{"code": fakeLoginCode}), http.StatusOK)
}

func TestLoginUnknownIDIsNotFound(t *testing.T) {
	b := newBridge(t, nil)

	expectLoginGone(t, b, strings.Repeat("0", 32))
	expectLoginGone(t, b, "not-an-id")

	resp := b.do(t, context.Background(), http.MethodGet, loginPathPrefix+strings.Repeat("0", 32), nil, nil)
	expectStatus(t, resp, http.StatusNotFound)
}

func TestLoginThirdStartEvictsTheOldest(t *testing.T) {
	b := newBridge(t, nil)

	first, second, third := b.startLogin(t), b.startLogin(t), b.startLogin(t)

	// The evicted login is dead and gone before the third start answers.
	if n := liveLogins(t); n != 2 {
		t.Fatalf("%d login processes, want 2", n)
	}

	if homes := b.loginHomes(t); len(homes) != 2 {
		t.Fatalf("login homes %v", homes)
	}

	expectLoginGone(t, b, first.LoginID)

	for _, kept := range []loginStarted{second, third} {
		expectStatus(t, b.sendCode(t, kept.LoginID, map[string]string{"code": fakeLoginCode}), http.StatusOK)
	}
}

func TestLoginExpires(t *testing.T) {
	setLoginTiming(t, &loginTTL, 200*time.Millisecond)

	logs := captureLog(t)
	b := newBridge(t, nil)
	started := b.startLogin(t)

	if started.ExpiresIn != 0 {
		t.Fatalf("expires_in %d", started.ExpiresIn)
	}

	eventually(t, "the login to expire", func() bool {
		return liveLogins(t) == 0 && len(b.loginHomes(t)) == 0
	})

	expectLoginGone(t, b, started.LoginID)

	if !strings.Contains(logs.String(), "login id="+started.LoginID[:8]+" outcome=expired") {
		t.Fatalf("log %q", logs.String())
	}
}

func TestLoginDeleteIsIdempotent(t *testing.T) {
	b := newBridge(t, nil)
	started := b.startLogin(t)

	for range 2 {
		resp := b.do(t, context.Background(), http.MethodDelete, loginPathPrefix+started.LoginID, nil, nil)
		expectStatus(t, resp, http.StatusNoContent)

		if n := liveLogins(t); n != 0 {
			t.Fatalf("%d login processes after delete", n)
		}

		if homes := b.loginHomes(t); len(homes) != 0 {
			t.Fatalf("login homes left %v", homes)
		}
	}

	expectLoginGone(t, b, started.LoginID)
}

func TestLoginWithoutAURLIsABadGateway(t *testing.T) {
	setLoginTiming(t, &loginStartTimeout, 300*time.Millisecond)
	t.Setenv(fakeLoginSilentEnv, "1")

	b := newBridge(t, nil)

	resp := b.do(t, context.Background(), http.MethodPost, loginPath, nil, nil)
	expectStatus(t, resp, http.StatusBadGateway)

	if got := decodeJSON[errorEnvelope](t, resp); got.Error.Type != "api_error" {
		t.Fatalf("error body = %+v", got)
	}

	if n := liveLogins(t); n != 0 {
		t.Fatalf("%d login processes", n)
	}

	if homes := b.loginHomes(t); len(homes) != 0 {
		t.Fatalf("login homes left %v", homes)
	}
}

func TestLoginStartLeftByTheClientLeaksNothing(t *testing.T) {
	t.Setenv(fakeLoginSilentEnv, "1")

	b := newBridge(t, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.ts.URL+loginPath, nil)
	if err != nil {
		t.Fatal(err)
	}

	if resp, err := b.ts.Client().Do(req); err == nil {
		_ = resp.Body.Close()
		t.Fatalf("status %d, want the client to give up", resp.StatusCode)
	}

	eventually(t, "the abandoned login to end", func() bool {
		return liveLogins(t) == 0 && len(b.loginHomes(t)) == 0
	})
}

func TestCloseEndsPendingLogins(t *testing.T) {
	b := newBridge(t, nil)

	b.startLogin(t)
	b.startLogin(t)

	b.srv.Close()

	if n := liveLogins(t); n != 0 {
		t.Fatalf("%d login processes after Close", n)
	}

	if homes := b.loginHomes(t); len(homes) != 0 {
		t.Fatalf("login homes left %v", homes)
	}

	assertNoGoroutines(t)

	resp := b.do(t, context.Background(), http.MethodPost, loginPath, nil, nil)
	expectStatus(t, resp, statusOverloaded)
}

func TestLoginRoutesNeedTheAPIKey(t *testing.T) {
	b := newBridge(t, map[string]string{"AGY_BRIDGE_API_KEY": "s3cret"})
	id := strings.Repeat("0", 32)

	for _, route := range []struct {
		method string
		path   string
		body   any
	}{
		{method: http.MethodPost, path: loginPath},
		{method: http.MethodPost, path: loginPathPrefix + id + "/code", body: map[string]string{"code": fakeLoginCode}},
		{method: http.MethodDelete, path: loginPathPrefix + id},
	} {
		resp := b.do(t, context.Background(), route.method, route.path, route.body, nil)
		expectStatus(t, resp, http.StatusUnauthorized)
	}

	if n := liveLogins(t); n != 0 {
		t.Fatalf("an unauthorised call started %d logins", n)
	}

	resp := b.do(t, context.Background(), http.MethodPost, loginPath, nil, http.Header{"X-Api-Key": {"s3cret"}})
	expectStatus(t, resp, http.StatusCreated)
}

func TestLoginLogsCarryNoSecret(t *testing.T) {
	logs := captureLog(t)
	b := newBridge(t, nil)

	good, bad := b.startLogin(t), b.startLogin(t)

	expectStatus(t, b.sendCode(t, good.LoginID, map[string]string{"code": fakeLoginCode}), http.StatusOK)
	refused := b.sendCode(t, bad.LoginID, map[string]string{"code": "bad-code-456"})
	expectStatus(t, refused, http.StatusUnprocessableEntity)

	got := logs.String()

	for _, want := range []string{
		"login id=" + good.LoginID[:8] + " outcome=started seconds=",
		"login id=" + good.LoginID[:8] + " outcome=ok seconds=",
		"login id=" + bad.LoginID[:8] + " outcome=failed seconds=",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("log lacks %q:\n%s", want, got)
		}
	}

	for _, secret := range []string{
		"accounts.google.com",
		fakeLoginCode,
		"bad-code-456",
		fakeLoginEmail,
		"fake-access-token",
		"fake-refresh-token",
		good.LoginID,
	} {
		if strings.Contains(got, secret) {
			t.Fatalf("log carries %q:\n%s", secret, got)
		}
	}
}
