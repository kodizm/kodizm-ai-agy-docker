package agybridge

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// agyLogin is a token file in the shape agy 1.5.2 writes.
const agyLogin = `{"token":{"access_token":"%s","token_type":"Bearer","refresh_token":"1//refresh",` +
	`"expiry":"2026-11-07T12:00:00.123456789+01:00","expires_in":3599},` +
	`"auth_method":"consumer","id_token_email":"me@example.test"}`

func writeLoginFile(t *testing.T, path, access string) {
	t.Helper()

	if err := os.WriteFile(path, []byte(strings.Replace(agyLogin, "%s", access, 1)), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestFileSourceServesTheLoginOfAnAgyTokenFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "antigravity-oauth-token")
	writeLoginFile(t, path, "ya29.first")

	src := &fileSource{path: path}

	accounts, err := src.Accounts(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	want := Account{
		ID:           "local",
		Email:        "me@example.test",
		AccessToken:  "ya29.first",
		RefreshToken: "1//refresh",
		ExpiresAt:    time.Date(2026, 11, 7, 11, 0, 0, 123456789, time.UTC),
	}

	if len(accounts) != 1 || accounts[0].ID != want.ID || accounts[0].Email != want.Email ||
		accounts[0].AccessToken != want.AccessToken || accounts[0].RefreshToken != want.RefreshToken ||
		!accounts[0].ExpiresAt.Equal(want.ExpiresAt) {
		t.Fatalf("accounts %+v", accounts)
	}

	// A new login is read on the next call.
	writeLoginFile(t, path, "ya29.second")

	if accounts, err := src.Accounts(context.Background()); err != nil || accounts[0].AccessToken != "ya29.second" {
		t.Fatalf("after a new login: %+v, %v", accounts, err)
	}
}

func TestFileSourceRefusesABrokenFile(t *testing.T) {
	dir := t.TempDir()

	tests := map[string]string{
		"not json":        `ya29.secret-value`,
		"no access token": `{"token":{"refresh_token":"1//secret-value"}}`,
		"bad expiry":      `{"token":{"access_token":"secret-value","expiry":"tomorrow"}}`,
	}

	for name, content := range tests {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(dir, strings.ReplaceAll(name, " ", "-"))
			if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
				t.Fatal(err)
			}

			_, err := (&fileSource{path: path}).Accounts(context.Background())
			if err == nil || strings.Contains(err.Error(), "secret-value") {
				t.Fatalf("error %v: want a refusal that carries no file content", err)
			}
		})
	}

	if _, err := (&fileSource{path: filepath.Join(dir, "missing")}).Accounts(context.Background()); err == nil {
		t.Fatal("a missing file was accepted")
	}
}

func TestNewSource(t *testing.T) {
	// 1. Without a token file the accounts are pushed: none until the first.
	src, err := newSource(Config{})
	if err != nil {
		t.Fatal(err)
	}

	pushed, ok := src.(*pushedSource)
	if !ok {
		t.Fatalf("source %T", src)
	}

	if accounts, err := pushed.Accounts(context.Background()); err != nil || len(accounts) != 0 {
		t.Fatalf("a cold pushed source holds %+v, %v", accounts, err)
	}

	// 2. A token file that cannot be read fails the start too.
	missing := filepath.Join(t.TempDir(), "missing")
	if _, err := newSource(Config{TokenFile: missing}); err == nil {
		t.Fatal("newSource accepted a missing token file")
	}

	// 3. A readable one is the source.
	path := filepath.Join(t.TempDir(), "token")
	writeLoginFile(t, path, "ya29.x")

	src, err = newSource(Config{TokenFile: path})
	if err != nil {
		t.Fatal(err)
	}

	if _, ok := src.(*fileSource); !ok {
		t.Fatalf("source %T", src)
	}
}
