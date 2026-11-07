package agybridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"time"
)

// Account is one Google account agy can run under.
type Account struct {
	// ID names the account in logs and directories: [A-Za-z0-9_.-] only.
	ID           string
	Email        string
	AccessToken  string
	RefreshToken string
	// ExpiresAt is when AccessToken expires. Without a RefreshToken the
	// account is only used while it is comfortably ahead.
	ExpiresAt time.Time
	// Cooldowns maps a quota group to the time the source wants it left alone
	// until; the pool adds its own on top.
	Cooldowns map[string]time.Time
	// Quota maps a quota group to the fraction left, 0 to 1; a group it does
	// not name counts as 1.
	Quota map[string]float64
}

// Outcome is what one agy session said about the account it ran under, for
// one quota group of its model. Kind is kindOK for a session that finished.
type Outcome struct {
	AccountID  string
	Group      string
	Kind       failureKind
	RetryAfter time.Duration
	// Detail is the failure text, cut and without an email address.
	Detail string
	// At is when the pool settled the outcome; a report may go out later.
	At time.Time
}

// AccountSource supplies the accounts and hears how they fared. Accounts
// returns only accounts the source considers enabled; the pool calls it at
// most every sourceRefreshEvery and when no account is eligible. Report may
// run on a request's goroutine, so it never blocks.
type AccountSource interface {
	Accounts(ctx context.Context) ([]Account, error)
	Report(ctx context.Context, o Outcome)
}

// newSource picks the account source the configuration names: for
// development an agy token file, else the accounts Kodizm pushes to the admin
// routes. There is no fallback to agy's own login.
func newSource(cfg Config) (AccountSource, error) {
	if cfg.TokenFile == "" {
		return newPushedSource(), nil
	}

	src := &fileSource{path: cfg.TokenFile}

	// A wrong path or file fails the start rather than every request.
	if _, err := src.Accounts(context.Background()); err != nil {
		return nil, fmt.Errorf("AGY_BRIDGE_TOKEN_FILE: %w", err)
	}

	return src, nil
}

// fileSource serves the one account of an agy token file, for development.
// The file is read on every call, so a fresh `agy login` is picked up. Its
// refresh token goes into the spawn home too: nothing else refreshes it.
type fileSource struct {
	path string
}

func (f *fileSource) Accounts(context.Context) ([]Account, error) {
	raw, err := os.ReadFile(f.path)
	if err != nil {
		return nil, fmt.Errorf("read token file: %w", err)
	}

	// The decoder's errors carry offsets and types, never file content.
	var login agyTokenFile
	if err := json.Unmarshal(raw, &login); err != nil {
		return nil, fmt.Errorf("token file is not agy's JSON: %w", err)
	}

	if login.Token.AccessToken == "" {
		return nil, errors.New("token file has no access_token")
	}

	acct := Account{
		ID:           "local",
		Email:        login.IDTokenEmail,
		AccessToken:  login.Token.AccessToken,
		RefreshToken: login.Token.RefreshToken,
	}

	if login.Token.Expiry != "" {
		if acct.ExpiresAt, err = time.Parse(time.RFC3339Nano, login.Token.Expiry); err != nil {
			return nil, errors.New("token file expiry is not an RFC 3339 time")
		}
	}

	return []Account{acct}, nil
}

func (f *fileSource) Report(context.Context, Outcome) {}
