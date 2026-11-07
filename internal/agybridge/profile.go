package agybridge

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

const (
	cleanAgentName = "clean-chat"

	// mcpServerName must equal the name of the server passed in session/new.
	mcpServerName = "client"

	// agyTokenPath is where agy 1.5.2 reads its login, below HOME.
	agyTokenPath = ".gemini/antigravity-cli/antigravity-oauth-token"

	installationIDFile = "installation_id"
)

// accountIDPattern is the charset an account id must keep to name a directory.
var accountIDPattern = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)

// cleanArgs selects the custom agent and drops agy's default customizations,
// so agy runs as a plain model without its roughly 6900-token harness prompt.
func cleanArgs() []string {
	return []string{"--agent", cleanAgentName, "--include-default-customizations=false"}
}

// writeCleanAgent writes the custom agent file agy discovers in the session's
// workspace. Only the client's MCP tools and, when web is set, agy's web
// search and fetch are enabled; the built-in file, shell, browser and
// subagent tools are not.
//
// Every client tool must be listed by name with eager set: a "*" wildcard does
// not register them. The names come from the client, so they are emitted as a
// JSON flow sequence, which is valid YAML whatever characters they carry. The
// web tool names are fixed because a misspelt name in tools can hang agy.
func writeCleanAgent(workspace string, toolNames []string, web bool) error {
	dir := filepath.Join(workspace, ".agents", "agents")
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("agent dir: %w", err)
	}

	tools := "[]"
	if web {
		tools = "[search_web, read_url_content]"
	}

	var agent strings.Builder

	agent.WriteString("---\n" +
		"name: " + cleanAgentName + "\n" +
		"description: Plain chat model\n" +
		"hidden: true\n" +
		"excludeDefaultComponents: true\n" +
		"mainAgent: true\n" +
		"subagent: false\n" +
		"tools: " + tools + "\n")

	servers, err := clientServers(toolNames)
	if err != nil {
		return err
	}

	if servers != "" {
		agent.WriteString("mcpServers: " + servers + "\n")
	}

	agent.WriteString("---\nYou are a helpful assistant.\n")

	if err := os.WriteFile(filepath.Join(dir, cleanAgentName+".md"), []byte(agent.String()), 0o600); err != nil {
		return fmt.Errorf("agent file: %w", err)
	}

	return nil
}

// clientServers renders the mcpServers value, or "" when there is no tool.
func clientServers(toolNames []string) (string, error) {
	if len(toolNames) == 0 {
		return "", nil
	}

	type eager struct {
		Eager bool `json:"eager"`
	}

	// A map marshals with sorted keys, which also deduplicates the names.
	tools := make(map[string]eager, len(toolNames))
	for _, name := range toolNames {
		tools[name] = eager{Eager: true}
	}

	raw, err := json.Marshal([]any{map[string]any{"name": mcpServerName, "tools": tools}})
	if err != nil {
		return "", fmt.Errorf("agent tools: %w", err)
	}

	return string(raw), nil
}

// agyTokenFile is agy's login file. The refresh token is left out for an
// account that has none: agy then runs on the access token alone.
type agyTokenFile struct {
	Token struct {
		AccessToken  string `json:"access_token"`
		TokenType    string `json:"token_type"`
		RefreshToken string `json:"refresh_token,omitempty"`
		Expiry       string `json:"expiry"`
		ExpiresIn    int    `json:"expires_in"`
	} `json:"token"`
	AuthMethod   string `json:"auth_method"`
	IDTokenEmail string `json:"id_token_email"`
}

func validAccountID(id string) bool {
	return id != "." && id != ".." && accountIDPattern.MatchString(id)
}

// spawnHomes makes a throwaway HOME for every agy process, below
// <base>/<account id>/, and removes it once agy is dead: agy keeps all its
// state under HOME, so nothing of a session outlives it.
type spawnHomes struct {
	base string

	// mu makes the creation of an account's installation id happen once.
	mu sync.Mutex
}

func newSpawnHomes(base string) (*spawnHomes, error) {
	if !filepath.IsAbs(base) {
		return nil, fmt.Errorf("homes dir must be absolute, got %q", base)
	}

	if err := os.MkdirAll(base, 0o700); err != nil {
		return nil, fmt.Errorf("homes dir: %w", err)
	}

	h := &spawnHomes{base: filepath.Clean(base)}
	h.sweep()

	return h, nil
}

// spawnHomePattern is the name prepare gives a home: randomHex(8).
var spawnHomePattern = regexp.MustCompile(`^[0-9a-f]{16}$`)

// sweep removes the homes a crashed or killed bridge left, holding agy logins.
// The base belongs to exactly one bridge process, and this one has not
// started any agy yet, so no live process can be using them. Every removal
// goes through remove's guards; an account's installation id stays.
func (h *spawnHomes) sweep() {
	accounts, err := os.ReadDir(h.base)
	if err != nil {
		log.Printf("spawn homes: sweep: %v", err)

		return
	}

	removed, failed := 0, 0

	for _, account := range accounts {
		if !account.IsDir() || !validAccountID(account.Name()) {
			continue
		}

		entries, err := os.ReadDir(filepath.Join(h.base, account.Name()))
		if err != nil {
			log.Printf("spawn homes: sweep %s: %v", account.Name(), err)
			failed++

			continue
		}

		for _, entry := range entries {
			if !spawnHomePattern.MatchString(entry.Name()) {
				continue
			}

			if err := h.remove(filepath.Join(h.base, account.Name(), entry.Name())); err != nil {
				log.Printf("spawn homes: sweep %s: %v", account.Name(), err)
				failed++

				continue
			}

			removed++
		}
	}

	log.Printf("spawn homes: removed %d leftover spawn home(s), %d failed", removed, failed)
}

// prepare makes a home holding only the account's login and its stable
// installation id, and returns its path.
func (h *spawnHomes) prepare(acct Account) (string, error) {
	home, err := h.prepareEmpty(acct.ID)
	if err != nil {
		return "", err
	}

	// agy makes a new installation id in every fresh home; seeding one per
	// account keeps it as stable as on the machine the account signed in on.
	// A half-made home goes again at once.
	installation, err := h.installationID(filepath.Dir(home))
	if err == nil {
		err = writeLogin(home, acct, installation)
	}

	if err != nil {
		return "", errors.Join(err, h.remove(home))
	}

	return home, nil
}

// prepareEmpty makes an empty home below <base>/<dir>/, the one an agy that
// signs in by itself gets, so remove and the startup sweep cover it too.
func (h *spawnHomes) prepareEmpty(dir string) (string, error) {
	accountDir, err := h.accountDir(dir)
	if err != nil {
		return "", err
	}

	home := filepath.Join(accountDir, randomHex(8))
	if err := os.Mkdir(home, 0o700); err != nil {
		return "", fmt.Errorf("spawn home: %w", err)
	}

	return home, nil
}

// accountDir makes <base>/<id> when missing and returns it. The id names a
// directory, so it may not carry a separator or dots.
func (h *spawnHomes) accountDir(id string) (string, error) {
	if !validAccountID(id) {
		return "", fmt.Errorf("account id %q is not a valid directory name", id)
	}

	dir := filepath.Join(h.base, id)
	if err := os.Mkdir(dir, 0o700); err != nil && !errors.Is(err, fs.ErrExist) {
		return "", fmt.Errorf("account dir: %w", err)
	}

	if err := realDir(dir); err != nil {
		return "", err
	}

	return dir, nil
}

func (h *spawnHomes) installationID(accountDir string) ([]byte, error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	path := filepath.Join(accountDir, installationIDFile)

	id, err := os.ReadFile(path)
	if err == nil {
		return id, nil
	}

	if !errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("installation id: %w", err)
	}

	id = []byte(uuidV4())
	if err := writeNew(path, id); err != nil {
		return nil, fmt.Errorf("installation id: %w", err)
	}

	return id, nil
}

func writeLogin(home string, acct Account, installation []byte) error {
	dir := filepath.Join(home, filepath.Dir(agyTokenPath))
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("agy dir: %w", err)
	}

	var login agyTokenFile

	login.Token.AccessToken = acct.AccessToken
	login.Token.TokenType = "Bearer"
	login.Token.RefreshToken = acct.RefreshToken
	login.Token.Expiry = acct.ExpiresAt.UTC().Format(time.RFC3339Nano)
	login.Token.ExpiresIn = max(0, int(time.Until(acct.ExpiresAt).Seconds()))
	login.AuthMethod = "consumer"
	login.IDTokenEmail = acct.Email

	raw, err := json.Marshal(login)
	if err != nil {
		return err
	}

	// Errors name the file, never its content.
	if err := writeNew(filepath.Join(home, agyTokenPath), raw); err != nil {
		return fmt.Errorf("agy token file: %w", err)
	}

	if err := writeNew(filepath.Join(dir, installationIDFile), installation); err != nil {
		return fmt.Errorf("agy installation id: %w", err)
	}

	return nil
}

// writeNew creates a 0600 file that must not exist yet.
func writeNew(path string, data []byte) error {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}

	_, err = file.Write(data)

	return errors.Join(err, file.Close())
}

// remove deletes a home prepare made. The path is checked by its text, never
// by resolving links: after Clean it must sit exactly two valid components
// below the base, and neither the account dir nor the home may be a symlink,
// so the removal never leaves the base. A home that is already gone is an
// error too: the bridge made it, so agy or something else removed it.
func (h *spawnHomes) remove(home string) error {
	rel, err := filepath.Rel(h.base, filepath.Clean(home))
	if err != nil {
		return fmt.Errorf("spawn home outside the homes dir: %w", err)
	}

	parts := strings.Split(rel, string(filepath.Separator))
	if len(parts) != 2 || !validAccountID(parts[0]) || !validAccountID(parts[1]) {
		return errors.New("spawn home is not <homes dir>/<account>/<home>")
	}

	for _, dir := range []string{filepath.Join(h.base, parts[0]), filepath.Join(h.base, rel)} {
		if err := realDir(dir); err != nil {
			return err
		}
	}

	return os.RemoveAll(filepath.Join(h.base, rel))
}

// realDir fails unless path is a directory itself, not a link to one.
func realDir(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}

	if !info.IsDir() {
		return fmt.Errorf("%s is not a directory", filepath.Base(path))
	}

	return nil
}

func uuidV4() string {
	b := make([]byte, 16)
	// crypto/rand.Read never returns an error.
	rand.Read(b)

	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80

	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}
