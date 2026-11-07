package agybridge

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
	"time"
)

func readCleanAgent(t *testing.T, workspace string) string {
	t.Helper()

	raw, err := os.ReadFile(filepath.Join(workspace, ".agents", "agents", "clean-chat.md"))
	if err != nil {
		t.Fatal(err)
	}

	return string(raw)
}

func TestWriteCleanAgentShape(t *testing.T) {
	workspace := t.TempDir()

	if err := writeCleanAgent(workspace, []string{"read_file", "Bash", "read_file", "mcp__x__y"}, true); err != nil {
		t.Fatal(err)
	}

	want := "---\n" +
		"name: clean-chat\n" +
		"description: Plain chat model\n" +
		"hidden: true\n" +
		"excludeDefaultComponents: true\n" +
		"mainAgent: true\n" +
		"subagent: false\n" +
		"tools: [search_web, read_url_content]\n" +
		`mcpServers: [{"name":"client","tools":{"Bash":{"eager":true},"mcp__x__y":{"eager":true},` +
		`"read_file":{"eager":true}}}]` + "\n" +
		"---\n" +
		"You are a helpful assistant.\n"

	if got := readCleanAgent(t, workspace); got != want {
		t.Fatalf("agent file\n%s\nwant\n%s", got, want)
	}
}

func TestWriteCleanAgentIsDeterministic(t *testing.T) {
	first, second := t.TempDir(), t.TempDir()

	if err := writeCleanAgent(first, []string{"b", "a", "c", "a"}, true); err != nil {
		t.Fatal(err)
	}

	if err := writeCleanAgent(second, []string{"a", "b", "c"}, true); err != nil {
		t.Fatal(err)
	}

	if readCleanAgent(t, first) != readCleanAgent(t, second) {
		t.Fatal("the same names in another order gave another file")
	}
}

func TestWriteCleanAgentEmitsHostileNamesSafely(t *testing.T) {
	names := []string{`say "hi"`, "a: b", "back\\slash", "new\nline", "- dash", "#hash", "<tag>&"}
	workspace := t.TempDir()

	if err := writeCleanAgent(workspace, names, true); err != nil {
		t.Fatal(err)
	}

	var line string

	for l := range strings.SplitSeq(readCleanAgent(t, workspace), "\n") {
		if value, ok := strings.CutPrefix(l, "mcpServers: "); ok {
			line = value
		}
	}

	var servers []struct {
		Name  string                    `json:"name"`
		Tools map[string]map[string]any `json:"tools"`
	}

	if err := json.Unmarshal([]byte(line), &servers); err != nil || len(servers) != 1 || servers[0].Name != "client" {
		t.Fatalf("mcpServers is not one JSON server: %v\n%s", err, line)
	}

	if len(servers[0].Tools) != len(names) {
		t.Fatalf("tools = %v", servers[0].Tools)
	}

	for _, name := range names {
		if servers[0].Tools[name]["eager"] != true {
			t.Fatalf("tool %q is not eager: %v", name, servers[0].Tools)
		}
	}
}

func TestWriteCleanAgentWithoutToolsOrWeb(t *testing.T) {
	workspace := t.TempDir()

	if err := writeCleanAgent(workspace, nil, false); err != nil {
		t.Fatal(err)
	}

	got := readCleanAgent(t, workspace)

	if strings.Contains(got, "mcpServers") || !strings.Contains(got, "\ntools: []\n") {
		t.Fatalf("agent file\n%s", got)
	}
}

func TestWriteCleanAgentIsPrivate(t *testing.T) {
	workspace := filepath.Join(t.TempDir(), "workspace")

	if err := writeCleanAgent(workspace, []string{"a"}, true); err != nil {
		t.Fatal(err)
	}

	for _, dir := range []string{workspace, filepath.Join(workspace, ".agents"), filepath.Join(workspace, ".agents", "agents")} {
		if info, err := os.Stat(dir); err != nil || info.Mode().Perm() != 0o700 {
			t.Fatalf("%s: %v, %v", dir, info, err)
		}
	}

	file := filepath.Join(workspace, ".agents", "agents", "clean-chat.md")
	if info, err := os.Stat(file); err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("agent file: %v, %v", info, err)
	}
}

func TestCleanLaunchSettings(t *testing.T) {
	want := []string{"--agent", "clean-chat", "--include-default-customizations=false"}
	if got := cleanArgs(); !slices.Equal(got, want) {
		t.Fatalf("cleanArgs = %v", got)
	}
}

// HOME comes last, so it wins over the bridge's own when exec reads the env.
func TestAgentEnvKeepsTheBridgeKeyFromAgyAndSetsTheHome(t *testing.T) {
	got := agentEnv([]string{"HOME=/h", "AGY_BRIDGE_API_KEY=secret", "AGY_BRIDGE_LISTEN=:1", "PATH=/bin"}, "/spawn")
	want := []string{
		"HOME=/h",
		"AGY_BRIDGE_LISTEN=:1",
		"PATH=/bin",
		"AGY_CLI_DISABLE_AUTO_UPDATE=true",
		"HOME=/spawn",
	}

	if !slices.Equal(got, want) {
		t.Fatalf("agentEnv = %v, want %v", got, want)
	}
}

func filesUnder(t *testing.T, dir string) []string {
	t.Helper()

	var files []string

	err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			rel, _ := filepath.Rel(dir, path)
			files = append(files, rel)
		}

		return err
	})
	if err != nil {
		t.Fatal(err)
	}

	slices.Sort(files)

	return files
}

func newTestHomes(t *testing.T) *spawnHomes {
	t.Helper()

	homes, err := newSpawnHomes(filepath.Join(t.TempDir(), "homes"))
	if err != nil {
		t.Fatal(err)
	}

	return homes
}

// tokenFileOf reads the agy token file of a spawn home as plain JSON, so a
// test sees exactly which keys agy would find.
func tokenFileOf(t *testing.T, home string) (map[string]any, os.FileMode) {
	t.Helper()

	path := filepath.Join(home, agyTokenPath)

	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	var file map[string]any
	if err := json.Unmarshal(raw, &file); err != nil {
		t.Fatal(err)
	}

	return file, info.Mode().Perm()
}

func TestSpawnHomeHoldsOnlyTheAccountsAccessToken(t *testing.T) {
	homes := newTestHomes(t)
	expiry := time.Date(2026, 11, 7, 12, 0, 0, 123, time.UTC)

	home, err := homes.prepare(Account{ID: "acct-1", Email: "a@example.test", AccessToken: "ya29.access", ExpiresAt: expiry})
	if err != nil {
		t.Fatal(err)
	}

	// 1. The home is <base>/<account id>/<random hex>, private to this user.
	rel, err := filepath.Rel(homes.base, home)
	if err != nil || !regexp.MustCompile(`^acct-1/[0-9a-f]{16}$`).MatchString(rel) {
		t.Fatalf("home %q, rel %q", home, rel)
	}

	for _, dir := range []string{
		filepath.Join(homes.base, "acct-1"),
		home,
		filepath.Join(home, ".gemini"),
		filepath.Join(home, ".gemini", "antigravity-cli"),
	} {
		if info, err := os.Stat(dir); err != nil || info.Mode().Perm() != 0o700 {
			t.Fatalf("%s: %v, %v", dir, info, err)
		}
	}

	// 2. The token file has agy's shape, 0600, and no refresh token.
	file, mode := tokenFileOf(t, home)
	token := file["token"].(map[string]any)

	if mode != 0o600 || file["auth_method"] != "consumer" || file["id_token_email"] != "a@example.test" {
		t.Fatalf("token file %v, mode %s", file, mode)
	}

	if _, ok := token["refresh_token"]; ok || token["access_token"] != "ya29.access" || token["token_type"] != "Bearer" ||
		token["expiry"] != "2026-11-07T12:00:00.000000123Z" {
		t.Fatalf("token %v", token)
	}

	if _, ok := token["expires_in"].(float64); !ok {
		t.Fatalf("expires_in %v", token["expires_in"])
	}

	// 3. Removal leaves only the account's installation id.
	if err := homes.remove(home); err != nil {
		t.Fatal(err)
	}

	if got := filesUnder(t, homes.base); !slices.Equal(got, []string{"acct-1/installation_id"}) {
		t.Fatalf("left behind %v", got)
	}
}

func TestSpawnHomeCarriesARefreshTokenWhenTheAccountHasOne(t *testing.T) {
	homes := newTestHomes(t)

	home, err := homes.prepare(Account{ID: "local", AccessToken: "a", RefreshToken: "1//refresh"})
	if err != nil {
		t.Fatal(err)
	}

	t.Cleanup(func() { _ = homes.remove(home) })

	file, _ := tokenFileOf(t, home)
	if token := file["token"].(map[string]any); token["refresh_token"] != "1//refresh" || token["expires_in"] != float64(0) {
		t.Fatalf("token %v", token)
	}
}

func TestEmptySpawnHomeHoldsNothingAndIsRemovable(t *testing.T) {
	homes := newTestHomes(t)

	home, err := homes.prepareEmpty("login")
	if err != nil {
		t.Fatal(err)
	}

	rel, err := filepath.Rel(homes.base, home)
	if err != nil || !regexp.MustCompile(`^login/[0-9a-f]{16}$`).MatchString(rel) {
		t.Fatalf("home %q, rel %q", home, rel)
	}

	if info, err := os.Stat(home); err != nil || info.Mode().Perm() != 0o700 {
		t.Fatalf("%s: %v, %v", home, info, err)
	}

	// No login and no installation id: agy writes its own.
	if got := filesUnder(t, homes.base); len(got) != 0 {
		t.Fatalf("empty home holds %v", got)
	}

	if err := homes.remove(home); err != nil {
		t.Fatal(err)
	}

	if _, err := homes.prepareEmpty("../x"); err == nil {
		t.Fatal("an invalid dir made a home")
	}
}

func TestInstallationIDIsStablePerAccount(t *testing.T) {
	homes := newTestHomes(t)

	read := func(id string) string {
		t.Helper()

		home, err := homes.prepare(Account{ID: id, AccessToken: "t"})
		if err != nil {
			t.Fatal(err)
		}

		defer func() { _ = homes.remove(home) }()

		raw, err := os.ReadFile(filepath.Join(home, ".gemini", "antigravity-cli", "installation_id"))
		if err != nil {
			t.Fatal(err)
		}

		return string(raw)
	}

	first, again, other := read("a"), read("a"), read("b")
	uuid := regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)

	if !uuid.MatchString(first) || first != again || first == other || !uuid.MatchString(other) {
		t.Fatalf("installation ids %q, %q, %q", first, again, other)
	}
}

func TestSpawnHomeRefusesAnInvalidAccountID(t *testing.T) {
	homes := newTestHomes(t)

	for _, id := range []string{"", ".", "..", "../x", "a/b", "a b", "é", "a\n"} {
		if home, err := homes.prepare(Account{ID: id, AccessToken: "t"}); err == nil {
			t.Fatalf("id %q made home %q", id, home)
		}
	}

	if left := filesUnder(t, filepath.Dir(homes.base)); len(left) != 0 {
		t.Fatalf("a refused id left %v", left)
	}
}

func TestNewSpawnHomesSweepsLeftoverHomes(t *testing.T) {
	logs := captureLog(t)
	base := filepath.Join(t.TempDir(), "homes")
	outside := t.TempDir()
	keep := filepath.Join(outside, "0123456789abcdef")

	mkdir := func(path string) {
		t.Helper()

		if err := os.MkdirAll(path, 0o700); err != nil {
			t.Fatal(err)
		}
	}

	write := func(path string) {
		t.Helper()

		if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
	}

	// 1. Two leftovers of a crashed bridge, each with agy state inside.
	for _, home := range []string{"acct-1/0123456789abcdef", "acct-2/fedcba9876543210"} {
		mkdir(filepath.Join(base, home, ".gemini", "antigravity-cli"))
		write(filepath.Join(base, home, agyTokenPath))
	}

	write(filepath.Join(base, "acct-1", installationIDFile))

	// 2. What the sweep must leave alone: a name that is no spawn home, a file
	// at the base, an invalid account dir, and symlinks at both levels.
	mkdir(filepath.Join(base, "acct-1", "not-a-home"))
	mkdir(filepath.Join(base, "acct-1", "ABCDEF0123456789"))
	mkdir(filepath.Join(base, "bad id", "0123456789abcdef"))
	write(filepath.Join(base, "stray"))
	mkdir(filepath.Join(keep, "inside"))

	if err := os.Symlink(outside, filepath.Join(base, "linked")); err != nil {
		t.Fatal(err)
	}

	if err := os.Symlink(keep, filepath.Join(base, "acct-2", "00000000000000aa")); err != nil {
		t.Fatal(err)
	}

	homes, err := newSpawnHomes(base)
	if err != nil {
		t.Fatal(err)
	}

	// 3. The leftovers are gone, the installation id stays, the rest is
	// untouched and nothing outside the base was reached.
	for _, gone := range []string{"acct-1/0123456789abcdef", "acct-2/fedcba9876543210"} {
		if _, err := os.Lstat(filepath.Join(homes.base, gone)); !os.IsNotExist(err) {
			t.Fatalf("%s survived the sweep: %v", gone, err)
		}
	}

	for _, kept := range []string{
		"acct-1/" + installationIDFile,
		"acct-1/not-a-home",
		"acct-1/ABCDEF0123456789",
		"bad id/0123456789abcdef",
		"stray",
		"linked",
		"acct-2/00000000000000aa",
	} {
		if _, err := os.Lstat(filepath.Join(homes.base, kept)); err != nil {
			t.Fatalf("%s was swept: %v", kept, err)
		}
	}

	if _, err := os.Stat(filepath.Join(keep, "inside")); err != nil {
		t.Fatalf("the sweep followed a symlink out of the base: %v", err)
	}

	if got := logs.String(); !strings.Contains(got, "removed 2 leftover spawn home(s), 1 failed") {
		t.Fatalf("log %q", got)
	}
}

func TestSpawnHomeRemovalIsGuarded(t *testing.T) {
	homes := newTestHomes(t)

	home, err := homes.prepare(Account{ID: "a", AccessToken: "t"})
	if err != nil {
		t.Fatal(err)
	}

	outside := t.TempDir()
	keep := filepath.Join(outside, "x", "y")

	if err := os.MkdirAll(keep, 0o700); err != nil {
		t.Fatal(err)
	}

	// 1. Anything but exactly <base>/<id>/<hex> is refused.
	for _, path := range []string{
		outside,
		keep,
		homes.base,
		filepath.Join(homes.base, "a"),
		filepath.Join(home, ".gemini"),
		filepath.Join(homes.base, "..", "x", "y"),
		homes.base + "/a/../../" + filepath.Base(outside) + "/x",
		filepath.Join(homes.base, "a", ".."),
	} {
		if err := homes.remove(path); err == nil {
			t.Fatalf("removed %q", path)
		}
	}

	if _, err := os.Stat(keep); err != nil {
		t.Fatalf("outside dir gone: %v", err)
	}

	// 2. A symlinked account dir is never followed out of the base.
	link := filepath.Join(homes.base, "evil")
	if err := os.Symlink(filepath.Join(outside, "x"), link); err != nil {
		t.Fatal(err)
	}

	if err := homes.remove(filepath.Join(link, "y")); err == nil {
		t.Fatal("removed through a symlinked account dir")
	}

	if _, err := os.Stat(keep); err != nil {
		t.Fatalf("outside dir gone: %v", err)
	}

	// 3. The real home goes, and a second removal of it is a failure: the
	// bridge made it, so a missing home is something it must hear about.
	if err := homes.remove(home); err != nil {
		t.Fatal(err)
	}

	if err := homes.remove(home); err == nil {
		t.Fatal("a missing home counted as removed")
	}
}
