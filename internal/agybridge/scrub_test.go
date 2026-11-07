package agybridge

import (
	"os"
	"path/filepath"
	"testing"
)

func TestScrubDefaults(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
		hits int
	}{
		{
			name: "billing line",
			in:   "x-anthropic-billing-header: cc_version=2.1.318.13b; cc_entrypoint=sdk-cli;\nKeep this.",
			want: "Keep this.",
			hits: 1,
		},
		{
			name: "billing line indented mid text",
			in:   "First.\n  X-Anthropic-Billing-Header: cc_version=1\nSecond.",
			want: "First.\nSecond.",
			hits: 1,
		},
		{
			name: "agent sdk sentence lower case",
			in:   "you are a claude agent, built on anthropic's claude agent sdk. Go on.",
			want: "You are an AI coding agent. Go on.",
			hits: 1,
		},
		{
			name: "claude code sentence",
			in:   "You are Claude Code, Anthropic's official CLI for Claude.",
			want: "You are an AI coding agent.",
			hits: 1,
		},
		{
			name: "claude code sentence, agent sdk variant",
			in:   "You are Claude Code, Anthropic's official CLI for Claude, running within the Claude Agent SDK.\nGo.",
			want: "You are an AI coding agent.\nGo.",
			hits: 1,
		},
		{
			name: "shared gateway rule, codex sentence",
			in:   "You are Codex, a coding agent based on GPT-5. Go.",
			want: "You are an AI coding agent. Go.",
			hits: 1,
		},
		{
			name: "header word mid line is kept",
			in:   "Mention x-anthropic-billing-header: here.",
			want: "Mention x-anthropic-billing-header: here.",
			hits: 0,
		},
		{
			name: "blank runs collapse",
			in:   "\n\nA\n\n\n\nB\n\n",
			want: "A\n\nB",
			hits: 0,
		},
	}

	scrub := newScrubber("")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, hits := scrub.apply(tt.in)

			if got != tt.want || hits != tt.hits {
				t.Fatalf("apply(%q) = %q, %d; want %q, %d", tt.in, got, hits, tt.want, tt.hits)
			}
		})
	}
}

func TestScrubFileRulesReloadEveryCall(t *testing.T) {
	file := filepath.Join(t.TempDir(), "scrub.json")
	scrub := newScrubber(file)

	// 1. A missing file leaves the defaults working.
	if got, hits := scrub.apply("Codename AZURE."); got != "Codename AZURE." || hits != 0 {
		t.Fatalf("missing file: got %q, %d", got, hits)
	}

	// 2. A valid file adds its rules, with Go expansion syntax in replace.
	writeFile(t, file, `[{"pattern": "codename (\\w+)", "replace": "project ${1}X"}]`)

	if got, hits := scrub.apply("Codename AZURE."); got != "project AZUREX." || hits != 1 {
		t.Fatalf("valid file: got %q, %d", got, hits)
	}

	// 3. An invalid file is ignored as a whole.
	writeFile(t, file, `[{"pattern": "codename (", "replace": ""}]`)

	if got, hits := scrub.apply("Codename AZURE."); got != "Codename AZURE." || hits != 0 {
		t.Fatalf("invalid file: got %q, %d", got, hits)
	}

	writeFile(t, file, `{"pattern": "not an array"}`)

	if got, _ := scrub.apply("Codename AZURE."); got != "Codename AZURE." {
		t.Fatalf("non-array file: got %q", got)
	}
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()

	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}
