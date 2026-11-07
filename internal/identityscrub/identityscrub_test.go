package identityscrub

import "testing"

func TestAntigravityRules(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
		hits int
	}{
		{
			name: "billing block alone",
			in:   "x-anthropic-billing-header: cc_version=2.1.318.13b; cc_entrypoint=sdk-cli;",
			want: "",
			hits: 1,
		},
		{
			name: "billing line before text",
			in:   "x-anthropic-billing-header: foo\nKeep this.",
			want: "Keep this.",
			hits: 1,
		},
		{
			name: "billing line indented, mixed case, mid text",
			in:   "First.\n \tX-Anthropic-Billing-Header : cc_version=1\nSecond.",
			want: "First.\nSecond.",
			hits: 1,
		},
		{
			name: "header word mid line is kept",
			in:   "Mention x-anthropic-billing-header: here.",
			want: "Mention x-anthropic-billing-header: here.",
		},
		{
			name: "agent sdk sentence, straight apostrophe",
			in:   "You are a Claude agent, built on Anthropic's Claude Agent SDK. Go on.",
			want: "You are an AI coding agent. Go on.",
			hits: 1,
		},
		{
			name: "agent sdk sentence lower case, curly apostrophe",
			in:   "you are a claude agent, built on anthropic’s claude agent sdk.",
			want: "You are an AI coding agent.",
			hits: 1,
		},
		{
			name: "claude code sentence",
			in:   "You are Claude Code, Anthropic's official CLI for Claude.",
			want: "You are an AI coding agent.",
			hits: 1,
		},
		{
			name: "claude code sentence, agent sdk variant, curly apostrophe",
			in:   "You are Claude Code, Anthropic’s official CLI for Claude, running within the Claude Agent SDK.\nGo.",
			want: "You are an AI coding agent.\nGo.",
			hits: 1,
		},
		{
			name: "codex agent sentence",
			in:   "You are Codex, an agent based on GPT-5. You share the workspace.",
			want: "You are an AI coding agent. You share the workspace.",
			hits: 1,
		},
		{
			name: "codex coding agent sentence with a model suffix",
			in:   "you are codex, a coding agent based on gpt-5.1-codex, running in a terminal.",
			want: "You are an AI coding agent, running in a terminal.",
			hits: 1,
		},
		{
			name: "claude code coordinator sentence",
			in: "You are Claude Code, an AI assistant that orchestrates software engineering tasks " +
				"across multiple workers.\n\n## 1. Your Role",
			want: "You are an AI coding agent.\n\n## 1. Your Role",
			hits: 1,
		},
		{
			name: "claude code coordinator sentence stops at its own period",
			in:   "you are claude code, an AI assistant that orchestrates work. Keep this.",
			want: "You are an AI coding agent. Keep this.",
			hits: 1,
		},
		{
			name: "claude code with another role passes",
			in:   "You are Claude Code, an AI assistant that helps.",
			want: "You are Claude Code, an AI assistant that helps.",
		},
		{
			name: "codex based on sentence",
			in:   "You are Codex, based on GPT-5. You are running as a coding agent in the Codex CLI.",
			want: "You are an AI coding agent. You are running as a coding agent in the Codex CLI.",
			hits: 1,
		},
		{
			name: "codex based on sentence with a model suffix",
			in:   "you are codex, based on gpt-5.2-codex.",
			want: "You are an AI coding agent.",
			hits: 1,
		},
		{
			name: "hermes sentence",
			in:   "You are Hermes Agent, an intelligent AI assistant created by Nous Research. Be kind.",
			want: "You are an AI assistant. Be kind.",
			hits: 1,
		},
		{
			name: "every rule at once",
			in: "x-anthropic-billing-header: cc_version=1;\n" +
				"You are a Claude agent, built on Anthropic's Claude Agent SDK.\n" +
				"You are Codex, a coding agent based on GPT-5.",
			want: "You are an AI coding agent.\nYou are an AI coding agent.",
			hits: 3,
		},
		{
			name: "plain identity words pass",
			in:   "Claude and Anthropic. cc_entrypoint=sdk-cli; Codex is a word.",
			want: "Claude and Anthropic. cc_entrypoint=sdk-cli; Codex is a word.",
		},
	}

	scrub, ok := Profile(ProfileAntigravity)
	if !ok {
		t.Fatal("antigravity profile is missing")
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, hits := scrub.Apply(tt.in)
			if got != tt.want || hits != tt.hits {
				t.Fatalf("Apply(%q) = %q, %d; want %q, %d", tt.in, got, hits, tt.want, tt.hits)
			}

			// A second pass finds nothing left to rewrite.
			if again, hits := scrub.Apply(got); again != got || hits != 0 {
				t.Fatalf("second Apply(%q) = %q, %d", got, again, hits)
			}
		})
	}
}

func TestProfileUnknown(t *testing.T) {
	for _, name := range []string{"", "Antigravity", "other"} {
		if scrub, ok := Profile(name); ok || scrub != nil {
			t.Fatalf("Profile(%q) = %v, %v", name, scrub, ok)
		}
	}
}

func TestCompile(t *testing.T) {
	scrub, err := Compile([]Rule{
		{
			Pattern: `codename (\w+)`,
			Replace: "project ${1}X",
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	if got, hits := scrub.Apply("CODENAME azure, codename blue"); got != "project azureX, project blueX" || hits != 2 {
		t.Fatalf("Apply = %q, %d", got, hits)
	}

	for name, rules := range map[string][]Rule{
		"empty pattern": {{Pattern: ""}},
		"bad pattern":   {{Pattern: "codename ("}},
	} {
		if _, err := Compile(rules); err == nil {
			t.Fatalf("%s compiled", name)
		}
	}
}
