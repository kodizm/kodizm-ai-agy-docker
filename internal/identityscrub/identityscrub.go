// Package identityscrub rewrites the client identity fingerprints an upstream
// refuses to serve. Google's Antigravity backend answers a system instruction
// that names a known agent harness with a fake 429 RESOURCE_EXHAUSTED, so a
// request bound for it has those sentences neutralised first.
package identityscrub

import (
	"fmt"
	"regexp"
)

// ProfileAntigravity is the provider profile of an Antigravity-fronting
// upstream.
const ProfileAntigravity = "antigravity"

// NeutralIdentity stands in for a harness identity sentence.
const NeutralIdentity = "You are an AI coding agent."

// Rule rewrites every match of Pattern with Replace. Patterns run
// case-insensitive and multiline; Replace uses Go expansion syntax ($1,
// ${name}), and an empty Replace removes the match.
type Rule struct {
	Pattern string `json:"pattern"`
	Replace string `json:"replace"`
}

// Scrubber is a compiled rule list, safe for concurrent use.
type Scrubber struct {
	rules []compiledRule
}

type compiledRule struct {
	re      *regexp.Regexp
	replace string
}

// antigravityRules are the measured triggers: each one alone in a system
// instruction draws the fake 429, while the same words in a message do not.
var antigravityRules = []Rule{
	{
		Pattern: `^[ \t]*x-anthropic-[a-z0-9-]*header[ \t]*:[^\n]*\n?`,
	},
	{
		Pattern: `You are a Claude agent, built on Anthropic['’]s Claude Agent SDK\.`,
		Replace: NeutralIdentity,
	},
	{
		Pattern: `You are Claude Code, Anthropic['’]s official CLI for Claude(?:, running within the Claude Agent SDK)?\.`,
		Replace: NeutralIdentity,
	},
	{
		// The coordinator prompt names its role after the comma, so the
		// match runs to the sentence's own period.
		Pattern: `You are Claude Code, an AI assistant that orchestrates[^.\n]*\.`,
		Replace: NeutralIdentity,
	},
	{
		// The sentence runs on after the model name, so the source's own
		// punctuation closes the replacement.
		Pattern: `You are Codex, (?:an|a coding) agent based on GPT-5(?:\.\d+)?(?:-[\w-]+)?`,
		Replace: "You are an AI coding agent",
	},
	{
		Pattern: `You are Codex, based on GPT-5(?:\.\d+)?(?:-[\w-]+)?`,
		Replace: "You are an AI coding agent",
	},
	{
		Pattern: `Hermes Agent, an intelligent AI assistant created by Nous Research`,
		Replace: "an AI assistant",
	},
}

var profiles = map[string]*Scrubber{
	ProfileAntigravity: mustCompile(antigravityRules),
}

// Profile is the built-in scrubber of a provider profile; false for a name
// with none.
func Profile(name string) (*Scrubber, bool) {
	scrubber, ok := profiles[name]

	return scrubber, ok
}

// Compile builds a scrubber from rules in order; an empty or invalid pattern
// fails the whole list, so a half-compiled rule set never runs.
func Compile(rules []Rule) (*Scrubber, error) {
	compiled := make([]compiledRule, 0, len(rules))

	for i, rule := range rules {
		if rule.Pattern == "" {
			return nil, fmt.Errorf("rule %d: empty pattern", i)
		}

		re, err := regexp.Compile("(?im)" + rule.Pattern)
		if err != nil {
			return nil, fmt.Errorf("rule %d: %w", i, err)
		}

		compiled = append(compiled, compiledRule{
			re:      re,
			replace: rule.Replace,
		})
	}

	return &Scrubber{rules: compiled}, nil
}

// Apply returns text with every rule applied in order and the number of
// matches rewritten. A text no rule matches comes back unchanged with zero.
func (s *Scrubber) Apply(text string) (string, int) {
	hits := 0

	for _, rule := range s.rules {
		n := len(rule.re.FindAllStringIndex(text, -1))
		if n == 0 {
			continue
		}

		hits += n
		text = rule.re.ReplaceAllString(text, rule.replace)
	}

	return text, hits
}

func mustCompile(rules []Rule) *Scrubber {
	scrubber, err := Compile(rules)
	if err != nil {
		panic(err)
	}

	return scrubber
}
