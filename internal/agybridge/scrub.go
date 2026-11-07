package agybridge

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"sync"

	"kodizm-agy-bridge/internal/identityscrub"
)

var (
	// defaultScrub is the gateway's own antigravity profile, so the bridge
	// and an Antigravity rung neutralise the same fingerprints.
	defaultScrub, _ = identityscrub.Profile(identityscrub.ProfileAntigravity)

	blankRun = regexp.MustCompile(`\n{3,}`)
)

// scrubber applies the default rules plus the rules of an optional JSON file,
// re-read on every call so an operator can react to a new trigger live.
type scrubber struct {
	file string

	mu          sync.Mutex
	lastProblem string
}

func newScrubber(file string) *scrubber {
	return &scrubber{file: file}
}

// apply returns the scrubbed text and how many matches the rules rewrote.
func (s *scrubber) apply(text string) (string, int) {
	text, hits := defaultScrub.Apply(text)

	if extra := s.fileRules(); extra != nil {
		var n int

		text, n = extra.Apply(text)
		hits += n
	}

	return strings.TrimSpace(blankRun.ReplaceAllString(text, "\n\n")), hits
}

// fileRules loads the extra rules; a broken file is ignored as a whole so a
// half-applied rule set never reaches agy.
func (s *scrubber) fileRules() *identityscrub.Scrubber {
	if s.file == "" {
		return nil
	}

	rules, err := loadScrubFile(s.file)
	s.report(err)

	if err != nil {
		return nil
	}

	return rules
}

// report logs a file problem once until it changes, not on every request.
func (s *scrubber) report(err error) {
	problem := ""
	if err != nil {
		problem = err.Error()
	}

	s.mu.Lock()
	changed := problem != s.lastProblem
	s.lastProblem = problem
	s.mu.Unlock()

	if changed && err != nil {
		log.Printf("scrub file ignored: %v", err)
	}
}

func loadScrubFile(path string) (*identityscrub.Scrubber, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read scrub file: %w", err)
	}

	var rules []identityscrub.Rule
	if err := json.Unmarshal(raw, &rules); err != nil {
		return nil, fmt.Errorf("parse scrub file %s: %w", path, err)
	}

	compiled, err := identityscrub.Compile(rules)
	if err != nil {
		return nil, fmt.Errorf("scrub file %s: %w", path, err)
	}

	return compiled, nil
}
