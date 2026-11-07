package agybridge

import (
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestClassifyBanner(t *testing.T) {
	tests := []struct {
		name       string
		chunk      string
		status     int
		kind       failureKind
		retryAfter time.Duration
	}{
		{
			name:   "resource exhausted is quota",
			chunk:  "⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): Resource has been exhausted.",
			status: http.StatusTooManyRequests,
			kind:   kindQuota,
		},
		{
			name:   "the word quota is quota",
			chunk:  "⚠ API error (attempt 1): code 429: Quota exceeded for Claude and GPT models",
			status: http.StatusTooManyRequests,
			kind:   kindQuota,
		},
		{
			name:   "any other 429 is a rate limit",
			chunk:  "⚠ API error (attempt 1): code 429: Too many requests",
			status: http.StatusTooManyRequests,
			kind:   kindRate,
		},
		{
			name:   "unavailable",
			chunk:  "⚠ API error (attempt 1): UNAVAILABLE (code 503): try later",
			status: statusOverloaded,
			kind:   kindCapacity,
		},
		{
			name:   "overloaded",
			chunk:  "  ⚠ API error (attempt 1): code 529 overloaded",
			status: statusOverloaded,
			kind:   kindCapacity,
		},
		{
			name:   "authentication required",
			chunk:  "Authentication required. Please visit the URL to log in: https://accounts.example.test/auth",
			status: http.StatusBadGateway,
			kind:   kindAuth,
		},
		{
			name:   "unauthenticated",
			chunk:  "⚠ API error (attempt 1): UNAUTHENTICATED (code 401): invalid credentials",
			status: http.StatusBadGateway,
			kind:   kindAuth,
		},
		{
			name:   "invalid grant",
			chunk:  "⚠ Agent execution terminated: oauth2: invalid_grant",
			status: http.StatusBadGateway,
			kind:   kindAuth,
		},
		{
			name:   "validation required",
			chunk:  "⚠ API error (attempt 1): PERMISSION_DENIED (code 403): validation_required",
			status: http.StatusBadGateway,
			kind:   kindVerification,
		},
		{
			name:   "verify your account",
			chunk:  "⚠ API error (attempt 1): code 403: Please verify your account to continue.",
			status: http.StatusBadGateway,
			kind:   kindVerification,
		},
		{
			name:   "a 403 with a validation url",
			chunk:  "⚠ API error (attempt 1): code 403: see https://accounts.example.test/signin/validate?x=1",
			status: http.StatusBadGateway,
			kind:   kindVerification,
		},
		{
			name:   "terminated",
			chunk:  "⚠ Agent execution terminated due to error.",
			status: http.StatusBadGateway,
			kind:   kindOther,
		},
		{
			name:       "a retry delay in seconds",
			chunk:      "⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): retry in 12.5s",
			status:     http.StatusTooManyRequests,
			kind:       kindQuota,
			retryAfter: 12500 * time.Millisecond,
		},
		{
			name:       "a JSON retryDelay",
			chunk:      `⚠ API error (attempt 1): code 429: {"details":[{"retryDelay": "30s"}]}`,
			status:     http.StatusTooManyRequests,
			kind:       kindRate,
			retryAfter: 30 * time.Second,
		},
		{
			name:       "a quotaResetDelay",
			chunk:      `⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): {"quotaResetDelay":"1h2m3s"}`,
			status:     http.StatusTooManyRequests,
			kind:       kindQuota,
			retryAfter: time.Hour + 2*time.Minute + 3*time.Second,
		},
		{
			name:       "resets in",
			chunk:      "⚠ API error (attempt 1): code 429: weekly quota for Gemini 4 Argon resets in 5h",
			status:     http.StatusTooManyRequests,
			kind:       kindQuota,
			retryAfter: 5 * time.Hour,
		},
		{
			name:       "a bare count after",
			chunk:      "⚠ API error (attempt 1): UNAVAILABLE (code 503): try again after 40",
			status:     statusOverloaded,
			kind:       kindCapacity,
			retryAfter: 40 * time.Second,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := classifyBanner(tt.chunk)

			if got == nil || got.Status != tt.status || got.Kind != tt.kind || got.RetryAfter != tt.retryAfter {
				t.Fatalf("classifyBanner = %+v, want %d %s %s", got, tt.status, tt.kind, tt.retryAfter)
			}
		})
	}
}

func TestClassifyBannerIgnoresOrdinaryText(t *testing.T) {
	for _, chunk := range []string{
		"The API error (code 429) means you are rate limited.",
		"Your quota resets in 5h.",
		"",
	} {
		if got := classifyBanner(chunk); got != nil {
			t.Fatalf("classifyBanner(%q) = %+v, want nil", chunk, got)
		}
	}
}

// A JSON-RPC error carries no banner prefix: the words alone classify it.
func TestClassifyText(t *testing.T) {
	tests := []struct {
		text string
		kind failureKind
	}{
		{text: "Authentication required. Please visit the URL to log in:", kind: kindAuth},
		{text: "request had invalid authentication credentials: 401", kind: kindAuth},
		{text: "RESOURCE_EXHAUSTED", kind: kindQuota},
		{text: "refused", kind: kindOther},
	}

	for _, tt := range tests {
		if got := classifyText(tt.text); got.Kind != tt.kind {
			t.Fatalf("classifyText(%q) = %+v, want %s", tt.text, got, tt.kind)
		}
	}
}

// The message is cut for the client, but the whole banner is classified: a
// delay past the cut still counts.
func TestClassifyBannerParsesBeforeTheCut(t *testing.T) {
	banner := "⚠ API error (attempt 1): RESOURCE_EXHAUSTED (code 429): " + strings.Repeat("é", 400) + " retry in 45s"

	if got := classifyBanner(banner); got == nil || got.RetryAfter != 45*time.Second {
		t.Fatalf("classifyBanner = %+v, want a 45s delay", got)
	}

	if n := len([]rune(bannerMessage(banner))); n != failureMessageRunes {
		t.Fatalf("message has %d runes, want %d", n, failureMessageRunes)
	}
}

func TestBannerMessageCarriesNoEmail(t *testing.T) {
	got := bannerMessage("⚠ API error (attempt 1): code 403: verify your account a.b+c@gmail.com at x?hint=a%40gmail.com")

	if strings.Contains(got, "gmail") || strings.Count(got, "<email>") != 2 {
		t.Fatalf("message %q", got)
	}
}

func TestFailureAPIError(t *testing.T) {
	tests := []struct {
		f      failure
		status int
		kind   string
	}{
		{f: failure{Status: http.StatusTooManyRequests, Kind: kindQuota}, status: http.StatusTooManyRequests, kind: "rate_limit_error"},
		{f: failure{Status: statusOverloaded, Kind: kindCapacity}, status: statusOverloaded, kind: "overloaded_error"},
		{f: failure{Status: http.StatusBadGateway, Kind: kindAuth}, status: http.StatusBadGateway, kind: "api_error"},
	}

	for _, tt := range tests {
		if got := tt.f.apiError("m"); got.status != tt.status || got.kind != tt.kind || got.message != "m" {
			t.Fatalf("apiError = %+v, want %d %s", got, tt.status, tt.kind)
		}
	}
}
