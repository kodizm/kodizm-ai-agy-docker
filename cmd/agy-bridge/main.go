// Command agy-bridge serves the Anthropic Messages API over Google
// Antigravity's `agy acp`. Run as `agy-bridge mcp-stub` it is the stdio MCP
// server agy starts to reach the client's tools.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"kodizm-agy-bridge/internal/agybridge"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "mcp-stub" {
		if err := agybridge.RunMCPStub(context.Background(), os.Stdin, os.Stdout, os.Getenv); err != nil {
			log.Fatalf("mcp-stub: %v", err)
		}

		return
	}

	if err := run(); err != nil {
		log.Fatalf("agy-bridge stopped: %v", err)
	}
}

// run serves until SIGINT or SIGTERM, which kill every agy session.
func run() error {
	cfg, err := agybridge.LoadConfig(os.Getenv)
	if err != nil {
		return err
	}

	flag.StringVar(&cfg.Listen, "listen", cfg.Listen, "listen address (AGY_BRIDGE_LISTEN)")
	flag.StringVar(&cfg.AgyBin, "agy", cfg.AgyBin, "agy binary (AGY_BIN)")
	flag.StringVar(&cfg.DefaultModel, "default-model", cfg.DefaultModel, "fallback model (AGY_BRIDGE_DEFAULT_MODEL)")
	flag.IntVar(&cfg.MaxConcurrent, "max-concurrent", cfg.MaxConcurrent, "live agy processes (AGY_BRIDGE_MAX_CONCURRENT)")
	flag.IntVar(&cfg.PerAccount, "per-account", cfg.PerAccount, "live agy processes per account (AGY_BRIDGE_PER_ACCOUNT)")
	flag.StringVar(&cfg.TokenFile, "token-file", cfg.TokenFile,
		"agy token file to serve instead of pushed accounts (AGY_BRIDGE_TOKEN_FILE)")
	flag.DurationVar(&cfg.UsageInterval, "usage-interval", cfg.UsageInterval,
		"agy quota poll interval, 0 off (AGY_BRIDGE_USAGE_INTERVAL)")
	flag.StringVar(&cfg.ScrubFile, "scrub-file", cfg.ScrubFile, "extra scrub rules (AGY_BRIDGE_SCRUB_FILE)")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	return agybridge.Run(ctx, cfg)
}
