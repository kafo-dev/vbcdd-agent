package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/openaicompat"

	"vbcdd.kafo.dev/agent/internal/shell"
)

// newTestAgent wires the agent to a fake endpoint. It fails the test when the
// provider or the model will not build.
func newTestAgent(
	t *testing.T,
	server *httptest.Server,
	runner shell.Runner,
	systemPrompt string,
	out, thinking, notices io.Writer,
) *Agent {
	t.Helper()
	provider, err := openaicompat.New(
		openaicompat.WithBaseURL(server.URL),
		openaicompat.WithAPIKey("key"),
	)
	if err != nil {
		t.Fatalf("provider: %v", err)
	}
	model, err := provider.LanguageModel(context.Background(), "model")
	if err != nil {
		t.Fatalf("model: %v", err)
	}
	return New(Config{
		Model:        model,
		Runner:       runner,
		SystemPrompt: systemPrompt,
		MaxTurns:     5,
		Out:          out,
		Thinking:     thinking,
		Notices:      notices,
	})
}

// scripted stands in for the endpoint: the first request asks for a shell
// command, the second answers in prose.
func scripted(t *testing.T, command string) *httptest.Server {
	t.Helper()
	turns := 0
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		turns++
		var body struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		streamHeaders(w)
		if turns == 1 {
			writeToolCall(t, w, command)
			return
		}
		last := body.Messages[len(body.Messages)-1]
		if last.Role != "tool" {
			t.Errorf("last role = %q, want %q", last.Role, "tool")
		}
		writeTextDelta(t, w, "done")
		writeFinish(t, w, "stop", nil)
	}))
}

// streamHeaders sets the content type every streamed reply carries.
func streamHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
}

// writeToolCall streams one complete tool call and the reason that ends the
// turn.
func writeToolCall(t *testing.T, w http.ResponseWriter, command string) {
	t.Helper()
	arguments, err := json.Marshal(map[string]any{"cmd": command})
	if err != nil {
		t.Errorf("encode arguments: %v", err)
	}
	chunk := chunk(map[string]any{
		"tool_calls": []any{map[string]any{
			"index": 0,
			"id":    "call_1",
			"type":  "function",
			"function": map[string]any{
				"name":      shellToolName,
				"arguments": string(arguments),
			},
		}},
	}, nil)
	writeEvent(t, w, chunk)
	writeFinish(t, w, "tool_calls", nil)
}

// writeTextDelta streams one content fragment.
func writeTextDelta(t *testing.T, w http.ResponseWriter, text string) {
	t.Helper()
	writeEvent(t, w, chunk(map[string]any{"content": text}, nil))
}

// writeReasoningDelta streams one reasoning fragment.
func writeReasoningDelta(t *testing.T, w http.ResponseWriter, text string) {
	t.Helper()
	writeEvent(t, w, chunk(map[string]any{"reasoning_content": text}, nil))
}

// writeFinish streams the final event with the reason the turn ended and the
// tokens it reported.
func writeFinish(t *testing.T, w http.ResponseWriter, reason string, usage map[string]any) {
	t.Helper()
	writeEvent(t, w, map[string]any{
		"id":      "chatcmpl-1",
		"object":  "chat.completion.chunk",
		"created": 1,
		"model":   "model",
		"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": reason}},
		"usage":   usage,
	})
}

// chunk renders one streamed chunk that carries delta.
func chunk(delta map[string]any, finish any) map[string]any {
	return map[string]any{
		"id":      "chatcmpl-1",
		"object":  "chat.completion.chunk",
		"created": 1,
		"model":   "model",
		"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}},
	}
}

// writeEvent writes one server-sent event.
func writeEvent(t *testing.T, w http.ResponseWriter, event map[string]any) {
	t.Helper()
	raw, err := json.Marshal(event)
	if err != nil {
		t.Errorf("encode event: %v", err)
	}
	fmt.Fprintf(w, "data: %s\n\n", raw)
}

func TestAskRunsShellTool(t *testing.T) {
	t.Parallel()
	server := scripted(t, "printf hi")
	defer server.Close()

	runner := shell.Runner{Shell: "/bin/sh", Dir: t.TempDir(), Timeout: 5 * time.Second, MaxOutputBytes: 4096}
	var streamed bytes.Buffer
	ag := newTestAgent(t, server, runner, "be brief", &streamed, nil, nil)

	reply, err := ag.Ask(context.Background(), "say hi")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if !strings.Contains(reply, "done") {
		t.Fatalf("reply = %q, want the assistant's text", reply)
	}
	if !strings.Contains(streamed.String(), "done") {
		t.Fatalf("streamed = %q, want the assistant's text", streamed.String())
	}
	if ag.Tokens() == 0 {
		t.Fatal("Tokens = 0, want an estimate when the endpoint reports none")
	}
	history := ag.History()
	if len(history) != 4 {
		t.Fatalf("history = %d messages, want 4", len(history))
	}
	if history[0].Role != fantasy.MessageRoleUser {
		t.Fatalf("first role = %q, want %q", history[0].Role, fantasy.MessageRoleUser)
	}
	if history[1].Role != fantasy.MessageRoleAssistant {
		t.Fatalf("second role = %q, want %q", history[1].Role, fantasy.MessageRoleAssistant)
	}
	if history[2].Role != fantasy.MessageRoleTool {
		t.Fatalf("third role = %q, want %q", history[2].Role, fantasy.MessageRoleTool)
	}
}

func TestHistoryDropsSystemPrompt(t *testing.T) {
	t.Parallel()
	ag := New(Config{SystemPrompt: "keep quiet", MaxTurns: 3})
	if got := ag.History(); len(got) != 0 {
		t.Fatalf("History = %d messages, want 0", len(got))
	}
	ag.Restore([]fantasy.Message{fantasy.NewUserMessage("earlier")})
	history := ag.History()
	if len(history) != 1 {
		t.Fatalf("History = %d messages, want 1", len(history))
	}
	text, ok := fantasy.AsMessagePart[fantasy.TextPart](history[0].Content[0])
	if !ok || text.Text != "earlier" {
		t.Fatalf("History[0] = %+v, want the message text", history[0])
	}
}

func TestAskRecordsReportedUsage(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		streamHeaders(w)
		writeReasoningDelta(t, w, "hmm")
		writeTextDelta(t, w, "hi")
		writeFinish(t, w, "stop", map[string]any{
			"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105,
		})
	}))
	defer server.Close()

	var thinking bytes.Buffer
	ag := newTestAgent(t, server, shell.Runner{}, "", io.Discard, &thinking, nil)
	if _, err := ag.Ask(context.Background(), "say hi"); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if got := ag.Tokens(); got != 105 {
		t.Fatalf("Tokens = %d, want 105", got)
	}
	if got := thinking.String(); got != "hmm\n" {
		t.Fatalf("thinking = %q, want %q", got, "hmm\n")
	}
}

// TestAskSeparatesSections checks that the reasoning, the command, what it
// prints, and the answer each start on their own line even when the model ends
// a fragment mid-line.
func TestAskSeparatesSections(t *testing.T) {
	t.Parallel()
	turns := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		turns++
		streamHeaders(w)
		if turns == 1 {
			writeReasoningDelta(t, w, "thinking one")
			writeToolCall(t, w, "printf 'hi there'")
		} else {
			writeReasoningDelta(t, w, "thinking two")
			writeTextDelta(t, w, "answer")
			writeFinish(t, w, "stop", nil)
		}
	}))
	defer server.Close()

	runner := shell.Runner{Shell: "/bin/sh", Dir: t.TempDir(), Timeout: 5 * time.Second, MaxOutputBytes: 4096}
	var answer, thinking, notices bytes.Buffer
	ag := newTestAgent(t, server, runner, "", &answer, &thinking, &notices)
	if _, err := ag.Ask(context.Background(), "say hi"); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if got, want := thinking.String(), "thinking one\nthinking two\n"; got != want {
		t.Fatalf("thinking = %q, want %q", got, want)
	}
	if got, want := notices.String(), "$ printf 'hi there'\nhi there\n[exit code 0]\n"; got != want {
		t.Fatalf("notices = %q, want %q", got, want)
	}
	if got, want := answer.String(), "answer\n"; got != want {
		t.Fatalf("answer = %q, want %q", got, want)
	}
}

// TestAskShowsCommandOutput checks that a command's output and exit code reach
// the notices writer, not only the model.
func TestAskShowsCommandOutput(t *testing.T) {
	t.Parallel()
	turns := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		turns++
		streamHeaders(w)
		if turns == 1 {
			writeToolCall(t, w, "printf hello; exit 3")
		} else {
			writeTextDelta(t, w, "done")
			writeFinish(t, w, "stop", nil)
		}
	}))
	defer server.Close()

	runner := shell.Runner{Shell: "/bin/sh", Dir: t.TempDir(), Timeout: 5 * time.Second, MaxOutputBytes: 4096}
	var answer, thinking, notices bytes.Buffer
	ag := newTestAgent(t, server, runner, "", &answer, &thinking, &notices)
	if _, err := ag.Ask(context.Background(), "say hi"); err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if got, want := notices.String(), "$ printf hello; exit 3\nhello\n[exit code 3]\n"; got != want {
		t.Fatalf("notices = %q, want %q", got, want)
	}
	if got, want := thinking.String(), ""; got != want {
		t.Fatalf("thinking = %q, want %q", got, want)
	}
	if got, want := answer.String(), "done\n"; got != want {
		t.Fatalf("answer = %q, want %q", got, want)
	}
}

func TestEstimateTokens(t *testing.T) {
	t.Parallel()
	messages := []fantasy.Message{{
		Role: fantasy.MessageRoleAssistant,
		Content: []fantasy.MessagePart{
			fantasy.TextPart{Text: strings.Repeat("a", 4)},
			fantasy.ReasoningPart{Text: strings.Repeat("b", 4)},
		},
	}}
	if got := estimateTokens(messages); got != 2 {
		t.Fatalf("estimateTokens = %d, want 2", got)
	}
}

func TestAskRejectsEmptyInput(t *testing.T) {
	t.Parallel()
	ag := New(Config{MaxTurns: 3})
	if _, err := ag.Ask(context.Background(), "   "); err == nil {
		t.Fatal("Ask accepted empty input")
	}
}

func TestToolDescriptionAbbreviatesHomeDir(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	dir := filepath.Join(home, "SomeDir")
	ag := New(Config{Runner: shell.Runner{Shell: "/bin/sh", Dir: dir}, MaxTurns: 3})
	got := ag.toolDescription()
	if !strings.Contains(got, "`~/SomeDir`") {
		t.Fatalf("description = %q, want the abbreviated working directory", got)
	}
	if strings.Contains(got, home) {
		t.Fatalf("description = %q, want no absolute home path", got)
	}
	if !strings.Contains(got, "`/bin/sh -c`") {
		t.Fatalf("description = %q, want the named interpreter", got)
	}
}

func TestAbbreviateHome(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	cases := []struct {
		name string
		path string
		want string
	}{
		{"home itself", home, "~"},
		{"below home", filepath.Join(home, "SomeDir"), "~/SomeDir"},
		{"outside home", filepath.Dir(home), filepath.Dir(home)},
		{"sibling with shared prefix", home + "x", home + "x"},
	}
	//nolint:paralleltest // t.Setenv rules out parallel subtests.
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := abbreviateHome(tc.path); got != tc.want {
				t.Fatalf("abbreviateHome(%q) = %q, want %q", tc.path, got, tc.want)
			}
		})
	}
}
