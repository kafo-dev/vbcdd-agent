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
	"strconv"
	"strings"
	"testing"
	"time"

	"vbcdd.kafo.dev/agent/internal/llm"
	"vbcdd.kafo.dev/agent/internal/shell"
)

// scripted stands in for the model: the first request asks for a shell
// command, the second answers in prose.
func scripted(t *testing.T, command string) *httptest.Server {
	t.Helper()
	turns := 0
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		turns++
		var body struct {
			Messages []llm.Message `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
		}
		if turns == 1 {
			writeReply(t, w, llm.Message{
				Role: llm.RoleAssistant,
				ToolCalls: []llm.ToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: llm.FunctionCall{
						Name:      shellToolName,
						Arguments: `{"cmd":` + strconv.Quote(command) + `}`,
					},
				}},
			})
			return
		}
		last := body.Messages[len(body.Messages)-1]
		if last.Role != llm.RoleTool {
			t.Errorf("last role = %q, want %q", last.Role, llm.RoleTool)
		}
		writeReply(t, w, llm.Message{Role: llm.RoleAssistant, Content: "done: " + last.Content})
	}))
}

// writeReply streams a message as the server-sent events the endpoint sends.
func writeReply(t *testing.T, w http.ResponseWriter, message llm.Message) {
	t.Helper()
	w.Header().Set("Content-Type", "text/event-stream")
	if message.Content != "" {
		delta, err := json.Marshal(map[string]any{
			"choices": []map[string]any{{"delta": map[string]any{"content": message.Content}}},
		})
		if err != nil {
			t.Errorf("encode delta: %v", err)
		}
		fmt.Fprintf(w, "data: %s\n\n", delta)
	}
	for index, call := range message.ToolCalls {
		fragment := map[string]any{
			"index": index,
			"id":    call.ID,
			"type":  call.Type,
			"function": map[string]any{
				"name":      call.Function.Name,
				"arguments": call.Function.Arguments,
			},
		}
		delta, err := json.Marshal(map[string]any{
			"choices": []map[string]any{{"delta": map[string]any{"tool_calls": []any{fragment}}}},
		})
		if err != nil {
			t.Errorf("encode delta: %v", err)
		}
		fmt.Fprintf(w, "data: %s\n\n", delta)
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
}

func TestAskRunsShellTool(t *testing.T) {
	t.Parallel()
	server := scripted(t, "printf hi")
	defer server.Close()

	runner := shell.Runner{Shell: "/bin/sh", Dir: t.TempDir(), Timeout: 5 * time.Second, MaxOutputBytes: 4096}
	client := llm.New(server.URL, "key", "model", nil, server.Client())
	var streamed bytes.Buffer
	ag := New(client, runner, "be brief", 5, &streamed, nil, nil)

	reply, err := ag.Ask(context.Background(), "say hi")
	if err != nil {
		t.Fatalf("Ask: %v", err)
	}
	if !strings.Contains(reply, "hi") {
		t.Fatalf("reply = %q, want the command output", reply)
	}
	if !strings.Contains(streamed.String(), "hi") {
		t.Fatalf("streamed = %q, want the command output", streamed.String())
	}
	if ag.Tokens() == 0 {
		t.Fatal("Tokens = 0, want an estimate when the endpoint reports none")
	}
	history := ag.History()
	if len(history) != 4 {
		t.Fatalf("history = %d messages, want 4", len(history))
	}
	if history[0].Role != llm.RoleUser {
		t.Fatalf("first role = %q, want %q", history[0].Role, llm.RoleUser)
	}
}

func TestHistoryDropsSystemPrompt(t *testing.T) {
	t.Parallel()
	ag := New(nil, shell.Runner{}, "keep quiet", 3, nil, nil, nil)
	if got := ag.History(); len(got) != 0 {
		t.Fatalf("History = %d messages, want 0", len(got))
	}
	ag.Restore([]llm.Message{{Role: llm.RoleUser, Content: "earlier"}})
	history := ag.History()
	if len(history) != 1 {
		t.Fatalf("History = %d messages, want 1", len(history))
	}
	if history[0].Content != "earlier" {
		t.Fatalf("History[0] = %q, want %q", history[0].Content, "earlier")
	}
}

func TestAskRecordsReportedUsage(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		reasoning, err := json.Marshal(map[string]any{
			"choices": []any{map[string]any{"delta": map[string]any{"reasoning_content": "hmm"}}},
		})
		if err != nil {
			t.Errorf("encode reasoning: %v", err)
		}
		fmt.Fprintf(w, "data: %s\n\n", reasoning)
		content, err := json.Marshal(map[string]any{
			"choices": []any{map[string]any{"delta": map[string]any{"content": "hi"}}},
		})
		if err != nil {
			t.Errorf("encode content: %v", err)
		}
		fmt.Fprintf(w, "data: %s\n\n", content)
		usage, err := json.Marshal(map[string]any{
			"choices": []any{},
			"usage":   map[string]any{"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105},
		})
		if err != nil {
			t.Errorf("encode usage: %v", err)
		}
		fmt.Fprintf(w, "data: %s\n\n", usage)
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	client := llm.New(server.URL, "key", "model", nil, server.Client())
	var thinking bytes.Buffer
	ag := New(client, shell.Runner{}, "", 3, io.Discard, &thinking, nil)
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
		w.Header().Set("Content-Type", "text/event-stream")
		if turns == 1 {
			writeDelta(t, w, map[string]any{"reasoning_content": "thinking one"})
			writeDelta(t, w, map[string]any{"tool_calls": []any{map[string]any{
				"index": 0,
				"id":    "call_1",
				"type":  "function",
				"function": map[string]any{
					"name":      shellToolName,
					"arguments": `{"cmd":"printf 'hi there'"}`,
				},
			}}})
		} else {
			writeDelta(t, w, map[string]any{"reasoning_content": "thinking two"})
			writeDelta(t, w, map[string]any{"content": "answer"})
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	runner := shell.Runner{Shell: "/bin/sh", Dir: t.TempDir(), Timeout: 5 * time.Second, MaxOutputBytes: 4096}
	client := llm.New(server.URL, "key", "model", nil, server.Client())
	var answer, thinking, notices bytes.Buffer
	ag := New(client, runner, "", 5, &answer, &thinking, &notices)
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
		w.Header().Set("Content-Type", "text/event-stream")
		if turns == 1 {
			writeDelta(t, w, map[string]any{"tool_calls": []any{map[string]any{
				"index": 0,
				"id":    "call_1",
				"type":  "function",
				"function": map[string]any{
					"name":      shellToolName,
					"arguments": `{"cmd":"printf hello; exit 3"}`,
				},
			}}})
		} else {
			writeDelta(t, w, map[string]any{"content": "done"})
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	runner := shell.Runner{Shell: "/bin/sh", Dir: t.TempDir(), Timeout: 5 * time.Second, MaxOutputBytes: 4096}
	client := llm.New(server.URL, "key", "model", nil, server.Client())
	var answer, thinking, notices bytes.Buffer
	ag := New(client, runner, "", 5, &answer, &thinking, &notices)
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

// writeDelta streams one delta to the endpoint's response.
func writeDelta(t *testing.T, w http.ResponseWriter, delta map[string]any) {
	t.Helper()
	event, err := json.Marshal(map[string]any{
		"choices": []any{map[string]any{"delta": delta}},
	})
	if err != nil {
		t.Errorf("encode delta: %v", err)
	}
	fmt.Fprintf(w, "data: %s\n\n", event)
}

func TestEstimateTokens(t *testing.T) {
	t.Parallel()
	messages := []llm.Message{{
		Role:             llm.RoleAssistant,
		Content:          strings.Repeat("a", 4),
		ReasoningContent: strings.Repeat("b", 4),
	}}
	if got := estimateTokens(messages); got != 2 {
		t.Fatalf("estimateTokens = %d, want 2", got)
	}
}

func TestNewOmitsEmptySystemPrompt(t *testing.T) {
	t.Parallel()
	ag := New(nil, shell.Runner{}, "   ", 3, nil, nil, nil)
	if len(ag.history) != 0 {
		t.Fatalf("history = %d messages, want none", len(ag.history))
	}
	ag.Restore([]llm.Message{{Role: llm.RoleUser, Content: "earlier"}})
	if len(ag.history) != 1 || ag.history[0].Role != llm.RoleUser {
		t.Fatalf("restored history = %+v, want one user message", ag.history)
	}
}

func TestAskRejectsEmptyInput(t *testing.T) {
	t.Parallel()
	ag := New(nil, shell.Runner{}, "", 3, nil, nil, nil)
	if _, err := ag.Ask(context.Background(), "   "); err == nil {
		t.Fatal("Ask accepted empty input")
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

func TestParametersAbbreviatesHomeDir(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)

	dir := filepath.Join(home, "SomeDir")
	ag := New(nil, shell.Runner{Shell: "/bin/sh", Dir: dir}, "", 3, nil, nil, nil)

	var schema struct {
		Properties struct {
			Cmd struct {
				Description string `json:"description"`
			} `json:"cmd"`
		} `json:"properties"`
	}
	if err := json.Unmarshal(ag.parameters(), &schema); err != nil {
		t.Fatalf("unmarshal parameters: %v", err)
	}
	got := schema.Properties.Cmd.Description
	if !strings.Contains(got, "`~/SomeDir`") {
		t.Fatalf("description = %q, want the abbreviated working directory", got)
	}
	if strings.Contains(got, home) {
		t.Fatalf("description = %q, want no absolute home path", got)
	}
}
