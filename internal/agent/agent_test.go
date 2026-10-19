package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
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
						Arguments: `{"command":` + strconv.Quote(command) + `}`,
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
	ag := New(client, runner, "be brief", 5, &streamed, nil)

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
	ag := New(nil, shell.Runner{}, "keep quiet", 3, nil, nil)
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

func TestAskRejectsEmptyInput(t *testing.T) {
	t.Parallel()
	ag := New(nil, shell.Runner{}, "", 3, nil, nil)
	if _, err := ag.Ask(context.Background(), "   "); err == nil {
		t.Fatal("Ask accepted empty input")
	}
}
