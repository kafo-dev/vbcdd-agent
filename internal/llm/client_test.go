package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// writeEvent writes one server-sent event and the blank line that ends it.
func writeEvent(t *testing.T, w http.ResponseWriter, payload string) {
	t.Helper()
	if _, err := fmt.Fprintf(w, "data: %s\n\n", payload); err != nil {
		t.Errorf("write event: %v", err)
	}
}

// writeDelta writes one event whose choice carries the given delta.
func writeDelta(t *testing.T, w http.ResponseWriter, delta map[string]any) {
	t.Helper()
	payload, err := json.Marshal(map[string]any{
		"choices": []any{map[string]any{"delta": delta}},
	})
	if err != nil {
		t.Errorf("encode delta: %v", err)
		return
	}
	writeEvent(t, w, string(payload))
}

func TestCompleteReadsToolCall(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer secret" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer secret")
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/v1/chat/completions")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeDelta(t, w, map[string]any{"tool_calls": []any{
			map[string]any{
				"index": 0, "id": "call_1", "type": "function",
				"function": map[string]any{"name": "shell", "arguments": `{"command":`},
			},
		}})
		writeDelta(t, w, map[string]any{"tool_calls": []any{
			map[string]any{"index": 0, "function": map[string]any{"arguments": `"ls"}`}},
		}})
		writeEvent(t, w, "[DONE]")
	}))
	defer server.Close()

	client := New(server.URL+"/v1/", "secret", "test-model", nil, server.Client())
	message, err := client.Complete(context.Background(), []Message{{Role: RoleUser, Content: "go"}}, nil, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(message.ToolCalls))
	}
	if message.ToolCalls[0].Function.Name != "shell" {
		t.Fatalf("tool name = %q, want %q", message.ToolCalls[0].Function.Name, "shell")
	}
	if got := message.ToolCalls[0].Function.Arguments; got != `{"command":"ls"}` {
		t.Fatalf("arguments = %q, want the joined fragments", got)
	}
}

func TestCompleteStreamsContent(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writeDelta(t, w, map[string]any{"content": "he"})
		writeDelta(t, w, map[string]any{"content": "llo"})
		writeEvent(t, w, "[DONE]")
	}))
	defer server.Close()

	client := New(server.URL, "", "test-model", nil, server.Client())
	var fragments []string
	onText := func(text string) { fragments = append(fragments, text) }
	messages := []Message{{Role: RoleUser, Content: "go"}}
	message, err := client.Complete(context.Background(), messages, nil, onText)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if message.Content != "hello" {
		t.Fatalf("content = %q, want %q", message.Content, "hello")
	}
	if got := strings.Join(fragments, ""); got != "hello" {
		t.Fatalf("streamed = %q, want %q", got, "hello")
	}
}

func TestCompleteSendsCustomHeaders(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Api-Key"); got != "custom" {
			t.Errorf("X-Api-Key = %q, want %q", got, "custom")
		}
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("Authorization = %q, want it unset", got)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		writeDelta(t, w, map[string]any{"content": "ok"})
		writeEvent(t, w, "[DONE]")
	}))
	defer server.Close()

	headers := map[string]string{"X-Api-Key": "custom"}
	client := New(server.URL, "", "test-model", headers, server.Client())
	messages := []Message{{Role: RoleUser, Content: "go"}}
	if _, err := client.Complete(context.Background(), messages, nil, nil); err != nil {
		t.Fatalf("Complete: %v", err)
	}
}

func TestCompleteReportsHTTPError(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"bad key","type":"invalid_request_error"}}`))
	}))
	defer server.Close()

	client := New(server.URL, "secret", "test-model", nil, server.Client())
	messages := []Message{{Role: RoleUser, Content: "go"}}
	if _, err := client.Complete(context.Background(), messages, nil, nil); err == nil {
		t.Fatal("Complete accepted a 401")
	}
}

func TestCompleteRejectsEmptyInput(t *testing.T) {
	t.Parallel()
	client := New("http://example.invalid", "key", "model", nil, nil)
	if _, err := client.Complete(context.Background(), nil, nil, nil); err == nil {
		t.Fatal("Complete accepted no messages")
	}
}
