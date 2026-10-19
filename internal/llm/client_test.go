package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCompleteReadsToolCall(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer secret" {
			t.Errorf("Authorization = %q, want %q", got, "Bearer secret")
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("path = %q, want %q", r.URL.Path, "/v1/chat/completions")
		}
		reply := map[string]any{
			"choices": []map[string]any{{
				"message": map[string]any{
					"role":    "assistant",
					"content": "",
					"tool_calls": []map[string]any{{
						"id":       "call_1",
						"type":     "function",
						"function": map[string]any{"name": "shell", "arguments": `{"command":"ls"}`},
					}},
				},
			}},
		}
		if err := json.NewEncoder(w).Encode(reply); err != nil {
			t.Errorf("encode: %v", err)
		}
	}))
	defer server.Close()

	client := New(server.URL+"/v1/", "secret", "test-model", nil, server.Client())
	message, err := client.Complete(context.Background(), []Message{{Role: RoleUser, Content: "go"}}, nil)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(message.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(message.ToolCalls))
	}
	if message.ToolCalls[0].Function.Name != "shell" {
		t.Fatalf("tool name = %q, want %q", message.ToolCalls[0].Function.Name, "shell")
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
		reply := map[string]any{
			"choices": []map[string]any{{"message": map[string]any{"role": "assistant", "content": "ok"}}},
		}
		if err := json.NewEncoder(w).Encode(reply); err != nil {
			t.Errorf("encode: %v", err)
		}
	}))
	defer server.Close()

	headers := map[string]string{"X-Api-Key": "custom"}
	client := New(server.URL, "", "test-model", headers, server.Client())
	if _, err := client.Complete(context.Background(), []Message{{Role: RoleUser, Content: "go"}}, nil); err != nil {
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
	if _, err := client.Complete(context.Background(), []Message{{Role: RoleUser, Content: "go"}}, nil); err == nil {
		t.Fatal("Complete accepted a 401")
	}
}

func TestCompleteRejectsEmptyInput(t *testing.T) {
	t.Parallel()
	client := New("http://example.invalid", "key", "model", nil, nil)
	if _, err := client.Complete(context.Background(), nil, nil); err == nil {
		t.Fatal("Complete accepted no messages")
	}
}
