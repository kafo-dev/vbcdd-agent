package transcript

import (
	"os"
	"path/filepath"
	"testing"

	"vbcdd.kafo.dev/agent/internal/llm"
)

func TestRoundTrip(t *testing.T) {
	t.Parallel()
	store := Store{Dir: filepath.Join(t.TempDir(), "sessions")}
	writer, err := store.Create()
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	sent := []llm.Message{
		{Role: llm.RoleSystem, Content: "dropped"},
		{Role: llm.RoleUser, Content: "list the files"},
		{Role: llm.RoleAssistant, Content: "", ToolCalls: []llm.ToolCall{{
			ID: "call_1", Type: "function", Function: llm.FunctionCall{Name: "shell", Arguments: `{"command":"ls"}`},
		}}},
		{Role: llm.RoleTool, ToolCallID: "call_1", Content: "main.go"},
	}
	for _, message := range sent {
		if err = writer.Append(message); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err = writer.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	session, messages, err := store.Open(writer.Session.ID)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if session.ID != writer.Session.ID {
		t.Fatalf("id = %q, want %q", session.ID, writer.Session.ID)
	}
	if len(messages) != 3 {
		t.Fatalf("messages = %d, want 3 (the system prompt stays out)", len(messages))
	}
	if messages[0].Content != "list the files" {
		t.Fatalf("first message = %q", messages[0].Content)
	}
	if len(messages[1].ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(messages[1].ToolCalls))
	}
	if messages[2].ToolCallID != "call_1" {
		t.Fatalf("tool call id = %q, want %q", messages[2].ToolCallID, "call_1")
	}
}

func TestLatestPicksNewest(t *testing.T) {
	t.Parallel()
	store := Store{Dir: filepath.Join(t.TempDir(), "sessions")}
	if _, _, err := store.Latest(); err != nil {
		t.Fatalf("Latest on a missing directory: %v", err)
	}
	first, err := store.Create()
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err = first.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	second, err := store.Create()
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err = second.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	session, ok, err := store.Latest()
	if err != nil {
		t.Fatalf("Latest: %v", err)
	}
	if !ok {
		t.Fatal("Latest found nothing")
	}
	if session.ID != second.Session.ID {
		t.Fatalf("latest = %q, want %q", session.ID, second.Session.ID)
	}
}

func TestOpenRejectsMissingSession(t *testing.T) {
	t.Parallel()
	store := Store{Dir: t.TempDir()}
	if _, _, err := store.Open("nope"); err == nil {
		t.Fatal("Open accepted a missing session")
	}
}

func TestCreateUsesFileMode(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	store := Store{Dir: dir}
	writer, err := store.Create()
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	defer writer.Close()
	info, err := os.Stat(writer.Session.Path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if perm := info.Mode().Perm(); perm != fileMode {
		t.Fatalf("file mode = %o, want %o", perm, fileMode)
	}
}
