package transcript

import (
	"os"
	"path/filepath"
	"testing"

	"charm.land/fantasy"
)

func TestRoundTrip(t *testing.T) {
	t.Parallel()
	store := Store{Dir: filepath.Join(t.TempDir(), "sessions")}
	writer, err := store.Create()
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	sent := []fantasy.Message{
		fantasy.NewSystemMessage("dropped"),
		fantasy.NewUserMessage("list the files"),
		{Role: fantasy.MessageRoleAssistant, Content: []fantasy.MessagePart{
			fantasy.ToolCallPart{ToolCallID: "call_1", ToolName: "shell", Input: `{"cmd":"ls"}`},
		}},
		{Role: fantasy.MessageRoleTool, Content: []fantasy.MessagePart{
			fantasy.ToolResultPart{
				ToolCallID: "call_1",
				Output:     fantasy.ToolResultOutputContentText{Text: "main.go"},
			},
		}},
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
	if text := messageText(t, messages[0]); text != "list the files" {
		t.Fatalf("first message = %q", text)
	}
	call, ok := fantasy.AsMessagePart[fantasy.ToolCallPart](messages[1].Content[0])
	if !ok {
		t.Fatalf("second message part = %T, want a tool call", messages[1].Content[0])
	}
	if call.ToolName != "shell" {
		t.Fatalf("tool name = %q, want %q", call.ToolName, "shell")
	}
	result, ok := fantasy.AsMessagePart[fantasy.ToolResultPart](messages[2].Content[0])
	if !ok {
		t.Fatalf("third message part = %T, want a tool result", messages[2].Content[0])
	}
	if result.ToolCallID != "call_1" {
		t.Fatalf("tool call id = %q, want %q", result.ToolCallID, "call_1")
	}
}

// messageText returns the text of a message's first text part.
func messageText(t *testing.T, message fantasy.Message) string {
	t.Helper()
	if len(message.Content) == 0 {
		t.Fatal("message has no content")
	}
	text, ok := fantasy.AsMessagePart[fantasy.TextPart](message.Content[0])
	if !ok {
		t.Fatalf("part = %T, want text", message.Content[0])
	}
	return text.Text
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
