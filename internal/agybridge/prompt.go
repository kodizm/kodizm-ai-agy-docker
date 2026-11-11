package agybridge

import (
	"bytes"
	"cmp"
	"encoding/json"
	"slices"
	"strconv"
	"strings"
)

// Placeholders stand where an attachment was when agy cannot take it.
const (
	imageOmitted    = "[image omitted]"
	documentOmitted = "[document omitted]"
)

// maxAttachmentNameBytes bounds the title part of a document's uri.
const maxAttachmentNameBytes = 64

// promptCapabilities is the content beyond text that agy said, at initialize,
// it takes in a prompt. ACP allows a client nothing else; text is always fine.
type promptCapabilities struct {
	image           bool
	embeddedContext bool
}

// parsePromptCapabilities reads agentCapabilities from an initialize result in
// either ACP shape: v1 promptCapabilities of booleans, or the v2 draft prompt
// of objects. Anything missing or of another shape is not advertised.
func parsePromptCapabilities(result json.RawMessage) promptCapabilities {
	var initialized struct {
		AgentCapabilities json.RawMessage `json:"agentCapabilities"`
	}

	var agent struct {
		V1 json.RawMessage `json:"promptCapabilities"`
		V2 json.RawMessage `json:"prompt"`
	}

	if json.Unmarshal(result, &initialized) != nil || json.Unmarshal(initialized.AgentCapabilities, &agent) != nil {
		return promptCapabilities{}
	}

	v1, v2 := capabilityFlags(agent.V1), capabilityFlags(agent.V2)
	isTrue := func(raw json.RawMessage) bool { return bytes.Equal(bytes.TrimSpace(raw), []byte("true")) }
	isObject := func(raw json.RawMessage) bool { return bytes.HasPrefix(bytes.TrimSpace(raw), []byte("{")) }

	return promptCapabilities{
		image:           isTrue(v1["image"]) || isObject(v2["image"]),
		embeddedContext: isTrue(v1["embeddedContext"]) || isObject(v2["embeddedContext"]),
	}
}

// capabilityFlags splits a capability object into its raw flags, none when it
// is not an object.
func capabilityFlags(raw json.RawMessage) map[string]json.RawMessage {
	var flags map[string]json.RawMessage

	_ = json.Unmarshal(raw, &flags)

	return flags
}

// promptBlock is one ACP content block of session/prompt: text, an image, or
// an embedded resource.
type promptBlock struct {
	Type     string
	Text     string
	MimeType string
	Data     string
	Resource *embeddedResource
}

// embeddedResource is the blob variant of an ACP embedded resource.
type embeddedResource struct {
	URI      string `json:"uri"`
	MimeType string `json:"mimeType"`
	Blob     string `json:"blob"`
}

// MarshalJSON writes each block in its own ACP shape: a text block always
// carries its text, even empty, and no other block carries any.
func (b promptBlock) MarshalJSON() ([]byte, error) {
	switch b.Type {
	case "image":
		return json.Marshal(struct {
			Type     string `json:"type"`
			MimeType string `json:"mimeType"`
			Data     string `json:"data"`
		}{b.Type, b.MimeType, b.Data})
	case "resource":
		return json.Marshal(struct {
			Type     string            `json:"type"`
			Resource *embeddedResource `json:"resource"`
		}{b.Type, b.Resource})
	}

	return json.Marshal(struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}{b.Type, b.Text})
}

// prompt is a rendered conversation: its text, with each attachment agy might
// take as a block of its own where it occurred. Neighbouring text is always
// one block.
type prompt []promptBlock

// textPrompt is text alone; empty text is an empty prompt.
func textPrompt(text string) prompt {
	var p prompt

	p.addText(text)

	return p
}

func (p *prompt) addText(text string) {
	if text == "" {
		return
	}

	if n := len(*p); n > 0 && (*p)[n-1].Type == "text" {
		(*p)[n-1].Text += text

		return
	}

	*p = append(*p, promptBlock{Type: "text", Text: text})
}

func (p *prompt) add(other prompt) {
	for _, b := range other {
		if b.Type == "text" {
			p.addText(b.Text)
		} else {
			*p = append(*p, b)
		}
	}
}

// joinLines joins the parts with a newline between each two, as the text
// prompt has always joined blocks.
func joinLines(parts []prompt) prompt {
	var joined prompt

	for i, part := range parts {
		if i > 0 {
			joined.addText("\n")
		}

		joined.add(part)
	}

	return joined
}

// hasMedia reports whether the prompt carries a block other than text.
func (p prompt) hasMedia() bool {
	for _, b := range p {
		if b.Type != "text" {
			return true
		}
	}

	return false
}

// forAgent is the prompt as agy may take it: an attachment it did not
// advertise turns into its placeholder. It is never empty, since
// session/prompt needs content.
func (p prompt) forAgent(capabilities promptCapabilities) []promptBlock {
	var out prompt

	for _, b := range p {
		switch {
		case b.Type == "image" && !capabilities.image:
			out.addText(imageOmitted)
		case b.Type == "resource" && !capabilities.embeddedContext:
			out.addText(documentOmitted)
		default:
			out.add(prompt{b})
		}
	}

	if len(out) == 0 {
		return []promptBlock{{Type: "text"}}
	}

	return out
}

// text is the prompt with every attachment as its placeholder.
func (p prompt) text() string {
	return p.forAgent(promptCapabilities{})[0].Text
}

// mediaSource is the source of an Anthropic image or document block.
type mediaSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// imageTypes are the image media types both Anthropic and Gemini take.
var imageTypes = []string{"image/jpeg", "image/png", "image/gif", "image/webp"}

// promptRenderer renders a conversation; it numbers the PDFs in order, so a
// renderer serves one conversation.
type promptRenderer struct {
	documents int
}

// image is an image block from a base64 source of a known type, or the
// placeholder. The payload is client input, so it is checked but never
// decoded, and a url is never fetched.
func (r *promptRenderer) image(raw json.RawMessage) prompt {
	var source mediaSource

	if json.Unmarshal(raw, &source) != nil || source.Type != "base64" || !isBase64(source.Data) {
		return textPrompt(imageOmitted)
	}

	mediaType := strings.ToLower(source.MediaType)
	if mediaType == "image/jpg" {
		mediaType = "image/jpeg"
	}

	if !slices.Contains(imageTypes, mediaType) {
		return textPrompt(imageOmitted)
	}

	return prompt{{Type: "image", MimeType: mediaType, Data: source.Data}}
}

// document is a base64 PDF as an embedded resource, a plain text document as
// its text, or the placeholder.
func (r *promptRenderer) document(b contentBlock) prompt {
	var source mediaSource

	if json.Unmarshal(b.Source, &source) != nil {
		return textPrompt(documentOmitted)
	}

	mediaType := strings.ToLower(source.MediaType)

	switch {
	case source.Type == "text" && mediaType == "text/plain":
		return textPrompt(source.Data)
	case source.Type == "base64" && mediaType == "application/pdf" && isBase64(source.Data):
		r.documents++

		var title string
		_ = json.Unmarshal(b.Title, &title)

		return prompt{{
			Type: "resource",
			Resource: &embeddedResource{
				URI:      attachmentURI(title, r.documents),
				MimeType: "application/pdf",
				Blob:     source.Data,
			},
		}}
	}

	return textPrompt(documentOmitted)
}

// attachmentURI names the nth PDF of the conversation after its title, cut to
// a safe set of characters: the title is client input and never a path.
func attachmentURI(title string, n int) string {
	name := strings.Map(func(r rune) rune {
		if r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '.' || r == '_' || r == '-' {
			return r
		}

		return '_'
	}, title)

	// Every rune left is one byte.
	name = name[:min(len(name), maxAttachmentNameBytes)]

	return "file:///attachments/" + cmp.Or(name, "document") + "-" + strconv.Itoa(n) + ".pdf"
}

// isBase64 reports whether data is non-empty standard padded base64, by its
// characters alone.
func isBase64(data string) bool {
	if data == "" || len(data)%4 != 0 {
		return false
	}

	body := strings.TrimSuffix(strings.TrimSuffix(data, "="), "=")

	return !strings.ContainsFunc(body, func(r rune) bool {
		return !(r >= 'A' && r <= 'Z' || r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '+' || r == '/')
	})
}
