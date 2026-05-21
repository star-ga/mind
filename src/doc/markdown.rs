// Copyright 2025 STARGA Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Minimal hand-rolled Markdown → HTML renderer for `mindc doc`.
//!
//! Covers the subset used in MIND `///` doc-comments:
//! - Paragraphs (blank-line separated)
//! - ATX headings (`# H1`, `## H2`, `### H3`)
//! - Fenced code blocks (triple backtick)
//! - Inline `code` (single backtick)
//! - **Bold** (`**text**` or `__text__`)
//! - *Italic* (`*text*` or `_text_`)
//! - `[link text](url)` hyperlinks
//! - Unordered lists (`- item` or `* item`)
//!
//! No external crate dependencies.

use super::html_escape;

/// Render `markdown` text to an HTML fragment string.
pub fn render_markdown(markdown: &str) -> String {
    let mut out = String::with_capacity(markdown.len() * 2);

    let lines: Vec<&str> = markdown.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Fenced code block
        if line.trim_start().starts_with("```") {
            let fence_len = count_leading_backticks(line.trim_start());
            let lang = line.trim_start().get(fence_len..).unwrap_or("").trim();
            out.push_str("<pre><code");
            if !lang.is_empty() {
                out.push_str(&format!(" class=\"language-{lang}\""));
            }
            out.push('>');
            i += 1;
            while i < lines.len() {
                let code_line = lines[i];
                if code_line.trim_start().starts_with(&"`".repeat(fence_len)) {
                    i += 1;
                    break;
                }
                out.push_str(&html_escape(code_line));
                out.push('\n');
                i += 1;
            }
            out.push_str("</code></pre>\n");
            continue;
        }

        // ATX headings
        if let Some(level) = heading_level(line) {
            let content = line.trim_start_matches('#').trim();
            out.push_str(&format!(
                "<h{level}>{}</h{level}>\n",
                inline_render(content)
            ));
            i += 1;
            continue;
        }

        // Unordered list items
        if is_list_item(line) {
            out.push_str("<ul>\n");
            while i < lines.len() && is_list_item(lines[i]) {
                let content = lines[i].trim_start_matches(['-', '*', ' ', '\t']);
                out.push_str(&format!("<li>{}</li>\n", inline_render(content.trim())));
                i += 1;
            }
            out.push_str("</ul>\n");
            continue;
        }

        // Blank line — paragraph break
        if line.trim().is_empty() {
            i += 1;
            // Skip multiple blank lines
            while i < lines.len() && lines[i].trim().is_empty() {
                i += 1;
            }
            continue;
        }

        // Paragraph: collect non-blank, non-heading, non-list, non-fence lines
        let mut para_lines: Vec<&str> = Vec::new();
        while i < lines.len() {
            let l = lines[i];
            if l.trim().is_empty()
                || heading_level(l).is_some()
                || is_list_item(l)
                || l.trim_start().starts_with("```")
            {
                break;
            }
            para_lines.push(l);
            i += 1;
        }
        if !para_lines.is_empty() {
            let text = para_lines.join(" ");
            out.push_str(&format!("<p>{}</p>\n", inline_render(&text)));
        }
    }

    out
}

/// Count consecutive backticks at the start of `s`.
fn count_leading_backticks(s: &str) -> usize {
    s.chars().take_while(|&c| c == '`').count()
}

/// Return the heading level (1–6) if `line` starts with `#` sequences, else None.
fn heading_level(line: &str) -> Option<usize> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let level = trimmed.chars().take_while(|&c| c == '#').count();
    if level > 6 {
        return None;
    }
    // Must be followed by space or end of line
    let after = &trimmed[level..];
    if after.is_empty() || after.starts_with(' ') {
        Some(level)
    } else {
        None
    }
}

/// Return true if `line` starts an unordered list item (`- ` or `* `).
fn is_list_item(line: &str) -> bool {
    let t = line.trim_start();
    (t.starts_with("- ") || t.starts_with("* "))
        && !t.starts_with("---")
        && !t.starts_with("***")
}

/// Render inline Markdown spans to HTML.
///
/// Handles: `code`, **bold**, *italic*, [link](url).
fn inline_render(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 32);
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut i = 0;

    while i < n {
        let ch = chars[i];

        // Inline code: `...`
        if ch == '`' {
            let end = chars[i + 1..].iter().position(|&c| c == '`');
            if let Some(rel) = end {
                let code: String = chars[i + 1..i + 1 + rel].iter().collect();
                out.push_str("<code>");
                out.push_str(&html_escape(&code));
                out.push_str("</code>");
                i += rel + 2;
                continue;
            }
        }

        // Bold: **...** or __...__
        if (ch == '*' && i + 1 < n && chars[i + 1] == '*')
            || (ch == '_' && i + 1 < n && chars[i + 1] == '_')
        {
            let delim = ch;
            let end = find_double_delim(&chars, i + 2, delim);
            if let Some(rel) = end {
                let inner: String = chars[i + 2..i + 2 + rel].iter().collect();
                out.push_str("<strong>");
                out.push_str(&html_escape(&inner));
                out.push_str("</strong>");
                i += rel + 4;
                continue;
            }
        }

        // Italic: *...* or _..._
        if ch == '*' || ch == '_' {
            let delim = ch;
            let end = find_single_delim(&chars, i + 1, delim);
            if let Some(rel) = end {
                let inner: String = chars[i + 1..i + 1 + rel].iter().collect();
                out.push_str("<em>");
                out.push_str(&html_escape(&inner));
                out.push_str("</em>");
                i += rel + 2;
                continue;
            }
        }

        // Link: [text](url)
        if ch == '[' {
            if let Some(link_html) = try_parse_link(&chars, i) {
                let (html_frag, consumed) = link_html;
                out.push_str(&html_frag);
                i += consumed;
                continue;
            }
        }

        // HTML escape everything else
        let mut buf = [0u8; 4];
        out.push_str(&html_escape(ch.encode_utf8(&mut buf)));
        i += 1;
    }

    out
}

/// Find the position (relative to `start`) of a `**` or `__` closing delimiter.
fn find_double_delim(chars: &[char], start: usize, delim: char) -> Option<usize> {
    let mut j = start;
    while j + 1 < chars.len() {
        if chars[j] == delim && chars[j + 1] == delim {
            return Some(j - start);
        }
        j += 1;
    }
    None
}

/// Find the position (relative to `start`) of a single closing delimiter.
fn find_single_delim(chars: &[char], start: usize, delim: char) -> Option<usize> {
    let mut j = start;
    while j < chars.len() {
        if chars[j] == delim {
            return Some(j - start);
        }
        if chars[j] == '\n' {
            return None;
        }
        j += 1;
    }
    None
}

/// Attempt to parse `[text](url)` starting at `i`.
/// Returns `(html_fragment, chars_consumed)` or `None`.
fn try_parse_link(chars: &[char], i: usize) -> Option<(String, usize)> {
    // Find closing `]`
    let close_bracket = chars[i + 1..].iter().position(|&c| c == ']')?;
    let text_end = i + 1 + close_bracket;
    // Must be followed by `(`
    if chars.get(text_end + 1) != Some(&'(') {
        return None;
    }
    // Find closing `)`
    let close_paren = chars[text_end + 2..]
        .iter()
        .position(|&c| c == ')')?;
    let url_end = text_end + 2 + close_paren;

    let text: String = chars[i + 1..text_end].iter().collect();
    let url: String = chars[text_end + 2..url_end].iter().collect();

    let consumed = url_end - i + 1;
    let html = format!(
        "<a href=\"{}\">{}</a>",
        html_escape(&url),
        html_escape(&text)
    );
    Some((html, consumed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_plain_paragraph() {
        let html = render_markdown("hello world");
        assert!(html.contains("<p>hello world</p>"), "got: {html}");
    }

    #[test]
    fn renders_inline_code() {
        let html = render_markdown("use `vec_new` to create");
        assert!(html.contains("<code>vec_new</code>"), "got: {html}");
    }

    #[test]
    fn renders_bold() {
        let html = render_markdown("**important** note");
        assert!(html.contains("<strong>important</strong>"), "got: {html}");
    }

    #[test]
    fn renders_italic() {
        let html = render_markdown("*note* this");
        assert!(html.contains("<em>note</em>"), "got: {html}");
    }

    #[test]
    fn renders_heading() {
        let html = render_markdown("## Examples");
        assert!(html.contains("<h2>Examples</h2>"), "got: {html}");
    }

    #[test]
    fn renders_fenced_code() {
        let md = "```mind\nfn foo() {}\n```";
        let html = render_markdown(md);
        assert!(html.contains("<pre><code"), "got: {html}");
        assert!(html.contains("fn foo()"), "got: {html}");
    }

    #[test]
    fn renders_list() {
        let md = "- first\n- second";
        let html = render_markdown(md);
        assert!(html.contains("<ul>"), "got: {html}");
        assert!(html.contains("<li>first</li>"), "got: {html}");
        assert!(html.contains("<li>second</li>"), "got: {html}");
    }

    #[test]
    fn renders_link() {
        let html = render_markdown("[MIND](https://mindlang.dev)");
        assert!(
            html.contains("<a href=\"https://mindlang.dev\">MIND</a>"),
            "got: {html}"
        );
    }
}
