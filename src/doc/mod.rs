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

//! `mindc doc` — rustdoc-style HTML documentation generator for MIND source.
//!
//! Phase 1 scope:
//! - Walk `*.mind` files and extract `pub` items + preceding `///` doc-comments.
//! - Render one HTML page per source file plus a top-level `index.html`.
//! - Emit `search-index.json` for client-side search.
//! - Optionally open the result in a browser (`--open`).
//!
//! Public entry point: [`run_doc`].

pub mod html;
pub mod markdown;

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command as OsCommand;

use crate::ast::{Node, Param, TypeAnn};
use crate::parser::parse_with_trivia;
use crate::parser::{Trivia, TriviaKind};

// ---------------------------------------------------------------------------
// Public options
// ---------------------------------------------------------------------------

/// Configuration for the `mindc doc` subcommand.
#[derive(Debug, Clone)]
pub struct DocOptions {
    /// Source files or directories to document.  Empty = current directory.
    pub paths: Vec<String>,
    /// Output directory (default: `./target/doc`).
    pub out_dir: PathBuf,
    /// When true, skip documenting external dependency files.
    pub no_deps: bool,
    /// Open the generated index in a browser after rendering.
    pub open: bool,
}

impl Default for DocOptions {
    fn default() -> Self {
        Self {
            paths: Vec::new(),
            out_dir: PathBuf::from("target/doc"),
            no_deps: false,
            open: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Doc item model
// ---------------------------------------------------------------------------

/// The syntactic kind of a documented item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemKind {
    Fn,
    Struct,
    Enum,
    Const,
    TypeAlias,
}

impl ItemKind {
    fn as_str(&self) -> &'static str {
        match self {
            ItemKind::Fn => "fn",
            ItemKind::Struct => "struct",
            ItemKind::Enum => "enum",
            ItemKind::Const => "const",
            ItemKind::TypeAlias => "type",
        }
    }
}

/// A single documented item extracted from a MIND source file.
#[derive(Debug, Clone)]
pub struct DocItem {
    /// Syntactic kind.
    pub kind: ItemKind,
    /// Item name.
    pub name: String,
    /// Canonical signature string (e.g. `fn vec_new() -> Vec`).
    pub signature: String,
    /// Accumulated `///` doc-comment text, without the `///` prefix.
    pub doc: String,
    /// 1-based line number in the source file.
    pub line: usize,
}

/// All documented items for one source file.
#[derive(Debug, Clone)]
pub struct FileDoc {
    /// Absolute path to the source file.
    pub path: PathBuf,
    /// All public documented items in source order.
    pub items: Vec<DocItem>,
}

// ---------------------------------------------------------------------------
// Top-level driver
// ---------------------------------------------------------------------------

/// Run `mindc doc` and return an exit code (0 = success, 1 = error, 2 = bad args).
pub fn run_doc(opts: &DocOptions) -> i32 {
    let files = match resolve_paths(&opts.paths) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error[doc]: {e}");
            return 1;
        }
    };

    if files.is_empty() {
        eprintln!("error[doc]: no *.mind files found");
        return 1;
    }

    // Extract documentation from each file.
    let mut file_docs: Vec<FileDoc> = Vec::new();
    for path in &files {
        match extract_file_doc(path) {
            Ok(fd) => file_docs.push(fd),
            Err(e) => {
                eprintln!("error[doc]: {}: {e}", path.display());
                return 1;
            }
        }
    }

    // Compute the common ancestor of all source files so we can strip it when
    // constructing relative output paths.
    let base_dir = common_ancestor(&files);

    // Create the output directory tree.
    if let Err(e) = fs::create_dir_all(&opts.out_dir) {
        eprintln!(
            "error[doc]: cannot create output dir {}: {e}",
            opts.out_dir.display()
        );
        return 1;
    }

    // Render HTML for each file.
    for fd in &file_docs {
        if let Err(e) = render_file_html(fd, &opts.out_dir, &base_dir) {
            eprintln!("error[doc]: render {}: {e}", fd.path.display());
            return 1;
        }
    }

    // Render index.
    if let Err(e) = render_index_html(&file_docs, &opts.out_dir, &base_dir) {
        eprintln!("error[doc]: render index: {e}");
        return 1;
    }

    // Emit search index.
    if let Err(e) = emit_search_index(&file_docs, &opts.out_dir, &base_dir) {
        eprintln!("error[doc]: search index: {e}");
        return 1;
    }

    let index_path = opts.out_dir.join("index.html");
    println!(
        "   Generated {} item(s) → {}",
        total_items(&file_docs),
        index_path.display()
    );

    if opts.open {
        open_browser(&index_path);
    }

    0
}

fn total_items(docs: &[FileDoc]) -> usize {
    docs.iter().map(|fd| fd.items.len()).sum()
}

// ---------------------------------------------------------------------------
// Path resolution (mirrors check/fmt pattern)
// ---------------------------------------------------------------------------

fn resolve_paths(paths: &[String]) -> Result<Vec<PathBuf>, String> {
    let roots: Vec<PathBuf> = if paths.is_empty() {
        vec![std::env::current_dir().map_err(|e| format!("cannot read current directory: {e}"))?]
    } else {
        paths.iter().map(PathBuf::from).collect()
    };

    let mut out: Vec<PathBuf> = Vec::new();
    for root in roots {
        if root.is_file() {
            out.push(root);
        } else if root.is_dir() {
            collect_mind_files(&root, &mut out).map_err(|e| format!("{}: {e}", root.display()))?;
        } else {
            return Err(format!("'{}' does not exist", root.display()));
        }
    }
    out.sort();
    Ok(out)
}

fn collect_mind_files(dir: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    let mut children: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    children.sort();
    for child in children {
        if child.is_dir() {
            collect_mind_files(&child, out)?;
        } else if child.extension().is_some_and(|ext| ext == "mind") {
            out.push(child);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Extraction pipeline
// ---------------------------------------------------------------------------

/// Extract [`FileDoc`] from one source file.
fn extract_file_doc(path: &Path) -> Result<FileDoc, String> {
    let source = fs::read_to_string(path).map_err(|e| format!("cannot read: {e}"))?;

    let (module, trivia) =
        parse_with_trivia(&source).map_err(|errs| format!("parse error: {}", errs[0].message))?;

    // Build a map from byte-offset → trivia index for fast preceding-comment
    // lookup when visiting AST nodes.
    let doc_trivia: Vec<&Trivia> = trivia
        .0
        .iter()
        .filter(|t| t.kind == TriviaKind::DocComment)
        .collect();

    let mut items: Vec<DocItem> = Vec::new();

    for node in &module.items {
        if let Some(item) = extract_item(node, &doc_trivia, &source) {
            items.push(item);
        }
    }

    Ok(FileDoc {
        path: path.to_path_buf(),
        items,
    })
}

/// Extract a [`DocItem`] from an AST node if it is a `pub` item.
fn extract_item(node: &Node, doc_trivia: &[&Trivia], source: &str) -> Option<DocItem> {
    match node {
        Node::FnDef {
            is_pub: true,
            name,
            params,
            ret_type,
            span,
            ..
        } => {
            let (kind, sig) = (ItemKind::Fn, render_fn_sig(name, params, ret_type.as_ref()));
            let line = offset_to_line(source, span.start());
            let doc = collect_doc_comments(doc_trivia, span.start(), source);
            Some(DocItem {
                kind,
                name: name.clone(),
                signature: sig,
                doc,
                line,
            })
        }
        Node::StructDef {
            is_pub: true,
            name,
            fields,
            span,
            ..
        } => {
            let sig = format!("struct {name} {{ {} }}", render_fields(fields));
            let line = offset_to_line(source, span.start());
            let doc = collect_doc_comments(doc_trivia, span.start(), source);
            Some(DocItem {
                kind: ItemKind::Struct,
                name: name.clone(),
                signature: sig,
                doc,
                line,
            })
        }
        Node::EnumDef {
            is_pub: true,
            name,
            variants,
            span,
            ..
        } => {
            let variant_names: Vec<String> = variants.iter().map(|v| v.name.clone()).collect();
            let sig = format!("enum {name} {{ {} }}", variant_names.join(", "));
            let line = offset_to_line(source, span.start());
            let doc = collect_doc_comments(doc_trivia, span.start(), source);
            Some(DocItem {
                kind: ItemKind::Enum,
                name: name.clone(),
                signature: sig,
                doc,
                line,
            })
        }
        Node::Const { name, ty, span, .. } => {
            let ty_str = ty
                .as_ref()
                .map(render_type)
                .unwrap_or_else(|| "_".to_string());
            let sig = format!("const {name}: {ty_str}");
            let line = offset_to_line(source, span.start());
            let doc = collect_doc_comments(doc_trivia, span.start(), source);
            Some(DocItem {
                kind: ItemKind::Const,
                name: name.clone(),
                signature: sig,
                doc,
                line,
            })
        }
        Node::TypeAlias {
            name, target, span, ..
        } => {
            let sig = format!("type {name} = {}", render_type(target));
            let line = offset_to_line(source, span.start());
            let doc = collect_doc_comments(doc_trivia, span.start(), source);
            Some(DocItem {
                kind: ItemKind::TypeAlias,
                name: name.clone(),
                signature: sig,
                doc,
                line,
            })
        }
        _ => None,
    }
}

/// Render a `fn` signature string.
fn render_fn_sig(name: &str, params: &[Param], ret: Option<&TypeAnn>) -> String {
    let params_str = params
        .iter()
        .map(|p| format!("{}: {}", p.name, render_type(&p.ty)))
        .collect::<Vec<_>>()
        .join(", ");

    match ret {
        Some(ty) => format!("fn {name}({params_str}) -> {}", render_type(ty)),
        None => format!("fn {name}({params_str})"),
    }
}

/// Render struct fields for signature.
fn render_fields(fields: &[crate::ast::Field]) -> String {
    fields
        .iter()
        .map(|f| format!("{}: {}", f.name, render_type(&f.ty)))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Render a [`TypeAnn`] to a canonical string.
pub fn render_type(ty: &TypeAnn) -> String {
    match ty {
        TypeAnn::ScalarI32 => "i32".to_string(),
        TypeAnn::ScalarI64 => "i64".to_string(),
        TypeAnn::ScalarF32 => "f32".to_string(),
        TypeAnn::ScalarF64 => "f64".to_string(),
        TypeAnn::ScalarBool => "bool".to_string(),
        TypeAnn::ScalarU32 => "u32".to_string(),
        TypeAnn::Named(n) => n.clone(),
        TypeAnn::Tensor { dtype, dims } => {
            let dims_str = dims.join(", ");
            format!("tensor<{dtype}[{dims_str}]>")
        }
        TypeAnn::DiffTensor { dtype, dims } => {
            let dims_str = dims.join(", ");
            format!("diff tensor<{dtype}[{dims_str}]>")
        }
        TypeAnn::Slice { mutable, element } => {
            let mut_str = if *mutable { "mut " } else { "" };
            format!("&{}[{}]", mut_str, render_type(element))
        }
        TypeAnn::Array { element, length } => {
            format!("[{}; {}]", render_type(element), length)
        }
        TypeAnn::Ref { mutable, target } => {
            let mut_str = if *mutable { "mut " } else { "" };
            format!("&{}{}", mut_str, render_type(target))
        }
        TypeAnn::Generic { name, args } => {
            let args_str = args.iter().map(render_type).collect::<Vec<_>>().join(", ");
            format!("{name}<{args_str}>")
        }
        TypeAnn::Tuple { elements } => {
            let elems = elements
                .iter()
                .map(render_type)
                .collect::<Vec<_>>()
                .join(", ");
            format!("({elems})")
        }
        TypeAnn::SparseTensor {
            layout,
            element,
            shape,
        } => {
            let layout_str = match layout {
                crate::ast::SparseLayout::Csr => "csr",
                crate::ast::SparseLayout::Csc => "csc",
                crate::ast::SparseLayout::Coo => "coo",
                crate::ast::SparseLayout::Bsr => "bsr",
            };
            let shape_str = shape
                .iter()
                .map(|d| match d {
                    crate::types::ShapeDim::Known(n) => n.to_string(),
                    crate::types::ShapeDim::Sym(s) => s.to_string(),
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "tensor<sparse[{layout_str}], {}[{shape_str}]>",
                render_type(element)
            )
        }
        TypeAnn::RawPtr { mutable, pointee } => {
            let mut_str = if *mutable { "mut" } else { "const" };
            format!("*{} {}", mut_str, render_type(pointee))
        }
        TypeAnn::FnPtr { params, ret } => {
            let params_str = params
                .iter()
                .map(render_type)
                .collect::<Vec<_>>()
                .join(", ");
            match ret {
                Some(r) => format!("extern \"C\" fn({params_str}) -> {}", render_type(r)),
                None => format!("extern \"C\" fn({params_str})"),
            }
        }
    }
}

/// Find `///` doc-comment trivia that immediately precede `item_offset`
/// (allowing only blank lines between the last doc comment and the item).
fn collect_doc_comments(doc_trivia: &[&Trivia], item_offset: usize, _source: &str) -> String {
    // Collect doc-comment trivia that come before item_offset, then find the
    // contiguous block ending just before item_offset.
    let preceding: Vec<&&Trivia> = doc_trivia
        .iter()
        .filter(|t| t.byte_offset < item_offset)
        .collect();

    if preceding.is_empty() {
        return String::new();
    }

    // Walk backward from the last preceding doc comment.  Accept a contiguous
    // run of DocComment trivia (blank lines are tolerated but line-comments
    // break the chain).
    let mut lines: Vec<&str> = Vec::new();
    for trivia in preceding.iter().rev() {
        if trivia.kind == TriviaKind::DocComment {
            let text = trivia.text.as_str();
            // Strip `/// ` or `///` prefix.
            let content = if let Some(rest) = text.strip_prefix("/// ") {
                rest
            } else if let Some(rest) = text.strip_prefix("///") {
                rest
            } else {
                text
            };
            lines.push(content);
        } else {
            // A non-doc-comment trivia breaks the block.
            break;
        }
    }

    lines.reverse();
    lines.join("\n")
}

/// Convert a byte offset in `source` to a 1-based line number.
fn offset_to_line(source: &str, offset: usize) -> usize {
    source[..offset.min(source.len())]
        .bytes()
        .filter(|&b| b == b'\n')
        .count()
        + 1
}

// ---------------------------------------------------------------------------
// HTML rendering
// ---------------------------------------------------------------------------

/// Render the HTML page for one source file.
fn render_file_html(fd: &FileDoc, out_dir: &Path, base_dir: &Path) -> Result<(), String> {
    // Compute output path: strip common prefix and turn into an HTML path.
    let rel_stem = file_rel_stem(&fd.path, base_dir);
    let html_path = out_dir.join(format!("{rel_stem}.html"));

    if let Some(parent) = html_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create dir {}: {e}", parent.display()))?;
    }

    let title = rel_stem.replace('/', "::");
    let mut body = String::new();

    body.push_str(&format!(
        "<h1 class=\"module-title\">{}</h1>\n",
        html_escape(&title)
    ));

    if fd.items.is_empty() {
        body.push_str("<p class=\"empty\">No public items.</p>\n");
    }

    for item in &fd.items {
        body.push_str(&render_item_card(item));
    }

    let depth = rel_stem.chars().filter(|&c| c == '/').count() + 1;
    let root_prefix = "../".repeat(depth);
    let html = html::page(&title, &body, &root_prefix);

    fs::write(&html_path, html)
        .map_err(|e| format!("cannot write {}: {e}", html_path.display()))?;

    Ok(())
}

/// Render the top-level `index.html` listing all modules.
fn render_index_html(docs: &[FileDoc], out_dir: &Path, base_dir: &Path) -> Result<(), String> {
    let mut body = String::new();
    body.push_str("<h1 class=\"module-title\">MIND Documentation</h1>\n");
    body.push_str("<table class=\"module-index\">\n");
    body.push_str("<thead><tr><th>Module</th><th>Items</th></tr></thead>\n<tbody>\n");

    for fd in docs {
        let stem = file_rel_stem(&fd.path, base_dir);
        let display = stem.replace('/', "::");
        let href = format!("{stem}.html");
        let count = fd.items.len();
        body.push_str(&format!(
            "<tr><td><a href=\"{href}\">{display}</a></td><td>{count}</td></tr>\n",
        ));
    }

    body.push_str("</tbody>\n</table>\n");

    // Per-module item listing in the index.
    for fd in docs {
        if fd.items.is_empty() {
            continue;
        }
        let stem = file_rel_stem(&fd.path, base_dir);
        let display = stem.replace('/', "::");
        let href_base = format!("{stem}.html");
        body.push_str(&format!(
            "<h2 class=\"mod-section\"><a href=\"{href_base}\">{display}</a></h2>\n<ul class=\"item-list\">\n"
        ));
        for item in &fd.items {
            body.push_str(&format!(
                "<li><span class=\"kind\">{}</span> <a href=\"{href_base}#{name}\">{name}</a></li>\n",
                item.kind.as_str(),
                name = html_escape(&item.name),
            ));
        }
        body.push_str("</ul>\n");
    }

    let html = html::page("MIND Documentation", &body, "");
    let index_path = out_dir.join("index.html");
    fs::write(&index_path, html).map_err(|e| format!("cannot write index.html: {e}"))?;

    Ok(())
}

/// Render a single item card to HTML.
fn render_item_card(item: &DocItem) -> String {
    let kind_str = item.kind.as_str();
    let name_esc = html_escape(&item.name);
    let doc_html = if item.doc.is_empty() {
        String::new()
    } else {
        format!(
            "<div class=\"docs\">{}</div>\n",
            markdown::render_markdown(&item.doc)
        )
    };

    format!(
        "<div class=\"item\" id=\"{name_esc}\">\n\
         <h3 class=\"signature\">\
         <span class=\"kind\">{kind_str}</span> \
         <span class=\"name\">{name_esc}</span>\
         <span class=\"params\">{params}</span>\
         </h3>\n\
         {doc_html}\
         </div>\n",
        params = sig_params_fragment(&item.signature, kind_str, &item.name),
    )
}

/// Extract the fragment of a signature that comes after `fn name` or `struct name`.
fn sig_params_fragment(sig: &str, kind: &str, name: &str) -> String {
    let prefix = format!("{kind} {name}");
    let rest = sig.strip_prefix(&prefix).unwrap_or("").trim_start();
    html_escape(rest)
}

/// Compute the common ancestor directory for a list of files.
///
/// For a single file, this is its parent directory.  For multiple files it is
/// the longest common prefix of all parent directories.
fn common_ancestor(files: &[PathBuf]) -> PathBuf {
    if files.is_empty() {
        return PathBuf::from(".");
    }
    if files.len() == 1 {
        return files[0]
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
    }

    // Collect parent components for all files.
    let parents: Vec<PathBuf> = files
        .iter()
        .map(|f| {
            f.parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."))
        })
        .collect();

    // Use the first parent as seed; repeatedly strip the last component until
    // it is a prefix of all other parents.
    let mut ancestor = parents[0].clone();
    'outer: loop {
        for parent in &parents[1..] {
            if !parent.starts_with(&ancestor) {
                if !ancestor.pop() {
                    return PathBuf::from(".");
                }
                continue 'outer;
            }
        }
        break;
    }
    ancestor
}

/// Turn a source path into a relative stem like `std/vec` or `src/lib`.
///
/// Strips `base_dir` as a prefix (if possible), removes the `.mind` extension,
/// and normalises separators to `/`.
fn file_rel_stem(path: &Path, base_dir: &Path) -> String {
    // Strip the common base prefix if possible.
    let rel = path.strip_prefix(base_dir).unwrap_or(path);
    let s = rel.to_string_lossy();
    // Remove `.mind` extension.
    let without_ext = if let Some(p) = s.strip_suffix(".mind") {
        p
    } else {
        &s
    };
    // Normalise to forward slashes and strip leading `./`.
    let normalised = without_ext.replace('\\', "/");
    let trimmed = normalised.trim_start_matches("./");
    trimmed.to_string()
}

// ---------------------------------------------------------------------------
// Search index
// ---------------------------------------------------------------------------

/// Emit `search-index.json` for client-side search.
fn emit_search_index(docs: &[FileDoc], out_dir: &Path, base_dir: &Path) -> Result<(), String> {
    let mut entries: Vec<HashMap<&str, serde_json::Value>> = Vec::new();
    for fd in docs {
        let stem = file_rel_stem(&fd.path, base_dir);
        for item in &fd.items {
            let mut m = HashMap::new();
            m.insert("name", serde_json::Value::String(item.name.clone()));
            m.insert(
                "kind",
                serde_json::Value::String(item.kind.as_str().to_string()),
            );
            m.insert("file", serde_json::Value::String(format!("{stem}.html")));
            m.insert("line", serde_json::Value::Number(item.line.into()));
            entries.push(m);
        }
    }

    let json =
        serde_json::to_string_pretty(&entries).map_err(|e| format!("JSON serialisation: {e}"))?;

    let path = out_dir.join("search-index.json");
    fs::write(&path, json).map_err(|e| format!("cannot write search-index.json: {e}"))?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Browser open
// ---------------------------------------------------------------------------

fn open_browser(path: &Path) {
    let path_str = path.to_string_lossy();
    // Try platform openers in order.
    for opener in &["xdg-open", "open", "start"] {
        if OsCommand::new(opener)
            .arg(path_str.as_ref())
            .spawn()
            .is_ok()
        {
            return;
        }
    }
    eprintln!("warning[doc]: could not open browser automatically");
}

// ---------------------------------------------------------------------------
// HTML escape
// ---------------------------------------------------------------------------

/// Escape HTML special characters.
pub fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_escape_special_chars() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("a & b"), "a &amp; b");
        assert_eq!(html_escape("\"hello\""), "&quot;hello&quot;");
    }

    #[test]
    fn render_fn_sig_no_ret() {
        use crate::ast::Span;
        let params = vec![Param {
            name: "x".to_string(),
            ty: TypeAnn::ScalarI64,
            span: Span::new(0, 0),
        }];
        let sig = render_fn_sig("foo", &params, None);
        assert_eq!(sig, "fn foo(x: i64)");
    }

    #[test]
    fn render_fn_sig_with_ret() {
        let sig = render_fn_sig("bar", &[], Some(&TypeAnn::ScalarF32));
        assert_eq!(sig, "fn bar() -> f32");
    }

    #[test]
    fn offset_to_line_first_line() {
        assert_eq!(offset_to_line("abc\ndef", 1), 1);
    }

    #[test]
    fn offset_to_line_second_line() {
        assert_eq!(offset_to_line("abc\ndef", 4), 2);
    }

    #[test]
    fn file_rel_stem_strips_extension() {
        let p = PathBuf::from("std/vec.mind");
        assert_eq!(file_rel_stem(&p, Path::new(".")), "std/vec");
    }

    #[test]
    fn file_rel_stem_strips_base_prefix() {
        let p = PathBuf::from("/tmp/work/std/vec.mind");
        let base = PathBuf::from("/tmp/work");
        assert_eq!(file_rel_stem(&p, &base), "std/vec");
    }

    #[test]
    fn common_ancestor_single_file() {
        let files = vec![PathBuf::from("/tmp/a/foo.mind")];
        assert_eq!(common_ancestor(&files), PathBuf::from("/tmp/a"));
    }

    #[test]
    fn common_ancestor_multiple_files_same_dir() {
        let files = vec![
            PathBuf::from("/tmp/a/foo.mind"),
            PathBuf::from("/tmp/a/bar.mind"),
        ];
        assert_eq!(common_ancestor(&files), PathBuf::from("/tmp/a"));
    }

    #[test]
    fn common_ancestor_multiple_files_different_dirs() {
        let files = vec![
            PathBuf::from("/tmp/a/std/foo.mind"),
            PathBuf::from("/tmp/a/src/bar.mind"),
        ];
        assert_eq!(common_ancestor(&files), PathBuf::from("/tmp/a"));
    }

    #[test]
    fn collect_doc_comments_empty_when_none() {
        let doc = collect_doc_comments(&[], 100, "");
        assert!(doc.is_empty());
    }
}
