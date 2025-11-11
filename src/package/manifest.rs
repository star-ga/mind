use std::collections::BTreeMap;

use anyhow::Result;
use serde::Deserialize;
use serde::Serialize;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct MindManifest {
    pub name: String,
    pub version: String,
    pub authors: Vec<String>,
    pub description: Option<String>,
    pub license: Option<String>,
    pub dependencies: Option<Vec<String>>,
    pub files: Vec<String>,
    pub checksums: Option<BTreeMap<String, String>>,
}

impl MindManifest {
    pub fn to_toml(&self) -> Result<String> {
        Ok(toml::to_string_pretty(self)?)
    }
}
