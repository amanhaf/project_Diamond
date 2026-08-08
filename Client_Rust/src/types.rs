use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct WorkerOk {
    pub status: String,
    pub original_path: String,
    pub normalized_path: String,
    pub converted: bool,
    pub render_path: String,
    pub phash_render: String,
    pub uuid: String,
    pub width_in: f64,
    pub height_in: f64,
    pub stitch_count: i64,
    pub color_count: Option<i64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct WorkerErr {
    pub status: String,
    pub original_path: Option<String>,
    pub error: String,
}