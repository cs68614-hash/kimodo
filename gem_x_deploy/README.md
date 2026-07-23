---
title: GEM-X Motion Capture
emoji: 🕺
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
python_version: 3.12
short_description: Video to SOMA motion and Blender BVH with NVIDIA GEM-X
---

# GEM-X Motion Capture

Upload a short, single-person video and recover NVIDIA SOMA whole-body motion.

Outputs:

- a 77-keypoint tracking preview;
- a Blender-friendly 77-joint SOMA BVH;
- a 78-node BVH retaining GEM-X's virtual root;
- the original GEM-X PyTorch result and a ZIP archive.

The source pipeline is NVIDIA GEM-X. Model weights are downloaded from
`nvidia/GEM-X` and are governed by the NVIDIA Open Model License Agreement.
The application code and GEM-X source are Apache-2.0.

For best results, use a 1–2 second clip with one fully visible person. The
first run is slower because model assets are downloaded and cached.
