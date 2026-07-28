---
title: Image to 3D ZeroGPU
emoji: 🧊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
suggested_hardware: zero-a10g
preload_from_hub:
  - tencent/Hunyuan3D-2 hunyuan3d-dit-v2-0/*
---

# Image to 3D — private ZeroGPU edition

Generate a 3D shape from one image and export GLB, OBJ, STL and 3MF files.
GLB and OBJ are intended for Blender; STL and 3MF use millimetre scale.

This adaptation keeps the shape-generation and lightweight vertex-colour
projection paths. The Hunyuan3D Paint texture pipeline is intentionally disabled
to keep ZeroGPU execution reliable.

## Licences and provider notice

This private Space is provided by its owner for personal testing. Tencent is not
affiliated with, associated with, sponsoring, or endorsing this Space.

- The original `animede/image-3d` code is licensed under the Polyform Small
  Business License 1.0.0. Its required notice is retained in `LICENSE`:
  `Required Notice: Copyright ゆずき (https://github.com/animede/image-3d)`.
- Tencent Hunyuan3D-2 is subject to the Tencent Hunyuan 3D 2.0 Community
  License Agreement. The agreement excludes the European Union, United Kingdom
  and South Korea and includes additional use, territory and scale restrictions.
  Read `LICENSES/HUNYUAN3D-LICENSE` and `LICENSES/HUNYUAN3D-NOTICE` before use.

Modified deployment files are `app.py`, this README and the dependency manifest.
The upstream source files copied into this Space are unmodified.
