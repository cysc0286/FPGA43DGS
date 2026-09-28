# Upstream reconstruction -> 3DGS Renderer v1

## Model

Binary little-endian PLY with 1..1,000,000 vertices and exactly 62 float32 properties in this order:

`x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3`.

`opacity` is stored before sigmoid, `scale` before exp, rotation is a w/x/y/z quaternion normalized by the front end. `f_dc/f_rest` use GraphDECO degree-3 SH layout: DC per RGB, then 15 remaining coefficients for each color channel. Normals are accepted but not used. This is not a generic XYZ/RGB point-cloud reader.

An upstream SH0 model can be adapted by zero-padding the 45 higher-order SH coefficients; this maintains the v1 format but does not compress its storage or skip v1 SH3 computations. Efficient low-order storage needs a separately versioned front end.

## Camera

Binary `FLCAM001`: 8-byte magic, four little-endian uint32 `(output_width, output_height, source_width, source_height)`, fourteen float64 `(camera_to_world_rotation[3][3], position[3], fx, fy)`. Total 136 bytes. The rotation is row-major in the file. Intrinsics refer to the source resolution and are scaled for output. Use the GraphDECO/COLMAP OpenCV-style local axes (x right, y down, z forward); use the supplied camera examples to validate conversions.

The renderer assumes the principal point at the image center under the existing pixel convention. Camera lens distortion must be removed upstream. An arbitrary non-centered principal point or distorted video frame requires an adapter or a new camera ABI; do not silently discard those parameters.

A target view requires position AND orientation AND intrinsics in the reconstruction's coordinate system. A position alone does not specify where the camera looks. Monocular reconstruction scale/origin must be aligned if a location is expressed in real-world meters. Rendering unobserved surfaces is not guaranteed to recover their true appearance.

The parser accepts dimensions up to 2048, but full validation is only recorded at 320×178; the parser limit is not a memory/performance guarantee.

## Output

`frame.bin`: magic `GSSOUT01`, uint32 width/height, then row-major pixels with float32 R/G/B/T and uint32 last contributor (20 bytes/pixel). RGB are linear/clamped to [0,1] only when writing the PPM preview; T is remaining transmittance. `frame.ppm` is an 8-bit RGB visualization. The internal `scene.bin` and Tile data are implementation details; future reconstruction should connect through model + camera, rather than create Gaussian-pixel commands.
