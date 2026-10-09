#pragma once
#include "../hls/renderer.hpp"
// One sorted Gaussian for one 8x8 sub-tile. q[143:0] retains the renderer
// features. Origins are uint16 at 144/160, valid width/height at 176/181,
// spiky classification at 186, and original ordinal at 192. Output adds
// the four mini-tile bits at 224. This module does not claim preprocessing.
void flicker_ctu(hls::stream<Word>& input,hls::stream<Word>& output,
                 unsigned count,unsigned mode);
