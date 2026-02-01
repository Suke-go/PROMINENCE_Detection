#ifndef BIQUAD_H
#define BIQUAD_H

typedef struct {
  float b0, b1, b2, a1, a2;
  float x1, x2, y1, y2;
} Biquad;

void biquad_reset(Biquad *f);
void biquad_config_bandpass(Biquad *f, float sample_rate, float center_freq,
                            float q_factor);
float biquad_process(Biquad *f, float in);

#endif
