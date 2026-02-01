#ifndef ENVELOPE_H
#define ENVELOPE_H

typedef struct {
  float output;
  float attack_coeff;
  float release_coeff;
} EnvelopeFollower;

void envelope_init(EnvelopeFollower *e, float sample_rate, float attack_ms,
                   float release_ms);
float envelope_process(EnvelopeFollower *e, float in);

#endif
