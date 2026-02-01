#ifndef ZFF_H
#define ZFF_H

typedef struct {
  double int1;
  double int2;
  float *trend_buffer;
  int trend_buf_size;
  int trend_write_pos;
  float trend_accum;
} ZFF;

#include <stddef.h>

void zff_init(ZFF *z, int sample_rate, float trend_window_ms,
              void *(*custom_alloc)(size_t));
void zff_process(ZFF *z, float in, float *zff_out, float *slope_out);
void zff_destroy(ZFF *z, void (*custom_free)(void *));

#endif
