void conv(float* in, float* weights, float* bias, float* out) {
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  kernel_h = $kernel_size;
  kernel_w = $kernel_size;
  int pad_h, pad_w;
  pad_h = $pad;
  pad_w = $pad;
  int stride_h, stride_w;
  stride_h = $stride;
  stride_w = $stride;
  // Groups
  int groups = $group;
  int o_g = $out_c / groups;
  int k_g = $in_c / groups;
  int o_head, k_head;
  // Convolution
  #pragma omp parallel for
  for (int n = 0; n < $out_num; n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int y = 0; y < $out_h; y++) {
            for (int x = 0; x < $out_w; x++) {
              for (int p = 0; p < kernel_h; p++) {
                for (int q = 0; q < kernel_w; q++) {
                  int in_y = y * stride_h - pad_h + p;
                  int in_x = x * stride_w - pad_w + q;
                  if (in_y >= 0 && in_y < $in_h
                      && in_x >= 0 && in_x < $in_w) {
                    int out_idx = n * $out_c * $out_h * $out_w + \
                      (o + o_head) * $out_h * $out_w + y * $out_w + x;
                    int in_idx = n * $in_c * $in_h * $in_w + \
                      (k + k_head) * $in_h * $in_w + in_y * $in_w + in_x;
                    int weight_idx = (o + o_head) * $weight_c * $weight_h * $weight_w + \
                      k * $weight_h * $weight_w + p * $weight_w + q;
                    out[out_idx] += in[in_idx] * weights[weight_idx];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if ($bias_term) {
    #pragma omp parallel for
    for (int n = 0; n < $out_num; n++) {
      for (int o = 0; o < $out_c; o++) {
        for (int y = 0; y < $out_h; y++) {
          for (int x = 0; x < $out_w; x++) {
            out[n * $out_c * $out_h * $out_w + o *
                $out_h * $out_w + y * $out_w + x] += bias[o];
          }
        }
      }
    }
  }
}
