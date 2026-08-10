/*
 * Regression test for forward/backward Gaussian density scaling.
 *
 * A log-domain scaling offset can legitimately be below log(DBL_MIN).
 * The backward pass must reuse that offset exactly: clamping it changes
 * the density scale and can underflow beta after repeated frames.
 */
#include <float.h>
#include <math.h>
#include <stdio.h>

#include <s3/gauden.h>
#include <s3/s3.h>
#include <sphinxbase/ckd_alloc.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,          \
                    __LINE__);                                              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int
main(void)
{
    gauden_t g = {0};
    float64 ***den =
        (float64 ***)ckd_calloc_3d(1, 1, 1, sizeof(float64));
    uint32 ***den_idx =
        (uint32 ***)ckd_calloc_3d(1, 1, 1, sizeof(uint32));
    uint32 cb[1] = {0};
    float64 forward_density;
    float64 *scale;

    g.n_feat = 1;
    g.n_density = 1;
    g.n_top = 1;
    den[0][0][0] = -800.0;

    scale = gauden_scale_densities_fwd(den, den_idx, cb, 1, &g);
    CHECK(scale != NULL, "forward density scaling");
    CHECK(scale[0] < log(DBL_MIN), "forward scale must cross DBL_MIN");
    forward_density = den[0][0][0];

    /* Backward recomputes the log density before applying forward's offset. */
    den[0][0][0] = -800.0;
    CHECK(gauden_scale_densities_bwd(den, den_idx, &scale, cb, 1, &g) ==
              S3_SUCCESS,
          "backward density scaling");
    CHECK(fabs(den[0][0][0] - forward_density) < 1e-9,
          "forward and backward scaled densities must agree");

    ckd_free(scale);
    ckd_free_3d((void ***)den);
    ckd_free_3d((void ***)den_idx);
    printf("PASS: log-domain density scale preserved\n");
    return 0;
}
