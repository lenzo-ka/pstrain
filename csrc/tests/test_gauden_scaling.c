/*
 * Regression test for forward/backward Gaussian density scaling.
 *
 * A log-domain scaling offset can legitimately be below log(DBL_MIN).
 * The backward pass must reuse that offset exactly: clamping it changes
 * the density scale and can underflow beta after repeated frames.
 */
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
    float64 saved_scale[1] = {-810.0};
    float64 *scale = saved_scale;

    g.n_feat = 1;
    g.n_density = 1;
    g.n_top = 1;
    den[0][0][0] = -800.0;

    CHECK(gauden_scale_densities_bwd(den, den_idx, &scale, cb, 1, &g) ==
              S3_SUCCESS,
          "backward density scaling");
    CHECK(saved_scale[0] == -810.0, "forward scale must remain unchanged");
    CHECK(fabs(den[0][0][0] - exp(10.0)) < 1e-9,
          "backward density must use the forward scale");

    ckd_free_3d((void ***)den);
    ckd_free_3d((void ***)den_idx);
    printf("PASS: log-domain density scale preserved\n");
    return 0;
}
