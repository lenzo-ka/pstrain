/*
 * Smoke test for libpstrainc model-file I/O (libio).
 *
 * Round-trips mixture weights, Gaussian parameters, and transition
 * matrices through the S3 binary format and asserts the data survives
 * unchanged. This is the minimal "does libpstrainc actually link and work"
 * check wired into ctest, so the C-tests CI job asserts something real
 * instead of passing on an empty test set.
 *
 * Exercises the same read/write/alloc/free paths that pstrain/lib/_cffi/io.py
 * drives from Python.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <s3/s3.h>
#include <s3/s3gau_io.h>
#include <s3/s3mixw_io.h>
#include <s3/s3tmat_io.h>
#include <s3/vector.h>
#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/prim_type.h>

#define EPS 1e-6f

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,          \
                    __LINE__);                                              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static float32
expect_mixw(uint32 i, uint32 j, uint32 k)
{
    return (float32)(i * 100 + j * 10 + k) + 0.5f;
}

static int
test_mixw(const char *fn)
{
    const uint32 n_mixw = 3, n_feat = 1, n_density = 4;
    uint32 i, j, k;

    float32 ***w =
        (float32 ***)ckd_calloc_3d(n_mixw, n_feat, n_density, sizeof(float32));
    for (i = 0; i < n_mixw; i++)
        for (j = 0; j < n_feat; j++)
            for (k = 0; k < n_density; k++)
                w[i][j][k] = expect_mixw(i, j, k);

    CHECK(s3mixw_write(fn, w, n_mixw, n_feat, n_density) == S3_SUCCESS,
          "s3mixw_write");
    ckd_free_3d((void *)w);

    float32 ***r = NULL;
    uint32 rn_mixw = 0, rn_feat = 0, rn_density = 0;
    CHECK(s3mixw_read(fn, &r, &rn_mixw, &rn_feat, &rn_density) == S3_SUCCESS,
          "s3mixw_read");
    CHECK(rn_mixw == n_mixw && rn_feat == n_feat && rn_density == n_density,
          "mixw dimensions");
    for (i = 0; i < n_mixw; i++)
        for (j = 0; j < n_feat; j++)
            for (k = 0; k < n_density; k++)
                CHECK(fabsf(r[i][j][k] - expect_mixw(i, j, k)) < EPS,
                      "mixw value");
    ckd_free_3d((void *)r);
    return 0;
}

static float32
expect_gau(uint32 m, uint32 d, uint32 v)
{
    return (float32)(m * 1000 + d * 10 + v) - 3.25f;
}

static int
test_gau(const char *fn)
{
    const uint32 n_mgau = 2, n_feat = 1, n_density = 3, veclen = 4;
    uint32 vl[1] = {veclen};
    uint32 m, f, d, v;

    /* s3gau_write serializes out[0][0][0] as one contiguous block of
     * n_mgau*n_feat*n_density*veclen floats, so the Gaussian data MUST be
     * allocated contiguously (the Python path passes a single numpy array).
     * Allocate the pointer structure and one data block, then point each
     * cell into it. */
    vector_t ***g =
        (vector_t ***)ckd_calloc_3d(n_mgau, n_feat, n_density, sizeof(vector_t));
    float32 *data =
        (float32 *)ckd_calloc((size_t)n_mgau * n_feat * n_density * veclen,
                              sizeof(float32));
    size_t off = 0;
    for (m = 0; m < n_mgau; m++)
        for (f = 0; f < n_feat; f++)
            for (d = 0; d < n_density; d++) {
                g[m][f][d] = data + off;
                off += veclen;
                for (v = 0; v < veclen; v++)
                    g[m][f][d][v] = expect_gau(m, d, v);
            }

    CHECK(s3gau_write(fn, (const vector_t ***)g, n_mgau, n_feat, n_density, vl) ==
              S3_SUCCESS,
          "s3gau_write");
    ckd_free((void *)data);
    ckd_free_3d((void *)g);

    vector_t ***rg = NULL;
    uint32 rn_mgau = 0, rn_feat = 0, rn_density = 0;
    uint32 *rvl = NULL;
    CHECK(s3gau_read(fn, &rg, &rn_mgau, &rn_feat, &rn_density, &rvl) ==
              S3_SUCCESS,
          "s3gau_read");
    CHECK(rn_mgau == n_mgau && rn_feat == n_feat && rn_density == n_density,
          "gau dimensions");
    CHECK(rvl[0] == veclen, "gau veclen");
    for (m = 0; m < n_mgau; m++)
        for (f = 0; f < n_feat; f++)
            for (d = 0; d < n_density; d++)
                for (v = 0; v < veclen; v++)
                    CHECK(fabsf(rg[m][f][d][v] - expect_gau(m, d, v)) < EPS,
                          "gau value");
    /* Read allocates the data as one contiguous block anchored at
     * rg[0][0][0]; free it and the pointer structure separately. */
    ckd_free((void *)rg[0][0][0]);
    ckd_free_3d((void *)rg);
    ckd_free((void *)rvl);
    return 0;
}

static float32
expect_tmat(uint32 t, uint32 i, uint32 j)
{
    return (float32)(t * 100 + i * 10 + j) + 0.125f;
}

static int
test_tmat(const char *fn)
{
    const uint32 n_tmat = 2, n_state = 4; /* n_state includes exit state */
    const uint32 n_rows = n_state - 1;
    uint32 t, i, j;

    float32 ***tm =
        (float32 ***)ckd_calloc_3d(n_tmat, n_rows, n_state, sizeof(float32));
    for (t = 0; t < n_tmat; t++)
        for (i = 0; i < n_rows; i++)
            for (j = 0; j < n_state; j++)
                tm[t][i][j] = expect_tmat(t, i, j);

    CHECK(s3tmat_write(fn, tm, n_tmat, n_state) == S3_SUCCESS, "s3tmat_write");
    ckd_free_3d((void *)tm);

    float32 ***rt = NULL;
    uint32 rn_tmat = 0, rn_state = 0;
    CHECK(s3tmat_read(fn, &rt, &rn_tmat, &rn_state) == S3_SUCCESS,
          "s3tmat_read");
    CHECK(rn_tmat == n_tmat && rn_state == n_state, "tmat dimensions");
    for (t = 0; t < n_tmat; t++)
        for (i = 0; i < n_rows; i++)
            for (j = 0; j < n_state; j++)
                CHECK(fabsf(rt[t][i][j] - expect_tmat(t, i, j)) < EPS,
                      "tmat value");
    ckd_free_3d((void *)rt);
    return 0;
}

int
main(int argc, char *argv[])
{
    const char *dir = (argc > 1) ? argv[1] : ".";
    char path[2048];

    snprintf(path, sizeof(path), "%s/roundtrip_mixw", dir);
    if (test_mixw(path))
        return 1;
    snprintf(path, sizeof(path), "%s/roundtrip_gau", dir);
    if (test_gau(path))
        return 1;
    snprintf(path, sizeof(path), "%s/roundtrip_tmat", dir);
    if (test_tmat(path))
        return 1;

    printf("PASS: libpstrainc I/O round-trip (mixw, gau, tmat)\n");
    return 0;
}
