#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <s3/gauden.h>
#include <s3/s3gau_io.h>
#include <s3/s3mixw_io.h>
#include <sphinxbase/ckd_alloc.h>

#include "../libs/libpstrain/pstrain_inc_comp.h"
#include "../libs/libpstrain/pstrain_kdtree.h"

#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", m); return 1; } } while (0)

static void
path(char *out, size_t n, const char *dir, const char *name)
{
    snprintf(out, n, "%s/%s", dir, name);
}

static int
kd_file_is_finite(const char *fn)
{
    FILE *fh = fopen(fn, "r");
    char token[256];
    if (fh == NULL)
        return 0;
    while (fscanf(fh, "%255s", token) == 1) {
        char *end;
        double value = strtod(token, &end);
        if (end != token && *end == '\0' && !isfinite(value)) {
            fclose(fh);
            return 0;
        }
    }
    fclose(fh);
    return 1;
}

int
main(int argc, char *argv[])
{
    uint32 veclen[1] = { 1 };
    vector_t ***mean, ***var, ***readback;
    float32 ***mixw, ***dnom;
    uint32 nm, nf, nd, *rvl;
    char meanfn[1024], varfn[1024], mixwfn[1024], dnomfn[1024];
    char kdout[1024], outmean[1024], outvar[1024], outmixw[1024];
    uint32 i;

    CHECK(argc == 2, "temporary directory argument");
    path(meanfn, sizeof(meanfn), argv[1], "floor_mean");
    path(varfn, sizeof(varfn), argv[1], "floor_var");
    path(mixwfn, sizeof(mixwfn), argv[1], "floor_mixw");
    path(dnomfn, sizeof(dnomfn), argv[1], "floor_dnom");
    path(kdout, sizeof(kdout), argv[1], "floor_kdtree");
    path(outmean, sizeof(outmean), argv[1], "floor_split_mean");
    path(outvar, sizeof(outvar), argv[1], "floor_split_var");
    path(outmixw, sizeof(outmixw), argv[1], "floor_split_mixw");

    mean = gauden_alloc_param(1, 1, 2, veclen);
    var = gauden_alloc_param(1, 1, 2, veclen);
    mixw = (float32 ***)ckd_calloc_3d(1, 1, 2, sizeof(float32));
    dnom = (float32 ***)ckd_calloc_3d(1, 1, 2, sizeof(float32));
    mean[0][0][0][0] = 0.0f;
    mean[0][0][1][0] = 1.0f;
    var[0][0][0][0] = 0.0f;
    var[0][0][1][0] = -1.1920929e-7f;
    mixw[0][0][0] = 0.6f;
    mixw[0][0][1] = 0.4f;
    dnom[0][0][0] = 10.0f;
    dnom[0][0][1] = 5.0f;

    CHECK(s3gau_write(meanfn, (const vector_t ***)mean, 1, 1, 2, veclen) == S3_SUCCESS, "write means");
    CHECK(s3gau_write(varfn, (const vector_t ***)var, 1, 1, 2, veclen) == S3_SUCCESS, "write raw variances");
    CHECK(s3mixw_write(mixwfn, mixw, 1, 1, 2) == S3_SUCCESS, "write mixture weights");
    CHECK(s3gaudnom_write(dnomfn, dnom, 1, 1, 2) == S3_SUCCESS, "write density counts");

    CHECK(pstrain_kdtree_build(meanfn, varfn, kdout, 0.2f, 2, 0) == 0, "KD-tree build");
    CHECK(kd_file_is_finite(kdout), "KD-tree geometry is finite");

    CHECK(pstrain_inc_comp(meanfn, varfn, mixwfn, dnomfn,
                           outmean, outvar, outmixw, 2) == 0,
          "component split");
    CHECK(s3gau_read(outmean, &readback, &nm, &nf, &nd, &rvl) == S3_SUCCESS, "read split means");
    for (i = 0; i < nd; ++i)
        CHECK(isfinite(readback[0][0][i][0]), "split mean is finite");
    gauden_free_param(readback);
    ckd_free(rvl);

    CHECK(s3gau_read(outvar, &readback, &nm, &nf, &nd, &rvl) == S3_SUCCESS, "read split variances");
    for (i = 0; i < nd; ++i) {
        CHECK(isfinite(readback[0][0][i][0]), "split variance is finite");
        CHECK(readback[0][0][i][0] >= GAUDEN_EVAL_VAR_FLOOR, "split variance honors floor");
    }
    gauden_free_param(readback);
    ckd_free(rvl);

    CHECK(s3gau_read(varfn, &readback, &nm, &nf, &nd, &rvl) == S3_SUCCESS, "reread raw variances");
    CHECK(readback[0][0][0][0] == 0.0f && readback[0][0][1][0] < 0.0f,
          "serialized input remains lossless");

    gauden_free_param(readback);
    ckd_free(rvl);
    gauden_free_param(mean);
    gauden_free_param(var);
    ckd_free_3d((void ***)mixw);
    ckd_free_3d((void ***)dnom);
    return 0;
}
