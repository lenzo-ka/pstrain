#include "pstrain_delint.h"

#include <s3/common.h>
#include <s3/model_def_io.h>
#include <s3/s3mixw_io.h>

#include <sphinxbase/matrix.h>
#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/err.h>

#include <sys_compat/file.h>

#include <stdio.h>
#include <string.h>

/* Forward declaration - from delint/main.c */
extern int32 smooth_mixw(float32 ****out_mixw,
                         float32 ***mixw_acc_a,
                         float32 ***mixw_acc_b,
                         uint32 n_mixw,
                         uint32 n_feat,
                         uint32 n_gau,
                         model_def_t *mdef,
                         float32 cilambda,
                         int32 maxiter);

static int
rd_param(uint32 *idx,
         const char **accumdirs,
         float32 ****out_mixw_acc,
         uint32 *out_n_mixw,
         uint32 *out_n_feat,
         uint32 *out_n_gau)
{
    char fn[MAXPATHLEN+1];
    const char *accum_dir;
    uint32 i;

    i = *idx;
    accum_dir = accumdirs[i];

    snprintf(fn, MAXPATHLEN, "%s/mixw_counts", accum_dir);

    E_INFO("Reading %s\n", fn);

    if (s3mixw_read(fn,
                    out_mixw_acc,
                    out_n_mixw,
                    out_n_feat,
                    out_n_gau) != S3_SUCCESS) {
        return S3_ERROR;
    }

    ++(*idx);

    return S3_SUCCESS;
}

int pstrain_delint(const char *moddeffn,
               const char *mixwfn,
               const char **accumdirs,
               float32 cilambda,
               int32 maxiter)
{
    model_def_t *mdef = NULL;
    float32 ***mixw_acc_in = NULL;
    float32 ***mixw_acc_a = NULL;
    float32 ***mixw_acc_b = NULL;
    float32 ***mixw = NULL;
    uint32 n_mixw, n_feat, n_gau;
    uint32 i;
    int ret = -1;

    if (moddeffn == NULL) {
        E_ERROR("Must specify model definition file\n");
        return -1;
    }

    if (mixwfn == NULL) {
        E_ERROR("Must specify output mixture weight file\n");
        return -1;
    }

    if (accumdirs == NULL || accumdirs[0] == NULL || accumdirs[1] == NULL) {
        E_ERROR("Must specify at least 2 accumulator directories\n");
        return -1;
    }

    /* Read model definition */
    if (model_def_read(&mdef, moddeffn) != S3_SUCCESS) {
        E_ERROR("Failed to read model definition from %s\n", moddeffn);
        return -1;
    }

    /* Read first two accumulator directories */
    i = 0;
    if (rd_param(&i, accumdirs, &mixw_acc_a, &n_mixw, &n_feat, &n_gau) != S3_SUCCESS) {
        E_ERROR("Failed to read first accumulator directory\n");
        goto cleanup;
    }

    if (rd_param(&i, accumdirs, &mixw_acc_b, &n_mixw, &n_feat, &n_gau) != S3_SUCCESS) {
        E_ERROR("Failed to read second accumulator directory\n");
        goto cleanup;
    }

    /* Read additional directories (must be even number) */
    while (accumdirs[i] != NULL) {
        if (rd_param(&i, accumdirs, &mixw_acc_in, &n_mixw, &n_feat, &n_gau) != S3_SUCCESS) {
            E_ERROR("Failed to read accumulator directory %d\n", i);
            goto cleanup;
        }

        /* Accumulate into A buffer */
        accum_3d(mixw_acc_a, mixw_acc_in, n_mixw, n_feat, n_gau);
        ckd_free_3d((void ***)mixw_acc_in);
        mixw_acc_in = NULL;

        /* Must have even number */
        if (accumdirs[i] == NULL) {
            E_ERROR("Must specify even number of accumulator directories\n");
            goto cleanup;
        }

        if (rd_param(&i, accumdirs, &mixw_acc_in, &n_mixw, &n_feat, &n_gau) != S3_SUCCESS) {
            E_ERROR("Failed to read accumulator directory %d\n", i);
            goto cleanup;
        }

        /* Accumulate into B buffer */
        accum_3d(mixw_acc_b, mixw_acc_in, n_mixw, n_feat, n_gau);
        ckd_free_3d((void ***)mixw_acc_in);
        mixw_acc_in = NULL;
    }

    /* Run deleted interpolation */
    if (smooth_mixw(&mixw,
                    mixw_acc_a, mixw_acc_b,
                    n_mixw, n_feat, n_gau,
                    mdef, cilambda, maxiter) != S3_SUCCESS) {
        E_ERROR("Deleted interpolation failed\n");
        goto cleanup;
    }

    /* Write output */
    E_INFO("Writing %s\n", mixwfn);
    if (s3mixw_write(mixwfn, mixw, n_mixw, n_feat, n_gau) != S3_SUCCESS) {
        E_ERROR("Failed to write mixture weights to %s\n", mixwfn);
        goto cleanup;
    }

    ret = 0;

cleanup:
    if (mdef != NULL)
        model_def_free(mdef);
    if (mixw_acc_in != NULL)
        ckd_free_3d((void ***)mixw_acc_in);
    /* Note: mixw_acc_a and mixw_acc_b are freed by smooth_mixw */

    return ret;
}
