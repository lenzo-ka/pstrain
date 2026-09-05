/* ====================================================================
 * Copyright (c) 1999-2016 Carnegie Mellon University.  All rights
 * reserved.
 *
 * This file is derived from CMU SphinxTrain/SphinxBase sources.
 * Modifications for pstrain are Copyright (c) 2026 Kevin Lenzo and are
 * licensed under the BSD 2-Clause license (see LICENSE at the repository
 * root). The repository history records the modifications.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * This work was supported in part by funding from the Defense Advanced
 * Research Projects Agency and the National Science Foundation of the
 * United States of America, and the CMU Sphinx Speech Consortium.
 *
 * THIS SOFTWARE IS PROVIDED BY CARNEGIE MELLON UNIVERSITY ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CARNEGIE MELLON UNIVERSITY
 * NOR ITS EMPLOYEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * ====================================================================
 */
/**
 * @file pstrain_flat.c
 * @brief Simplified flat initialization API for CFFI
 *
 * Wraps SphinxTrain's mk_flat and init_gau functionality.
 */

#include "pstrain_flat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys_compat/file.h>

#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/cmd_ln.h>
#include <sphinxbase/err.h>
#include <sphinxbase/feat.h>
#include <sphinxbase/cmn.h>
#include <sphinxbase/agc.h>

#include <s3/corpus.h>
#include <s3/lexicon.h>
#include <s3/model_def_io.h>
#include <s3/s3mixw_io.h>
#include <s3/s3tmat_io.h>
#include <s3/s3gau_io.h>
#include <s3/s3ts2cb_io.h>
#include <s3/ts2cb.h>
#include <s3/topo_read.h>
#include <s3/gauden.h>
#include <s3/s3acc_io.h>
#include <s3/s3.h>

/* Implementation detail of the public pstrain_init_gau wrapper. */
static int pstrain_init_gau_core(lexicon_t *lex, model_def_t *mdef, feat_t *feat,
                                 const char *accumdir, const char *meanfn);

int
pstrain_flat_tmat(const char *mdef_path,
              const char *topo_path,
              const char *output_path)
{
    model_def_t *mdef = NULL;
    float32 **proto_tmat = NULL;
    float32 ***tmat = NULL;
    uint32 n_tmat;
    uint32 n_state_pm;
    uint32 i, j, k;
    int ret = 0;

    if (!mdef_path || !topo_path || !output_path) {
        E_ERROR("Invalid arguments to pstrain_flat_tmat\n");
        return -1;
    }

    /* Read model definition */
    E_INFO("Reading model definition from %s\n", mdef_path);
    if (model_def_read(&mdef, mdef_path) != S3_SUCCESS) {
        E_ERROR("Failed to read model definition\n");
        return -1;
    }

    /* Read topology file */
    E_INFO("Reading topology from %s\n", topo_path);
    if (topo_read(&proto_tmat, &n_state_pm, topo_path) != S3_SUCCESS) {
        E_ERROR("Failed to read topology: %s\n", topo_path);
        model_def_free(mdef);
        return -1;
    }

    /* Create transition matrices */
    n_tmat = mdef->n_tied_tmat;
    tmat = (float32 ***)ckd_calloc_3d(n_tmat, n_state_pm - 1, n_state_pm,
                                       sizeof(float32));

    for (k = 0; k < n_tmat; k++) {
        for (i = 0; i < n_state_pm - 1; i++) {
            for (j = 0; j < n_state_pm; j++) {
                tmat[k][i][j] = proto_tmat[i][j];
            }
        }
    }

    /* Write transition matrices */
    E_INFO("Writing transition matrices to %s\n", output_path);
    if (s3tmat_write(output_path, tmat, n_tmat, n_state_pm) != S3_SUCCESS) {
        E_ERROR("Failed to write transition matrices\n");
        ret = -1;
    }

    ckd_free_3d((void ***)tmat);
    ckd_free_2d((void **)proto_tmat);
    model_def_free(mdef);

    return ret;
}

int
pstrain_flat_mixw(uint32 n_tied_state,
              uint32 n_stream,
              uint32 n_density,
              const char *output_path)
{
    float32 ***mixw;
    float32 mixw_ini;
    uint32 i, j, k;

    if (n_tied_state == 0 || n_stream == 0 || n_density == 0 || !output_path) {
        E_ERROR("Invalid arguments to pstrain_flat_mixw\n");
        return -1;
    }

    /* Allocate mixture weights */
    mixw = (float32 ***)ckd_calloc_3d(n_tied_state, n_stream, n_density,
                                       sizeof(float32));

    /* Initialize uniformly */
    mixw_ini = 1.0f / (float32)n_density;

    for (i = 0; i < n_tied_state; i++) {
        for (j = 0; j < n_stream; j++) {
            for (k = 0; k < n_density; k++) {
                mixw[i][j][k] = mixw_ini;
            }
        }
    }

    /* Write mixture weights */
    E_INFO("Writing mixture weights to %s [%ux%ux%u]\n",
           output_path, n_tied_state, n_stream, n_density);
    if (s3mixw_write(output_path, mixw, n_tied_state, n_stream, n_density) != S3_SUCCESS) {
        E_ERROR("Failed to write mixture weights\n");
        ckd_free_3d((void ***)mixw);
        return -1;
    }

    ckd_free_3d((void ***)mixw);
    return 0;
}

/**
 * Initialize Gaussian parameters from features.
 *
 * This wraps the SphinxTrain init_gau functionality. It computes
 * means (first pass) or variances (second pass, when mean_path provided)
 * from feature data.
 *
 * Global mode (mdef_path=NULL): computes single global mean/var
 * Per-state mode: requires segmentation files
 */
int
pstrain_init_gau(const char *mdef_path,
             const char *dict_path,
             const char *filler_dict_path,
             const char *feat_type,
             int32 ceplen,
             const char *ctl_path,
             const char *cep_dir,
             const char *cep_ext,
             const char *lsn_path,
             const char *seg_dir,
             const char *seg_ext,
             const char *accum_dir,
             const char *mean_path)
{
    model_def_t *mdef = NULL;
    lexicon_t *lex = NULL;
    feat_t *feat = NULL;
    int ret = -1;

    if (!ctl_path || !cep_dir || !accum_dir) {
        E_ERROR("Invalid arguments to pstrain_init_gau\n");
        return -1;
    }

    /* Initialize feature extraction - use CMN_BATCH to match BW training */
    feat = feat_init(feat_type ? feat_type : "1s_c_d_dd",
                     CMN_BATCH,
                     FALSE,
                     AGC_NONE,
                     1, ceplen);
    if (!feat) {
        E_ERROR("Failed to initialize feature extraction\n");
        return -1;
    }

    /* Set up corpus */
    corpus_set_mfcc_dir(cep_dir);
    corpus_set_mfcc_ext(cep_ext ? cep_ext : ".mfc");
    corpus_set_ctl_filename(ctl_path);

    if (lsn_path)
        corpus_set_lsn_filename(lsn_path);
    if (seg_dir)
        corpus_set_seg_dir(seg_dir);
    if (seg_ext)
        corpus_set_seg_ext(seg_ext);

    if (corpus_init() != S3_SUCCESS) {
        E_ERROR("Failed to initialize corpus\n");
        goto cleanup;
    }

    /* Read model definition if provided */
    if (mdef_path) {
        E_INFO("Reading model definition from %s\n", mdef_path);
        if (model_def_read(&mdef, mdef_path) != S3_SUCCESS) {
            E_ERROR("Failed to read model definition\n");
            goto cleanup;
        }

        /* Set up continuous tied-state to codebook mapping */
        mdef->cb = cont_ts2cb(mdef->n_tied_state);

        /* Read dictionary */
        if (dict_path) {
            E_INFO("Reading dictionary from %s\n", dict_path);
            lex = lexicon_read(NULL, dict_path, mdef->acmod_set);
            if (!lex) {
                E_ERROR("Failed to read dictionary\n");
                goto cleanup;
            }

            if (filler_dict_path) {
                E_INFO("Reading filler dictionary from %s\n", filler_dict_path);
                lexicon_read(lex, filler_dict_path, mdef->acmod_set);
            }
        }
    }

    /* Run init_gau core */
    ret = pstrain_init_gau_core(lex, mdef, feat, accum_dir, mean_path);

cleanup:
    if (lex)
        lexicon_free(lex);
    if (mdef)
        model_def_free(mdef);
    if (feat)
        feat_free(feat);

    return ret;
}

/**
 * Core init_gau computation - accumulates mean or variance statistics.
 *
 * This is adapted from SphinxTrain's init_gau.c
 */
static int
pstrain_init_gau_core(lexicon_t *lex, model_def_t *mdef, feat_t *feat,
                  const char *accumdir, const char *meanfn)
{
    char *trans = NULL;
    char *fn;
    acmod_set_t *acmod_set;

    vector_t *mfcc = NULL;
    int32 n_frame;
    int32 feat_n_frame;
    int32 tmp;

    uint16 *seg = NULL;
    uint32 *sseq = NULL;
    uint32 *ci_sseq = NULL;

    uint32 attempted_cnt = 0;
    uint32 processed_cnt = 0;
    uint32 skip_transcript = 0;
    uint32 skip_segmentation = 0;
    uint32 skip_feature = 0;
    uint32 skip_frame_mismatch = 0;
    uint32 skip_too_short = 0;
    uint32 skip_feature_frames = 0;

    char **word = NULL;
    uint32 n_word;

    acmod_id_t *phone = NULL;
    uint32 n_phone;
    char *btw_mark = NULL;

    vector_t ***mean_acc = NULL;
    vector_t ***mean = NULL;
    vector_t ***var_acc = NULL;
    float32 ***dnom = NULL;

    const uint32 *veclen;

    uint32 n_ts;

    uint32 *r_veclen;
    uint32 r_n_ts;
    uint32 r_n_feat;
    uint32 r_n_density;

    mfcc_t ***f = NULL;

    uint32 ceplen = 13; /* Default */

    if (mdef) {
        acmod_set = mdef->acmod_set;
        n_ts = mdef->n_tied_state;
    }
    else {
        acmod_set = NULL;
        n_ts = 1;  /* Global mean/var */
    }

    veclen = (uint32 *)feat_stream_lengths(feat);

    if (meanfn == NULL) {
        E_INFO("Computing %ux%ux1 mean estimates\n", n_ts, feat_dimension1(feat));

        mean_acc = gauden_alloc_param(n_ts,
                                      feat_dimension1(feat),
                                      1,
                                      veclen);
        var_acc = NULL;
    }
    else {
        E_INFO("Computing %ux%ux1 variance estimates\n", n_ts, feat_dimension1(feat));

        if (s3gau_read(meanfn,
                       &mean,
                       &r_n_ts,
                       &r_n_feat,
                       &r_n_density,
                       &r_veclen) != S3_SUCCESS) {
            E_ERROR("Unable to read means from %s\n", meanfn);
            return -1;
        }
        ckd_free(r_veclen);

        mean_acc = NULL;
        var_acc = gauden_alloc_param(n_ts,
                                     feat_dimension1(feat),
                                     1,
                                     veclen);
    }

    dnom = (float32 ***)ckd_calloc_3d(n_ts, feat_dimension1(feat), 1, sizeof(float32));

    while (corpus_next_utt()) {
        if (mfcc) {
            free(mfcc[0]);
            ckd_free(mfcc);
            mfcc = NULL;
        }
        if (trans) {
            free(trans);
            trans = NULL;
        }
        if (seg) {
            free(seg);
            seg = NULL;
        }
        if (word) {
            ckd_free(word);
            word = NULL;
        }
        if (phone) {
            ckd_free(phone);
            phone = NULL;
        }
        if (btw_mark) {
            ckd_free(btw_mark);
            btw_mark = NULL;
        }
        if (ci_sseq) {
            ckd_free(ci_sseq);
            ci_sseq = NULL;
        }
        if (sseq) {
            ckd_free(sseq);
            sseq = NULL;
        }
        if (f) {
            feat_array_free(f);
            f = NULL;
        }

        if ((++attempted_cnt % 100) == 0) {
            E_INFO("[%u] utterances attempted\n", attempted_cnt);
        }

        /* For global mode (mdef=NULL), we don't need transcripts/segmentation */
        if (mdef) {
            if (corpus_get_sent(&trans) != S3_SUCCESS) {
                E_WARN("Unable to read transcript for %s, skipping\n",
                       corpus_utt_brief_name());
                ++skip_transcript;
                continue;
            }

            if (corpus_get_seg(&seg, &n_frame) != S3_SUCCESS) {
                E_WARN("Unable to read segmentation for %s, skipping\n",
                       corpus_utt_brief_name());
                ++skip_segmentation;
                continue;
            }
        }

        if (sys_compat_access(corpus_mfcc_filename(), R_OK) != 0 ||
            corpus_get_generic_featurevec(&mfcc, &tmp, ceplen) < 0) {
            E_WARN("Can't read features for %s, skipping\n",
                   corpus_utt_brief_name());
            ++skip_feature;
            continue;
        }

        if (mdef == NULL) n_frame = tmp;

        if (mdef && tmp != n_frame) {
            E_WARN("Frame count mismatch for %s, skipping\n",
                   corpus_utt_brief_name());
            ++skip_frame_mismatch;
            continue;
        }

        feat_n_frame = n_frame;

        if (n_frame < 9) {
            E_WARN("Utterance %s too short (%d frames), skipping\n",
                   corpus_utt_brief_name(), n_frame);
            ++skip_too_short;
            continue;
        }

        f = feat_array_alloc(feat, feat_n_frame + feat_window_size(feat));
        feat_s2mfc2feat_live(feat, mfcc, &feat_n_frame, TRUE, TRUE, f);

        if (feat_n_frame != n_frame) {
            E_WARN("Feature frame count changed for %s, skipping\n",
                   corpus_utt_brief_name());
            feat_array_free(f);
            f = NULL;
            ++skip_feature_frames;
            continue;
        }

        /* Accumulate statistics */
        if (mean_acc) {
            /* Accumulate mean sums (no segmentation needed for global mode) */
            uint32 t, s, ff, c;
            for (t = 0; t < (uint32)n_frame; t++) {
                s = (sseq) ? sseq[t] : 0;
                for (ff = 0; ff < feat_dimension1(feat); ff++) {
                    dnom[s][ff][0] += 1.0;
                    for (c = 0; c < veclen[ff]; c++) {
                        mean_acc[s][ff][0][c] += f[t][ff][c];
                    }
                }
            }
        }
        else if (var_acc) {
            /* Accumulate variance sums */
            uint32 t, s, ff, c;
            float32 diff;
            for (t = 0; t < (uint32)n_frame; t++) {
                s = (sseq) ? sseq[t] : 0;
                for (ff = 0; ff < feat_dimension1(feat); ff++) {
                    dnom[s][ff][0] += 1.0;
                    for (c = 0; c < veclen[ff]; c++) {
                        diff = f[t][ff][c] - mean[s][ff][0][c];
                        var_acc[s][ff][0][c] += diff * diff;
                    }
                }
            }
        }
        ++processed_cnt;
    }

    E_INFO("flat_init: processed %u, skipped %u (transcript read: %u, "
           "segmentation read: %u, feature read: %u, frame mismatch: %u, "
           "too short: %u, feature frame change: %u)\n",
           processed_cnt, attempted_cnt - processed_cnt, skip_transcript,
           skip_segmentation, skip_feature, skip_frame_mismatch, skip_too_short,
           skip_feature_frames);

    /* Write accumulator counts */
    fn = ckd_calloc(strlen(accumdir) + strlen("/gauden_counts") + 1, 1);
    sprintf(fn, "%s/gauden_counts", accumdir);

    if (s3gaucnt_write(fn, mean_acc, var_acc, (var_acc != NULL), dnom,
                       n_ts, feat_dimension1(feat), 1, veclen) != 0) {
        E_ERROR("Failed to write Gaussian counts to %s\n", fn);
        ckd_free(fn);
        return -1;
    }

    E_INFO("Wrote Gaussian counts to %s\n", fn);
    ckd_free(fn);

    /* Cleanup */
    if (mfcc) {
        free(mfcc[0]);
        ckd_free(mfcc);
    }
    if (trans) free(trans);
    if (seg) free(seg);
    if (word) ckd_free(word);
    if (phone) ckd_free(phone);
    if (btw_mark) ckd_free(btw_mark);
    if (ci_sseq) ckd_free(ci_sseq);
    if (sseq) ckd_free(sseq);
    if (f) feat_array_free(f);

    if (mean_acc) gauden_free_param(mean_acc);
    if (var_acc) gauden_free_param(var_acc);
    if (mean) gauden_free_param(mean);
    ckd_free_3d((void ***)dnom);

    return attempted_cnt - processed_cnt;
}

/**
 * Normalize accumulated counts to get model parameters.
 *
 * Adapted from SphinxTrain's norm program.
 */
int
pstrain_norm_gau(const char *accum_dir,
             const char *mean_path,
             const char *var_path)
{
    char file_name[4096];
    vector_t ***wt_mean = NULL;
    vector_t ***wt_var = NULL;
    int32 pass2var = FALSE;
    float32 ***dnom = NULL;
    uint32 n_mgau;
    uint32 n_stream;
    uint32 n_density;
    uint32 *veclen = NULL;

    if (!accum_dir) {
        E_ERROR("Invalid arguments to pstrain_norm_gau\n");
        return -1;
    }

    /* Read accumulated counts */
    E_INFO("Reading Gaussian counts from %s\n", accum_dir);
    snprintf(file_name, sizeof(file_name), "%s/gauden_counts", accum_dir);

    if (rdacc_den(accum_dir,
                  &wt_mean,
                  &wt_var,
                  &pass2var,
                  &dnom,
                  &n_mgau,
                  &n_stream,
                  &n_density,
                  &veclen) != S3_SUCCESS) {
        E_ERROR("Failed to read Gaussian counts from %s\n", accum_dir);
        return -1;
    }

    E_INFO("Normalizing for n_mgau=%u, n_stream=%u, n_density=%u\n",
           n_mgau, n_stream, n_density);

    /* Normalize means */
    if (mean_path && wt_mean) {
        gauden_norm_wt_mean(NULL, wt_mean, dnom,
                           n_mgau, n_stream, n_density, veclen);

        E_INFO("Writing means to %s\n", mean_path);
        if (s3gau_write(mean_path,
                        (const vector_t ***)wt_mean,
                        n_mgau,
                        n_stream,
                        n_density,
                        veclen) != S3_SUCCESS) {
            E_ERROR("Failed to write means to %s\n", mean_path);
            return -1;
        }
    }

    /* Normalize variances (if pass2var is set) */
    if (var_path && wt_var && pass2var) {
        gauden_norm_wt_var(NULL, wt_var, pass2var, dnom,
                          wt_mean,  /* wt_mean is now just mean */
                          n_mgau, n_stream, n_density, veclen,
                          FALSE);  /* not tied var */

        E_INFO("Writing variances to %s\n", var_path);
        if (s3gau_write(var_path,
                        (const vector_t ***)wt_var,
                        n_mgau,
                        n_stream,
                        n_density,
                        veclen) != S3_SUCCESS) {
            E_ERROR("Failed to write variances to %s\n", var_path);
            return -1;
        }
    }

    /* Cleanup */
    if (wt_mean) gauden_free_param(wt_mean);
    if (wt_var) gauden_free_param(wt_var);
    if (dnom) ckd_free_3d((void ***)dnom);
    if (veclen) ckd_free(veclen);

    return 0;
}
