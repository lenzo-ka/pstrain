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
 * @file pstrain_bw.c
 * @brief Simplified Baum-Welch training API for CFFI
 *
 * Provides a simplified interface to SphinxTrain's BW training,
 * hiding the complex struct initialization from Python.
 */

#include "pstrain_bw.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/err.h>
#include <sphinxbase/feat.h>
#include <sphinxbase/cmd_ln.h>

#include <s3/model_inventory.h>
#include <s3/model_def_io.h>
#include <s3/s3gau_io.h>
#include <s3/s3mixw_io.h>
#include <s3/s3tmat_io.h>
#include <s3/ts2cb.h>
#include <s3/gauden.h>
#include <s3/vector.h>
#include <s3/state_seq.h>
#include <s3/state.h>
#include <s3/lexicon.h>

/* The forced-alignment / utterance-HMM builders live alongside the
 * standalone bw binary; we link them through libpstrainc. */
#include "next_utt_states.h"

/* Forward declarations from bw code */
extern int32 baum_welch_update(float64 *log_forw_prob,
                               vector_t **feature,
                               uint32 n_obs,
                               void *state,
                               uint32 n_state,
                               model_inventory_t *inv,
                               float64 a_beam,
                               float64 b_beam,
                               float32 spthresh,
                               void *phseg,
                               int32 mixw_reest,
                               int32 tmat_reest,
                               int32 mean_reest,
                               int32 var_reest,
                               int32 pass2var,
                               int32 var_is_full,
                               FILE *pdumpfh,
                               void *timers,
                               feat_t *fcb);

/**
 * Training context - holds all state for a training session
 */
struct pstrain_bw_context_s {
    model_inventory_t *inv;
    model_def_t *mdef;
    feat_t *feat;
    lexicon_t *lex;  /* Dictionary/lexicon for transcript processing */

    /* Config */
    float64 a_beam;
    float64 b_beam;
    uint32 topn;
    float32 spthresh;
    float32 mixw_floor;
    float32 tmat_floor;
    int32 mixw_reest;
    int32 tmat_reest;
    int32 mean_reest;
    int32 var_reest;
    int32 pass2var;  /* Use 2-pass variance estimation for numerical stability */
    int32 multipron; /* Multi-pron training: build wide utterance graphs */

    /* Stats */
    float64 total_log_lik;
    uint32 total_frames;
    uint32 total_utts;
};

pstrain_bw_context_t *
pstrain_bw_init(const char *mdef_path,
            const char *means_path,
            const char *vars_path,
            const char *mixw_path,
            const char *tmat_path,
            const pstrain_bw_config_t *config)
{
    pstrain_bw_context_t *ctx;
    uint32 n_ts, n_cb;

    ctx = ckd_calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    /* Set config with defaults */
    ctx->a_beam = config ? config->a_beam : 1e-90;
    ctx->b_beam = config ? config->b_beam : 1e-10;  /* SphinxTrain default */
    ctx->topn = config ? config->topn : 1;
    ctx->spthresh = config ? config->spthresh : 0.0;
    ctx->mixw_floor = config ? config->mixw_floor : 1e-8;
    /* Provenance: upstream bw/train_cmd_ln.c supplies the live -tpfloor
     * default because the SphinxTrain Perl stage drivers do not override it. */
    ctx->tmat_floor = config ? config->tmat_floor : 1e-4;
    ctx->mixw_reest = config ? config->mixw_reest : 1;
    ctx->tmat_reest = config ? config->tmat_reest : 1;
    ctx->mean_reest = config ? config->mean_reest : 1;
    ctx->var_reest = config ? config->var_reest : 1;
    ctx->pass2var = config ? config->pass2var : 1;  /* Match SphinxTrain -2passvar yes */
    ctx->multipron = 1;  /* On by default; disable via pstrain_bw_set_multipron(ctx, 0). */

    /* Initialize cmd_ln with default values - required for gauden_alloc_acc etc.
     * and baum_welch_update which queries cmd_ln internally. */
    {
        static const arg_t args[] = {
            { "-meanreest", ARG_BOOLEAN, "yes", "Re-estimate means" },
            { "-varreest", ARG_BOOLEAN, "yes", "Re-estimate variances" },
            { "-mixwreest", ARG_BOOLEAN, "yes", "Re-estimate mixture weights" },
            { "-tmatreest", ARG_BOOLEAN, "yes", "Re-estimate transition matrices" },
            { "-fullvar", ARG_BOOLEAN, "no", "Use full covariance" },
            { "-outphsegdir", ARG_STRING, NULL, "Phone segmentation output dir" },
            { "-phsegext", ARG_STRING, "phseg", "Phone segmentation extension" },
            { "-cepext", ARG_STRING, ".mfc", "Cepstrum file extension" },
            { "-timing", ARG_BOOLEAN, "no", "Enable timing" },
            { NULL, 0, NULL, NULL }
        };
        cmd_ln_parse(args, 0, NULL, FALSE);
    }

    /* Initialize feature extraction - 1s_c_d_dd is 39-dim (13 * 3) */
    ctx->feat = feat_init("1s_c_d_dd", CMN_BATCH, 0, AGC_NONE, 1, 13);
    if (!ctx->feat) {
        E_ERROR("Failed to initialize feature module\n");
        goto error;
    }

    /* Read model definition */
    E_INFO("Reading model definition from %s\n", mdef_path);
    if (model_def_read(&ctx->mdef, mdef_path) != S3_SUCCESS) {
        E_ERROR("Failed to read model definition\n");
        goto error;
    }

    /* Set up continuous tied-state to codebook mapping */
    ctx->mdef->cb = cont_ts2cb(ctx->mdef->n_tied_state);
    n_ts = ctx->mdef->n_tied_state;
    n_cb = ctx->mdef->n_tied_state;

    /* Create model inventory */
    ctx->inv = mod_inv_new();
    if (!ctx->inv) {
        E_ERROR("Failed to create model inventory\n");
        goto error;
    }

    mod_inv_set_n_feat(ctx->inv, feat_dimension1(ctx->feat));
    ctx->inv->acmod_set = ctx->mdef->acmod_set;
    ctx->inv->mdef = ctx->mdef;

    /* Read mixture weights */
    E_INFO("Reading mixture weights from %s\n", mixw_path);
    if (mod_inv_read_mixw(ctx->inv, ctx->mdef, mixw_path, ctx->mixw_floor) != S3_SUCCESS) {
        E_ERROR("Failed to read mixture weights\n");
        goto error;
    }

    /* Read transition matrices */
    E_INFO("Reading transition matrices from %s\n", tmat_path);
    if (mod_inv_read_tmat(ctx->inv, tmat_path, ctx->tmat_floor) != S3_SUCCESS) {
        E_ERROR("Failed to read transition matrices\n");
        goto error;
    }

    /* Larger topn evaluates more densities.  Upstream uses every density for
     * CI and CD-tied BW, and one density for CD-untied BW. */
    E_INFO("Reading Gaussians from %s and %s\n", means_path, vars_path);
    if (mod_inv_read_gauden(ctx->inv, means_path, vars_path, 0.0001, ctx->topn, 0) != S3_SUCCESS) {
        E_ERROR("Failed to read Gaussians\n");
        goto error;
    }

    E_INFO("Gaussians loaded: n_mgau=%u, n_feat=%u, n_density=%u\n",
           ctx->inv->gauden->n_mgau, ctx->inv->gauden->n_feat,
           ctx->inv->gauden->n_density);

    /* Precompute Gaussian evaluation values */
    E_INFO("Precomputing Gaussian evaluation values...\n");
    if (gauden_eval_precomp(ctx->inv->gauden) != S3_SUCCESS) {
        E_ERROR("Failed to precompute Gaussian values\n");
        goto error;
    }
    E_INFO("Gaussian precomputation complete\n");

    /* Allocate accumulators */
    E_INFO("Allocating Gaussian accumulators...\n");
    if (mod_inv_alloc_gauden_acc(ctx->inv) != S3_SUCCESS) {
        E_ERROR("Failed to allocate Gaussian accumulators\n");
        goto error;
    }
    E_INFO("Allocating mixture weight accumulators...\n");
    if (mod_inv_alloc_mixw_acc(ctx->inv) != S3_SUCCESS) {
        E_ERROR("Failed to allocate mixture weight accumulators\n");
        goto error;
    }
    E_INFO("Allocating transition matrix accumulators...\n");
    if (mod_inv_alloc_tmat_acc(ctx->inv) != S3_SUCCESS) {
        E_ERROR("Failed to allocate transition matrix accumulators\n");
        goto error;
    }

    E_INFO("BW context initialized: %u tied states, %u codebooks\n", n_ts, n_cb);
    return ctx;

error:
    pstrain_bw_free(ctx);
    return NULL;
}

void
pstrain_bw_free(pstrain_bw_context_t *ctx)
{
    if (!ctx) return;

    if (ctx->lex) lexicon_free(ctx->lex);
    if (ctx->inv) mod_inv_free(ctx->inv);
    if (ctx->mdef) model_def_free(ctx->mdef);
    if (ctx->feat) feat_free(ctx->feat);

    ckd_free(ctx);
}

int
pstrain_bw_set_dict(pstrain_bw_context_t *ctx,
                const char *dict_path,
                const char *filler_dict_path)
{
    if (!ctx || !dict_path) {
        E_ERROR("Invalid arguments to pstrain_bw_set_dict\n");
        return -1;
    }

    /* Free existing lexicon if any */
    if (ctx->lex) {
        lexicon_free(ctx->lex);
        ctx->lex = NULL;
    }

    /* Read main dictionary */
    E_INFO("Reading dictionary from %s\n", dict_path);
    ctx->lex = lexicon_read(NULL, dict_path, ctx->mdef->acmod_set);
    if (!ctx->lex) {
        E_ERROR("Failed to read dictionary\n");
        return -1;
    }

    /* Read filler dictionary if provided */
    if (filler_dict_path) {
        E_INFO("Reading filler dictionary from %s\n", filler_dict_path);
        ctx->lex = lexicon_read(ctx->lex, filler_dict_path, ctx->mdef->acmod_set);
        if (!ctx->lex) {
            E_ERROR("Failed to read filler dictionary\n");
            return -1;
        }
    }

    E_INFO("Dictionary loaded: %u entries\n", ctx->lex->entry_cnt);
    return 0;
}

int
pstrain_bw_set_multipron(pstrain_bw_context_t *ctx, int enable)
{
    if (!ctx) return -1;
    ctx->multipron = enable ? 1 : 0;
    return 0;
}

float64
pstrain_bw_set_a_beam(pstrain_bw_context_t *ctx, float64 a_beam)
{
    float64 previous;

    if (!ctx || a_beam <= 0.0)
        return 0.0;
    previous = ctx->a_beam;
    ctx->a_beam = a_beam;
    return previous;
}

/*
 * Build a state sequence for one utterance, dispatching to either the
 * historical linear builder (next_utt_states) or the multi-pronunciation
 * graph builder (next_utt_states_graph). Both helpers live next to the
 * standalone bw binary so the CLI and CFFI paths share one
 * implementation; see csrc/programs/bw/next_utt_states.{c,h}.
 *
 * `trans_copy` must be a writable buffer (next_utt_states mutates it).
 * `*needs_free` is set on return: nonzero means the caller must call
 * state_seq_free(state_seq, *n_state) once the BW update is done. Zero
 * means the state_seq is backed by the static arrays inside
 * state_seq_make() and must NOT be freed.
 */
static state_t *
build_utt_state_seq(pstrain_bw_context_t *ctx,
                    char *trans_copy,
                    uint32 *n_state,
                    int *needs_free)
{
    if (!ctx->multipron) {
        *needs_free = 0;
        return next_utt_states(n_state,
                               ctx->lex,
                               ctx->inv,
                               ctx->mdef,
                               trans_copy);
    }

    *needs_free = 1;
    return next_utt_states_graph(n_state,
                                 ctx->lex,
                                 ctx->inv,
                                 ctx->mdef,
                                 trans_copy);
}

state_t *
pstrain_bw_build_state_seq(pstrain_bw_context_t *ctx,
                           const char *transcript,
                           uint32 *n_state)
{
    state_t *state_seq;
    char *trans_copy;
    int needs_free;

    if (!ctx || !ctx->lex || !transcript || !n_state || !ctx->multipron)
        return NULL;
    trans_copy = ckd_salloc(transcript);
    state_seq = build_utt_state_seq(ctx, trans_copy, n_state, &needs_free);
    ckd_free(trans_copy);
    return state_seq;
}

void
pstrain_bw_free_state_seq(state_t *state_seq, uint32 n_state)
{
    if (state_seq)
        state_seq_free(state_seq, n_state);
}

int
pstrain_bw_process_utt_text(pstrain_bw_context_t *ctx,
                        const float *features,
                        uint32 n_frames,
                        const char *transcript)
{
    vector_t **feat_vecs = NULL;
    state_t *state_seq = NULL;
    uint32 n_state;
    uint32 n_feat_stream;
    uint32 *veclen;
    float64 log_forw_prob;
    int ret;
    uint32 f, s;
    char *trans_copy;

    if (!ctx || !features || n_frames == 0 || !transcript) {
        E_ERROR("Invalid arguments to pstrain_bw_process_utt_text\n");
        return -1;
    }

    if (!ctx->lex) {
        E_ERROR("Dictionary not loaded - call pstrain_bw_set_dict first\n");
        return -1;
    }

    n_feat_stream = feat_dimension1(ctx->feat);
    veclen = feat_stream_lengths(ctx->feat);

    /* Allocate feature array */
    feat_vecs = ckd_calloc_2d(n_frames, n_feat_stream, sizeof(vector_t));
    for (f = 0; f < n_frames; f++) {
        for (s = 0; s < n_feat_stream; s++) {
            feat_vecs[f][s] = (vector_t)&features[f * veclen[s]];
        }
    }

    /* Build the state sequence for this utterance. The helper handles
     * both the linear and the multi-pron-graph paths and reports
     * whether the result must be freed by the caller. trans_copy is
     * mutable input (str2words / next_utt_states overwrite it). */
    {
        int needs_free = 0;
        trans_copy = ckd_salloc(transcript);
        state_seq = build_utt_state_seq(ctx, trans_copy, &n_state, &needs_free);
        ckd_free(trans_copy);

        if (!state_seq) {
            E_ERROR("Failed to build state sequence from transcript\n");
            ckd_free_2d((void **)feat_vecs);
            return -1;
        }

        ret = baum_welch_update(&log_forw_prob,
                                feat_vecs,
                                n_frames,
                                state_seq,
                                n_state,
                                ctx->inv,
                                ctx->a_beam,
                                ctx->b_beam,
                                ctx->spthresh,
                                NULL,
                                ctx->mixw_reest,
                                ctx->tmat_reest,
                                ctx->mean_reest,
                                ctx->var_reest,
                                0,
                                0,
                                NULL,
                                NULL,
                                ctx->feat);

        ckd_free_2d((void **)feat_vecs);

        /* Linear path uses static internal buffers (do NOT free).
         * Graph path allocates fresh per utterance (DO free). */
        if (needs_free) {
            state_seq_free(state_seq, n_state);
        }
    }

    if (ret != S3_SUCCESS) {
        E_ERROR("Baum-Welch update failed\n");
        return -1;
    }

    ctx->total_log_lik += log_forw_prob;
    ctx->total_frames += n_frames;
    ctx->total_utts++;

    return 0;
}

/**
 * Process utterance from raw MFCC features (13-dim).
 * Uses C feat module to apply CMN and compute deltas, exactly like SphinxTrain.
 */
int
pstrain_bw_process_utt_mfcc(pstrain_bw_context_t *ctx,
                        const float *mfcc,
                        uint32 n_mfcc_frames,
                        const char *transcript)
{
    mfcc_t **mfcc_buf = NULL;
    mfcc_t ***feat_buf = NULL;
    state_t *state_seq = NULL;
    uint32 n_state;
    int32 n_feat_frames;
    int32 ncep_in;  /* for feat_s2mfc2feat_live */
    float64 log_forw_prob;
    int ret;
    uint32 f, d;
    char *trans_copy;
    int32 ceplen = 13;

    if (!ctx || !mfcc || n_mfcc_frames == 0 || !transcript) {
        E_ERROR("Invalid arguments to pstrain_bw_process_utt_mfcc\n");
        return -1;
    }

    if (!ctx->lex) {
        E_ERROR("Dictionary not loaded - call pstrain_bw_set_dict first\n");
        return -1;
    }

    /* Convert input float array to mfcc_t format */
    mfcc_buf = ckd_calloc_2d(n_mfcc_frames, ceplen, sizeof(mfcc_t));
    for (f = 0; f < n_mfcc_frames; f++) {
        for (d = 0; d < ceplen; d++) {
            mfcc_buf[f][d] = mfcc[f * ceplen + d];
        }
    }

    /* Use feat module to compute features (applies CMN, deltas) */
    /* This is exactly what SphinxTrain's bw does */
    feat_buf = feat_array_alloc(ctx->feat, n_mfcc_frames + 10);
    if (!feat_buf) {
        E_ERROR("Failed to allocate feature buffer\n");
        ckd_free_2d((void **)mfcc_buf);
        return -1;
    }

    /* Process MFCCs through feat module - this applies CMN and deltas */
    ncep_in = (int32)n_mfcc_frames;
    n_feat_frames = feat_s2mfc2feat_live(ctx->feat,
                                          mfcc_buf,
                                          &ncep_in,
                                          TRUE,  /* beginutt */
                                          TRUE,  /* endutt */
                                          feat_buf);

    ckd_free_2d((void **)mfcc_buf);

    if (n_feat_frames == 0) {
        E_ERROR("feat_s2mfc2feat_live returned 0 frames\n");
        feat_array_free(feat_buf);
        return -1;
    }

    /* Build the state sequence; helper handles linear vs. graph path
     * and reports whether the result must be freed by us. */
    {
        int needs_free = 0;
        trans_copy = ckd_salloc(transcript);
        state_seq = build_utt_state_seq(ctx, trans_copy, &n_state, &needs_free);
        ckd_free(trans_copy);

        if (!state_seq) {
            E_ERROR("Failed to build state sequence from transcript\n");
            feat_array_free(feat_buf);
            return -1;
        }

        ret = baum_welch_update(&log_forw_prob,
                                feat_buf,
                                n_feat_frames,
                                state_seq,
                                n_state,
                                ctx->inv,
                                ctx->a_beam,
                                ctx->b_beam,
                                ctx->spthresh,
                                NULL,  /* phseg */
                                ctx->mixw_reest,
                                ctx->tmat_reest,
                                ctx->mean_reest,
                                ctx->var_reest,
                                ctx->pass2var,
                                0,     /* var_is_full */
                                NULL,  /* pdumpfh */
                                NULL,  /* latfh */
                                ctx->feat);

        feat_array_free(feat_buf);

        if (needs_free) {
            state_seq_free(state_seq, n_state);
        }
    }

    if (ret != S3_SUCCESS) {
        E_ERROR("baum_welch_update failed\n");
        return ret == PSTRAIN_BW_FINAL_STATE_NOT_REACHED
            ? PSTRAIN_BW_FINAL_STATE_NOT_REACHED : -1;
    }

    ctx->total_log_lik += log_forw_prob;
    ctx->total_frames += n_feat_frames;
    ctx->total_utts++;

    return 0;
}

int
pstrain_bw_process_utt(pstrain_bw_context_t *ctx,
                   const float *features,
                   uint32 n_frames,
                   const uint32 *phone_ids,
                   uint32 n_phones)
{
    vector_t **feat_vecs = NULL;
    state_t *state_seq = NULL;
    uint32 n_state;
    uint32 n_feat_stream;
    uint32 *veclen;
    float64 log_forw_prob;
    int ret;
    uint32 f, s;

    if (!ctx || !features || n_frames == 0 || !phone_ids || n_phones == 0) {
        E_ERROR("Invalid arguments to pstrain_bw_process_utt\n");
        return -1;
    }

    n_feat_stream = feat_dimension1(ctx->feat);
    veclen = feat_stream_lengths(ctx->feat);

    /* Allocate feature array in vector_t** format
     * vector_t** is [n_frames][n_streams] where each element is a vector */
    feat_vecs = ckd_calloc_2d(n_frames, n_feat_stream, sizeof(vector_t));

    /* Copy features - for 1s_c_d_dd, n_feat_stream=1, veclen[0]=39 */
    for (f = 0; f < n_frames; f++) {
        for (s = 0; s < n_feat_stream; s++) {
            feat_vecs[f][s] = (vector_t)&features[f * veclen[s]];
        }
    }

    /* Build state sequence from phone IDs */
    state_seq = state_seq_make(&n_state,
                               (acmod_id_t *)phone_ids,
                               n_phones,
                               ctx->inv,
                               ctx->mdef);
    if (!state_seq) {
        E_ERROR("Failed to build state sequence\n");
        ckd_free_2d((void **)feat_vecs);
        return -1;
    }

    /* Run Baum-Welch update */
    ret = baum_welch_update(&log_forw_prob,
                            feat_vecs,
                            n_frames,
                            state_seq,
                            n_state,
                            ctx->inv,
                            ctx->a_beam,
                            ctx->b_beam,
                            ctx->spthresh,
                            NULL,  /* phseg - not using phone segmentation */
                            ctx->mixw_reest,
                            ctx->tmat_reest,
                            ctx->mean_reest,
                            ctx->var_reest,
                            ctx->pass2var,
                            0,     /* var_is_full */
                            NULL,  /* pdumpfh */
                            NULL,  /* timers */
                            ctx->feat);

    /* Free the vector pointer array (not the data - it's owned by caller) */
    ckd_free_2d((void **)feat_vecs);
    /* Note: Don't call state_seq_free() - state_seq_make() uses static arrays */

    if (ret != S3_SUCCESS) {
        E_ERROR("Baum-Welch update failed\n");
        return -1;
    }

    /* Update stats */
    ctx->total_log_lik += log_forw_prob;
    ctx->total_frames += n_frames;
    ctx->total_utts++;

    return 0;
}

int
pstrain_bw_normalize(pstrain_bw_context_t *ctx)
{
    gauden_t *g = ctx->inv->gauden;
    uint32 n_mgau = g->n_mgau;
    uint32 n_feat = g->n_feat;
    uint32 n_density = g->n_density;
    uint32 *veclen = g->veclen;
    uint32 i, j, k, l;

    /* Normalize Gaussians: mean and variance */
    E_INFO("Normalizing Gaussians...\n");
    for (i = 0; i < n_mgau; i++) {
        for (j = 0; j < n_feat; j++) {
            for (k = 0; k < n_density; k++) {
                float32 d = g->dnom[i][j][k];
                if (d > 0) {
                    for (l = 0; l < veclen[j]; l++) {
                        /* Normalize mean */
                        g->mean[i][j][k][l] = g->macc[i][j][k][l] / d;
                        /* Normalize variance */
                        {
                            float32 v;
                            if (ctx->pass2var) {
                                /* 2-pass: vacc contains E[(x-μ_old)²], just divide by count */
                                v = g->vacc[i][j][k][l] / d;
                            } else {
                                /* 1-pass: vacc contains E[x²], compute E[x²] - E[x]² */
                                v = (g->vacc[i][j][k][l] / d) -
                                    (g->mean[i][j][k][l] * g->mean[i][j][k][l]);
                            }
                            /* Floor variance to prevent numerical issues.
                             * Use 1e-4 as minimum floor (matching SphinxTrain). */
                            if (v < 1e-4f) v = 1e-4f;
                            g->var[i][j][k][l] = v;
                        }
                    }
                } else {
                    /* No data for this senone - keep old values from previous model */
                    E_WARN("mgau %u feat %u density %u has no data, keeping old values\n", i, j, k);
                }
            }
        }
    }

    /* Normalize mixture weights */
    E_INFO("Normalizing mixture weights...\n");
    for (i = 0; i < ctx->inv->n_mixw; i++) {
        for (j = 0; j < ctx->inv->n_feat; j++) {
            float32 sum = 0;
            for (k = 0; k < ctx->inv->n_density; k++) {
                sum += ctx->inv->mixw_acc[i][j][k];
            }
            if (sum > 0) {
                for (k = 0; k < ctx->inv->n_density; k++) {
                    ctx->inv->mixw[i][j][k] = ctx->inv->mixw_acc[i][j][k] / sum;
                }
            } else {
                E_WARN("mixw %u feat %u has no data, keeping old values\n", i, j);
            }
        }
    }

    /* Normalize transition matrices */
    E_INFO("Normalizing transition matrices...\n");
    for (i = 0; i < ctx->inv->n_tmat; i++) {
        for (j = 0; j < ctx->inv->n_state_pm - 1; j++) {  /* n_state_pm-1 emitting states */
            float32 sum = 0;
            for (k = 0; k < ctx->inv->n_state_pm; k++) {
                sum += ctx->inv->tmat_acc[i][j][k];
            }
            if (sum > 0) {
                for (k = 0; k < ctx->inv->n_state_pm; k++) {
                    ctx->inv->tmat[i][j][k] = ctx->inv->tmat_acc[i][j][k] / sum;
                }
            } else {
                E_WARN("tmat %u state %u has no data, keeping old values\n", i, j);
            }
        }
    }

    /* Recompute Gaussian precomputation values */
    E_INFO("Recomputing Gaussian precomputation values...\n");
    if (gauden_eval_precomp(g) != S3_SUCCESS) {
        E_ERROR("Failed to recompute Gaussian values\n");
        return -1;
    }

    /* Clear accumulators for next iteration */
    E_INFO("Clearing accumulators...\n");
    for (i = 0; i < n_mgau; i++) {
        for (j = 0; j < n_feat; j++) {
            for (k = 0; k < n_density; k++) {
                g->dnom[i][j][k] = 0;
                for (l = 0; l < veclen[j]; l++) {
                    g->macc[i][j][k][l] = 0;
                    g->vacc[i][j][k][l] = 0;
                }
            }
        }
    }
    for (i = 0; i < ctx->inv->n_mixw; i++) {
        for (j = 0; j < ctx->inv->n_feat; j++) {
            for (k = 0; k < ctx->inv->n_density; k++) {
                ctx->inv->mixw_acc[i][j][k] = 0;
            }
        }
    }
    for (i = 0; i < ctx->inv->n_tmat; i++) {
        for (j = 0; j < ctx->inv->n_state_pm - 1; j++) {
            for (k = 0; k < ctx->inv->n_state_pm; k++) {
                ctx->inv->tmat_acc[i][j][k] = 0;
            }
        }
    }

    /* Note: gauden_eval_precomp was already called above (line 634).
     * Do NOT call it again here - it would flip variances back to raw form,
     * but pstrain_bw_save expects precomputed form (1/(2*sigma^2)). */

    /* Reset stats for next iteration */
    ctx->total_log_lik = 0;
    ctx->total_frames = 0;
    ctx->total_utts = 0;

    E_INFO("Normalization complete\n");
    return 0;
}

int
pstrain_bw_save(pstrain_bw_context_t *ctx,
            const char *means_path,
            const char *vars_path,
            const char *mixw_path,
            const char *tmat_path)
{
    gauden_t *g = ctx->inv->gauden;
    uint32 i, j, k, l;

    E_INFO("Saving Gaussians to %s and %s\n", means_path, vars_path);
    if (s3gau_write(means_path,
                    (const vector_t ***)g->mean,
                    g->n_mgau,
                    g->n_feat,
                    g->n_density,
                    g->veclen) != S3_SUCCESS) {
        E_ERROR("Failed to write means\n");
        return -1;
    }

    /* After gauden_eval_precomp, g->var contains 1/(2*sigma^2).
     * We need to convert back to sigma^2 for saving.
     * var_original = 1 / (2 * var_precomputed) */
    for (i = 0; i < g->n_mgau; i++) {
        for (j = 0; j < g->n_feat; j++) {
            for (k = 0; k < g->n_density; k++) {
                for (l = 0; l < g->veclen[j]; l++) {
                    if (g->var[i][j][k][l] > 0) {
                        g->var[i][j][k][l] = 1.0f / (2.0f * g->var[i][j][k][l]);
                    }
                }
            }
        }
    }

    if (s3gau_write(vars_path,
                    (const vector_t ***)g->var,
                    g->n_mgau,
                    g->n_feat,
                    g->n_density,
                    g->veclen) != S3_SUCCESS) {
        E_ERROR("Failed to write variances\n");
        return -1;
    }

    /* Convert back to precomputed form for subsequent iterations */
    for (i = 0; i < g->n_mgau; i++) {
        for (j = 0; j < g->n_feat; j++) {
            for (k = 0; k < g->n_density; k++) {
                for (l = 0; l < g->veclen[j]; l++) {
                    if (g->var[i][j][k][l] > 0) {
                        g->var[i][j][k][l] = 1.0f / (2.0f * g->var[i][j][k][l]);
                    }
                }
            }
        }
    }

    E_INFO("Saving mixture weights to %s\n", mixw_path);
    if (s3mixw_write(mixw_path,
                     ctx->inv->mixw,
                     ctx->inv->n_mixw,
                     ctx->inv->n_feat,
                     ctx->inv->n_density) != S3_SUCCESS) {
        E_ERROR("Failed to write mixture weights\n");
        return -1;
    }

    E_INFO("Saving transition matrices to %s\n", tmat_path);
    if (s3tmat_write(tmat_path,
                     ctx->inv->tmat,
                     ctx->inv->n_tmat,
                     ctx->inv->n_state_pm) != S3_SUCCESS) {
        E_ERROR("Failed to write transition matrices\n");
        return -1;
    }

    return 0;
}

void
pstrain_bw_get_stats(pstrain_bw_context_t *ctx,
                 float64 *total_log_lik,
                 uint32 *total_frames,
                 uint32 *total_utts)
{
    if (total_log_lik) *total_log_lik = ctx->total_log_lik;
    if (total_frames) *total_frames = ctx->total_frames;
    if (total_utts) *total_utts = ctx->total_utts;
}

int
pstrain_bw_save_counts(pstrain_bw_context_t *ctx, const char *counts_path)
{
    gauden_t *g = ctx->inv->gauden;

    E_INFO("Saving density counts to %s\n", counts_path);
    /* Use s3gaudnom_write to write just dnom (density counts).
     * This matches the format expected by pstrain_inc_comp (Gaussian splitting). */
    if (s3gaudnom_write(counts_path,
                        g->dnom,
                        g->n_mgau,
                        g->n_feat,
                        g->n_density) != S3_SUCCESS) {
        E_ERROR("Failed to write density counts\n");
        return -1;
    }

    return 0;
}
