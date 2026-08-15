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
 * @file pstrain_align.c
 * @brief In-process forced-alignment session wrapper.
 *
 * Drives the sphinx3 aligner vendored under csrc/programs/sphinx3_align/
 * without going through main_align.c's argv path or the on-disk control
 * file. Replaces the subprocess wrapper and the PocketSphinx-based
 * aligner; see docs/sphinx3-align-cffi-plan.md for context.
 */

#include "pstrain_align.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/cmd_ln.h>
#include <sphinxbase/err.h>
#include <sphinxbase/feat.h>
#include <sphinxbase/prim_type.h>

#include "main_align.h"
#include "dict.h"
#include "kbcore.h"
#include "mdef.h"
#include "s3_align.h"

/* The static defn[] table lives in main_align.c. We synthesize a cmd_ln_t
 * directly from the user's config struct using the same option names; the
 * sphinx3 cmd_ln machinery validates the keys against the registered
 * argument table that main_align.c installs. */
extern arg_t *cmd_ln_get_defn_for_align(void);

/* Maximum cepstrum frames we'll allocate for a single utterance.  Matches
 * the constant baked into main_align.c so we stay aligned with the CLI's
 * limits. */
#define PSTRAIN_ALIGN_MAX_FRAMES 32768

struct pstrain_align_context_s {
    cmd_ln_t *config;
    int32 ncep;
    int32 want_phones;
    int32 want_states;
};

/* Single-instance enforcement: the underlying aligner holds module-static
 * state (kbc, ascr, etc.) over in main_align.c. Promoting to a per-context
 * model is mechanical but unnecessary today. */
static pstrain_align_context_t *g_ctx = NULL;
static char g_last_error[1024];

static void
set_error(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error), fmt, ap);
    va_end(ap);
}

void
pstrain_align_config_default(pstrain_align_config_t *config)
{
    if (config == NULL) return;
    config->beam = 1e-64;
    config->insert_sil = 1;
    config->compute_phones = 1;
    config->compute_states = 0;
    config->feat_type = "1s_c_d_dd";
    config->cmn = "current";
    config->cmninit = "40,3,-1";
    config->agc = "none";
    config->varnorm = 0;
    config->ceplen = 13;
    config->frate = 100;
    config->lts_mismatch = 0;
}

static const char *
nz(const char *s, const char *fallback)
{
    return (s && *s) ? s : fallback;
}

/* Build a cmd_ln_t with all the keys models_init / align_utt_capture
 * read. We register the same defn[] table the CLI binary uses so cmd_ln
 * validation is bit-identical and unknown keys fail fast. */
static cmd_ln_t *
build_config(const char *mdef_path,
             const char *mean_path,
             const char *var_path,
             const char *mixw_path,
             const char *tmat_path,
             const char *feat_params_path,
             const char *dict_path,
             const char *fdict_path,
             const pstrain_align_config_t *cfg)
{
    char beam_str[64];
    char insert_sil_str[16];
    char ceplen_str[16];
    char frate_str[16];
    char lts_str[8];
    snprintf(beam_str, sizeof(beam_str), "%g", cfg->beam);
    snprintf(insert_sil_str, sizeof(insert_sil_str), "%d", cfg->insert_sil);
    snprintf(ceplen_str, sizeof(ceplen_str), "%d", cfg->ceplen);
    snprintf(frate_str, sizeof(frate_str), "%d", cfg->frate);
    snprintf(lts_str, sizeof(lts_str), "%s", cfg->lts_mismatch ? "yes" : "no");

    arg_t *defn = cmd_ln_get_defn_for_align();
    cmd_ln_t *c = cmd_ln_init(NULL, defn, FALSE,
        "-mdef", mdef_path,
        "-mean", mean_path,
        "-var", var_path,
        "-mixw", mixw_path,
        "-tmat", tmat_path,
        "-dict", dict_path,
        "-fdict", fdict_path ? fdict_path : "",
        "-featparams", feat_params_path ? feat_params_path : "",
        "-feat", nz(cfg->feat_type, "1s_c_d_dd"),
        "-cmn", nz(cfg->cmn, "current"),
        "-cmninit", nz(cfg->cmninit, "40,3,-1"),
        "-agc", nz(cfg->agc, "none"),
        "-varnorm", cfg->varnorm ? "yes" : "no",
        "-ceplen", ceplen_str,
        "-frate", frate_str,
        "-beam", beam_str,
        "-insert_sil", insert_sil_str,
        "-lts_mismatch", lts_str,
        "-insent", "<unused>",
        NULL);
    return c;
}

pstrain_align_context_t *
pstrain_align_init(const char *mdef_path,
               const char *mean_path,
               const char *var_path,
               const char *mixw_path,
               const char *tmat_path,
               const char *feat_params_path,
               const char *dict_path,
               const char *fdict_path,
               const pstrain_align_config_t *config)
{
    if (g_ctx != NULL) {
        set_error("pstrain_align: another aligner is already active; free it first");
        return NULL;
    }
    if (mdef_path == NULL || mean_path == NULL || var_path == NULL ||
        mixw_path == NULL || tmat_path == NULL || dict_path == NULL) {
        set_error("pstrain_align: required path argument missing");
        return NULL;
    }

    pstrain_align_config_t defaults;
    pstrain_align_config_default(&defaults);
    const pstrain_align_config_t *cfg = config ? config : &defaults;

    cmd_ln_t *c = build_config(mdef_path, mean_path, var_path, mixw_path,
                               tmat_path, feat_params_path,
                               dict_path, fdict_path, cfg);
    if (c == NULL) {
        set_error("pstrain_align: cmd_ln_init failed");
        return NULL;
    }

    /* models_init / align_init below call E_FATAL on bad input. There's no
     * way to recover cleanly from inside this process once that fires, so
     * we rely on the caller to provide valid paths. Setting a longjmp
     * trampoline is the right next step but lives outside this first cut.
     */
    models_init(c);

    if (feat == NULL) {
        feat = feat_array_alloc(kbcore_fcb(kbc), PSTRAIN_ALIGN_MAX_FRAMES);
    }

    align_init(kbc->mdef, kbc->tmat, dict, c, kbc->logmath);

    pstrain_align_context_t *ctx = ckd_calloc(1, sizeof(*ctx));
    ctx->config = c;
    ctx->ncep = feat_cepsize(kbcore_fcb(kbc));
    ctx->want_phones = cfg->compute_phones;
    ctx->want_states = cfg->compute_states;
    g_ctx = ctx;
    g_last_error[0] = '\0';
    return ctx;
}

void
pstrain_align_free(pstrain_align_context_t *ctx)
{
    if (ctx == NULL) return;

    align_free();
    if (feat) {
        feat_array_free(feat);
        feat = NULL;
    }
    models_free();
    cmd_ln_free_r(ctx->config);
    ckd_free(ctx);
    if (g_ctx == ctx) {
        g_ctx = NULL;
    }
}

/* Trim leading/trailing whitespace and return a fresh malloc'd copy.
 *
 * Deliberately preserves <s>/</s> sentence markers when present: the
 * standalone sphinx3_align CLI passes them straight through to
 * align_build_sent_hmm (they resolve to the corresponding filler-dict
 * entries and produce the same start/end silence labels the CLI emits).
 * Stripping them would change which filler word ID lands on those
 * frames and break parity with the binary. */
static char *
clean_transcript(const char *raw)
{
    if (raw == NULL) return NULL;
    const char *start = raw;
    while (*start == ' ' || *start == '\t' || *start == '\n') start++;
    size_t len = strlen(start);
    char *out = ckd_calloc(len + 1, sizeof(char));
    memcpy(out, start, len);
    while (len > 0 && (out[len - 1] == ' ' || out[len - 1] == '\t' ||
                       out[len - 1] == '\n' || out[len - 1] == '\r')) {
        out[--len] = '\0';
    }
    return out;
}

/* Count the entries in a linked seg list. Each variant of align_*_t has
 * the same {.next} accessor at the same offset; we use type-specific
 * helpers below rather than depending on layout. */
static uint32
count_st(const align_stseg_t *s)
{
    uint32 n = 0;
    while (s) { n++; s = s->next; }
    return n;
}
static uint32
count_ph(const align_phseg_t *p)
{
    uint32 n = 0;
    while (p) { n++; p = p->next; }
    return n;
}
static uint32
count_wd(const align_wdseg_t *w)
{
    uint32 n = 0;
    while (w) { n++; w = w->next; }
    return n;
}

/* Internal arena: a single contiguous string buffer attached to the result
 * struct. Holds all the seg labels; freed in pstrain_align_result_free. */
struct seg_arena {
    char *buf;
    size_t len;
    size_t cap;
};

static const char *
arena_strdup(struct seg_arena *a, const char *s)
{
    if (s == NULL) s = "";
    size_t n = strlen(s) + 1;
    if (a->len + n > a->cap) {
        size_t new_cap = a->cap ? a->cap * 2 : 256;
        while (new_cap < a->len + n) new_cap *= 2;
        a->buf = ckd_realloc(a->buf, new_cap);
        a->cap = new_cap;
    }
    char *out = a->buf + a->len;
    memcpy(out, s, n);
    a->len += n;
    return out;
}

/* The arena's buf may move when arena_strdup grows it, so we record byte
 * offsets first and resolve them to pointers at the end. */
static void
flatten_wd(const align_wdseg_t *src, pstrain_align_seg_t *dst, uint32 n,
           size_t *name_offs, struct seg_arena *arena)
{
    uint32 i = 0;
    for (const align_wdseg_t *w = src; w && i < n; w = w->next, i++) {
        const char *name = dict_wordstr(dict, w->wid);
        name_offs[i] = arena->len;
        arena_strdup(arena, name);
        dst[i].start_frame = w->sf;
        dst[i].end_frame = w->ef;
        dst[i].score = w->score;
    }
}

static void
flatten_ph(const align_phseg_t *src, pstrain_align_seg_t *dst, uint32 n,
           size_t *name_offs, struct seg_arena *arena)
{
    char buf[64];
    uint32 i = 0;
    for (const align_phseg_t *p = src; p && i < n; p = p->next, i++) {
        mdef_phone_str(kbc->mdef, p->pid, buf);
        name_offs[i] = arena->len;
        arena_strdup(arena, buf);
        dst[i].start_frame = p->sf;
        dst[i].end_frame = p->ef;
        dst[i].score = p->score;
    }
}

static void
flatten_st(const align_stseg_t *src, pstrain_align_seg_t *dst, uint32 n,
           size_t *name_offs, struct seg_arena *arena)
{
    char buf[64];
    uint32 i = 0;
    for (const align_stseg_t *s = src; s && i < n; s = s->next, i++) {
        snprintf(buf, sizeof(buf), "p%d.s%d", (int)s->pid, (int)s->state);
        name_offs[i] = arena->len;
        arena_strdup(arena, buf);
        dst[i].start_frame = (int32)i;        /* per-frame */
        dst[i].end_frame = (int32)i;
        dst[i].score = s->score;
    }
}

static int
build_result(int want_phones, int want_states,
             align_stseg_t *stseg, align_phseg_t *phseg, align_wdseg_t *wdseg,
             int32 n_frames,
             pstrain_align_result_t **out_result)
{
    pstrain_align_result_t *r = ckd_calloc(1, sizeof(*r));
    struct seg_arena *arena = ckd_calloc(1, sizeof(*arena));
    r->_arena = arena;

    r->n_words = count_wd(wdseg);
    r->n_phones = want_phones ? count_ph(phseg) : 0;
    r->n_states = want_states ? count_st(stseg) : 0;

    size_t *wd_offs = r->n_words ? ckd_calloc(r->n_words, sizeof(size_t)) : NULL;
    size_t *ph_offs = r->n_phones ? ckd_calloc(r->n_phones, sizeof(size_t)) : NULL;
    size_t *st_offs = r->n_states ? ckd_calloc(r->n_states, sizeof(size_t)) : NULL;

    if (r->n_words) {
        r->words = ckd_calloc(r->n_words, sizeof(pstrain_align_seg_t));
        flatten_wd(wdseg, r->words, r->n_words, wd_offs, arena);
    }
    if (r->n_phones) {
        r->phones = ckd_calloc(r->n_phones, sizeof(pstrain_align_seg_t));
        flatten_ph(phseg, r->phones, r->n_phones, ph_offs, arena);
    }
    if (r->n_states) {
        r->states = ckd_calloc(r->n_states, sizeof(pstrain_align_seg_t));
        flatten_st(stseg, r->states, r->n_states, st_offs, arena);
    }

    for (uint32 i = 0; i < r->n_words; i++) {
        r->words[i].name = arena->buf + wd_offs[i];
    }
    for (uint32 i = 0; i < r->n_phones; i++) {
        r->phones[i].name = arena->buf + ph_offs[i];
    }
    for (uint32 i = 0; i < r->n_states; i++) {
        r->states[i].name = arena->buf + st_offs[i];
    }

    int32 total = 0;
    for (const align_wdseg_t *w = wdseg; w; w = w->next) total += w->score;
    r->total_score = total;
    r->n_frames = n_frames;

    if (wd_offs) ckd_free(wd_offs);
    if (ph_offs) ckd_free(ph_offs);
    if (st_offs) ckd_free(st_offs);
    *out_result = r;
    return 0;
}

void
pstrain_align_result_free(pstrain_align_result_t *result)
{
    if (result == NULL) return;
    if (result->_arena) {
        struct seg_arena *a = (struct seg_arena *)result->_arena;
        if (a->buf) ckd_free(a->buf);
        ckd_free(a);
    }
    if (result->words) ckd_free(result->words);
    if (result->phones) ckd_free(result->phones);
    if (result->states) ckd_free(result->states);
    ckd_free(result);
}

const char *
pstrain_align_last_error(void)
{
    return g_last_error[0] ? g_last_error : NULL;
}

/* feat_s2mfc2feat_live wants mfcc_t** (an array of frame pointers). We
 * accept a contiguous float buffer from the caller, so we set up the
 * row-pointer table on the fly. */
static int
prepare_feat_from_mfcc(const float *mfcc, uint32 n_frames, uint32 ncep,
                       int32 *out_nfr)
{
    if ((int32)ncep != feat_cepsize(kbcore_fcb(kbc))) {
        set_error("pstrain_align: ncep=%u does not match model cepsize=%d",
                  ncep, feat_cepsize(kbcore_fcb(kbc)));
        return -1;
    }
    if (n_frames > PSTRAIN_ALIGN_MAX_FRAMES) {
        set_error("pstrain_align: n_frames=%u exceeds max=%d",
                  n_frames, PSTRAIN_ALIGN_MAX_FRAMES);
        return -1;
    }
    mfcc_t **rows = ckd_calloc(n_frames, sizeof(mfcc_t *));
    for (uint32 i = 0; i < n_frames; i++) {
        rows[i] = (mfcc_t *)(mfcc + i * ncep);
    }
    int32 nfr = (int32)n_frames;
    int32 produced = feat_s2mfc2feat_live(kbcore_fcb(kbc), rows, &nfr,
                                          TRUE, TRUE, feat);
    ckd_free(rows);
    if (produced < 0) {
        set_error("pstrain_align: feat_s2mfc2feat_live failed");
        return -1;
    }
    *out_nfr = produced;
    return 0;
}

int
pstrain_align_mfcc(pstrain_align_context_t *ctx,
               const float *mfcc,
               uint32 n_frames,
               uint32 ncep,
               const char *transcript,
               const char *utt_id,
               pstrain_align_result_t **out_result)
{
    if (ctx == NULL || mfcc == NULL || transcript == NULL || out_result == NULL) {
        set_error("pstrain_align_mfcc: NULL argument");
        return -1;
    }
    *out_result = NULL;

    int32 nfr = 0;
    if (prepare_feat_from_mfcc(mfcc, n_frames, ncep, &nfr) < 0) {
        return -1;
    }

    char *sent = clean_transcript(transcript);
    align_stseg_t *stseg = NULL;
    align_phseg_t *phseg = NULL;
    align_wdseg_t *wdseg = NULL;
    int rc = align_utt_capture(sent, nfr,
                               (char *)(utt_id ? utt_id : "utt"),
                               &stseg, &phseg, &wdseg);
    if (rc != 0) {
        ckd_free(sent);
        set_error("pstrain_align_mfcc: align_utt_capture failed (rc=%d)", rc);
        return -1;
    }

    int built = build_result(ctx->want_phones, ctx->want_states,
                             stseg, phseg, wdseg, nfr, out_result);

    align_utt_release();
    ckd_free(sent);
    return built;
}

int
pstrain_align_mfc_file(pstrain_align_context_t *ctx,
                   const char *mfc_path,
                   const char *transcript,
                   const char *utt_id,
                   pstrain_align_result_t **out_result)
{
    if (ctx == NULL || mfc_path == NULL || transcript == NULL || out_result == NULL) {
        set_error("pstrain_align_mfc_file: NULL argument");
        return -1;
    }
    *out_result = NULL;

    int32 nfr = feat_s2mfc2feat(kbcore_fcb(kbc), mfc_path, NULL, "",
                                0, -1, feat, PSTRAIN_ALIGN_MAX_FRAMES);
    if (nfr <= 0) {
        set_error("pstrain_align_mfc_file: failed to read %s", mfc_path);
        return -1;
    }

    char *sent = clean_transcript(transcript);
    align_stseg_t *stseg = NULL;
    align_phseg_t *phseg = NULL;
    align_wdseg_t *wdseg = NULL;
    int rc = align_utt_capture(sent, nfr,
                               (char *)(utt_id ? utt_id : "utt"),
                               &stseg, &phseg, &wdseg);
    if (rc != 0) {
        ckd_free(sent);
        set_error("pstrain_align_mfc_file: align_utt_capture failed (rc=%d)", rc);
        return -1;
    }

    int built = build_result(ctx->want_phones, ctx->want_states,
                             stseg, phseg, wdseg, nfr, out_result);

    align_utt_release();
    ckd_free(sent);
    return built;
}
