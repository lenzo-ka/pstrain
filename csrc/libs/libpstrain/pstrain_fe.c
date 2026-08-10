/**
 * pstrain_fe.c - Simplified FE initialization for pstrain
 *
 * Provides a simple interface to create a front-end without needing
 * to go through the cmd_ln infrastructure.
 */

#include <sphinxbase/fe.h>
#include <sphinxbase/cmd_ln.h>
#include <sphinxbase/ckd_alloc.h>
#include <sphinxbase/err.h>
#include <stdio.h>

/**
 * Create a front-end with explicit parameters.
 *
 * This bypasses the cmd_ln parsing and creates an FE directly with
 * the given parameters. Uses defaults for any unspecified values.
 *
 * @param samprate Sample rate in Hz (e.g., 16000)
 * @param nfilt Number of mel filters (e.g., 25 for SphinxTrain wideband)
 * @param nfft FFT size (e.g., 512)
 * @param lowerf Lower frequency bound (e.g., 130)
 * @param upperf Upper frequency bound (e.g., 6800)
 * @param ncep Number of cepstral coefficients (e.g., 13)
 * @param alpha Pre-emphasis coefficient (e.g., 0.97)
 * @param lifter Liftering coefficient (e.g., 22)
 * @return Initialized fe_t*, or NULL on failure
 */
fe_t *
pstrain_fe_create(float samprate, int nfilt, int nfft,
              float lowerf, float upperf, int ncep,
              float alpha, int lifter)
{
    cmd_ln_t *config;
    fe_t *fe;
    char samprate_str[32], nfilt_str[16], nfft_str[16];
    char lowerf_str[32], upperf_str[32], ncep_str[16];
    char alpha_str[32], lifter_str[16], frate_str[16], wlen_str[32];

    /* Convert numeric values to strings */
    snprintf(samprate_str, sizeof(samprate_str), "%f", samprate);
    snprintf(nfilt_str, sizeof(nfilt_str), "%d", nfilt);
    snprintf(nfft_str, sizeof(nfft_str), "%d", nfft);
    snprintf(lowerf_str, sizeof(lowerf_str), "%f", lowerf);
    snprintf(upperf_str, sizeof(upperf_str), "%f", upperf);
    snprintf(ncep_str, sizeof(ncep_str), "%d", ncep);
    snprintf(alpha_str, sizeof(alpha_str), "%f", alpha);
    snprintf(lifter_str, sizeof(lifter_str), "%d", lifter);
    snprintf(frate_str, sizeof(frate_str), "%d", 100);
    snprintf(wlen_str, sizeof(wlen_str), "%f", 0.025625);

    /* Create a command line with FE defaults plus our overrides */
    /* Match sphinx_fe/SphinxTrain defaults exactly */
    config = cmd_ln_init(NULL, fe_get_args(), FALSE,
                         "-samprate", samprate_str,
                         "-nfilt", nfilt_str,
                         "-nfft", nfft_str,
                         "-lowerf", lowerf_str,
                         "-upperf", upperf_str,
                         "-ncep", ncep_str,
                         "-alpha", alpha_str,
                         "-lifter", lifter_str,
                         "-transform", "dct",
                         "-dither", "no",
                         "-remove_dc", "no",
                         "-remove_noise", "yes",
                         "-frate", frate_str,
                         "-wlen", wlen_str,
                         NULL);

    if (config == NULL) {
        E_ERROR("Failed to create FE config\n");
        return NULL;
    }

    fe = fe_init_auto_r(config);
    /* Note: fe_init_auto_r may or may not retain config; for safety
     * we rely on the FE to handle config lifecycle */

    return fe;
}

/**
 * Create a front-end with default parameters for 16kHz audio.
 *
 * @return Initialized fe_t*, or NULL on failure
 */
fe_t *
pstrain_fe_create_default(void)
{
    return pstrain_fe_create(
        16000.0f,  /* samprate */
        25,        /* nfilt: SphinxTrain wideband training default */
        512,       /* nfft */
        130.0f,    /* lowerf */
        6800.0f,   /* upperf */
        13,        /* ncep */
        0.97f,     /* alpha */
        22         /* lifter */
    );
}
/* ====================================================================
 * Copyright (c) 1999-2016 Carnegie Mellon University.  All rights
 * reserved.
 *
 * This file is derived from CMU SphinxTrain/SphinxBase sources.
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
