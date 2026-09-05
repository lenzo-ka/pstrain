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
 * @file pstrain_agg_seg.c
 * @brief CFFI-friendly wrapper for segment aggregation.
 */

#include "pstrain_agg_seg.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sphinxbase/cmd_ln.h>
#include <sphinxbase/err.h>
#include <sphinxbase/feat.h>

/* External function from agg_seg/main.c */
extern int agg_seg_run(void);

int
pstrain_agg_seg(const char *mdef_path,
            const char *dict_path,
            const char *fdict_path,
            const char *ctl_path,
            const char *cep_dir,
            const char *cep_ext,
            const char *seg_dir,
            const char *seg_ext,
            const char *output_path,
            const char *index_path,
            const char *ts2cb_path,
            const char *cnt_path,
            int32 segtype,
            const char *feat_type,
            int32 ceplen,
            int32 stride,
            int32 cachesz)
{
    char stride_str[16], cachesz_str[16], ceplen_str[16];
    const char *segtype_str;
    int ret;

    if (!ctl_path || !cep_dir || !output_path) {
        E_ERROR("NULL required path argument\n");
        return -1;
    }

    /* Convert segtype enum to string */
    switch (segtype) {
        case PSTRAIN_SEGTYPE_ALL:
            segtype_str = "all";
            break;
        case PSTRAIN_SEGTYPE_ST:
            segtype_str = "st";
            if (!mdef_path || !ts2cb_path) {
                E_ERROR("segtype 'st' requires mdef_path and ts2cb_path\n");
                return -1;
            }
            break;
        case PSTRAIN_SEGTYPE_PHN:
            segtype_str = "phn";
            if (!mdef_path || !dict_path) {
                E_ERROR("segtype 'phn' requires mdef_path and dict_path\n");
                return -1;
            }
            break;
        default:
            E_ERROR("Invalid segtype: %d\n", segtype);
            return -1;
    }

    /* Set up string arguments */
    snprintf(stride_str, sizeof(stride_str), "%d", stride > 0 ? stride : 1);
    snprintf(cachesz_str, sizeof(cachesz_str), "%d", cachesz > 0 ? cachesz : 200);
    snprintf(ceplen_str, sizeof(ceplen_str), "%d", ceplen > 0 ? ceplen : 13);

    /* Build command line */
    static const arg_t args[] = {
        { "-moddeffn", ARG_STRING, NULL, "Model definition file" },
        { "-dictfn", ARG_STRING, NULL, "Dictionary file" },
        { "-fdictfn", ARG_STRING, NULL, "Filler dictionary file" },
        { "-ctlfn", ARG_STRING, NULL, "Control file" },
        { "-cepdir", ARG_STRING, NULL, "Cepstrum directory" },
        { "-cepext", ARG_STRING, "mfc", "Cepstrum extension" },
        { "-segdir", ARG_STRING, NULL, "Segmentation directory" },
        { "-segext", ARG_STRING, "v8_seg", "Segmentation extension" },
        { "-segdmpfn", ARG_STRING, NULL, "Output dump file" },
        { "-segidxfn", ARG_STRING, NULL, "Index file" },
        { "-segdmpdirs", ARG_STRING_LIST, NULL, "Dump directories" },
        { "-ts2cbfn", ARG_STRING, NULL, "TS2CB file" },
        { "-cntfn", ARG_STRING, NULL, "Count file" },
        { "-segtype", ARG_STRING, "st", "Segment type" },
        { "-feat", ARG_STRING, "1s_c_d_dd", "Feature type" },
        { "-ceplen", ARG_INT32, "13", "Cepstrum length" },
        { "-stride", ARG_INT32, "1", "Frame stride" },
        { "-cachesz", ARG_INT32, "200", "Cache size in MB" },
        { "-allowomit", ARG_BOOLEAN, "yes", "Allow omitted utterances" },
        /* Keep agg_seg's inherited live default: cmn_live subtracts the
         * running mean that existed before the current frames, then updates
         * its accumulated/windowed mean.  The shipped CFFI aggregation API
         * reaches this path but exposes no CMN argument, while the standalone
         * agg_seg command can override -cmn.  No comparative evidence is
         * known for this default; in particular, the stage-2 TIMIT result was
         * for forced alignment's batch/current aliases, not aggregation or
         * live CMN.  Its effect on aggregated training data, later models,
         * decoding, and other corpora or conditions is not established. */
        { "-cmn", ARG_STRING, "live", "CMN type" },
        { "-varnorm", ARG_BOOLEAN, "no", "Variance normalization" },
        { "-agc", ARG_STRING, "none", "AGC type" },
        { "-lda", ARG_STRING, NULL, "LDA file" },
        { "-ldadim", ARG_INT32, "0", "LDA dimension" },
        { "-svspec", ARG_STRING, NULL, "Subvector spec" },
        { "-lsnfn", ARG_STRING, NULL, "LSN file" },
        { "-sentdir", ARG_STRING, NULL, "Sentence directory" },
        { "-sentext", ARG_STRING, NULL, "Sentence extension" },
        { "-mllrctlfn", ARG_STRING, NULL, "MLLR control file" },
        { "-mllrdir", ARG_STRING, NULL, "MLLR directory" },
        { "-cb2mllrfn", ARG_STRING, ".1cls.", "CB to MLLR file" },
        { "-nskip", ARG_INT32, "0", "Skip utterances" },
        { "-runlen", ARG_INT32, "-1", "Run length" },
        { "-part", ARG_INT32, NULL, "Partition" },
        { "-npart", ARG_INT32, NULL, "Number of partitions" },
        { "-help", ARG_BOOLEAN, "no", "Help" },
        { "-example", ARG_BOOLEAN, "no", "Example" },
        { NULL, 0, NULL, NULL }
    };

    int argc = 0;
    const char *argv[64];
    argv[argc++] = "pstrain_agg_seg";

    if (mdef_path) {
        argv[argc++] = "-moddeffn";
        argv[argc++] = mdef_path;
    }
    if (dict_path) {
        argv[argc++] = "-dictfn";
        argv[argc++] = dict_path;
    }
    if (fdict_path) {
        argv[argc++] = "-fdictfn";
        argv[argc++] = fdict_path;
    }
    argv[argc++] = "-ctlfn";
    argv[argc++] = ctl_path;
    argv[argc++] = "-cepdir";
    argv[argc++] = cep_dir;
    if (cep_ext) {
        argv[argc++] = "-cepext";
        argv[argc++] = cep_ext;
    }
    if (seg_dir) {
        argv[argc++] = "-segdir";
        argv[argc++] = seg_dir;
    }
    if (seg_ext) {
        argv[argc++] = "-segext";
        argv[argc++] = seg_ext;
    }
    argv[argc++] = "-segdmpfn";
    argv[argc++] = output_path;
    if (index_path) {
        argv[argc++] = "-segidxfn";
        argv[argc++] = index_path;
    }
    if (ts2cb_path) {
        argv[argc++] = "-ts2cbfn";
        argv[argc++] = ts2cb_path;
    }
    if (cnt_path) {
        argv[argc++] = "-cntfn";
        argv[argc++] = cnt_path;
    }
    argv[argc++] = "-segtype";
    argv[argc++] = segtype_str;
    if (feat_type) {
        argv[argc++] = "-feat";
        argv[argc++] = feat_type;
    }
    argv[argc++] = "-ceplen";
    argv[argc++] = ceplen_str;
    argv[argc++] = "-stride";
    argv[argc++] = stride_str;
    argv[argc++] = "-cachesz";
    argv[argc++] = cachesz_str;

    cmd_ln_parse(args, argc, (char **)argv, TRUE);

    ret = agg_seg_run();

    if (ret != 0) {
        E_ERROR("agg_seg_run failed\n");
        return -1;
    }

    E_INFO("Segment aggregation completed: %s\n", output_path);
    return 0;
}
