#include "pstrain_param_cnt.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sphinxbase/cmd_ln.h>
#include <sphinxbase/err.h>

/* External function from param_cnt/main.c */
extern int param_cnt_run(void);

int pstrain_param_cnt(const char *mdef_path,
                  const char *dict_path,
                  const char *fdict_path,
                  const char *ctl_path,
                  const char *lsn_path,
                  const char *ts2cb_path,
                  const char *seg_dir,
                  const char *seg_ext,
                  const char *output_path,
                  int32 param_type,
                  uint32 n_skip,
                  int32 run_len,
                  uint32 part,
                  uint32 n_part)
{
    int ret;
    char n_skip_str[16], run_len_str[16], part_str[16], n_part_str[16];

    /* Validate required arguments */
    if (!mdef_path || !dict_path || !ctl_path || !lsn_path) {
        E_ERROR("NULL required path argument (mdef_path, dict_path, ctl_path, lsn_path)\n");
        return -1;
    }

    /* Map param_type enum to string */
    const char *param_type_str;
    if (param_type == PARAM_CNT_STATE) param_type_str = "state";
    else if (param_type == PARAM_CNT_CB) param_type_str = "cb";
    else if (param_type == PARAM_CNT_PHONE) param_type_str = "phone";
    else {
        E_ERROR("Invalid param_type: %d\n", param_type);
        return -1;
    }

    /* Convert numeric arguments to strings */
    snprintf(n_skip_str, sizeof(n_skip_str), "%u", n_skip);
    snprintf(run_len_str, sizeof(run_len_str), "%d", run_len);
    if (part > 0) snprintf(part_str, sizeof(part_str), "%u", part);
    if (n_part > 0) snprintf(n_part_str, sizeof(n_part_str), "%u", n_part);

    /* Define the command line arguments */
    static const arg_t defn[] = {
        { "-help", ARG_BOOLEAN, "no", "Shows the usage of the tool" },
        { "-example", ARG_BOOLEAN, "no", "Shows example of how to use the tool" },
        { "-moddeffn", ARG_STRING, NULL, "Model definition file" },
        { "-ts2cbfn", ARG_STRING, NULL, "Tied-state-to-codebook mapping file" },
        { "-ctlfn", ARG_STRING, NULL, "Control file of the training corpus" },
        { "-part", ARG_INT32, NULL, "Corpus part number (range 1..NPART)" },
        { "-npart", ARG_INT32, NULL, "Partition the corpus into this many equal sized subsets" },
        { "-nskip", ARG_INT32, NULL, "# of lines to skip in the control file" },
        { "-runlen", ARG_INT32, NULL, "# of lines to process in the control file (after any skip)" },
        { "-lsnfn", ARG_STRING, NULL, "All word transcripts for the training corpus" },
        { "-dictfn", ARG_STRING, NULL, "Dictionary for the content words" },
        { "-fdictfn", ARG_STRING, NULL, "Dictionary for the filler words" },
        { "-segdir", ARG_STRING, NULL, "Root directory of the training corpus state segmentation files" },
        { "-segext", ARG_STRING, "v8_seg", "Extension of the training corpus state segmentation files" },
        { "-paramtype", ARG_STRING, "state", "Parameter type to count {'state', 'cb', 'phone'}" },
        { "-outputfn", ARG_STRING, NULL, "If specified, write counts to this file" },
        { NULL, 0, NULL, NULL }
    };

    /* Construct argv for cmd_ln_parse */
    const char *argv[40];
    int argc = 0;

    argv[argc++] = "pstrain_param_cnt";
    argv[argc++] = "-moddeffn"; argv[argc++] = mdef_path;
    argv[argc++] = "-dictfn"; argv[argc++] = dict_path;
    argv[argc++] = "-ctlfn"; argv[argc++] = ctl_path;
    argv[argc++] = "-lsnfn"; argv[argc++] = lsn_path;
    argv[argc++] = "-paramtype"; argv[argc++] = param_type_str;

    if (fdict_path) { argv[argc++] = "-fdictfn"; argv[argc++] = fdict_path; }
    if (ts2cb_path) { argv[argc++] = "-ts2cbfn"; argv[argc++] = ts2cb_path; }
    if (seg_dir) { argv[argc++] = "-segdir"; argv[argc++] = seg_dir; }
    if (seg_ext) { argv[argc++] = "-segext"; argv[argc++] = seg_ext; }
    if (output_path) { argv[argc++] = "-outputfn"; argv[argc++] = output_path; }
    if (n_skip > 0) { argv[argc++] = "-nskip"; argv[argc++] = n_skip_str; }
    if (run_len >= 0) { argv[argc++] = "-runlen"; argv[argc++] = run_len_str; }
    if (part > 0) { argv[argc++] = "-part"; argv[argc++] = part_str; }
    if (n_part > 0) { argv[argc++] = "-npart"; argv[argc++] = n_part_str; }

    cmd_ln_parse(defn, argc, (char **)argv, FALSE);

    ret = param_cnt_run();

    if (ret != 0) {
        E_ERROR("param_cnt_run failed\n");
        return -1;
    }

    E_INFO("Parameter counting completed successfully\n");
    return 0;
}
