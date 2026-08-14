#include <stdio.h>

#include "../libs/libpstrain/pstrain_align.h"
#include "../programs/sphinx3_align/tmat.h"
#include "../programs/sphinx3_align/mdef.h"
#include "../programs/sphinx3_align/dict.h"
#include "../programs/sphinx3_align/s3_align.h"

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,          \
                    __LINE__);                                              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int
main(int argc, char *argv[])
{
    pstrain_align_config_t config;
    pstrain_align_context_t *ctx;
    int32 initial, final;
    char transcript_off[] = "<s> a and </s>";
    char transcript_on[] = "<s> a and </s>";

    CHECK(argc == 8, "expected model and dictionary fixture paths");
    pstrain_align_config_default(&config);
    config.optional_boundary_silence = 0;
    ctx = pstrain_align_init(argv[1], argv[2], argv[3], argv[4], argv[5], NULL,
                             argv[6], argv[7], &config);
    CHECK(ctx != NULL, "initialize aligner fixture");

    CHECK(align_build_sent_hmm(transcript_off, 1, 0) == 0, "build off-mode DAG");
    CHECK(align_has_boundary_bypasses(&initial, &final) == 0, "inspect off-mode DAG");
    CHECK(!initial && !final, "off mode requires both boundary HMMs");
    align_destroy_sent_hmm();

    CHECK(align_build_sent_hmm(transcript_on, 1, 1) == 0, "build on-mode DAG");
    CHECK(align_has_boundary_bypasses(&initial, &final) == 0, "inspect on-mode DAG");
    CHECK(initial && final, "on mode permits both boundary bypasses");
    align_destroy_sent_hmm();

    pstrain_align_free(ctx);
    puts("PASS: aligner boundary SIL bypass is gated at both ends");
    return 0;
}
