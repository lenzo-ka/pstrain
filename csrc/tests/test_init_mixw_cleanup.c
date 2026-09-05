/* Exercise successful init_mixw cleanup under the native sanitizers. */

#include <stdio.h>

int pstrain_init_mixw(const char *src_mdef_path,
                      const char *src_mixw_path,
                      const char *src_mean_path,
                      const char *src_var_path,
                      const char *src_tmat_path,
                      const char *dest_mdef_path,
                      const char *dest_mixw_path,
                      const char *dest_mean_path,
                      const char *dest_var_path,
                      const char *dest_tmat_path,
                      int continuous);

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,         \
                    __LINE__);                                              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

int
main(int argc, char *argv[])
{
    char mixw[1024], means[1024], variances[1024], tmat[1024];
    int i;

    CHECK(argc == 7, "expected model fixture paths and output directory");
    CHECK(snprintf(mixw, sizeof(mixw), "%s/mixture_weights", argv[6]) > 0,
          "format mixture-weight output path");
    CHECK(snprintf(means, sizeof(means), "%s/means", argv[6]) > 0,
          "format mean output path");
    CHECK(snprintf(variances, sizeof(variances), "%s/variances", argv[6]) > 0,
          "format variance output path");
    CHECK(snprintf(tmat, sizeof(tmat), "%s/transition_matrices", argv[6]) > 0,
          "format transition-matrix output path");

    for (i = 0; i < 5; ++i) {
        CHECK(pstrain_init_mixw(argv[1], argv[2], argv[3], argv[4], argv[5],
                                argv[1], mixw, means, variances, tmat, 1) == 0,
              "repeated initialization succeeds");
    }
    printf("PASS: repeated init_mixw calls release initialization tracking\n");
    return 0;
}
