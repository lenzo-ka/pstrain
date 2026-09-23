/*
 * The alignment dictionary drops a pronunciation that uses a phone the
 * acoustic model does not define.  When that was a word's unsuffixed
 * pronunciation and an alternative survives, the first surviving alternative
 * in file order becomes the word's base entry, as the training loader
 * resolves it, instead of ending the process.  Check that the unsuffixed
 * spelling reaches it, that later alternatives link to it, that the load
 * says which alternative is being used, and that dict_free releases
 * everything.
 *
 * Usage: test_align_dict_dropped_base <mdef> <scratch dir>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sphinxbase/err.h>

#include "dict.h"
#include "mdef.h"

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,          \
                    __LINE__);                                              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* AH, B and F are in every ARPABET inventory; OE is in none. */
static const char *DICT_TEXT =
    "good B AH F\n"
    "boeuf B OE F\n"
    "boeuf(2) B AH F\n"
    "boeuf(3) F AH B\n"
    "still F AH B\n";

static const char *FILLER_TEXT =
    "<s> SIL\n"
    "</s> SIL\n"
    "<sil> SIL\n";

static int
write_file(const char *path, const char *text)
{
    FILE *fp = fopen(path, "w");

    if (fp == NULL)
        return -1;
    fputs(text, fp);
    fclose(fp);
    return 0;
}

int
main(int argc, char *argv[])
{
    const char *dir;
    char dict_path[1024];
    char filler_path[1024];
    char log_path[1024];
    mdef_t *mdef;
    dict_t *d;
    s3wid_t base, second, third;
    FILE *fp;
    char *log_text;
    long log_size;

    CHECK(argc > 2, "usage: test_align_dict_dropped_base <mdef> <scratch dir>");
    dir = argv[2];

    snprintf(dict_path, sizeof(dict_path), "%s/dropped_base.dict", dir);
    snprintf(filler_path, sizeof(filler_path), "%s/dropped_base.filler", dir);
    snprintf(log_path, sizeof(log_path), "%s/dropped_base.log", dir);
    CHECK(write_file(dict_path, DICT_TEXT) == 0, "could not write the dictionary");
    CHECK(write_file(filler_path, FILLER_TEXT) == 0, "could not write the filler dictionary");

    mdef = mdef_init(argv[1], 0);
    CHECK(mdef != NULL, "could not read the model definition");

    /* Capture the library's diagnostics; stderr stays for failures and for
     * anything a sanitizer has to say. */
    fp = fopen(log_path, "w");
    CHECK(fp != NULL, "could not open the diagnostics file");
    err_set_logfp(fp);
    d = dict_init(mdef, dict_path, filler_path, 0, 0, 0, 0);
    err_set_logfp(stderr);
    fclose(fp);
    CHECK(d != NULL, "dict_init returned no dictionary");

    base = dict_wordid(d, "boeuf");
    second = dict_wordid(d, "boeuf(2)");
    third = dict_wordid(d, "boeuf(3)");
    CHECK(IS_S3WID(base), "the unsuffixed spelling must resolve");
    CHECK(IS_S3WID(second) && IS_S3WID(third), "both alternatives must load");
    CHECK(base == second, "the first surviving alternative is the word's base");
    CHECK(dict_basewid(d, second) == second, "the survivor is its own base");
    CHECK(dict_basewid(d, third) == second, "a later alternative links to the survivor");
    CHECK(dict_nextalt(d, second) == third, "the survivor's alternative chain holds the rest");
    CHECK(NOT_S3WID(dict_nextalt(d, third)), "the chain ends after the last alternative");
    CHECK(strcmp(dict_wordstr(d, base), "boeuf(2)") == 0,
          "the survivor keeps its own spelling");
    CHECK(d->word[base].pronlen == 3, "the survivor keeps its own pronunciation");
    CHECK(IS_S3WID(dict_wordid(d, "good")) && IS_S3WID(dict_wordid(d, "still")),
          "the rest of the dictionary loads");

    fp = fopen(log_path, "rb");
    CHECK(fp != NULL, "could not read the captured diagnostics");
    fseek(fp, 0, SEEK_END);
    log_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    log_text = malloc((size_t)log_size + 1);
    CHECK(log_text != NULL, "could not allocate the captured diagnostics");
    CHECK(fread(log_text, 1, (size_t)log_size, fp) == (size_t)log_size,
          "short read of the captured diagnostics");
    log_text[log_size] = '\0';
    fclose(fp);

    if (strstr(log_text, "Using 'boeuf(2)' as the pronunciation of 'boeuf'") == NULL) {
        fprintf(stderr, "FAIL: the load does not say which alternative is "
                        "used (%s:%d)\n", __FILE__, __LINE__);
        fputs(log_text, stderr);
        return 1;
    }

    free(log_text);
    dict_free(d);
    mdef_free(mdef);

    printf("PASS: %s\n", __FILE__);
    return 0;
}
