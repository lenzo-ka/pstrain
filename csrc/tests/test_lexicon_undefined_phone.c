/*
 * A pronunciation that uses a phone the acoustic model does not define is
 * dropped by lexicon_read.  Check that the entry really is dropped, that the
 * rest of the file still loads, and that the end-of-load summary states the
 * shortfall instead of reporting only what survived.
 *
 * Usage: test_lexicon_undefined_phone <mdef> <scratch dir>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <s3/acmod_set.h>
#include <s3/lexicon.h>
#include <s3/model_def_io.h>
#include <s3/s3.h>
#include <sphinxbase/err.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,          \
                    __LINE__);                                              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/* AH, B and F are in every ARPABET inventory; OE and Y2 are in none. */
static const char *DICT_TEXT =
    "GOOD B AH F\n"
    "BAD B OE F\n"
    "ALSOBAD OE Y2\n"
    "STILLGOOD F AH B\n";

int
main(int argc, char *argv[])
{
    const char *mdef_path;
    const char *dir;
    char dict_path[1024];
    char log_path[1024];
    model_def_t *mdef = NULL;
    lexicon_t *lex;
    FILE *fp;
    char *log_text;
    long log_size;

    CHECK(argc > 2, "usage: test_lexicon_undefined_phone <mdef> <scratch dir>");
    mdef_path = argv[1];
    dir = argv[2];

    snprintf(dict_path, sizeof(dict_path), "%s/undefined_phone.dict", dir);
    snprintf(log_path, sizeof(log_path), "%s/undefined_phone.log", dir);

    fp = fopen(dict_path, "w");
    CHECK(fp != NULL, "could not write the test dictionary");
    fputs(DICT_TEXT, fp);
    fclose(fp);

    CHECK(model_def_read(&mdef, mdef_path) == S3_SUCCESS,
          "could not read the model definition");

    /* Route only the library's own diagnostics into a file, so the summary
     * can be inspected while the process stderr stays free for this test's
     * failures and for anything a sanitizer has to say. */
    fp = fopen(log_path, "w");
    CHECK(fp != NULL, "could not open the diagnostics file");
    err_set_logfp(fp);
    lex = lexicon_read(NULL, dict_path, mdef->acmod_set);
    err_set_logfp(stderr);
    fclose(fp);

    CHECK(lex != NULL, "lexicon_read returned no lexicon");
    CHECK(lexicon_lookup(lex, "GOOD") != NULL, "GOOD should have loaded");
    CHECK(lexicon_lookup(lex, "STILLGOOD") != NULL, "STILLGOOD should have loaded");
    CHECK(lexicon_lookup(lex, "BAD") == NULL, "BAD uses an undefined phone");
    CHECK(lexicon_lookup(lex, "ALSOBAD") == NULL, "ALSOBAD uses undefined phones");

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

    if (strstr(log_text, "2 entries added from") == NULL
        || strstr(log_text, "2 pronunciations in") == NULL
        || strstr(log_text, "phones the model does not define") == NULL) {
        fprintf(stderr, "FAIL: the end-of-load summary does not state the "
                        "dropped pronunciations (%s:%d)\n", __FILE__, __LINE__);
        fputs(log_text, stderr);
        return 1;
    }

    free(log_text);
    lexicon_free(lex);
    model_def_free(mdef);

    printf("PASS: %s\n", __FILE__);
    return 0;
}
