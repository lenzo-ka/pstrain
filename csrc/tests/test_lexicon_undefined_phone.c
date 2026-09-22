/*
 * A pronunciation that uses a phone the acoustic model does not define is
 * dropped by lexicon_read.  Check that the entry really is dropped, that the
 * rest of the file still loads, and that the end-of-load summary states the
 * shortfall instead of reporting only what survived.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <s3/acmod_set.h>
#include <s3/lexicon.h>
#include <s3/s3.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stdout, "FAIL: %s (%s:%d)\n", (msg), __FILE__,          \
                    __LINE__);                                              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static const char *DICT_TEXT =
    "GOOD B AH F\n"
    "BAD B OE F\n"
    "ALSOBAD OE Y2\n"
    "STILLGOOD F AH B\n";

int
main(int argc, char *argv[])
{
    static const char *attr[] = { "ci", "base", NULL };
    static const char *ci_phones[] = { "AH", "B", "F", "SIL" };
    const char *dir = (argc > 1) ? argv[1] : ".";
    char dict_path[1024];
    char log_path[1024];
    acmod_set_t *acmod_set;
    lexicon_t *lex;
    FILE *fp;
    char *log_text;
    long log_size;
    size_t i;

    snprintf(dict_path, sizeof(dict_path), "%s/undefined_phone.dict", dir);
    snprintf(log_path, sizeof(log_path), "%s/undefined_phone.log", dir);

    fp = fopen(dict_path, "w");
    CHECK(fp != NULL, "could not write the test dictionary");
    fputs(DICT_TEXT, fp);
    fclose(fp);

    acmod_set = acmod_set_new();
    acmod_set_set_n_ci_hint(acmod_set, 4);
    for (i = 0; i < sizeof(ci_phones) / sizeof(ci_phones[0]); ++i)
        acmod_set_add_ci(acmod_set, ci_phones[i], attr);

    /* Capture the loader's diagnostics so the summary can be inspected.
     * stderr stays redirected for the rest of the run; every check in this
     * test reports on stdout. */
    fflush(stderr);
    CHECK(freopen(log_path, "w", stderr) != NULL, "could not redirect stderr");
    lex = lexicon_read(NULL, dict_path, acmod_set);
    fflush(stderr);

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

    CHECK(strstr(log_text, "2 entries added from") != NULL,
          "the summary should still count what loaded");
    CHECK(strstr(log_text, "2 pronunciations in") != NULL,
          "the summary should count the dropped pronunciations");
    CHECK(strstr(log_text, "phones the model does not define") != NULL,
          "the summary should name the cause");

    free(log_text);
    lexicon_free(lex);

    printf("PASS: %s\n", __FILE__);
    return 0;
}
