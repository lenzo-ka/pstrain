/*
 * The library aligner accepts utterances up to PSTRAIN_ALIGN_MAX_FRAMES
 * (32,768) frames, and every per-frame buffer on its path must hold that
 * many.  The aligner's per-frame score buffer used to hold only the
 * standalone program's 15,000, so any longer utterance wrote past the end
 * of a heap buffer on every frame after the 15,000th; under the sanitizers
 * that is an immediate failure here.
 *
 * Align a real utterance tiled end to end to just under the limit, through
 * both the matrix and the file entry points, and check that the alignment
 * covers every frame.  The fixture model is a flat snapshot, so where the
 * words land says nothing; what matters is that every frame is aligned.
 * Then check that one frame over the limit is refused with an error naming
 * the frame count and the limit, from both entry points, and that the
 * context still aligns afterwards.
 *
 * Usage: test_align_frame_limit <fixture dir> <scratch dir>
 *   <fixture dir> holds model/, dictionary.dict, filler.dict and
 *   arctic_a0257.mfc (tests/fixtures/multipron_final_state).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pstrain_align.h"

#define LIMIT 32768
#define NCEP 13

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            const char *e_ = pstrain_align_last_error();                    \
            fprintf(stderr, "FAIL: %s (%s:%d)%s%s\n", (msg), __FILE__,      \
                    __LINE__, e_ ? ": " : "", e_ ? e_ : "");                \
            return 1;                                                       \
        }                                                                   \
    } while (0)

static const char *SENTENCE = "they are coming ashore whoever they are";

static void
swap32(void *p)
{
    unsigned char *b = p, t;
    t = b[0]; b[0] = b[3]; b[3] = t;
    t = b[1]; b[1] = b[2]; b[2] = t;
}

/* Read a Sphinx cepstrum file (int32 float count, then floats), in either
 * byte order.  Returns the frame count, or -1. */
static long
read_mfc(const char *path, float **out)
{
    FILE *fp = fopen(path, "rb");
    int32 n;
    long size, i;
    int swap = 0;
    float *data;

    if (fp == NULL)
        return -1;
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (fread(&n, sizeof(n), 1, fp) != 1) {
        fclose(fp);
        return -1;
    }
    if ((long)n * 4 + 4 != size) {
        swap32(&n);
        swap = 1;
    }
    if ((long)n * 4 + 4 != size || n % NCEP != 0) {
        fclose(fp);
        return -1;
    }
    data = malloc((size_t)n * sizeof(float));
    if (data == NULL || fread(data, sizeof(float), (size_t)n, fp) != (size_t)n) {
        free(data);
        fclose(fp);
        return -1;
    }
    fclose(fp);
    if (swap)
        for (i = 0; i < n; i++)
            swap32(&data[i]);
    *out = data;
    return n / NCEP;
}

/* Write frames as a native-order cepstrum file. */
static int
write_mfc(const char *path, const float *data, long n_frames)
{
    FILE *fp = fopen(path, "wb");
    int32 n = (int32)(n_frames * NCEP);
    int ok;

    if (fp == NULL)
        return -1;
    ok = fwrite(&n, sizeof(n), 1, fp) == 1 &&
         fwrite(data, sizeof(float), (size_t)n, fp) == (size_t)n;
    return (fclose(fp) == 0 && ok) ? 0 : -1;
}

/* Tile `frames` of `src` end to end into a new buffer of `total` frames. */
static float *
tile(const float *src, long frames, long total)
{
    float *out = malloc((size_t)total * NCEP * sizeof(float));
    long f;

    if (out == NULL)
        return NULL;
    for (f = 0; f < total; f++)
        memcpy(out + f * NCEP, src + (f % frames) * NCEP, NCEP * sizeof(float));
    return out;
}

static int
check_aligned(const pstrain_align_result_t *r, long frames)
{
    CHECK(r != NULL, "no result");
    CHECK(r->n_frames == frames, "the result covers every frame");
    CHECK(r->n_words > 0, "the result has words");
    CHECK(r->words[0].start_frame == 0, "the alignment starts at the first frame");
    CHECK(r->words[r->n_words - 1].end_frame == frames - 1,
          "the alignment reaches the last frame");
    return 0;
}

int
main(int argc, char *argv[])
{
    char path[7][1024];
    const char *fixture, *scratch;
    pstrain_align_context_t *ctx;
    pstrain_align_result_t *r = NULL;
    float *utt, *longest, *over;
    char sent[256];
    long frames, total;
    const char *err;
    int rc;

    CHECK(argc > 2, "usage: test_align_frame_limit <fixture dir> <scratch dir>");
    fixture = argv[1];
    scratch = argv[2];
    snprintf(path[0], sizeof(path[0]), "%s/model/mdef", fixture);
    snprintf(path[1], sizeof(path[1]), "%s/model/means", fixture);
    snprintf(path[2], sizeof(path[2]), "%s/model/variances", fixture);
    snprintf(path[3], sizeof(path[3]), "%s/model/mixture_weights", fixture);
    snprintf(path[4], sizeof(path[4]), "%s/model/transition_matrices", fixture);
    snprintf(path[5], sizeof(path[5]), "%s/dictionary.dict", fixture);
    snprintf(path[6], sizeof(path[6]), "%s/filler.dict", fixture);

    {
        char mfc[1024];
        snprintf(mfc, sizeof(mfc), "%s/arctic_a0257.mfc", fixture);
        frames = read_mfc(mfc, &utt);
        CHECK(frames > 0, "could not read the fixture utterance");
    }

    /* The most whole copies that fit under the limit, well past the 15,000
     * frames the score buffer used to hold. */
    total = (LIMIT / frames) * frames;
    CHECK(total > 15000, "the tiled utterance must pass 15,000 frames");
    longest = tile(utt, frames, total);
    over = tile(utt, frames, LIMIT + 1);
    CHECK(longest && over, "out of memory");
    snprintf(sent, sizeof(sent), "%s", SENTENCE);

    ctx = pstrain_align_init(path[0], path[1], path[2], path[3], path[4],
                             NULL, path[5], path[6], NULL);
    CHECK(ctx != NULL, "pstrain_align_init failed");

    /* Just under the limit, from memory. */
    rc = pstrain_align_mfcc(ctx, longest, (uint32)total, NCEP, sent, "longest", &r);
    CHECK(rc == 0, "an utterance under the limit aligns from memory");
    if (check_aligned(r, total))
        return 1;
    pstrain_align_result_free(r);
    r = NULL;

    /* One frame over the limit, from memory: refused, naming the limit. */
    rc = pstrain_align_mfcc(ctx, over, LIMIT + 1, NCEP, sent, "over", &r);
    CHECK(rc != 0 && r == NULL, "one frame over the limit is refused");
    err = pstrain_align_last_error();
    CHECK(err && strstr(err, "utterance over has 32769 frames, more than the "
                             "32768 frames the aligner accepts"),
          "the refusal names the frame count and the limit");

    /* The same two from files. */
    {
        char longest_path[1024], over_path[1024];
        snprintf(longest_path, sizeof(longest_path), "%s/longest.mfc", scratch);
        snprintf(over_path, sizeof(over_path), "%s/over.mfc", scratch);
        CHECK(write_mfc(longest_path, longest, total) == 0, "could not write longest.mfc");
        CHECK(write_mfc(over_path, over, LIMIT + 1) == 0, "could not write over.mfc");

        rc = pstrain_align_mfc_file(ctx, over_path, sent, "over", &r);
        CHECK(rc != 0 && r == NULL, "one frame over the limit is refused from a file");
        err = pstrain_align_last_error();
        CHECK(err && strstr(err, "utterance over has 32769 frames, more than the "
                                 "32768 frames the aligner accepts"),
              "the file refusal names the frame count and the limit");

        rc = pstrain_align_mfc_file(ctx, longest_path, sent, "longest", &r);
        CHECK(rc == 0, "an utterance under the limit aligns from a file");
        if (check_aligned(r, total))
            return 1;
        pstrain_align_result_free(r);
        r = NULL;
    }

    pstrain_align_free(ctx);
    free(over);
    free(longest);
    free(utt);

    printf("PASS: %s (%ld frames aligned)\n", __FILE__, total);
    return 0;
}
