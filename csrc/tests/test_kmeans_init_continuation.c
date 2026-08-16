#include <s3/s3gau_io.h>
#include <s3/s3.h>

#include <sphinxbase/ckd_alloc.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#define make_dir(path) _mkdir(path)
#else
#include <sys/stat.h>
#define make_dir(path) mkdir(path, 0777)
#endif

static int
make_path(char *path, size_t size, const char *dir, const char *name)
{
    int n = snprintf(path, size, "%s/%s", dir, name);
    return n < 0 || (size_t)n >= size;
}

static int
write_header(FILE *fh)
{
    static const char header[] = "s3\n      endhdr\n";
    uint32_t byte_order = UINT32_C(0x11223344);

    return fwrite(header, 1, sizeof(header) - 1, fh) != sizeof(header) - 1 ||
           fwrite(&byte_order, sizeof(byte_order), 1, fh) != 1;
}

static int
write_u32(FILE *fh, uint32_t value)
{
    return fwrite(&value, sizeof(value), 1, fh) != 1;
}

static int
write_inputs(const char *dir)
{
    char path[4096];
    FILE *fh;
    uint32_t i;

    (void)make_dir(dir);
    if (make_path(path, sizeof(path), dir, "mdef"))
        return 1;
    fh = fopen(path, "w");
    if (fh == NULL)
        return 1;
    fputs("0.3\n"
          "1 n_base\n"
          "0 n_tri\n"
          "3 n_state_map\n"
          "2 n_tied_state\n"
          "2 n_tied_ci_state\n"
          "1 n_tied_tmat\n"
          "#base lft rt p attrib tmat ... state id's ...\n"
          "X - - - n/a 0 0 1 N\n",
          fh);
    if (fclose(fh) != 0)
        return 1;

    if (make_path(path, sizeof(path), dir, "obs.dump"))
        return 1;
    fh = fopen(path, "wb");
    if (fh == NULL || write_header(fh))
        return 1;
    for (i = 0; i < 1000; ++i) {
        float value = (float)i;
        if (fwrite(&value, sizeof(value), 1, fh) != 1)
            return 1;
    }
    for (i = 0; i < 1000; ++i) {
        float value = (float)(i + 2000);
        if (fwrite(&value, sizeof(value), 1, fh) != 1)
            return 1;
    }
    if (fclose(fh) != 0)
        return 1;

    if (make_path(path, sizeof(path), dir, "obs.idx"))
        return 1;
    fh = fopen(path, "wb");
    if (fh == NULL || write_header(fh) ||
        write_u32(fh, 2) ||    /* SEGDMP_TYPE_FEAT */
        write_u32(fh, 2) ||    /* two tied states */
        write_u32(fh, 1) ||    /* one frame per segment */
        write_u32(fh, 4) ||    /* one float per frame */
        write_u32(fh, 0) || write_u32(fh, 2) ||
        write_u32(fh, 1000) || write_u32(fh, 20) ||
        write_u32(fh, 1000) || write_u32(fh, 4020))
        return 1;
    return fclose(fh) != 0;
}

static int
check_means(const char *dir)
{
    char path[4096];
    vector_t ***means;
    uint32 n_state, n_stream, n_density, *veclen;
    float state2_first, state2_second;

    if (make_path(path, sizeof(path), dir, "means") ||
        s3gau_read(path, &means, &n_state, &n_stream, &n_density,
                   &veclen) != S3_SUCCESS) {
        fputs("failed to read kmeans_init means\n", stderr);
        return 1;
    }
    if (n_state != 2 || n_stream != 1 || n_density != 2 || veclen[0] != 1) {
        fputs("unexpected kmeans_init means dimensions\n", stderr);
        return 1;
    }

    state2_first = means[1][0][0][0];
    state2_second = means[1][0][1][0];
    ckd_free(veclen);
    ckd_free_3d((void ***)means);

    /* State 1 consumes draws 1-10.  Its successor must begin with draw 11
     * (0.691004395), not restart at draw 1 (0.396464765). */
    if (state2_first != 2741.5f || state2_second != 2241.5f) {
        fprintf(stderr,
                "state 2 did not continue the default RNG stream: "
                "means are %.9g, %.9g (reset signature: 2243, 2743)\n",
                state2_first, state2_second);
        return 1;
    }
    return 0;
}

int
main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s setup|check DIRECTORY\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "setup") == 0)
        return write_inputs(argv[2]);
    if (strcmp(argv[1], "check") == 0)
        return check_means(argv[2]);
    fprintf(stderr, "unknown action: %s\n", argv[1]);
    return 2;
}
