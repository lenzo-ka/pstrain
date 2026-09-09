/* Exercise byte swaps directly and through the v8_seg array reader. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <s3/s3io.h>
#include <sphinxbase/byteorder.h>

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__,         \
                    __LINE__);                                              \
            return 1;                                                       \
        }                                                                   \
    } while (0)

/*
 * Bit patterns that a byte swap has to carry through untouched: signalling
 * and quiet NaNs, both infinities, both zeros, the smallest and largest
 * subnormals, and the four single-byte patterns that walk a set byte across
 * the word.  Stored big-endian, the way a Sphinx cepstrum file stores them.
 */
static const unsigned char float32_patterns[] = {
    0x3f, 0x80, 0x00, 0x00,     /* 1.0 */
    0x00, 0x00, 0x00, 0x00,     /* +0.0 */
    0x80, 0x00, 0x00, 0x00,     /* -0.0 */
    0x7f, 0x80, 0x00, 0x00,     /* +infinity */
    0xff, 0x80, 0x00, 0x00,     /* -infinity */
    0x7f, 0x80, 0x00, 0x01,     /* signalling NaN */
    0x7f, 0xc0, 0x00, 0x00,     /* quiet NaN */
    0xff, 0xc0, 0x00, 0x01,     /* negative quiet NaN */
    0x00, 0x00, 0x00, 0x01,     /* smallest subnormal */
    0x00, 0x7f, 0xff, 0xff,     /* largest subnormal */
    0x00, 0x00, 0x00, 0xff,
    0x00, 0x00, 0xff, 0x00,
    0x00, 0xff, 0x00, 0x00,
    0xff, 0x00, 0x00, 0x00,
    0x80, 0x00, 0x00, 0x01,
    0x7f, 0xff, 0xff, 0xff,
    0x01, 0x02, 0x03, 0x04,
    0xaa, 0x55, 0xaa, 0x55
};

static const unsigned char float64_patterns[] = {
    0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     /* 1.0 */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     /* +0.0 */
    0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     /* -0.0 */
    0x7f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     /* +infinity */
    0xff, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     /* -infinity */
    0x7f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,     /* signalling NaN */
    0x7f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,     /* quiet NaN */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,     /* smallest subnormal */
    0x00, 0x0f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,     /* largest subnormal */
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff,
    0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10
};

/*
 * Asymmetric patterns that are finite whichever way round the bytes are
 * read: swapping any of them changes the value, and neither the original
 * nor the swapped result is a NaN, so a plain float comparison of the two
 * is meaningful on a host of either endianness.  Used by the witness below.
 */
static const unsigned char float32_asymmetric[] = {
    0x3f, 0x80, 0x00, 0x00,
    0x01, 0x02, 0x03, 0x04,
    0x40, 0x49, 0x0f, 0xdb,
    0x3e, 0xaa, 0xaa, 0xab,
    0x41, 0x20, 0x00, 0x00,
    0x12, 0x34, 0x56, 0x78
};

#define MAX_FLOAT32 (sizeof(float32_patterns) / 4)
#define MAX_FLOAT64 (sizeof(float64_patterns) / 8)

/*
 * Load a raw byte image into typed storage the way the production callers
 * do -- fread straight into a float array -- so the swap below runs on
 * storage whose declared type is float, with nothing between the read and
 * the swap to hide the access from the optimizer.
 */
static int
load_raw(const char *path, const unsigned char *bytes, size_t nbytes,
         void *dest)
{
    FILE *fh = fopen(path, "wb");

    if (fh == NULL)
        return -1;
    if (fwrite(bytes, 1, nbytes, fh) != nbytes || fclose(fh) != 0)
        return -1;
    fh = fopen(path, "rb");
    if (fh == NULL)
        return -1;
    if (fread(dest, 1, nbytes, fh) != nbytes) {
        fclose(fh);
        return -1;
    }
    if (fclose(fh) != 0)
        return -1;
    return remove(path) == 0 ? 0 : -1;
}

/*
 * Swap in place across a whole array, the shape used by
 * feat_s2mfc_read_norm_pad(), sphinx_fe's output writer, and cepview.
 */
static void
swap_float32_array(float32 *values, size_t count)
{
    size_t i;

    for (i = 0; i < count; ++i)
        SWAP_FLOAT32(&values[i]);
}

static void
swap_float64_array(float64 *values, size_t count)
{
    size_t i;

    for (i = 0; i < count; ++i)
        SWAP_FLOAT64(&values[i]);
}

/*
 * Exercise the macros the way production does and check the bytes that come
 * out.  The earlier version of this test wrapped every SWAP_FLOAT call in a
 * pair of memcpy calls, which hid the access from the compiler entirely: it
 * could not have caught the type punning the macros used to do, and it could
 * not catch a future reintroduction either.  Here the storage is a plain
 * float array, read and written through float lvalues on both sides of the
 * swap, and the bytes are inspected through an unsigned char view -- which
 * is the one aliasing the standard always permits.
 */
static int
check_float_swaps(const char *dir)
{
    char path[1024];
    float32 mfc[MAX_FLOAT32];
    float64 dbl[MAX_FLOAT64];
    float32 original[sizeof(float32_asymmetric) / 4];
    const unsigned char *view;
    size_t i, j;

    CHECK(snprintf(path, sizeof(path), "%s/byteorder.raw", dir) > 0,
          "format raw path");

    /* float32: fread, swap the array, inspect the bytes. */
    CHECK(load_raw(path, float32_patterns, sizeof(float32_patterns), mfc) == 0,
          "load float32 patterns");
    swap_float32_array(mfc, MAX_FLOAT32);
    view = (const unsigned char *)mfc;
    for (i = 0; i < MAX_FLOAT32; ++i)
        for (j = 0; j < 4; ++j)
            CHECK(view[i * 4 + j] == float32_patterns[i * 4 + (3 - j)],
                  "float32 swap reverses every byte");

    /* Swapping again must restore the file image exactly. */
    swap_float32_array(mfc, MAX_FLOAT32);
    CHECK(memcmp(mfc, float32_patterns, sizeof(float32_patterns)) == 0,
          "float32 double swap restores every bit pattern");

    /* float64: same, over eight bytes. */
    CHECK(load_raw(path, float64_patterns, sizeof(float64_patterns), dbl) == 0,
          "load float64 patterns");
    swap_float64_array(dbl, MAX_FLOAT64);
    view = (const unsigned char *)dbl;
    for (i = 0; i < MAX_FLOAT64; ++i)
        for (j = 0; j < 8; ++j)
            CHECK(view[i * 8 + j] == float64_patterns[i * 8 + (7 - j)],
                  "float64 swap reverses every byte");
    swap_float64_array(dbl, MAX_FLOAT64);
    CHECK(memcmp(dbl, float64_patterns, sizeof(float64_patterns)) == 0,
          "float64 double swap restores every bit pattern");

    /*
     * The witness.  Both reads go through float lvalues with only the macro
     * between them.  If the macro reaches the storage through an integer
     * lvalue, a compiler applying -fstrict-aliasing is entitled to reuse the
     * value it loaded before the swap, and these comparisons come out equal.
     */
    CHECK(load_raw(path, float32_asymmetric, sizeof(float32_asymmetric), mfc)
          == 0, "load asymmetric float32 patterns");
    for (i = 0; i < sizeof(float32_asymmetric) / 4; ++i) {
        original[i] = mfc[i];
        SWAP_FLOAT32(&mfc[i]);
        CHECK(mfc[i] != original[i],
              "a float read after the swap must see the swapped bytes");
        SWAP_FLOAT32(&mfc[i]);
        CHECK(mfc[i] == original[i],
              "a float read after the second swap must see the original");
    }

    /*
     * awritefloat()'s shape: swap the array, hand the bytes to an opaque
     * consumer, swap it back, then keep using it as floats.
     */
    swap_float32_array(mfc, sizeof(float32_asymmetric) / 4);
    {
        FILE *fh = fopen(path, "wb");
        CHECK(fh != NULL, "open swapped output");
        CHECK(fwrite(mfc, 4, sizeof(float32_asymmetric) / 4, fh)
              == sizeof(float32_asymmetric) / 4, "write swapped output");
        CHECK(fclose(fh) == 0, "close swapped output");
        CHECK(remove(path) == 0, "remove swapped output");
    }
    swap_float32_array(mfc, sizeof(float32_asymmetric) / 4);
    for (i = 0; i < sizeof(float32_asymmetric) / 4; ++i)
        CHECK(mfc[i] == original[i],
              "swap out and back must leave the float values unchanged");

    return 0;
}

int
main(int argc, char *argv[])
{
    static const unsigned char contents[] = {
        0x00, 0x00, 0x00, 0x0a,
        0x00, 0x01, 0xff, 0xfe, 0x12, 0x34, 0x80, 0x01, 0x7f, 0xff,
        0x00, 0x00, 0xff, 0xff, 0x40, 0x00, 0xaa, 0x55, 0x01, 0x02
    };
    static const short expected[] = {
        1, -2, 0x1234, (short)0x8001, 0x7fff, 0, -1, 0x4000,
        (short)0xaa55, 0x0102
    };
    char path[1024];
    FILE *fh;
    short *frames = NULL;
    int length = 0;
    int16 value16;
    int32 value32;
    float32 value_float;
    float64 value_double;
    uint32_t float_bits;
    uint64_t double_bits;
    size_t i;

    CHECK(argc == 2, "expected temporary output directory");
    CHECK(snprintf(path, sizeof(path), "%s/byteorder.v8_seg", argv[1]) > 0,
          "format v8_seg path");
    fh = fopen(path, "wb");
    CHECK(fh != NULL, "open v8_seg fixture");
    CHECK(fwrite(contents, 1, sizeof(contents), fh) == sizeof(contents),
          "write v8_seg fixture");
    CHECK(fclose(fh) == 0, "close v8_seg fixture");

    CHECK(areadshort(path, &frames, &length) == 10, "read v8_seg fixture");
    CHECK(length == 10, "preserve v8_seg sample count");
    for (i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i)
        CHECK(frames[i] == expected[i], "swap v8_seg frame");
    free(frames);
    CHECK(remove(path) == 0, "remove v8_seg fixture");

    value16 = (int16)0x8001U;
    SWAP_INT16(&value16);
    CHECK((uint16_t)value16 == UINT16_C(0x0180), "swap 0x8001");
    value32 = (int32)0x80000001U;
    SWAP_INT32(&value32);
    CHECK((uint32_t)value32 == UINT32_C(0x01000080),
          "swap 0x80000001");
    value32 = (int32)0xff000000U;
    SWAP_INT32(&value32);
    CHECK((uint32_t)value32 == UINT32_C(0x000000ff),
          "swap 0xff000000");

    float_bits = UINT32_C(0x3f800000);
    memcpy(&value_float, &float_bits, sizeof(value_float));
    SWAP_FLOAT32(&value_float);
    memcpy(&float_bits, &value_float, sizeof(float_bits));
    CHECK(float_bits == UINT32_C(0x0000803f), "swap float bit pattern");

    double_bits = UINT64_C(0x3ff0000000000000);
    memcpy(&value_double, &double_bits, sizeof(value_double));
    SWAP_FLOAT64(&value_double);
    memcpy(&double_bits, &value_double, sizeof(double_bits));
    CHECK(double_bits == UINT64_C(0x000000000000f03f),
          "swap double bit pattern");

    if (check_float_swaps(argv[1]) != 0)
        return 1;

    printf("PASS: byte swaps preserve integer, float, and v8_seg bits\n");
    return 0;
}
