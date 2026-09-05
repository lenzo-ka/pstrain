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

    printf("PASS: byte swaps preserve integer, float, and v8_seg bits\n");
    return 0;
}
