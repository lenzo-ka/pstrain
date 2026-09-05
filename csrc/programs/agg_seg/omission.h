#ifndef AGG_SEG_OMISSION_H
#define AGG_SEG_OMISSION_H

#include <s3/s3.h>

typedef enum {
    AGG_OMIT_FEATURE_READ,
    AGG_OMIT_TOO_SHORT,
    AGG_OMIT_TRANSCRIPT_READ,
    AGG_OMIT_SEGMENTATION_READ,
    AGG_OMIT_SEGMENTATION_MISMATCH,
    AGG_OMIT_TRIPHONE_CONVERSION,
    AGG_OMIT_SEGMENT_GENERATION,
    AGG_OMIT_SENONE_SEQUENCE,
    AGG_OMIT_REASON_COUNT
} agg_omit_reason_t;

void agg_omission_reset(void);
void agg_omission_processed(void);
void agg_omission_record(agg_omit_reason_t reason);
uint32 agg_omission_count(void);
void agg_omission_report(void);

#endif
