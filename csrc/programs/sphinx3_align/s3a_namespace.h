/**
 * @file s3a_namespace.h
 * @brief Prefix conflicting public symbols from the vendored sphinx3
 *        aligner so they don't collide with the SphinxTrain copies in
 *        libcommon/libclust/libmodinv when both end up in libst2c.
 *
 * The two code bases have functions that share names but very different
 * implementations (libcommon/vector.c is the SphinxTrain build-time
 * version; sphinx3_align/vector.c is the runtime decoder version). Both
 * are needed: SphinxTrain code uses its copy for clustering, the aligner
 * uses its copy for Gaussian evaluation.
 *
 * Force-included on every sphinx3_align translation unit via the
 * compiler-aware forced-include hook in csrc/CMakeLists.txt
 * in csrc/CMakeLists.txt.
 */

#ifndef _ST2_S3A_NAMESPACE_H_
#define _ST2_S3A_NAMESPACE_H_

#define vector_floor       s3a_vector_floor
#define vector_normalize   s3a_vector_normalize
#define vector_nz_floor    s3a_vector_nz_floor
#define vector_print       s3a_vector_print
#define vector_sum_norm    s3a_vector_sum_norm

#define free_kd_tree       s3a_free_kd_tree
#define read_kd_trees      s3a_read_kd_trees

#define gauden_free        s3a_gauden_free

#endif /* _ST2_S3A_NAMESPACE_H_ */
