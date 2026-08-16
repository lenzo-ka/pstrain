/**
 * pstrain_fe.h - Simplified FE initialization for pstrain
 *
 * Provides a simple interface to create a front-end without needing
 * to go through the cmd_ln infrastructure.
 */

#ifndef PSTRAIN_FE_H
#define PSTRAIN_FE_H

#include <sphinxbase/fe.h>

/**
 * Create a front-end with explicit parameters.
 *
 * This bypasses the cmd_ln parsing and creates an FE directly with
 * the given parameters.
 *
 * @param samprate Sample rate in Hz (e.g., 16000)
 * @param nfilt Number of mel filters (e.g., 25 for SphinxTrain wideband)
 * @param nfft FFT size (e.g., 512)
 * @param lowerf Lower frequency bound (e.g., 130)
 * @param upperf Upper frequency bound (e.g., 6800)
 * @param ncep Number of cepstral coefficients (e.g., 13)
 * @param alpha Pre-emphasis coefficient (e.g., 0.97)
 * @param lifter Liftering coefficient (e.g., 22)
 * @param dither Whether to add half-bit dither
 * @param remove_dc Whether to remove DC offset from each frame
 * @return Initialized fe_t*, or NULL on failure
 */
fe_t *pstrain_fe_create(float samprate, int nfilt, int nfft,
                        float lowerf, float upperf, int ncep,
                        float alpha, int lifter, int dither, int seed, int remove_dc,
                        int remove_noise, const char *transform, int frate, float wlen);

/**
 * Create a front-end with default parameters for 16kHz audio.
 *
 * @return Initialized fe_t*, or NULL on failure
 */
fe_t *pstrain_fe_create_default(void);

#endif /* PSTRAIN_FE_H */
