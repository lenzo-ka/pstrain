#ifndef PSTRAIN_MAP_ADAPT_H
#define PSTRAIN_MAP_ADAPT_H

#include <sphinxbase/prim_type.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Perform MAP (Maximum A Posteriori) adaptation of acoustic models.
 *
 * Given a baseline model and forward-backward statistics from adaptation
 * data, update the model parameters to maximize the a posteriori probability.
 *
 * @param meanfn Baseline Gaussian mean file (required)
 * @param varfn Baseline Gaussian variance file (required)
 * @param mixwfn Baseline mixture weight file (required for mixw adaptation)
 * @param tmatfn Baseline transition matrix file (required for tmat adaptation)
 * @param accumdirs NULL-terminated array of accumulator directories
 * @param mapmeanfn Output MAP mean file (required)
 * @param mapvarfn Output MAP variance file (NULL to skip)
 * @param mapmixwfn Output MAP mixture weight file (NULL to skip)
 * @param maptmatfn Output MAP transition matrix file (NULL to skip)
 * @param moddeffn Model definition file (required for tied-state models)
 * @param ts2cbfn Tied-state to codebook mapping file
 * @param tau Prior weight hyperparameter (default 10.0)
 * @param fixed_tau Use fixed tau value (1) or estimate from data (0)
 * @param bayes_mean Use Bayesian mean estimation (1) or MAP (0)
 * @param mwfloor Mixture weight floor (default 1e-5)
 * @param varfloor Variance floor (default 1e-5)
 * @param tpfloor Transition probability floor (default 1e-4)
 * @return 0 on success, non-zero on error
 */
int pstrain_map_adapt(const char *meanfn,
                  const char *varfn,
                  const char *mixwfn,
                  const char *tmatfn,
                  const char **accumdirs,
                  const char *mapmeanfn,
                  const char *mapvarfn,
                  const char *mapmixwfn,
                  const char *maptmatfn,
                  const char *moddeffn,
                  const char *ts2cbfn,
                  float32 tau,
                  int32 fixed_tau,
                  int32 bayes_mean,
                  float32 mwfloor,
                  float32 varfloor,
                  float32 tpfloor);

#ifdef __cplusplus
}
#endif

#endif /* PSTRAIN_MAP_ADAPT_H */
