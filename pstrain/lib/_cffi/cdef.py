"""C type definitions for CFFI.

This module contains the CDEF string with all C type and function declarations
needed for the pstrainc library bindings.
"""

# Core type definitions for cffi
CDEF = """
// Basic types
typedef float float32;
typedef double float64;
typedef int int32;
typedef unsigned int uint32;
typedef short int16;
typedef unsigned short uint16;
typedef signed char int8;
typedef unsigned char uint8;
typedef float32 *vector_t;
typedef uint32 acmod_id_t;
typedef uint32 word_posn_t;
typedef void *glist_t;
typedef void *gnode_t;
typedef void *bitvec_t;
typedef float32 mfcc_t;

// Opaque pointers - in ABI mode we just use void*
// (The actual struct contents are unknown to cffi)
typedef void *cmd_ln_t;
typedef void *logmath_t;
typedef void *fe_t;
typedef void *feat_t;
typedef void *hash_table_t;
typedef void *hash_iter_t;
typedef void *hash_entry_t;
typedef void *listelem_alloc_t;
typedef void *yin_t;
typedef void *acmod_set_t;
typedef void *acmod_t;
typedef void *ci_acmod_t;
typedef void *lexicon_t;
typedef void *lex_entry_t;
typedef void *model_def_t;
typedef void *model_def_entry_t;
typedef void *model_inventory_t;
typedef void *dtree_t;
typedef void *dtree_node_t;
typedef void *pset_t;
typedef void *quest_t;
typedef void *comp_quest_t;
typedef void *s3lattice_t;
typedef void *s3phseg_t;
typedef void *mllr_reg_t;
typedef void *gauden_t;
typedef void *agc_t;
typedef void *cmn_t;
typedef void *fsg_model_t;
typedef void *fsg_link_t;
typedef void *fsg_arciter_t;
typedef void *ngram_model_t;
typedef void *ngram_iter_t;
typedef void *jsgf_t;
typedef void *jsgf_rule_t;
typedef void *kd_tree_node_t;
typedef void *heap_t;
typedef void *priority_queue_t;
typedef void *mmio_file_t;
typedef void *bit_encode_t;
typedef void *itree_t;
typedef void *arg_t;
typedef void *lineiter_t;
typedef void *logadd_t;
typedef void *ptmr_t;
typedef void *pctr_t;
typedef void *pstrain_decoder_t;

typedef struct {
    const char *hmm;
    const char *dict;
    const char *fdict;
    const char *lm;
    double beam;
    double wbeam;
    double lw;
    double wip;
    double pbeam;
    double lpbeam;
    double lponlybeam;
    double fwdflatbeam;
    double fwdflatwbeam;
    long pl_window;
    long samprate;
    const char *agc;
    const char *cmn;
    const char *cmninit;
    int varnorm;
} pstrain_decoder_config_t;

pstrain_decoder_t *pstrain_decoder_create(const pstrain_decoder_config_t *config);
int pstrain_decoder_start_utt(pstrain_decoder_t *decoder);
int pstrain_decoder_process_raw(pstrain_decoder_t *decoder,
                                const int16 *samples, size_t nsamples,
                                int no_search, int full_utt);
int pstrain_decoder_end_utt(pstrain_decoder_t *decoder);
const char *pstrain_decoder_hyp(pstrain_decoder_t *decoder);
const char *pstrain_decoder_config_str(pstrain_decoder_t *decoder, const char *name);
long pstrain_decoder_config_int(pstrain_decoder_t *decoder, const char *name);
void pstrain_decoder_free(pstrain_decoder_t *decoder);
const char *pstrain_pocketsphinx_version(void);
const char *pstrain_fp_contract_policy(void);

// Enum types (must come before functions that use them)
typedef enum { CMN_NONE, CMN_LIVE, CMN_BATCH } cmn_type_t;
typedef enum { AGC_NONE, AGC_MAX, AGC_EMAX, AGC_NOISE } agc_type_t;

// ============================================================
// CORE FILE I/O FUNCTIONS
// ============================================================

// Gaussian I/O (s3gau_io.h)
int s3gau_read(const char *fn,
               vector_t ****out,
               uint32 *out_n_mgau,
               uint32 *out_n_feat,
               uint32 *out_n_density,
               uint32 **out_veclen);

int s3gau_write(const char *fn,
                const vector_t ***out,
                uint32 n_mgau,
                uint32 n_feat,
                uint32 n_density,
                const uint32 *veclen);

int s3gau_read_full(const char *fn,
                    vector_t *****out,
                    uint32 *out_n_mgau,
                    uint32 *out_n_feat,
                    uint32 *out_n_density,
                    uint32 **out_veclen);

int s3gau_write_full(const char *fn,
                     const vector_t ****out,
                     uint32 n_mgau,
                     uint32 n_feat,
                     uint32 n_density,
                     const uint32 *veclen);

// Mixture weight I/O (s3mixw_io.h)
int s3mixw_read(const char *fn,
                float32 ****out_mixw,
                uint32 *out_n_mixw,
                uint32 *out_n_feat,
                uint32 *out_n_density);

int s3mixw_write(const char *fn,
                 float32 ***mixw,
                 uint32 n_mixw,
                 uint32 n_feat,
                 uint32 n_density);

// Gaussian density count I/O (for inc_comp)
int s3gaudnom_read(const char *fn,
                   float32 ****out_dnom,
                   uint32 *out_n_cb,
                   uint32 *out_n_feat,
                   uint32 *out_n_density);

int s3gaudnom_write(const char *fn,
                    float32 ***dnom,
                    uint32 n_cb,
                    uint32 n_feat,
                    uint32 n_density);

// Transition matrix I/O (s3tmat_io.h)
int s3tmat_read(const char *fn,
                float32 ****out_tmat,
                uint32 *out_n_tmat,
                uint32 *out_n_state);

int s3tmat_write(const char *fn,
                 float32 ***tmat,
                 uint32 n_tmat,
                 uint32 n_state);

// Model definition I/O (model_def_io.h)
int model_def_read(model_def_t **out_mdef, const char *fn);
int model_def_write(model_def_t *mdef, const char *fn);
void model_def_free(model_def_t *mdef);

// ============================================================
// FEATURE EXTRACTION (Front End)
// ============================================================

void pstrain_cffi_fe_start_stream(fe_t *fe);
int pstrain_cffi_fe_start_utt(fe_t *fe);
int pstrain_cffi_fe_process_frames(fe_t *fe,
                      int16 const **inout_spch,
                      size_t *inout_nsamps,
                      mfcc_t **buf_cep,
                      int32 *inout_nframes,
                      int32 *out_frameidx);
int pstrain_cffi_fe_end_utt(fe_t *fe, mfcc_t *out_cepvector, int32 *out_nframes);
int pstrain_cffi_fe_free(fe_t *fe);
int pstrain_cffi_fe_get_output_size(fe_t *fe);
int pstrain_cffi_fe_mfcc_to_float(fe_t *fe, mfcc_t **input,
                                 float32 **output, int32 nframes);

// pstrain helper functions - simplified FE initialization
fe_t *pstrain_fe_create(float samprate, int nfilt, int nfft,
                    float lowerf, float upperf, int ncep,
                    float alpha, int lifter, int dither, int remove_dc,
                    int remove_noise, const char *transform, int frate, float wlen);
fe_t *pstrain_fe_create_default(void);

// pstrain flat initialization
int pstrain_flat_tmat(const char *mdef_path,
                  const char *topo_path,
                  const char *output_path);
int pstrain_flat_mixw(uint32 n_tied_state,
                  uint32 n_stream,
                  uint32 n_density,
                  const char *output_path);
int pstrain_init_gau(const char *mdef_path,
                 const char *dict_path,
                 const char *filler_dict_path,
                 const char *feat_type,
                 int32 ceplen,
                 const char *ctl_path,
                 const char *cep_dir,
                 const char *cep_ext,
                 const char *lsn_path,
                 const char *seg_dir,
                 const char *seg_ext,
                 const char *accum_dir,
                 const char *mean_path);
int pstrain_norm_gau(const char *accum_dir,
                 const char *mean_path,
                 const char *var_path);

// pstrain Gaussian splitting (increase components)
int pstrain_inc_comp(const char *in_mean_path,
                 const char *in_var_path,
                 const char *in_mixw_path,
                 const char *dcount_path,
                 const char *out_mean_path,
                 const char *out_var_path,
                 const char *out_mixw_path,
                 uint32 n_inc);

// pstrain K-means clustering
float64 pstrain_kmeans(const float32 *observations,
                   uint32 n_obs,
                   uint32 veclen,
                   uint32 n_clusters,
                   uint32 max_iter,
                   float32 min_ratio,
                   float32 *out_centroids,
                   uint32 *out_labels);

int pstrain_kmeans_init(const float32 *features,
                    uint32 n_frames,
                    uint32 veclen,
                    uint32 n_density,
                    uint32 max_iter,
                    float32 min_ratio,
                    float32 *out_means,
                    float32 *out_vars,
                    float32 *out_weights);

// pstrain BW training context and config
typedef struct pstrain_bw_config_s {
    float64 a_beam;
    float64 b_beam;
    uint32 topn;
    float32 spthresh;
    float32 mixw_floor;
    float32 tmat_floor;
    int32 mixw_reest;
    int32 tmat_reest;
    int32 mean_reest;
    int32 var_reest;
    int32 pass2var;
    int32 unobserved_gaussian_policy;
} pstrain_bw_config_t;

typedef struct pstrain_bw_context_s pstrain_bw_context_t;

typedef struct state_s {
    uint32 mixw;
    uint32 cb;
    uint32 ci_cb;
    uint32 ci_mixw;
    uint32 n_prior;
    uint32 *prior_state;
    float32 *prior_tprob;
    uint32 n_next;
    uint32 *next_state;
    float32 *next_tprob;
    uint32 tmat;
    uint32 m_state;
    uint32 l_mixw;
    uint32 l_cb;
    uint32 l_ci_mixw;
    uint32 l_ci_cb;
    float32 *tacc;
    uint32 phn;
} state_t;

pstrain_bw_context_t *pstrain_bw_init(const char *mdef_path,
                              const char *means_path,
                              const char *vars_path,
                              const char *mixw_path,
                              const char *tmat_path,
                              const pstrain_bw_config_t *config);
void pstrain_bw_free(pstrain_bw_context_t *ctx);
int pstrain_bw_set_dict(pstrain_bw_context_t *ctx,
                    const char *dict_path,
                    const char *filler_dict_path);
int pstrain_bw_set_multipron(pstrain_bw_context_t *ctx, int enable);
float64 pstrain_bw_set_a_beam(pstrain_bw_context_t *ctx, float64 a_beam);
state_t *pstrain_bw_build_state_seq(pstrain_bw_context_t *ctx,
                                const char *transcript,
                                uint32 *n_state);
void pstrain_bw_free_state_seq(state_t *state_seq, uint32 n_state);
uint32 pstrain_bw_count_active_fallback_senones(pstrain_bw_context_t *ctx);
int pstrain_bw_fallback_senone_active(pstrain_bw_context_t *ctx, uint32 senone);
int pstrain_bw_process_utt_text(pstrain_bw_context_t *ctx,
                            const float *features,
                            uint32 n_frames,
                            const char *transcript);
int pstrain_bw_process_utt_mfcc(pstrain_bw_context_t *ctx,
                            const float *mfcc,
                            uint32 n_mfcc_frames,
                            const char *transcript);
int pstrain_bw_process_utt(pstrain_bw_context_t *ctx,
                       const float *features,
                       uint32 n_frames,
                       const uint32 *phone_ids,
                       uint32 n_phones);
int pstrain_bw_normalize(pstrain_bw_context_t *ctx);
int pstrain_bw_dump_accum(pstrain_bw_context_t *ctx, const char *accum_dir);
int pstrain_bw_restore_accumdirs(pstrain_bw_context_t *ctx,
                              const char *const *accum_dirs,
                              uint32 n_accum_dirs);
int pstrain_bw_save(pstrain_bw_context_t *ctx,
                const char *means_path,
                const char *vars_path,
                const char *mixw_path,
                const char *tmat_path);
void pstrain_bw_get_stats(pstrain_bw_context_t *ctx,
                      float64 *total_log_lik,
                      uint32 *total_frames,
                      uint32 *total_utts);
int pstrain_bw_save_counts(pstrain_bw_context_t *ctx, const char *counts_path);

// ============================================================
// LOG MATH
// ============================================================

logmath_t *pstrain_cffi_logmath_init(float64 base, int shift, int use_table);
void pstrain_cffi_logmath_free(logmath_t *lmath);
int32 pstrain_cffi_logmath_log(logmath_t *lmath, float64 p);
float64 pstrain_cffi_logmath_exp(logmath_t *lmath, int32 p);
int32 pstrain_cffi_logmath_add(logmath_t *lmath, int32 p, int32 q);
float64 pstrain_cffi_logmath_get_base(logmath_t *lmath);

// ============================================================
// MEMORY ALLOCATION (ckd_alloc)
// ============================================================

void pstrain_cffi_ckd_free(void *ptr);
void pstrain_cffi_ckd_free_3d(void *ptr);

// ============================================================
// ERROR HANDLING
// ============================================================

void pstrain_session_reset(void);
int pstrain_session_probe_set(void);
int pstrain_session_probe_is_set(void);

// ============================================================
// MDEF GENERATION (mk_mdef_gen)
// ============================================================

// Generate CI (context-independent) mdef from phone list
int pstrain_mdef_gen_ci(const char *phone_list_path,
                    const char *output_path,
                    uint32 n_state);

// Generate all-triphones mdef from dictionary
int pstrain_mdef_gen_alltriphones(const char *phone_list_path,
                              const char *dict_path,
                              const char *filler_dict_path,
                              const char *output_path,
                              uint32 n_state,
                              int32 ignore_wpos);

// Generate untied mdef from transcripts
int pstrain_mdef_gen_untied(const char *phone_list_path,
                        const char *dict_path,
                        const char *filler_dict_path,
                        const char *transcript_path,
                        const char *output_path,
                        uint32 n_state,
                        int32 ignore_wpos,
                        int32 inventory_policy,
                        int32 multipron,
                        const char **utterances,
                        uint32 n_utterances);

// Count triphones in transcripts
int pstrain_mdef_count_triphones(const char *phone_list_path,
                             const char *dict_path,
                             const char *filler_dict_path,
                             const char *transcript_path,
                             const char *output_path,
                             int32 ignore_wpos);

// ============================================================
// DECISION TREE FUNCTIONS (pstrain_dtree.h)
// ============================================================

// Read phone set (question) file
pset_t *pstrain_read_pset(const char *filename,
                      const char *mdef_path,
                      uint32 *out_n_pset);

// Free phone set array
void pstrain_free_pset(pset_t *pset, uint32 n_pset);

// Build decision tree for triphones
int pstrain_build_tree(const char *mdef_path,
                   const char *mixw_path,
                   const char *mean_path,
                   const char *var_path,
                   const char *pset_path,
                   const char *output_path,
                   const char *phone,
                   uint32 state,
                   int32 continuous,
                   uint32 ssplitmin,
                   uint32 ssplitmax,
                   float32 ssplitthr,
                   uint32 csplitmin,
                   uint32 csplitmax,
                   float32 csplitthr,
                   float32 mwfloor,
                   float32 varfloor,
                   float32 cntthresh,
                   float32 *stwt,
                   uint32 n_stwt,
                   int32 rotate_state_weights,
                   int32 directional_questions,
                   int32 allphones,
                   int32 intermediate_dumps);

// Tie states using decision trees
int pstrain_tie_states(const char *input_mdef_path,
                   const char *output_mdef_path,
                   const char *tree_dir,
                   const char *pset_path,
                   const char *phone,
                   int32 allphones);

// Generate phonetic questions
int pstrain_make_quests(const char *mdef_path,
                    const char *mixw_path,
                    const char *mean_path,
                    const char *var_path,
                    const char *output_path,
                    int32 continuous,
                    uint32 npermute,
                    uint32 quests_per_state,
                    float32 varfloor,
                    uint32 niter);

// Prune decision trees to target number of senones
int pstrain_prune_tree(const char *mdef_path,
                   const char *pset_path,
                   const char *input_tree_dir,
                   const char *output_tree_dir,
                   uint32 n_seno_target,
                   float32 min_occ,
                   int32 allphones);

int pstrain_init_mixw(const char *src_mdef_path,
                  const char *src_mixw_path,
                  const char *src_mean_path,
                  const char *src_var_path,
                  const char *src_tmat_path,
                  const char *dest_mdef_path,
                  const char *dest_mixw_path,
                  const char *dest_mean_path,
                  const char *dest_var_path,
                  const char *dest_tmat_path,
                  int32 continuous);

// ============================================================
// SEGMENT AGGREGATION (pstrain_agg_seg.h)
// ============================================================

// Segment type enum
#define PSTRAIN_SEGTYPE_ALL 0
#define PSTRAIN_SEGTYPE_ST 1
#define PSTRAIN_SEGTYPE_PHN 2

// Aggregate feature segments from training corpus
int pstrain_agg_seg(const char *mdef_path,
                const char *dict_path,
                const char *fdict_path,
                const char *ctl_path,
                const char *cep_dir,
                const char *cep_ext,
                const char *seg_dir,
                const char *seg_ext,
                const char *output_path,
                const char *index_path,
                const char *ts2cb_path,
                const char *cnt_path,
                int32 segtype,
                const char *feat_type,
                int32 ceplen,
                int32 stride,
                int32 cachesz);

// ============================================================
// PARAMETER COUNTING (pstrain_param_cnt.h)
// ============================================================

// Parameter count types
#define PARAM_CNT_STATE 0
#define PARAM_CNT_CB    1
#define PARAM_CNT_PHONE 2

// Count parameter occurrences in the training corpus
int pstrain_param_cnt(const char *mdef_path,
                  const char *dict_path,
                  const char *fdict_path,
                  const char *ctl_path,
                  const char *lsn_path,
                  const char *ts2cb_path,
                  const char *seg_dir,
                  const char *seg_ext,
                  const char *output_path,
                  int32 param_type,
                  uint32 n_skip,
                  int32 run_len,
                  uint32 part,
                  uint32 n_part);

// ============================================================
// TIED-STATE TO CODEBOOK MAPPING (ts2cb.h, s3ts2cb_io.h)
// ============================================================

// Create semi-continuous ts2cb mapping (all states -> codebook 0)
uint32 *semi_ts2cb(uint32 n_ts);

// Create continuous ts2cb mapping (identity: state i -> codebook i)
uint32 *cont_ts2cb(uint32 n_ts);

// Create phone-tied mixture ts2cb mapping
uint32 *ptm_ts2cb(model_def_t mdef);

// Read ts2cb mapping from file
int s3ts2cb_read(const char *fn,
                 uint32 **out_ts2cb,
                 uint32 *out_n_ts,
                 uint32 *out_n_cb);

// Write ts2cb mapping to file
int s3ts2cb_write(const char *fn,
                  uint32 *ts2cb,
                  uint32 n_ts,
                  uint32 n_cb);

// ============================================================
// MLLR (Maximum Likelihood Linear Regression) FUNCTIONS
// ============================================================

// mllr.h - MLLR computation and transform
int32 compute_mllr(float32 *****regl,
                   float32 ****regr,
                   const uint32 *veclen,
                   uint32 nclass,
                   uint32 nfeat,
                   uint32 mllr_mult,
                   uint32 mllr_add,
                   float32 *****A,
                   float32 ****B);

int32 mllr_transform_mean(vector_t ***mean,
                          vector_t ***var,
                          uint32 gau_begin,
                          uint32 n_mgau,
                          uint32 n_feat,
                          uint32 n_density,
                          const uint32 *veclen,
                          float32 ****A,
                          float32 ***B,
                          int32 *cb2mllr,
                          uint32 n_mllr_class);

// mllr_io.h - MLLR matrix I/O
int32 store_reg_mat(const char *regmatfn,
                    const uint32 *veclen,
                    uint32 n_class,
                    uint32 n_stream,
                    float32 ****A,
                    float32 ***B);

int32 read_reg_mat(const char *regmatfn,
                   uint32 **veclen,
                   uint32 *n_class,
                   uint32 *n_stream,
                   float32 *****A,
                   float32 ****B);

int32 free_mllr_A(float32 ****A,
                  uint32 n_class,
                  uint32 n_stream);

int32 free_mllr_B(float32 ***B,
                  uint32 n_class,
                  uint32 n_stream);

int32 free_mllr_reg(float32 *****regl,
                    float32 ****regr,
                    uint32 n_class,
                    uint32 n_stream);

// s3cb2mllr_io.h - Codebook to MLLR class mapping I/O
int s3cb2mllr_read(const char *fn,
                   int32 **out_cb2mllr,
                   uint32 *out_n_cb,
                   uint32 *out_n_mllr);

int s3cb2mllr_write(const char *fn,
                    int32 *cb2mllr,
                    uint32 n_cb,
                    uint32 n_mllr);

// ============================================================
// KD-TREE FUNCTIONS (pstrain_kdtree.h)
// ============================================================

// Build KD-trees for fast Gaussian selection
int pstrain_kdtree_build(const char *meanfn,
                     const char *varfn,
                     const char *outfn,
                     float32 threshold,
                     int32 depth,
                     int32 absolute);

// ============================================================
// DELETED INTERPOLATION (pstrain_delint.h)
// ============================================================

// Perform deleted interpolation to smooth mixture weights
int pstrain_delint(const char *moddeffn,
               const char *mixwfn,
               const char **accumdirs,
               float32 cilambda,
               int32 maxiter);

// ============================================================
// MAP ADAPTATION (pstrain_map_adapt.h)
// ============================================================

// Perform MAP adaptation of acoustic models
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

// ============================================================
// FORCED ALIGNMENT (pstrain_align.h)
// ============================================================

typedef struct pstrain_align_config_s {
    double  beam;
    int     insert_sil;
    int     compute_phones;
    int     compute_states;
    const char *feat_type;
    const char *cmn;
    const char *agc;
    int     varnorm;
    int     frate;
    int     lts_mismatch;
} pstrain_align_config_t;

void pstrain_align_config_default(pstrain_align_config_t *config);

typedef struct pstrain_align_context_s pstrain_align_context_t;

typedef struct pstrain_align_seg_s {
    const char *name;
    int32 start_frame;
    int32 end_frame;
    int32 score;
} pstrain_align_seg_t;

typedef struct pstrain_align_result_s {
    pstrain_align_seg_t *words;
    uint32 n_words;
    pstrain_align_seg_t *phones;
    uint32 n_phones;
    pstrain_align_seg_t *states;
    uint32 n_states;
    int32 total_score;
    int32 n_frames;
    void *_arena;
} pstrain_align_result_t;

pstrain_align_context_t *
pstrain_align_init(const char *mdef_path,
               const char *mean_path,
               const char *var_path,
               const char *mixw_path,
               const char *tmat_path,
               const char *feat_params_path,
               const char *dict_path,
               const char *fdict_path,
               const pstrain_align_config_t *config);

void pstrain_align_free(pstrain_align_context_t *ctx);

int
pstrain_align_mfcc(pstrain_align_context_t *ctx,
               const float *mfcc,
               uint32 n_frames,
               uint32 ncep,
               const char *transcript,
               const char *utt_id,
               pstrain_align_result_t **out_result);

int
pstrain_align_mfc_file(pstrain_align_context_t *ctx,
                   const char *mfc_path,
                   const char *transcript,
                   const char *utt_id,
                   pstrain_align_result_t **out_result);

void pstrain_align_result_free(pstrain_align_result_t *result);

const char *pstrain_align_last_error(void);

"""
