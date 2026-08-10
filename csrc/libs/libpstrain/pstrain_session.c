/* Per-request cleanup for the persistent native worker. */

#include <sphinxbase/cmd_ln.h>

void pstrain_dtree_session_reset(void);
void pstrain_kmeans_session_reset(void);

void
pstrain_session_reset(void)
{
    cmd_ln_free();
    pstrain_dtree_session_reset();
    pstrain_kmeans_session_reset();
}
