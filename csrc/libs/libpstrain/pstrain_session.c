/* Per-request cleanup for the persistent native worker. */

#include <sphinxbase/cmd_ln.h>

static const arg_t session_probe_args[] = {
    { "-sessionprobe", ARG_BOOLEAN, "no", "Session reset test probe" },
    { NULL, 0, NULL, NULL }
};

void pstrain_dtree_session_reset(void);
void pstrain_kmeans_session_reset(void);

void
pstrain_session_reset(void)
{
    cmd_ln_free();
    pstrain_dtree_session_reset();
    pstrain_kmeans_session_reset();
}

int
pstrain_session_probe_set(void)
{
    char *argv[] = { "pstrain-session-probe", "-sessionprobe", "yes" };
    return cmd_ln_parse(session_probe_args, 3, argv, TRUE) < 0 ? -1 : 0;
}

int
pstrain_session_probe_is_set(void)
{
    return cmd_ln_exists("-sessionprobe") && cmd_ln_boolean("-sessionprobe");
}
