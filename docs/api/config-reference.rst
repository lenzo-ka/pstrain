Configuration Reference
=======================

All configuration parameters for pstrain.


description
-----------

``description``
   :Type: ``str``
   :Default: ``''``
   :Description: Human-readable profile purpose


features
--------

``features.agc``
   :Type: ``str``
   :Default: ``'none'``
   :Description: Automatic gain-control mode

``features.alpha``
   :Type: ``float``
   :Default: ``0.97``
   :Description: Pre-emphasis coefficient

``features.cmn``
   :Type: ``str``
   :Default: ``'batch'``
   :Description: Cepstral mean-normalization mode

``features.cmninit``
   :Type: ``str``
   :Default: ``'40,3,-1'``
   :Description: Initial cepstral mean vector for live CMN

``features.dither``
   :Type: ``bool``
   :Default: ``True``
   :Description: Add half-bit dither to input audio

``features.feat_type``
   :Type: ``str``
   :Default: ``'1s_c_d_dd'``
   :Description: Sphinx feature stream type

``features.frate``
   :Type: ``int``
   :Default: ``100``
   :Description: Feature frame rate in Hz

``features.lifter``
   :Type: ``int``
   :Default: ``22``
   :Description: Cepstral lifter window

``features.lowerf``
   :Type: ``int``
   :Default: ``130``
   :Description: Lower filter-bank frequency in Hz

``features.ncep``
   :Type: ``int``
   :Default: ``13``
   :Description: Number of cepstral coefficients

``features.nfft``
   :Type: ``int``
   :Default: ``512``
   :Description: FFT size

``features.nfilt``
   :Type: ``int``
   :Default: ``25``
   :Description: Number of mel filters

``features.remove_dc``
   :Type: ``bool``
   :Default: ``True``
   :Description: Remove DC offset from each frame

``features.remove_noise``
   :Type: ``bool``
   :Default: ``True``
   :Description: Remove noise with spectral subtraction

``features.samprate``
   :Type: ``int``
   :Default: ``16000``
   :Description: Audio sample rate in Hz

``features.transform``
   :Type: ``str``
   :Default: ``'dct'``
   :Description: Filter-bank transform

``features.upperf``
   :Type: ``int``
   :Default: ``6800``
   :Description: Upper filter-bank frequency in Hz

``features.varnorm``
   :Type: ``str``
   :Default: ``'no'``
   :Description: Cepstral variance-normalization mode

``features.wlen``
   :Type: ``float``
   :Default: ``0.025625``
   :Description: Analysis window length in seconds


runner
------

``runner.jobs``
   :Type: ``int | None``
   :Default: ``None``
   :Description: Parallel workers; null means auto

``runner.nice``
   :Type: ``int``
   :Default: ``5``
   :Description: Worker niceness increment


split
-----

``split.seed``
   :Type: ``int``
   :Default: ``42``
   :Description: Deterministic split seed

``split.test_count``
   :Type: ``int | None``
   :Default: ``None``
   :Description: Fixed test utterance count; zero disables an additional holdout

``split.train_ratio``
   :Type: ``float | None``
   :Default: ``None``
   :Description: Training fraction


training
--------

``training.a_beam``
   :Type: ``float``
   :Default: ``1e-90``
   :Description: Forward alignment beam

``training.accept_arctic_a0587_known_skip``
   :Type: ``bool``
   :Default: ``False``
   :Description: Honor only the two ratified Arctic a0587 stage/pass skip events

``training.arctic_a0302_zero_codebook_band``
   :Type: ``tuple[int, int] | None``
   :Default: ``None``
   :Description: Accepted inclusive exact-zero codebook occupancy band for the singular Arctic a0302 terminal-alignment exception

``training.b_beam``
   :Type: ``float``
   :Default: ``1e-10``
   :Description: Backward alignment beam

``training.ci.convergence_ratio``
   :Type: ``float``
   :Default: ``0.001``
   :Description: Absolute likelihood-delta convergence threshold

``training.ci.max_iterations``
   :Type: ``int``
   :Default: ``10``
   :Description: Maximum training passes

``training.ci.min_iterations``
   :Type: ``int``
   :Default: ``1``
   :Description: Minimum training passes

``training.exclusion_schedule``
   :Type: ``dict``
   :Default: ``{}``
   :Description: Experimental stage/pass utterance exclusions

``training.failed_alignment``
   :Type: ``recover | abort | omit``
   :Default: ``'recover'``
   :Description: Action when Baum-Welch alignment fails; ``omit`` reports and excludes the utterance while continuing

``training.max_skip_fraction``
   :Type: ``float``
   :Default: ``0.05``
   :Description: Maximum skipped-update fraction

``training.multipron_training``
   :Type: ``bool``
   :Default: ``True``
   :Description: Sum posteriors over pronunciation variants

``training.n_senones``
   :Type: ``int``
   :Default: ``200``
   :Description: Target tied-state count

``training.n_state``
   :Type: ``int``
   :Default: ``3``
   :Description: Emitting states per HMM

``training.optional_final_silence``
   :Type: ``bool``
   :Default: ``True``
   :Description: Permit final transcript silence to consume zero frames

``training.question_niter``
   :Type: ``int``
   :Default: ``1``
   :Description: Question generation iterations

``training.question_npermute``
   :Type: ``int``
   :Default: ``12``
   :Description: Question permutations

``training.question_quests_per_state``
   :Type: ``int``
   :Default: ``20``
   :Description: Questions generated per state

``training.retry_beam_factor``
   :Type: ``float``
   :Default: ``10000000000.0``
   :Description: Beam widening factor for one retry

``training.tied.convergence_ratio``
   :Type: ``float``
   :Default: ``0.001``
   :Description: Absolute likelihood-delta convergence threshold

``training.tied.max_iterations``
   :Type: ``int``
   :Default: ``10``
   :Description: Maximum training passes

``training.tied.min_iterations``
   :Type: ``int``
   :Default: ``1``
   :Description: Minimum training passes

``training.tree_csplitmax``
   :Type: ``int``
   :Default: ``2000``
   :Description: Maximum phone-context splits

``training.tree_csplitthr``
   :Type: ``float``
   :Default: ``0.0``
   :Description: Phone-context split threshold

``training.tree_directional_questions``
   :Type: ``bool``
   :Default: ``True``
   :Description: Honor _L/_R tree-question suffixes; disable only for isolation measurements

``training.tree_intermediate_dumps``
   :Type: ``bool``
   :Default: ``False``
   :Description: Dump intermediate decision trees to worker diagnostics

``training.tree_mwfloor``
   :Type: ``float``
   :Default: ``1e-08``
   :Description: Tree mixture-weight floor

``training.tree_rotate_state_weights``
   :Type: ``bool``
   :Default: ``True``
   :Description: Apply target-relative tree state weights; disable only for isolation measurements

``training.tree_ssplitmax``
   :Type: ``int``
   :Default: ``7``
   :Description: Maximum state splits

``training.tree_ssplitthr``
   :Type: ``float``
   :Default: ``0.0``
   :Description: State split threshold

``training.tree_state_weights``
   :Type: ``tuple``
   :Default: ``(1.0, 0.05, 0.0)``
   :Description: Decision-tree state weights

``training.untied.convergence_ratio``
   :Type: ``float``
   :Default: ``0.001``
   :Description: Absolute likelihood-delta convergence threshold

``training.untied.max_iterations``
   :Type: ``int``
   :Default: ``10``
   :Description: Maximum training passes

``training.untied.min_iterations``
   :Type: ``int``
   :Default: ``1``
   :Description: Minimum training passes

``training.untied_inventory``
   :Type: ``all-triphone | transcript-reachable | linear``
   :Default: ``'all-triphone'``
   :Description: Untied-model phone inventory policy
