Configuration Reference
=======================

All configuration parameters for pstrain.


audio
-----

``audio.format``
   :Type: ``Literal``
   :Default: ``wav``
   :Description: Audio format

``audio.sample_rate``
   :Type: ``int``
   :Default: ``16000``
   :Description: Audio sample rate in Hz


corpus
------

``corpus.audio_dir``
   :Type: ``str | None``
   :Default: ``None``
   :Description: Audio directory (if None, uses project_dir/audio/)

``corpus.transcript_file``
   :Type: ``str | None``
   :Default: ``None``
   :Description: Transcript file (if None, uses etc/all.transcription)


description
-----------

``description``
   :Type: ``str | None``
   :Default: ``None``
   :Description: Experiment description


dictionary
----------

``dictionary.case_sensitive``
   :Type: ``bool``
   :Default: ``False``
   :Description: Case-sensitive word lookup

``dictionary.filler_dict``
   :Type: ``str``
   :Default: ``shared/filler.dict``
   :Description: Filler dictionary (sentence boundaries)

``dictionary.main_dict``
   :Type: ``str``
   :Default: ``shared/dictionary.dict``
   :Description: Main dictionary file (relative to project)

``dictionary.phoneset``
   :Type: ``str``
   :Default: ``shared/phoneset.txt``
   :Description: Phoneset file

``dictionary.silence_phone``
   :Type: ``str``
   :Default: ``SIL``
   :Description: Silence phone symbol


features
--------

``features.agc``
   :Type: ``Literal``
   :Default: ``none``
   :Description: Automatic Gain Control

``features.cmn``
   :Type: ``Literal``
   :Default: ``batch``
   :Description: Cepstral Mean Normalization

``features.delta``
   :Type: ``bool``
   :Default: ``True``
   :Description: Compute delta features

``features.delta_delta``
   :Type: ``bool``
   :Default: ``True``
   :Description: Compute delta-delta features

``features.feature_type``
   :Type: ``Literal``
   :Default: ``1s_c_d_dd``
   :Description: Sphinx feature stream type (1s_c_d_dd=continuous, s2_4x=semi-continuous)

``features.frame_length_ms``
   :Type: ``float``
   :Default: ``25.0``
   :Description: Frame length in milliseconds

``features.frame_shift_ms``
   :Type: ``float``
   :Default: ``10.0``
   :Description: Frame shift in milliseconds

``features.lifter``
   :Type: ``int``
   :Default: ``22``
   :Description: Liftering parameter (0=no liftering)

``features.lower_freq``
   :Type: ``float``
   :Default: ``130.0``
   :Description: Lower frequency cutoff (Hz) - 130 wideband, 200 telephone

``features.nfft``
   :Type: ``int``
   :Default: ``512``
   :Description: FFT size

``features.num_ceps``
   :Type: ``int``
   :Default: ``13``
   :Description: Number of cepstral coefficients

``features.num_filters``
   :Type: ``int``
   :Default: ``25``
   :Description: Number of mel filters (25 SphinxTrain wideband; 15 for 8 kHz telephone)

``features.preemphasis``
   :Type: ``float``
   :Default: ``0.97``
   :Description: Preemphasis coefficient

``features.transform``
   :Type: ``Literal``
   :Default: ``dct``
   :Description: Transform type

``features.type``
   :Type: ``Literal``
   :Default: ``mfcc``
   :Description: Feature type

``features.upper_freq``
   :Type: ``float``
   :Default: ``6800.0``
   :Description: Upper frequency cutoff (Hz) - 6800 wideband, 3500 telephone

``features.use_energy``
   :Type: ``bool``
   :Default: ``True``
   :Description: Include energy feature

``features.varnorm``
   :Type: ``bool``
   :Default: ``False``
   :Description: Variance normalization


name
----

``name``
   :Type: ``str | None``
   :Default: ``None``
   :Description: Experiment name


parallel
--------

``parallel.n_jobs``
   :Type: ``int``
   :Default: ``-1``
   :Description: Number of parallel jobs: 1=serial, -1=all cores minus 1, N=specific count

``parallel.show_progress``
   :Type: ``bool``
   :Default: ``True``
   :Description: Show progress bars during parallel execution


split
-----

``split.seed``
   :Type: ``int | None``
   :Default: ``None``
   :Description: Random seed for reproducible splits (None = random)

``split.train_ratio``
   :Type: ``float``
   :Default: ``0.9``
   :Description: Fraction for training (0.9 = 90% train, 10% test)


training
--------

``training.cd.tied.abeam``
   :Type: ``float``
   :Default: ``1e-90``
   :Description: Alpha beam

``training.cd.tied.bbeam``
   :Type: ``float``
   :Default: ``1e-10``
   :Description: Beta beam

``training.cd.tied.convergence_threshold``
   :Type: ``float``
   :Default: ``0.001``
   :Description: Maximum signed absolute change in average log likelihood per frame

``training.cd.tied.min_iterations``
   :Type: ``int``
   :Default: ``1``
   :Description: Minimum iterations

``training.cd.tied.mixw_floor``
   :Type: ``float``
   :Default: ``1e-05``
   :Description: Mixture weight floor

``training.cd.tied.n_gaussians``
   :Type: ``int``
   :Default: ``8``
   :Description: Number of Gaussians per state

``training.cd.tied.n_iterations``
   :Type: ``int``
   :Default: ``10``
   :Description: Maximum training iterations

``training.cd.tied.n_senones``
   :Type: ``int``
   :Default: ``200``
   :Description: Target number of senones (tied states)

``training.cd.tied.tmat_floor``
   :Type: ``float``
   :Default: ``1e-05``
   :Description: Transition probability floor

``training.cd.tied.topn``
   :Type: ``int``
   :Default: ``8``
   :Description: Number of top Gaussians

``training.cd.tied.varfloor``
   :Type: ``float``
   :Default: ``0.0001``
   :Description: Variance floor

``training.cd.untied.abeam``
   :Type: ``float``
   :Default: ``1e-90``
   :Description: Alpha beam

``training.cd.untied.bbeam``
   :Type: ``float``
   :Default: ``1e-10``
   :Description: Beta beam

``training.cd.untied.convergence_threshold``
   :Type: ``float``
   :Default: ``0.001``
   :Description: Maximum signed absolute change in average log likelihood per frame

``training.cd.untied.min_iterations``
   :Type: ``int``
   :Default: ``1``
   :Description: Minimum iterations

``training.cd.untied.mixw_floor``
   :Type: ``float``
   :Default: ``1e-05``
   :Description: Mixture weight floor

``training.cd.untied.n_gaussians``
   :Type: ``int``
   :Default: ``1``
   :Description: Number of Gaussians for untied models

``training.cd.untied.n_iterations``
   :Type: ``int``
   :Default: ``10``
   :Description: Maximum training iterations

``training.cd.untied.tmat_floor``
   :Type: ``float``
   :Default: ``0.0001``
   :Description: Transition probability floor

``training.cd.untied.topn``
   :Type: ``int``
   :Default: ``8``
   :Description: Number of top Gaussians

``training.cd.untied.varfloor``
   :Type: ``float``
   :Default: ``0.0001``
   :Description: Variance floor

``training.ci.abeam``
   :Type: ``float``
   :Default: ``1e-90``
   :Description: Alpha beam for BW forward pass

``training.ci.bbeam``
   :Type: ``float``
   :Default: ``1e-10``
   :Description: Beta beam for BW backward pass

``training.ci.convergence_threshold``
   :Type: ``float``
   :Default: ``0.001``
   :Description: Maximum signed absolute change in average log likelihood per frame

``training.ci.min_iterations``
   :Type: ``int``
   :Default: ``1``
   :Description: Minimum iterations before checking convergence

``training.ci.mixw_floor``
   :Type: ``float``
   :Default: ``1e-08``
   :Description: Mixture weight floor

``training.ci.n_gaussians``
   :Type: ``int``
   :Default: ``1``
   :Description: Initial number of Gaussians per state

``training.ci.n_iterations``
   :Type: ``int``
   :Default: ``10``
   :Description: Maximum training iterations

``training.ci.topn``
   :Type: ``int``
   :Default: ``1``
   :Description: Number of top Gaussians to use in BW

``training.ci.varfloor``
   :Type: ``float``
   :Default: ``0.0001``
   :Description: Variance floor

``training.gaussian_increment.enabled``
   :Type: ``bool``
   :Default: ``False``
   :Description: Enable Gaussian splitting

``training.gaussian_increment.n_iterations_after_split``
   :Type: ``int``
   :Default: ``10``
   :Description: Re-training iterations after split

``training.gaussian_increment.schedule``
   :Type: ``list``
   :Default: ``[1, 2, 4, 8]``
   :Description: Gaussian splitting schedule (powers of 2)

``training.n_states``
   :Type: ``int``
   :Default: ``3``
   :Description: Number of emitting states per HMM (3 or 5 typical)

``training.tree.max_depth``
   :Type: ``int``
   :Default: ``50``
   :Description: Maximum tree depth

``training.tree.min_observations``
   :Type: ``int``
   :Default: ``100``
   :Description: Minimum observations per leaf node

``training.tree.questions_file``
   :Type: ``pathlib.Path | None``
   :Default: ``None``
   :Description: Path to questions file (auto-generate if None)
