from base_processor.timeseries.tests import TimeSeriesTest

# -----------------------------
# parameters for global tests
# -----------------------------

params_global = [
    TimeSeriesTest(
        name      = 'test',
        nchannels = 11,
        nsamples  = 120000,
        result    = 'pass',
        rate      = 200,
        inputs    = {
            'file': '/data/input/test.edf'
        }),
    # EDF+D
    TimeSeriesTest(
        name      = 'test2',
        nchannels = 43,
        nsamples  = 497400,
        result    = 'pass',
        rate      = 200,
        inputs    = {
            'file': '/data/input/103-002_EEG_01_17_2019.edf'
        }),
    TimeSeriesTest(
        name      = 'template',
        template  = True,
        nchannels = 2,
        nsamples  = 12000,
        result    = 'pass',
        rate      = 800,
        inputs    = {
            'file': '/data/input/sin_wave.edf'
        }),
]
