import os
import glob
import pytest
import math
import numpy as np

# ------------------- test models -------------------

class ChannelTest(object):
    '''
    Test channel object
    '''
    def __init__(self, name, nsamples, rate, channel_type):
        self.name = name                # channel label
        self.nsamples = nsamples        # number of samples
        self.rate = rate                # sampling rate
        self.channel_type = channel_type              # "CONTINUOUS" -OR- "UNIT"


class TimeSeriesTest(object):
    '''
    Test Timeseries file
    '''
    def __init__(self, name, nchannels, nsamples=None, channels=None, rate=None, inputs=None, spikes=False, result=None, template=False):
        self.name      = name       # channel label
        self.nchannels = nchannels  # number of channels (all channels)
        self.nsamples  = nsamples   # number of samples (all channels)
        self.rate      = rate       # sampling rate (all channels)
        self.inputs    = inputs
        self.spikes    = spikes
        self.template  = template 
        self.result    = "pass" if result is None else result

        if channels is None:
            self.channels = self.channels_to_dict([
                ChannelTest('test-channel-{}'.format(i), nsamples, rate, 'CONTINUOUS')
                for i in range(nchannels)
            ])
        else:
            self.channels = self.channels_to_dict(channels)

    def channels_to_dict(self, channels):
        '''
        organizes a list of ChannelTest objects into a dictionary
        like this:

            { 'Fp1' : ChannelTest(),
              'Fp2' : ChannelTest()
              ...
            }
        This form makes it easy to index channels during tests.
        '''
        c_dict = dict()
        # if channels is a list
        if isinstance(channels, list):
            for channel in channels:
                c_dict[str(channel.name)] = channel
        # if channels is a single object
        elif isinstance(channels, ChannelTest):
            c_dict[str(channel.name)] = channel
        # if channels is none of the above
        else:
            return channels
        return c_dict

# ------------------- helper methods -------------------

def inferred_rate_test(channel):
    if channel.type == 'CONTINUOUS':
        inferred_rate = float(channel.num_values)/((channel.end-channel.start)/1e6)

        assert channel.rate == pytest.approx(inferred_rate, 0.01)

def sin_wav(t,amp,frequency):
    return amp * math.sin( 2 * math.pi * frequency * t)

def sin_wav_array(duration, sampling, amplitude, frequency):
    sin_wav_arr=[]
    for x in range(duration*int(sampling)):
        t=float( x) /sampling
        sin_wav_arr.append(sin_wav(t,amplitude,frequency))
    return sin_wav_arr

# ------------------- general setup and test functions -------------------

def number_channels_test(task_channels, num_channels):
    '''
    check number of channels
    '''
    assert len(task_channels) == num_channels


def number_samples_test(task, expected):
    ''' check number of values/bytes per channel '''
    if not expected.spikes:
        for channel in task.channels:
            num_bytes = os.stat(channel.data_file).st_size
            nsamples  = num_bytes / 8
            try:
                # channel-specific sample size
                exp_chan = expected.channels[str(channel.name)]
                if exp_chan.channel_type != 'UNIT':
                    assert nsamples           == exp_chan.nsamples
                    assert channel.num_values == exp_chan.nsamples
            except KeyError:
                # package-specific sample size
                assert nsamples           == expected.nsamples
                assert channel.num_values == expected.nsamples

def bin_test(task, result_files):
    for channel in task.channels:
        inferred_rate_test(channel)

# ------------------- methods for global tests -------------------

def exception_test(task, expected):
    '''
    failing test
    '''
    initialize(task)
    with pytest.raises(Exception):
        task.run()

def timeseries_test(task, expected):
    '''
    run tests
    '''
    if str(expected.result) == str("pass"):
        channels_test(task, expected)
    elif str(expected.result) == str("fail"):
        exception_test(task, expected)

# ------------------- methods for tests per channel -------------------

def rate_per_channel_test(task, expected):
    for channel in task.channels:
        try:
            # channel-specific rate
            assert pytest.approx(channel.rate, 0.01) == expected.channels[str(channel.name)].rate
        except KeyError:
            # package-specific rate
            assert pytest.approx(channel.rate, 0.01) == expected.rate

def sin_wave_test(task, expected):
    # for testing values, we need to create a test file with 2 channels of 15 seconds, 400 amplitude
    # one channel is "Sin 10Hz" and the other is "Sin 20Hz"
    # because of the transformation and encoding of the different file format, we allow for a 1% difference in value
    for channel in task.channels:
        file_values = np.fromfile(channel.data_file, dtype=np.float64)
        template_values = []
        try:
            rate = expected.channels[str(channel.name)].rate
        except KeyError:
            rate = expected.rate
        if str(channel.name).lower() == str("sin 10hz") or str(channel.name).lower() == str("sin_10hz"):
            template_values = sin_wav_array(15, rate, 400, 10)
        elif str(channel.name).lower() == str("sin 20hz") or str(channel.name).lower() == str("sin_20hz"):
            template_values = sin_wav_array(15, rate, 400, 20)
        np.testing.assert_allclose(template_values,file_values, rtol=0.01, atol=0.1)


def channels_test(task, ts_test):
    task.run()
    number_channels_test(task.channels, len(ts_test.channels))
    assert str(ts_test.name) == str(ts_test.name)
    number_samples_test(task, ts_test)
    rate_per_channel_test(task, ts_test)
    if ts_test.template:
        sin_wave_test(task, ts_test)
    # make sure output bin files and number of channels match
    result_files = glob.glob('*ts.bin')
    assert len(task.channels) == len(result_files)

    bin_test(task, result_files)

    # cleanup
    [os.remove(f) for f in result_files]
