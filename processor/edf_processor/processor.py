import traceback
import numpy as np
from base_processor.timeseries import BaseTimeSeriesProcessor
from base_processor.timeseries.base import chunks
import base_processor.timeseries.utils as utils
from edf_processor.edf import EdfReader

class EdfProcessor(BaseTimeSeriesProcessor):

    def task(self):
        file_path = self.inputs['file']

        try:
            with EdfReader(file_path) as edf_file:
                n_samples = edf_file.get_n_samples()
                start_time = utils.usecs_since_epoch(edf_file.get_start_datetime())

                for signal_number, signal_name in enumerate(edf_file.get_signal_labels()):
                    if signal_name == 'EDF Annotations':
                        continue

                    sample_rate = edf_file.get_sample_frequency(signal_number)
                    unit = edf_file.get_physical_dimension(signal_number)
                    nb_samples_per_record = edf_file.get_nr_samples(signal_number)

                    channel = self.get_or_create_channel(
                        name=str(signal_name).strip(),
                        unit=str(unit),
                        rate=sample_rate,
                        type='continuous'
                    )

                    timestamps = []
                    values = []

                    # if the EDF file is_discontiguous then it includes timestamp per values for each signal/record
                    if edf_file.is_discontiguous():
                        for index in range(edf_file.get_number_of_data_records()):
                            vals = edf_file.read_signal(signal_number, index * nb_samples_per_record, (index + 1) * nb_samples_per_record)
                            ts = edf_file.get_timestamps(index, signal_number, start_time)

                            timestamps.append(ts[:len(vals)])
                            values.append(vals)

                    # otherwise the data is contiguous and we need to create a
                    # set of corresponding timestamps per value using the
                    # sampling rate and start / end times
                    else:
                        nsamples = n_samples[signal_number]
                        length = (n_samples[signal_number] - 1) / sample_rate
                        end_time = int(start_time + length * 1e6)

                        # setting the chunk_size = nsamples should yield exactly one chunk
                        chunk = next(chunks(start_time, end_time, nsamples, nsamples))
                        vals = edf_file.read_signal(signal_number, chunk.start_index, chunk.end_index)

                        timestamps.append(chunk.timestamps[:len(vals)])
                        values.append(vals)

                    self.write_channel_data(
                        channel=channel,
                        timestamps=np.concatenate(timestamps),
                        values=np.concatenate(values)
                    )

        except Exception as e:
            print(traceback.format_exc())

        self.finalize()
