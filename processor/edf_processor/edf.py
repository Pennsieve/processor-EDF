import struct
import re
import numpy as np
from datetime import datetime

SECOND_TO_USECOND = 1000000

def twos_comp(val, bits):
    """Compute the 2's complement of int value val."""
    if (val & (1 << (bits - 1))) != 0:  # If sign bit is set, e.g., 8bit: 128-255
        val -= (1 << bits)              # Compute negative value
    return val                          # Return positive value as is

def transform_value(val, phys_min, phys_max, dig_min, dig_max, bits):
    two_comped_val = twos_comp(val, bits)
    bit_value = (phys_max - phys_min) / (dig_max - dig_min)
    offset = (phys_max / bit_value) - dig_max
    return bit_value * (offset + float(two_comped_val))

class EdfReader:
    """
    Simple interface for EDF, EDF+C, and EDF+D files.
    """

    def __init__ (self, filename):
        self.open(filename)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        return

    def open(self, filename):
        with open(filename, "rb") as f:
            self.version = f.read(8).strip().decode('utf-8', 'ignore')
            self.patient_id = f.read(80).strip().decode('utf-8', 'ignore')
            self.record_id = f.read(80).strip().decode('utf-8', 'ignore')
            self.start_date = f.read(8).strip().decode('utf-8', 'ignore')
            self.start_time = f.read(8).strip().decode('utf-8', 'ignore')
            self.nb_bytes = int(f.read(8))
            self.reserved = f.read(44).strip()
            self.nb_data_rec = int(f.read(8))
            self.duration = float(f.read(8))
            self.nb_signal = int(f.read(4))
            self.labels = [y.strip().decode('utf-8', 'ignore') for y in re.findall(b'.{1,16}', f.read(16 * self.nb_signal))]
            self.transducer_type = [y.strip().decode('utf-8', 'ignore') for y in re.findall(b'.{1,80}', f.read(80 * self.nb_signal))]
            self.phy_dim = [y.strip().decode('utf-8', 'ignore') for y in re.findall(b'.{1,8}', f.read(8 * self.nb_signal))]
            self.phy_min = [float(y) for y in re.findall(b'.{1,8}', f.read(8 * self.nb_signal))]
            self.phy_max = [float(y) for y in re.findall(b'.{1,8}', f.read(8 * self.nb_signal))]
            self.dig_min = [float(y) for y in re.findall(b'.{1,8}', f.read(8 * self.nb_signal))]
            self.dig_max = [float(y) for y in re.findall(b'.{1,8}', f.read(8 * self.nb_signal))]
            self.prefiltering = [y.strip().decode('utf-8', 'ignore') for y in re.findall(b'.{1,80}', f.read(80 * self.nb_signal))]
            self.nr_samples = [int(y) for y in re.findall(b'.{1,8}', f.read(8 * self.nb_signal))]
            self.reserved_signal = [y.strip().decode('utf-8', 'ignore') for y in re.findall(b'.{1,32}', f.read(32 * self.nb_signal))]
            self.data_signal = []
            self.rec_start_time = []
            self.annotations = []
            for x in range(self.nb_signal):
                if self.labels[x] != "EDF Annotations":
                    self.data_signal.append(np.array([]))
            for y in range(self.nb_data_rec):
                self.rec_start_time.append([])
                for x in range(self.nb_signal):
                    if self.labels[x] == "EDF Annotations":
                        segment = f.read(2 * self.nr_samples[x])
                        self.annotations.append(segment)
                        if self.reserved == "EDF+D":
                            split_segment = segment.split(b"\x14\x14")
                            self.rec_start_time[y] = float(split_segment[0])
                    else:
                        block = f.read(2 * self.nr_samples[x])
                        segment = [block[i:i + 2] for i in range(0, len(block), 2)]
                        self.data_signal[x] = np.append(self.data_signal[x], [
                            transform_value(struct.unpack('<H', z)[0], self.phy_min[x], self.phy_max[x], self.dig_min[x],
                                            self.dig_max[x], 16) for z in segment])
    
    def get_n_samples(self):
        return [x * self.nb_data_rec for x in self.nr_samples] 

    def get_signal_labels(self):
        return self.labels
   
    def get_record_start_time(self, i):
        return self.rec_start_time[i]
    
    def get_number_of_data_records(self):
        return self.nb_data_rec

    def get_nr_samples(self, i):
        return self.nr_samples[i]

    def get_sample_frequency(self, i):
        return float(self.nr_samples[i]) / self.duration

    def get_physical_dimension(self, i):
        return self.phy_dim[i]

    def get_start_datetime(self):
        day, month, year = map(int, self.start_date.split("."))
        hour, minute, second = map(int, self.start_time.split("."))
        year = year + 1900 if year < 85 else year + 2000

        return datetime(year, month, day, hour, minute, second)

    def get_timestamps(self, index, signal_number, start_time) :
        # Start_time is in usecs since epoch
        start_time_record = start_time + self.get_record_start_time(index) * SECOND_TO_USECOND  
        end_time_record = start_time_record + SECOND_TO_USECOND * self.get_duration() 
        return np.linspace(start_time_record, end_time_record, num=self.get_nr_samples(signal_number), endpoint=False) 
  
    def is_discontiguous(self):
        return self.reserved == "EDF+D"

    def get_duration(self):
        return self.duration
    
    def get_annotations(self):
        return self.annotations

    def read_signal(self, i, start, end):
        return self.data_signal[i][start:end]
