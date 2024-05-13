#!/usr/bin/env python3.12

import os

from edf_processor import EdfProcessor

input_dir = os.environ.get('INPUT_DIR')

input_files = [
    f.path
    for f in os.scandir(input_dir)
    if f.is_file() and os.path.splitext(f.name)[1].lower() == '.edf'
]

if (len(input_files) != 1):
    raise Exception("EDF processor only supports a single file as input")

task = EdfProcessor(inputs = { 'file': input_files[0] })

task.run()
