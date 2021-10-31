#!/bin/bash
make
pip3 install ./py[extra]
pytest ./tests/.
