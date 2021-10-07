import tensorflow as tf
import numpy as np
import sys
from mpi4py import MPI
from tqdm import tqdm

comm = MPI.COMM_WORLD;
nprs = comm.Get_size();
rank = comm.Get_rank()

NBase = np.array_split(range(10),nprs);
H = np.zeros((10,10));
for i in NBase[rank]:
    H[i,i] = 1.0;

H = sum(comm.gather(H, root=0));
if rank == 0:
    print(H)



