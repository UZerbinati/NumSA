import pytest
#STDLIB
from math import pi

#NGSOLVE/NETGEN
from netgen.geom2d import unit_square
from ngsolve import *
from numsa.NGSlepc import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
npro = comm.size

@pytest.mark.mpi
def test_mpi():
    h = 1e-2;
    if comm.rank == 0:
        ngmesh = unit_square.GenerateMesh(maxh=h).Distribute(comm)
    else:
        ngmesh = netgen.meshing.Mesh.Receive(comm)

    mesh = Mesh(ngmesh)

    fes = H1(mesh, order=1, dirichlet=".*")
    u = fes.TrialFunction()
    v = fes.TestFunction()

    a = BilinearForm(fes)
    a += grad(u)*grad(v)*dx
    
    m = BilinearForm(fes)
    m += u*v*dx

    a.Assemble()
    m.Assemble()

    EP = SLEPcEigenProblem("GHEP","krylovschur")
    EP.SpectralTransformation("sinvert")
    
    PC = EP.KSP.getPC();
    PC.setType("lu");
    PC.setFactorSolverType("mumps");
    
    EP.setOperators([a.mat,m.mat],fes.FreeDofs())
    EP.setWhich(1);
    
    EP.Solve()
    
    lam, gfur, gfui = EP.getPairs(fes) 
    assert (abs(lam[0]/pi**2-2)<1e-3)
