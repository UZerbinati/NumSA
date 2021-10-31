import ngsolve as ngs
from ngsolve.ngs2petsc import *
import slepc4py.SLEPc as spc
from mpi4py import MPI
import numpy as np

class SLEPcEigenProblem():
    def __init__(self,PType,SType):
        self.E = spc.EPS().create()
        if PType == "GHEP":
            self.E.setProblemType(spc.EPS.ProblemType.GHEP) 
        elif PType == "HEP":
            self.E.setProblemType(spc.EPS.ProblemType.HEP)
        elif PType == "NHEP":
            self.E.setProblemType(spc.EPS.ProblemType.NHEP)
        elif PType == "GNHEP":
            self.E.setProblemType(spc.EPS.ProblemType.GNHEP)
        self.E.setType(SType)
    def setOperators(self,mats,Dofs):
        if len(mats)==1:
            self.A = CreatePETScMatrix(mats[0],Dofs);
            self.E.setOperators(self.A)
        elif len(mats)==2:
            self.A = CreatePETScMatrix(mats[0],Dofs);
            self.M = CreatePETScMatrix(mats[1],Dofs);
            self.E.setOperators(self.A,self.M)
    def SpectralTransformation(self,opt):
        self.ST = self.E.getST();
        self.ST.setType(opt);
        self.KSP = self.ST.getKSP();
    def setWhich(self,n):
        self.N = n
        self.E.setDimensions(n,psc.DECIDE)
    def getPair(self,fes,s):
        xr, xi = self.A.createVecs()
        lam = self.E.getEigenpair(s, xr, xi)
        vecmap = VectorMapping(fes.ParallelDofs(), fes.FreeDofs())
        vr =  ngs.GridFunction(fes)
        vecmap.P2N(xr, vr.vec)
        vi =  ngs.GridFunction(fes)
        vecmap.P2N(xi, vi.vec)
        return lam,vr,vi
    def getPairs(self,fes):
        vr =  ngs.GridFunction(fes, multidim=self.N)
        vi =  ngs.GridFunction(fes, multidim=self.N)
        lam = [];
        for s in range(self.N):
            xr, xi = self.A.createVecs()
            lam.append(self.E.getEigenpair(s, xr, xi))
            vecmap = VectorMapping(fes.ParallelDofs(), fes.FreeDofs())
            vecmap.P2N(xr, vr.vecs[s])
            vecmap.P2N(xi, vi.vecs[s])
        return lam,vr,vi
    def Solve(self):
        self.E.setST(self.ST);
        self.E.solve();
        self.Iterations = self.E.getIterationNumber()
        self.Converged = self.E.getConverged()
        sol_type = self.E.getType()
        nev, ncv, mpd = self.E.getDimensions()
        self.tolerance, Mit = self.E.getTolerances()
