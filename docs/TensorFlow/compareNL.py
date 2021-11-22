def compareNL(b,bias):
    d=119;
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    #############################| VANILLA NEWTON|##############################
    #Vanilla Newton Number Of CG Iteration per Non Linear Iteration
    VanillaNewtonLIT = [0,7,9,10,13]
    VanillaNewtonErr = [0.9306789636611938,0.02549976110458374,0.0007881820201873779,4.1425228118896484e-06,5.960464477539063e-08]
    VanillaNewtonCOM = [];
    VanillaNewtonMeM = [];

    for l in range(len(VanillaNewtonLIT)):
        VanillaNewtonCOM = VanillaNewtonCOM + [bias+4*2*sum([LIT*d for LIT in VanillaNewtonLIT[0:l+1]])]
    for l in range(len(VanillaNewtonLIT)):
        VanillaNewtonMeM = VanillaNewtonMeM + [bias+3*d*l*4]

    #############################| FEDNL WK 2|##############################

    FEDNL2It = 12;
    FEDNL2Err = [0.9306789636611938,0.02344226837158203,0.004081755876541138,0.001575171947479248,
                 0.0007283687591552734,0.00016567111015319824,3.674626350402832e-05,8.821487426757812e-06,
                 3.0100345611572266e-06,1.0728836059570312e-06,3.5762786865234375e-07,1.4901161193847656e-07,
                 5.960464477539063e-08];
    FEDNL2COM = []
    FEDNL2MeM = []
    for l in range(FEDNL2It):
        FEDNL2COM = FEDNL2COM + [d*2*l*4+d*d*4];
    FEDNL2COM=[bias]+FEDNL2COM;
    for l in range(FEDNL2It):
        FEDNL2MeM = FEDNL2MeM + [(l+1)*(d+2*d**2)*4];
    FEDNL2MeM=[bias]+FEDNL2MeM;

    #############################| FEDNL WK 8|##############################

    FEDNL8It = 12;
    FEDNL8Err = [0.93067896, 0.02343294, 0.00401628, 0.0016137362, 0.0006377995, 
                 0.00018271804, 4.401803e-05, 1.0579824e-05, 3.0994415e-06, 9.536743e-07,
                 3.874302e-07, 2.0861626e-07, 1.4901161e-07]
    FEDNL8COM = []
    FEDNL8MeM = []
    for l in range(FEDNL8It):
        FEDNL8COM = FEDNL8COM + [d*2*l*4+d*d*4];
    FEDNL8COM=[bias]+FEDNL8COM;
    for l in range(FEDNL8It):
        FEDNL8MeM = FEDNL8MeM + [(l+1)*(d+2*d**2)*4];
    FEDNL8MeM=[bias]+FEDNL8MeM;

    #############################| FEDNL WK 16|##############################
    FEDNL16It = 13;
    FEDNL16Err = [0.93067896, 0.023391873, 0.0038612187, 0.0014861226, 0.0005749166, 
                  0.00017276406, 3.901124e-05, 9.745359e-06, 2.95043e-06, 1.1026859e-06, 
                  5.662441e-07, 4.172325e-07, 4.172325e-07, 3.874302e-07]

    FEDNL16COM = []
    FEDNL16MeM = []
    for l in range(FEDNL16It):
        FEDNL16COM = FEDNL16COM + [d*2*l*4+d*d*4];
    FEDNL16COM=[bias]+FEDNL16COM;
    for l in range(FEDNL16It):
        FEDNL16MeM = FEDNL16MeM + [(l+1)*(d+2*d**2)*4];
    FEDNL16MeM=[bias]+FEDNL16MeM;

    #############################| QuasiNewton WK 2|##############################

    QuasiNewton80HalkoRank = 80;
    QuasiNewton80LIT = 20;
    QuasiNewton80Err = [
        0.9306789636611938,0.024927228689193726,0.0018192827701568604,0.0005114078521728516,
        0.0002964437007904053,0.0002022385597229004,9.566545486450195e-05,6.0111284255981445e-05,
        3.415346145629883e-05,1.582503318786621e-05,8.493661880493164e-06,5.930662155151367e-06,
        3.427267074584961e-06,2.3245811462402344e-06,1.3113021850585938e-06,6.258487701416016e-07,
        3.8743019104003906e-07,2.682209014892578e-07,2.086162567138672e-07,8.940696716308594e-08
    ]
    QuasiNewton80COM = [];
    QuasiNewton80MeM = [];
    for l in range(QuasiNewton80LIT):
        QuasiNewton80COM = QuasiNewton80COM + [bias+l*d*2*QuasiNewton80HalkoRank*4];
    for l in range(QuasiNewton80LIT):
        QuasiNewton80MeM = QuasiNewton80MeM + [bias+l*d*QuasiNewton80HalkoRank*4+l*d*4];
    #############################| FEDQNL WK 2|##############################
    FEDQNLHalkoRank = 1;
    FEDQNLWk = 2;
    FEDQNLInitRank = 119;
    FEDQNLLIT = 57;
    FEDQNLErr = [0.93067896, 0.13985145, 0.061779857, 0.033067346, 0.020624846, 0.01451695, 
                 0.010643721, 0.007771462, 0.005669892, 0.0041793287, 0.0031336546, 0.002387941,
                 0.0018449724, 0.0014425814, 0.0011403263, 0.00090909004, 0.0007298291, 0.00058978796,
                 0.00047910213, 0.00039055943, 0.0003195703, 0.00026223063, 0.00021582842, 0.00017812848,
                 0.00014734268, 0.00012212992, 0.0001013577, 8.4251165e-05, 7.009506e-05, 5.8412552e-05,
                 4.87566e-05, 4.0650368e-05, 3.400445e-05, 2.8431416e-05, 2.3782253e-05, 1.9878149e-05,
                 1.66893e-05, 1.3977289e-05, 1.1742115e-05, 9.834766e-06, 8.255243e-06, 6.943941e-06,
                 5.811453e-06, 4.917383e-06, 4.1127205e-06, 3.4570694e-06, 2.9206276e-06, 2.4735928e-06,
                 2.115965e-06, 1.758337e-06, 1.4603138e-06, 1.2516975e-06, 1.0728836e-06, 9.23872e-07,
                 7.4505806e-07, 6.2584877e-07, 5.662441e-07, 5.066395e-07]
    FEDQNLCOM = [];
    FEDQNLMeM = [];
    for l in range(FEDQNLLIT):
        FEDQNLCOM = FEDQNLCOM + [l*d*4*FEDQNLHalkoRank*4+d*d*4];
    FEDQNLCOM=[bias]+FEDQNLCOM;
    for l in range(FEDQNLLIT):
        FEDQNLMeM = FEDQNLMeM + [(l+1)*4*(FEDQNLWk*d*FEDQNLHalkoRank+2*d*FEDQNLHalkoRank)+4*FEDQNLInitRank*d*2+4*FEDQNLLIT];
    FEDQNLMeM=[bias]+FEDQNLMeM;

    def B2MB(values):
        new_values = [1e-6*v for v in values];
        return new_values;

    fig,ax = plt.subplots()
    ax.semilogy(VanillaNewtonLIT[b:],VanillaNewtonErr[b:],"o-")
    ax.semilogy(range(FEDNL2It+1)[b:],FEDNL2Err[b:],"o-")
    ax.set_yscale("log", base=10)
    ax.legend(["Newton Method WK. 2","FEDNL Rank 1 WK. 2","Quasi Newton Rank 80 WK. 2"])
    ax.set_title("Iteration To Convergence")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")

    fig,ax = plt.subplots()
    ax.semilogy(B2MB(VanillaNewtonCOM[b:]),VanillaNewtonErr[b:],"o-")
    ax.semilogy(B2MB(FEDNL2COM[b:]),FEDNL2Err[b:],"o-")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.legend(["Newton Method WK. 2","FEDNL Rank 1 WK. 2","Quasi Newton Rank 80 WK. 2"])
    ax.set_title("Communication Volume")
    ax.set_xlabel("MB")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")

    fig,ax = plt.subplots()
    ax.semilogy(B2MB(VanillaNewtonMeM[b:]),VanillaNewtonErr[b:],"o-")
    ax.semilogy(B2MB(FEDNL2MeM[b:]),FEDNL2Err[b:],"o-")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.legend(["Newton Method WK. 2","FEDNL Rank 1 WK. 2","Quasi Newton Rank 80 WK. 2"])
    ax.set_title("Memory Footprint")
    ax.set_xlabel("MB")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")




    fig,ax = plt.subplots()
    ax.semilogy(range(QuasiNewton80LIT)[b:],QuasiNewton80Err[b:],"o-")
    ax.semilogy(range(FEDNL2It+1)[b:],FEDNL2Err[b:],"o-")
    ax.legend(["Quasi Newton Method WK. 2","FEDNL Rank 1 WK. 2"])
    ax.set_yscale("log", base=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Iterations To Convergence")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")

    fig,ax = plt.subplots()
    ax.semilogy(B2MB(QuasiNewton80COM[b:]),QuasiNewton80Err[b:],"o-")
    ax.semilogy(B2MB(FEDNL2COM[b:]),FEDNL2Err[b:],"o-")
    ax.legend(["Quasi Newton Method WK. 2","FEDNL Rank 1 WK. 2"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_title("Communication Volume")
    ax.set_xlabel("MB")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")

    fig,ax = plt.subplots()
    ax.semilogy(B2MB(QuasiNewton80MeM[b:]),QuasiNewton80Err[b:],"o-")
    ax.semilogy(B2MB(FEDNL2MeM[b:]),FEDNL2Err[b:],"o-")
    ax.legend(["Quasi Newton Method WK. 2","FEDNL Rank 1 WK. 2"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_title("Memory Footprint")
    ax.set_xlabel("MB")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")




    fig,ax = plt.subplots()
    ax.semilogy(range(FEDNL2It+1)[b:],FEDNL2Err[b:],"o-")
    ax.semilogy(range(FEDNL8It+1)[b:],FEDNL8Err[b:],"o-")
    ax.semilogy(range(FEDNL16It+1)[b:],FEDNL16Err[b:],"o-")
    ax.legend(["FEDNL Rank 1 WK. 2","FEDNL Rank 1 WK. 8","FEDNL Rank 1 WK. 16"])
    ax.set_yscale("log", base=10)
    ax.set_title("Iterations To Convergence")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")

    fig,ax = plt.subplots()
    ax.semilogy(B2MB(FEDNL2COM[b:]),FEDNL2Err[b:],"o-")
    ax.semilogy(B2MB(FEDNL8COM[b:]),FEDNL8Err[b:],"o-")
    ax.semilogy(B2MB(FEDNL16COM[b:]),FEDNL16Err[b:],"o-")
    ax.legend(["FEDNL Rank 1 WK. 2","FEDNL Rank 1 WK. 8","FEDNL Rank 1 WK. 16"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_title("Communication Volume")
    ax.set_xlabel("MB")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")

    fig,ax = plt.subplots()
    ax.semilogy(B2MB(FEDNL2MeM[b:]),FEDNL2Err[b:],"o-")
    ax.semilogy(B2MB(FEDNL8MeM[b:]),FEDNL8Err[b:],"o-")
    ax.semilogy(B2MB(FEDNL16MeM[b:]),FEDNL16Err[b:],"o-")
    ax.legend(["FEDNL Rank 1 WK. 2","FEDNL Rank 1 WK. 8","FEDNL Rank 1 WK. 16"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_title("Memory Footprint")
    ax.set_xlabel("MB")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")




    fig,ax = plt.subplots()
    ax.semilogy(range(QuasiNewton80LIT)[b:],QuasiNewton80Err[b:],"o-")
    ax.semilogy(range(FEDQNLLIT+1)[b:],FEDQNLErr[b:],"o-")
    ax.legend(["Quasi Newton Rank 80 WK. 2","FEDQNL Rank 1 WK. 2"])
    ax.set_yscale("log", base=10)
    ax.set_title("Iterations To Convergence")
    ax.set_xlabel("Iterations")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")

    fig,ax = plt.subplots()
    ax.semilogy(B2MB(QuasiNewton80COM[b:]),QuasiNewton80Err[b:],"o-")
    ax.semilogy(B2MB(FEDQNLCOM[b:]),FEDQNLErr[b:],"o-")
    ax.legend(["Quasi Newton Rank 80 WK. 2","FEDQNL Rank 1 WK. 2"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_title("Communication Volume")
    ax.set_xlabel("MB")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")


    fig,ax = plt.subplots()
    ax.semilogy(B2MB(QuasiNewton80MeM[b:]),QuasiNewton80Err[b:],"o-")
    ax.semilogy(B2MB(FEDQNLMeM[b:]),FEDQNLErr[b:],"o-")
    ax.legend(["Quasi Newton Rank 80 WK. 2","FEDQNL Rank 1 WK. 2"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_title("Memory Foorprint")
    ax.set_xlabel("MB")
    ax.set_ylabel(r"$|\mathcal{L}(x^k)-\mathcal{L}(x^*)|$")