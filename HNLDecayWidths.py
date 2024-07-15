import siren
from siren.SIREN_DarkNews import PyDarkNewsInteractionCollection
import os
import numpy as np
import rundec
import siren
from scipy.integrate import quad
import siren.utilities.Constants as Constants
import numpy as np

crd = rundec.CRunDec()

# see https://pdg.lbl.gov/2020/reviews/rpp2020-rev-quark-masses.pdf for quark masses
mu = 2.32e-3
md = 4.71e-3
ms = 92.9e-3
me = 0.511e-3
mmu = 105.6e-3
mtau = 1.776
mK = 0.493677

def lam(a,b,c):
    return a**2 + b**2 + c**2 - 2*a*b - 2*a*c - 2*b*c

def integrand(x,xu,xd,xl):
    return 1./x * (x - xl**2 - xd**2) * (1 + xu**2 - x) * np.sqrt(lam(x,xl**2,xd**2)*lam(1,x,xu**2))

def I(xu,xd,xl):
    return 12 * quad(integrand,(xd+xl)**2,(1-xu)**2,args=(xu,xd,xl))[0]

def GammaCC(U,mN,mu,md,ml,Nw=1):
    xu = mu/mN
    xd = md/mN
    xl = ml/mN
    if(xu+xd+xl>1): return 0
    return Nw * Constants.FermiConstant**2 * mN**5 / (192*Constants.pi**3) * U**2 * I(xu,xd,xl)

def L(x,thresh=3e-3):
    num = 1 - 3 * x**2 - (1 - x**2) * np.sqrt(1 - 4 * x**2)
    num = np.where(x<thresh,2*x**6 + 6*x**8 + 18*x**10,num)
    denom = x**2 * (1 + np.sqrt(1 - 4 * x**2))
    return np.log(num/denom)
    
def GammaNC(U,mN,mf,Nz=1,fs="qup"):
        
    if fs=="qup":
        C1f = 1./4. * (1 - 8./3. * Constants.thetaWeinberg + 32./9. * Constants.thetaWeinberg**2)
        C2f = 1./3. * Constants.thetaWeinberg * (4./3. * Constants.thetaWeinberg - 1)
    elif fs=="qdown":
        C1f = 1./4. * (1 - 4./3. * Constants.thetaWeinberg + 8./9. * Constants.thetaWeinberg**2)
        C2f = 1./6. * Constants.thetaWeinberg * (2./3. * Constants.thetaWeinberg - 1)
    else:
        return 0
    
    x = mf/mN
    if (2*x>=1): return 0
    prefactor = Nz * Constants.FermiConstant**2 * mN**5 / (192*Constants.pi**3) * U**2
    factor1 = (1 - 14 * x**2 - 2 * x**4 - 12 * x**6)*np.sqrt(1 - 4 * x**2) + 12 * x**4 * (x**4 - 1) * L(x)
    factor2 = x**2 * (2 + 10 * x*82 - 12 * x**4) * np.sqrt(1 - 4 * x**2) + 6 * x**4 * (1 - 2 * x**2 + 2 * x**4) * L(x)
    return prefactor * (C1f*factor1 + 4*C2f*factor2)

def DeltaQCD(m_N,
             nl=5,
             flavor_thresholds= np.array([1.5,4.8,173.21])):
    
    nf = int(3 + sum(m_N>flavor_thresholds))
    if nf<=4:
        # use tau mass for reference
        m_ref = 1.7768
        alpha_s_ref = 0.332
    else:
        # use Z mass for reference
        m_ref = 91.1876
        alpha_s_ref = 0.1179
    
    alpha_s = crd.AlphasExact(alpha_s_ref,m_ref,m_N,nf,nl)
    # from https://arxiv.org/abs/2007.03701
    return alpha_s / np.pi + 5.2 * alpha_s**2 / np.pi**2 + 26.4 * alpha_s**3 / np.pi**3

def GammaHadronsCC(m_N,m_l):
    Gamma_ud = GammaCC(1,m_N,mu,md,m_l,Nw=3*Constants.Vud)
    Gamma_us = GammaCC(1,m_N,mu,ms,m_l,Nw=3*Constants.Vus)
    return (1+DeltaQCD(m_N)) * (Gamma_ud + np.sqrt(1 - 4*(mK/m_N)**2) * Gamma_us)

def GammaHadronsNC(m_N):
    Gamma_uu = GammaNC(1,m_N,mu,Nz=3,fs="qup")
    Gamma_dd = GammaNC(1,m_N,md,Nz=3,fs="qdown")
    Gamma_ss = GammaNC(1,m_N,ms,Nz=3,fs="qdown")
    return (1+DeltaQCD(m_N)) * (Gamma_uu + Gamma_dd + np.sqrt(1 - 4*(mK/m_N)**2) * Gamma_ss)



def get_decay_widths(m4, Ue4, Umu4, Utau4):
    
    xs_path = siren.utilities.get_cross_section_model_path(f"DarkNewsTables-v{siren.utilities.darknews_version()}", must_exist=False)
    table_dir = os.path.join(
        xs_path,
        "HNL_M%2.2e_e+%2.2e_mu%2.2e_tau%2.2e"%(m4,Ue4,Umu4,Utau4),
    )
    
    # Define a DarkNews model
    model_kwargs = {
        "m4": m4,
        "Ue4": Ue4,
        "Umu4": Umu4,
        "Utau4": Utau4,
        "gD":0,
        "epsilon":0,
        "mzprime":10,
        "noHC": True,
        "HNLtype": "dirac",
        "include_nelastic": True,
        "nuclear_targets": ["H1"]
    }
    if m4 > 2*0.1057:
        model_kwargs["decay_product"] = "mu+mu-"
        DN_processes = PyDarkNewsInteractionCollection(table_dir = table_dir, **model_kwargs)
        dimuon_decay = DN_processes.decays[0]
    else:
        dimuon_decay = None
    if m4 > 2*0.000511:
        model_kwargs["decay_product"] = "e+e-"
        DN_processes = PyDarkNewsInteractionCollection(table_dir = table_dir, **model_kwargs)
        dielectron_decay = DN_processes.decays[0]
    else:
        dielectron_decay = None
    two_body_decay = siren.interactions.HNLTwoBodyDecay(m4, [Ue4, Umu4, Utau4], siren.interactions.HNLTwoBodyDecay.ChiralNature.Dirac)
    record = siren.dataclasses.InteractionRecord()
    record.signature.primary_type = siren.dataclasses.Particle.N4
    record.signature.target_type = siren.dataclasses.Particle.Decay
    
    decay_widths = {}
    
    for decay in [dimuon_decay,dielectron_decay,two_body_decay]:
        if decay is None: continue
        for signature in decay.GetPossibleSignaturesFromParent(siren.dataclasses.Particle.N4):
            record.signature.secondary_types = signature.secondary_types
            decay_widths[tuple(signature.secondary_types)] = decay.TotalDecayWidthForFinalState(record)
    
    # 3 Body hadronic decay width
    if m4 > 1:
        GammaHadron_2Body_NC = 0
        GammaHadron_2Body_CC = {siren.dataclasses.Particle.EMinus:0,
                                siren.dataclasses.Particle.MuMinus:0,
                                siren.dataclasses.Particle.TauMinus:0}
        for sec_types,decay_width in decay_widths.items():
            if len(sec_types) != 2: continue
            for charged_lepton in GammaHadron_2Body_CC.keys():  
                if sec_types[0] == charged_lepton:
                    GammaHadron_2Body_CC[charged_lepton] += decay_width
            if sec_types[0] in [siren.dataclasses.Particle.NuE,
                                siren.dataclasses.Particle.NuMu,
                                siren.dataclasses.Particle.NuTau]:
                GammaHadron_2Body_NC += decay_width
        decay_widths[tuple([siren.dataclasses.Particle.NuLight,
                            siren.dataclasses.Particle.Hadrons])] = (Ue4**2 + Umu4**2 + Utau4**2) * GammaHadronsNC(m4) - GammaHadron_2Body_NC
        for charged_lepton,ml,U in zip(GammaHadron_2Body_CC.keys(),
                                       [me,mmu,mtau],
                                       [Ue4,Umu4,Utau4]):
            if U<=0: continue
            decay_widths[tuple([charged_lepton,
                                siren.dataclasses.Particle.Hadrons])] = U**2 * GammaHadronsCC(m4,ml) - GammaHadron_2Body_CC[charged_lepton]
    decay_widths["total"] = sum(decay_widths.values())
    
    return decay_widths
