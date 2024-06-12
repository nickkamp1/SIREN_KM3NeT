import siren
from siren.SIREN_Controller import SIREN_Controller
import uproot
import awkward as ak
import numpy as np

c = 3e-1 # m / ns

def get_targetZA(siren_file,dummy_val=-1):
    target_list = siren_file.target_type.to_numpy()
    ZA = np.array([[str(x)[3:6],str(x)[6:9]] if int(x[0])==1 else [dummy_val,dummy_val] for x in target_list],dtype=int)
    return ZA[:,0],ZA[:,1]

def four_vector_dot(p1,p2):
    return p1[...,0]*p2[...,0] - (p1[...,1]*p2[...,1] + p1[...,2]*p2[...,2] + p1[...,3]*p2[...,3])

def get_interaction_variables(siren_file):
    
    # Primary interaction momenta
    nu_momentum = (siren_file.primary_momentum).to_numpy()[:,0]
    secondary_momenta = (siren_file.secondary_momenta).to_numpy()[:,0]
    lepton_momentum = secondary_momenta[:,0]
    lepton_3momentum = np.linalg.norm(lepton_momentum[:,1:],axis=-1)
    target_momentum = secondary_momenta[:,1]
    
    # Primary interaction parameters
    y = 1 - (lepton_momentum[:,0]/nu_momentum[:,0])
    q = nu_momentum - lepton_momentum
    Q2 = -1 * four_vector_dot(q,q)
    p2 = lepton_momentum + target_momentum - nu_momentum
    x = Q2 / (2*p2[:,0]*(nu_momentum[:,0] - lepton_momentum[:,0]))
    
    # Secondary particle variables
    NTracks = np.sum(siren_file.num_secondaries,axis=-1)
    assert(np.all(NTracks==NTracks[0])) # make sure we have rectangular data
    secondary_momenta = siren_file.secondary_momenta.to_numpy()
    Id_tr = siren_file.secondary_types.to_numpy().reshape(len(NTracks),max(NTracks)) #TODO: same as pdg id?
    E_tr = secondary_momenta[:,:,:,0].reshape(len(NTracks),max(NTracks))
    Pdg_tr =  siren_file.secondary_types.to_numpy().reshape(len(NTracks),max(NTracks))
    vertex = siren_file.vertex.to_numpy()
    num_secondaries = siren_file.num_secondaries.to_numpy()
    Vx_tr = np.repeat(vertex[:,:,0],num_secondaries[0],axis=-1)
    Vy_tr = np.repeat(vertex[:,:,1],num_secondaries[0],axis=-1)
    Vz_tr = np.repeat(vertex[:,:,2],num_secondaries[0],axis=-1)
    secondary_3momenta = np.linalg.norm(secondary_momenta[:,:,:,1:],axis=-1)
    Dx_tr = (secondary_momenta[:,:,:,1]/secondary_3momenta).reshape(len(NTracks),max(NTracks))
    Dy_tr = (secondary_momenta[:,:,:,2]/secondary_3momenta).reshape(len(NTracks),max(NTracks))
    Dz_tr = (secondary_momenta[:,:,:,3]/secondary_3momenta).reshape(len(NTracks),max(NTracks))
    
    # Vertex timing
    primary_momentum = siren_file.primary_momentum.to_numpy()
    primary_mass = np.sqrt(four_vector_dot(primary_momentum,primary_momentum))
    primary_mass = np.where(np.isnan(primary_mass),0,primary_mass)
    gamma = primary_momentum[...,0]/primary_mass
    beta = np.sqrt(1-gamma**-2)
    delta_vertex = vertex - vertex[:,np.newaxis,0]
    vertex_dist = np.linalg.norm(delta_vertex,axis=-1)
    vertex_time = beta * c * vertex_dist
    T_tr = np.repeat(vertex_time,num_secondaries[0],axis=-1)
    
    return {"Bx":x,
            "By":y,
            "E_nu":nu_momentum[:,0],
            "Dx_nu":nu_momentum[:,1]/nu_momentum[:,0],
            "Dy_nu":nu_momentum[:,2]/nu_momentum[:,0],
            "Dz_nu":nu_momentum[:,3]/nu_momentum[:,0],
            "E_pl":lepton_momentum[:,0],
            "Dx_pl":lepton_momentum[:,1]/lepton_3momentum,
            "Dy_pl":lepton_momentum[:,2]/lepton_3momentum,
            "Dz_pl":lepton_momentum[:,3]/lepton_3momentum,
            "NTracks":NTracks,
            "Id_tr":Id_tr,
            "E_tr":E_tr,
            "Pdg_tr":Pdg_tr,
            "Vx_tr":Vx_tr,
            "Vy_tr":Vy_tr,
            "Vz_tr":Vz_tr,
            "Dx_tr":Dx_tr,
            "Dy_tr":Dy_tr,
            "Dz_tr":Dz_tr,
            "T_tr":T_tr
           }



class SIREN_to_gSeaGen:
    
    def output(self,root_file,tree_name="Events"):
        root_file[tree_name] = self.Events_dict
    
    def __init__(self,siren_file,dummy_int = -1):
        
        Z,A = get_targetZA(siren_file)
        interaction_dict = get_interaction_variables(siren_file)
        Nevents = len((siren_file.event_weight).to_list())
        dummy_val = dummy_int*np.ones(Nevents)

        self.Events_dict =    {"Evt":range(Nevents),
                               "PScale":dummy_val,
                               "TargetZ":Z,
                               "TargetA":A,
                               "InterID":dummy_val,
                               "Bx":interaction_dict["Bx"], #TODO: check SIREN calculation here
                               "By":interaction_dict["By"],
                               "LST":dummy_val,
                               "MJD":dummy_val,
                               "VerInCan":siren_file.in_fiducial.to_numpy()[:,0],
                               "WaterXSec":dummy_val, #TODO: use SIREN injector to compute cross section for each event
                               "WaterIntLen":dummy_val, #TODO: same as above
                               "PEarth":dummy_val, #TODO: Interface with NuSquids?
                               "ColumnDepth":dummy_val, #TODO: same as above?
                               "XSecMean":dummy_val, #TODO: use SIREN injector or ignore and allow reweighting to happen at the SIREN level
                               "GenWeight":dummy_val, #TODO: use SIREN to get gen weight only, or ignore as above
                               "EvtWeight":siren_file.event_weight,
                               "E_nu":interaction_dict["E_nu"],
                               "Pdg_nu":siren_file.primary_type[:,0],
                               "Vx_nu":siren_file.vertex[:,0,0],
                               "Vy_nu":siren_file.vertex[:,0,1],
                               "Vz_nu":siren_file.vertex[:,0,2],
                               "Dx_nu":interaction_dict["Dx_nu"],
                               "Dy_nu":interaction_dict["Dy_nu"],
                               "Dz_nu":interaction_dict["Dz_nu"],
                               "T_nu":np.zeros(Nevents), #TODO: is there a better interaction time?
                               "E_pl":interaction_dict["E_pl"],
                               "Dx_pl":interaction_dict["Dx_pl"],
                               "Dy_pl":interaction_dict["Dy_pl"],
                               "Dz_pl":interaction_dict["Dz_pl"],
                               "NTracks":interaction_dict["NTracks"],
                               "Id_tr":interaction_dict["Id_tr"],
                               "E_tr":interaction_dict["E_tr"],
                               "Pdg_tr":interaction_dict["Pdg_tr"],
                               "Vx_tr":interaction_dict["Vx_tr"],
                               "Vy_tr":interaction_dict["Vy_tr"],
                               "Vz_tr":interaction_dict["Vz_tr"],
                               "Dx_tr":interaction_dict["Dx_tr"],
                               "Dy_tr":interaction_dict["Dy_tr"],
                               "Dz_tr":interaction_dict["Dz_tr"],
                               "T_tr":interaction_dict["T_tr"],
                               "NSysWgt":dummy_val,
                               "WSys":dummy_val,
                               "NSysWgt_ParName":dummy_val,
                               "WSys_ParName":dummy_val
                              }