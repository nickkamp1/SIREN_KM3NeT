from siren.SIREN_Controller import SIREN_Controller
import siren
import os

def RunMinimalHNLSimulation(events_to_inject,outfile,
                            m4="1000",Ue4=0, Umu4=0, Utau4=1):

    # Expeirment to run
    experiment = "KM3NeTORCA"

    # Define the controller
    controller = SIREN_Controller(events_to_inject, experiment)

    # Particle to inject
    primary_type = siren.dataclasses.Particle.ParticleType.NuTau
    hnl_type = siren.dataclasses.Particle.ParticleType.N4

    # Primary distributions
    primary_injection_distributions = {}
    primary_physical_distributions = {}

    siren_input_file = "Data/SIREN/Input/LHCb_%s_%s_%s.txt"%(prefix,generator,parent)
    assert(os.path.isfile(siren_input_file))
    with open(siren_input_file, "rbU") as f:
        num_input_events = sum(1 for _ in f) - 1

    primary_external_dist = siren.distributions.PrimaryExternalDistribution(siren_input_file)
    primary_injection_distributions["external"] = primary_external_dist


    fid_vol = controller.GetFiducialVolume()
    position_distribution = siren.distributions.PrimaryBoundedVertexDistribution(fid_vol)
    primary_injection_distributions["position"] = position_distribution

    secondary_position_distribution = siren.distributions.SecondaryBoundedVertexDistribution(fid_vol)

    # SetProcesses
    controller.SetProcesses(
        primary_type, primary_injection_distributions, primary_physical_distributions,
        [hnl_type], [[]], [[]]
    )
    for process in controller.secondary_injection_processes:
        print(process.primary_type)

    ##################

    # Now include DIS interaction
    cross_section_model = "HNLDISSplines"

    xsfiledir = siren.utilities.get_cross_section_model_path(cross_section_model)

    # Cross Section Model
    target_type = siren.dataclasses.Particle.ParticleType.Nucleon

    DIS_xs = siren.interactions.HNLDISFromSpline(
        os.path.join(xsfiledir, "M_0000MeV/dsdxdy-nu-N-nc-GRV98lo_patched_central.fits"),
        os.path.join(xsfiledir, "M_%sMeV/sigma-nu-N-nc-GRV98lo_patched_central.fits"%m4),
        float(m4)*1e-3,
        [Ue4,Umu4,Utau4],
        siren.utilities.Constants.isoscalarMass,
        1,
        [primary_type],
        [target_type],
    )

    DIS_interaction_collection = siren.interactions.InteractionCollection(primary_type, [DIS_xs])


    # Decay Model
    two_body_decay = siren.interactions.HNLTwoBodyDecay(float(m4)*1e-3, [Ue4, Umu4, Utau4], siren.interactions.HNLTwoBodyDecay.ChiralNature.Majorana)
    Decay_interaction_collection = siren.interactions.InteractionCollection(hnl_type, [two_body_decay])

    controller.SetInteractions(primary_interaction_collection=DIS_interaction_collection,
                               secondary_interaction_collections=[Decay_interaction_collection])

    # if we are below the W mass, use DarkNews for dimuon decay
    if float(m4)*1e-3 < siren.utilities.Constants.wMass:

        # Define a DarkNews model
        model_kwargs = {
            "m4": float(m4)*1e-3,
            "Ue4": Ue4,
            "Umu4": Umu4,
            "Utau4": Utau4,
            "gD":0,
            "epsilon":0,
            "mzprime":0.1,
            "noHC": True,
            "HNLtype": "majorana",
            "include_nelastic": True,
            "decay_product":"mu+mu-"
        }

        xs_path = siren.utilities.get_cross_section_model_path(f"DarkNewsTables-v{siren.utilities.darknews_version()}", must_exist=False)

        # Define DarkNews Model
        table_dir = os.path.join(
            xs_path,
            "HNL_M%2.2e_e+%2.2e_mu%2.2e_tau%2.2e"%(float(m4),Ue4,Umu4,Utau4),
        )
        controller.InputDarkNewsModel(primary_type, table_dir, upscattering=False, **model_kwargs)

    ##################


    # Run generation and save events
    controller.Initialize()

    for process in controller.secondary_injection_processes:
        print(process.primary_type)
        for interaction in process.interactions.GetDecays():
            for signature in interaction.GetPossibleSignatures():
                print(signature.secondary_types)

    def stop(datum, i):
        secondary_type = datum.record.signature.secondary_types[i]
        return secondary_type != siren.dataclasses.Particle.ParticleType.N4

    controller.SetInjectorStoppingCondition(stop)

    controller.GenerateEvents()
    controller.SaveEvents(outfile,
                          save_int_probs=True,
                          save_int_params=True)





def RunDipoleHNLSimulation(events_to_inject,outfile,
                           m4,mu_tr_mu4):

    # Define a DarkNews model
        model_kwargs = {
            "m4": float(m4)*1e-3,
            "mu_tr_mu4": mu_tr_mu4, # GeV^-1
            "UD4": 0,
            "Umu4": 0,
            "epsilon": 0.0,
            "gD": 0.0,
            "decay_product": "photon",
            "noHC": True,
            "HNLtype": "dirac",
        }

        # Expeirment to run
        experiment = "KM3NeTORCA"

        # Define the controller
        controller = SIREN_Controller(events_to_inject, experiment)

        # Particle to inject
        primary_type = siren.dataclasses.Particle.ParticleType.NuMu

        xs_path = siren.utilities.get_cross_section_model_path(f"DarkNewsTables-v{siren.utilities.darknews_version()}", must_exist=False)
        # Define DarkNews Model
        table_dir = os.path.join(
            xs_path,
            "Dipole_M%2.2e_mu%2.2e" % (model_kwargs["m4"], model_kwargs["mu_tr_mu4"]),
        )
        controller.InputDarkNewsModel(primary_type, table_dir, **model_kwargs, upscattering=False)

        # Now include DIS interaction
        cross_section_model = "DipoleHNLDISSplines"

        xsfiledir = siren.utilities.get_cross_section_model_path(cross_section_model)

        # Cross Section Model
        target_type = siren.dataclasses.Particle.ParticleType.Nucleon

        DIS_xs = siren.interactions.HNLDipoleDISFromSpline(
            os.path.join(xsfiledir, "M_0000MeV/dsdxdy-nu-N-em-GRV98lo_patched_central.fits"),
            os.path.join(xsfiledir, "M_%sMeV/sigma-nu-N-em-GRV98lo_patched_central.fits"%m4),
            model_kwargs["m4"],
            [0,model_kwargs["mu_tr_mu4"],0],
            [primary_type],
            [target_type],
        )

        DIS_interaction_collection = siren.interactions.InteractionCollection(
                    primary_type, [DIS_xs]
                )

        controller.SetInteractions(DIS_interaction_collection)

        # Primary distributions
        primary_injection_distributions = {}
        primary_physical_distributions = {}

        # energy distribution
        edist = siren.distributions.PowerLaw(2, 1e1, 1.7e4)
        edist_phys = siren.distributions.TabulatedFluxDistribution(1e1,1e5,"daemonflux_numu.txt", True)
        primary_injection_distributions["energy"] = edist
        primary_physical_distributions["energy"] = edist_phys

        # direction distribution
        direction_distribution = siren.distributions.IsotropicDirection()
        primary_injection_distributions["direction"] = direction_distribution
        primary_physical_distributions["direction"] = direction_distribution

        # position distribution
        position_distribution = controller.GetCylinderVolumePositionDistributionFromSector("orca")
        primary_injection_distributions["position"] = position_distribution

        # SetProcesses
        controller.SetProcesses(
            primary_type, primary_injection_distributions, primary_physical_distributions
        )

        controller.Initialize()

        def stop(datum, i):
            secondary_type = datum.record.signature.secondary_types[i]
            return secondary_type != siren.dataclasses.Particle.ParticleType.N4

        controller.SetInjectorStoppingCondition(stop)

        events = controller.GenerateEvents(fill_tables_at_exit=False)

        os.makedirs("output", exist_ok=True)

        controller.SaveEvents(outfile,fill_tables_at_exit=False,hdf5=False,siren_events=False)


def main():
    events_to_inject = int(1e5)
    for m4 in ["0300","0600","1000"]:
        for mu_tr_mu4 in [1e-6,5e-7,1e-7]:
            outfile = "input/ORCA_DipoleHNL_m4_%s_mu_%2.2e"%(m4,mu_tr_mu4)
            RunDipoleHNLSimulation(events_to_inject,outfile,m4,mu_tr_mu4)

if __name__=="__main__":
    main()