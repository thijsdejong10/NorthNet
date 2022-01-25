from rdkit import Chem
from NorthNet import network_generation as n_gen

def extend_network_specific(network, reagents, reaction_template):
    '''
    Extend the network using a single reagent set and reaction template.

    Parameters
    ----------
    network: NorthNet Network object
        Network to be extrapolated from. Modified in place.
    reagents: NorthNet Compound objects
        Reagents to be applied to the network.
    reaction_template: NorthNet Reaction_Template object.
        Reaction template to be used on the network.

    Returns
    -------
    None
    '''

    template_substructures = reaction_template.ReactantSubstructures
    compounds_in_network = list(network.NetworkCompounds.values())

    reactive_substrs = [Chem.MolFromSmarts(x) for x in template_substructures]

    reactants = n_gen.get_reactive_compounds(
                                            compounds_in_network, 
                                            reactive_substrs
                                            )

    for reactant in reactants:
        insert = [reactant] + reagents

        input_valid = n_gen.check_reaction_input(
                                                insert, 
                                                reactive_substrs
                                                )

        if input_valid:
            resulting_reactions = n_gen.run_rdkit_reaction(
                                                            insert, 
                                                            reaction_template
                                                            )
            network.add_reactions(resulting_reactions)
        else:
            pass

def extend_network_self(network, reaction_template):
    '''
    Extend the network using any members of the network which can interact
    according to the reaction template provided.

    Parameters
    ----------
    network: NorthNet Network object
        Network to be extrapolated from. Modified in place.
    reagents: NorthNet Compound objects
        Reagents to be applied to the network.
    reaction_template: NorthNet Reaction_Template object.
        Reaction template to be used on the network.
    secondary_substructure: NorthNet/NetGen Substructure object
        Substructure of the second reaction component.

    Returns
    -------
    None
    '''
    compounds_in_network = list(network.NetworkCompounds.values())

    substructures = [Chem.MolFromSmarts(x)
                            for x in reaction_template.ReactantSubstructures]

    reactants1 = n_gen.get_reactive_compounds(
                                            compounds_in_network, 
                                            [substructures[0]]
                                            )

    reactants2 = n_gen.get_reactive_compounds(
                                            compounds_in_network, 
                                            [substructures[1]]
                                            )

    for reactant_1 in reactants1:
        for reactant_2 in reactants2:

            insert = [reactant_1] + [reactant_2]
            resulting_reactions = n_gen.run_rdkit_reaction(
                                                            insert, 
                                                            reaction_template
                                                            )
            network.add_reactions(resulting_reactions)

def extend_network_task(network, reaction_template):
    '''
    Extend the network using any members of the network which can interact
    according to the reaction template provided.

    Parameters
    ----------
    network: NorthNet Network object
        Network to be extrapolated from. Modified in place.
    reagents: NorthNet Compound objects
        Reagents to be applied to the network.
    reaction_template: NorthNet Reaction_Template object.
        Reaction template to be used on the network.
    secondary_substructure: NorthNet Substructure object
        Substructure of the second reaction component.

    Returns
    -------
    None
    '''
    import itertools

    current_compounds = list(network.NetworkCompounds.values())
    reactants = []
    for substruct in reaction_template.ReactantSubstructures:
        substructure = Chem.MolFromSmarts(substruct)
        matches = n_gen.get_reactive_compounds(
                                                current_compounds,
                                                [substructure]
                                                )
        reactants.append(matches)

    # Build reactant combinations
    inputs = list(itertools.product(*reactants))
    for input in inputs:
        resulting_reactions = n_gen.run_rdkit_reaction(
                                                        input, 
                                                        reaction_template
                                                        )
        network.add_reactions(resulting_reactions)
