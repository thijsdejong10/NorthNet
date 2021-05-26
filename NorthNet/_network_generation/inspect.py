from itertools import compress

def reactive_species(species_list, substructures):
    '''
    Find species in 'species_list'  with a given
    'group' (SMARTS string). Returns a list of 'matches' (list of SMILES strings)

    Parameters
    ----------
    species_list: list
        List of NorthNet Compound objects from which reactive molecules are extracted.

    substructure: list of rdkit mol objects
        Reactive substructures.

    Returns
    -------
    matches: list
        List of NorthNet Compound objects which contain the substructure.
    '''
    matches = [] # empty container to put matching molecules in
    for mol in species_list: # testing each molecule in the list of species for reaction group matches
        for s in substructures:
            if mol.Mol.HasSubstructMatch(s):
                matches.append(mol) # append the molecule to the matches if in contains the reactive group
    return matches

def remove_invalid_reactions(reactions,invalid_substructures):

    '''
    Designed to remove reactions with products which contain invalid substructures
    as defined by the user.

    Parameters
    ----------
    reactions: list
        List of NorthNet Generated_Reaction objects.
    invalid_substructures: list
        list of NorthNet Substructure objects.

    Returns
    -------
    reactions: list
        list of reactions with those that produce invalid substrructures removed.
    '''

    sortlist = [] # A list which stores reaction exception violations (True)
    for r in reactions:
        tag = True
        for exc in invalid_substructures: # iterate through list of exception reactions
            if any([p.HasSubstructMatch(exc.Mol) for p in r.Reaction.GetProducts()]):
                tag = False

        sortlist.append(tag)

    reactions = list(compress(reactions, sortlist)) # use itertools compress to remove reactions with same index as a False value in filter.

    return reactions

def check_reaction_input(reactant_list, reactive_substructs):

    '''
    reactant_list: NorthNet Compound Object

    reaction_template: NorthNet ReactionTemplate object
    '''
    test_list = [False]*len(reactant_list)
    for c,r in enumerate(reactant_list):
        for s in reactive_substructs:
            if r.Mol.HasSubstructMatch(s):
                test_list[c] = True
            else:
                pass

    return all(test_list)
