import sys
from NorthNet import Classes

class Network:
    '''
    An object which stored compounds and reactions and the connections
    between them. 
    '''
    def __init__(self, reactions, name, description):
        '''
        The Network object is initialised with a list of Reaction objects. If
        the list is empty, then the network is initialised as an empty network.

        Parameters
        ----------
        reactions: list of NorthNet Reaction objects
        name: string
            A label name for the network.
        description: string
            A description for the network.
        '''

        if isinstance(reactions, list):
            check_reactions = [isinstance(r, Classes.Reaction) for r in reactions]
            if not all(check_reactions):
                sys.exit('''class Network: the reactions arg must be a list of
                NorthNet Reaction objects.''')
        else:
            sys.exit('''class Network: the reactions arg must be a list of
            NorthNet Reaction objects.''')

        if not isinstance(name, str):
            sys.exit('''class Network: name arg should be a string.''')
        if not isinstance(name, str):
            sys.exit('''class Network: description arg should be a string.''')

        self.Name = name
        self.Description = description

        '''1. create a dictionary of lists of Reaction objects, which are
        organised by their reaction SMILES as keys. '''
        self.NetworkReactions = {}

        '''2. Create dictionary of Compound objects used as reactants
        and products from the reaction objects (SMILES used as keys).'''
        self.NetworkCompounds = {}

        '''3. Create dictionary to store network inputs'''
        self.NetworkInputs = {}

        '''4. Create dictionary of network outputs'''
        self.NetworkOutputs = {}

        if len(reactions) == 0:
            pass
        else:
            self.add_reactions(reactions)

    def add_compound(self, compound):
        '''
        Add a compound to the network.
        compoud: NorthNet Compound object
        '''

        if not isinstance(compound, Classes.Compound):
            sys.exit('''Network.add_compound():
                compound arg should be a NorthNet Compound object''')

        if compound.SMILES not in self.NetworkCompounds:
            self.NetworkCompounds[compound.SMILES] = compound

    def add_compounds(self,compounds):
        '''
        Add list of NorthNet Compound objects to the Network

        Parameters
        ----------
        compounds: list of NorthNet Compound objects
            compounds to be added
        '''
        if isinstance(compounds, list):
            check_compounds = [isinstance(c, Classes.Compound) for c in compounds]
            if not all(check_compounds):
                sys.exit('''Network.add_compounds():
                compounds arg should be a list of NorthNet Compound objects''')
        else:
            sys.exit('''Network.add_compounds():
            compounds arg should be a list of NorthNet Compound objects''')

        for compound in compounds:
            self.add_compound(compound)

    def remove_compounds(self, compounds):
        '''
        Remove list of compounds and the reactions in which they are involved
        from the network.

        Parameters
        ----------
        compounds: list of NorthNet Compound objects
            compounds to be removed
        '''

        if isinstance(compounds, list):
            check_compounds = [isinstance(c, Classes.Compound) for c in compounds]
            if not all(check_compounds):
                sys.exit('''Network.remove_compounds():
                compounds arg should be a list of NorthNet Compound objects''')
        else:
            sys.exit('''Network.remove_compounds():
            compounds arg should be a list of NorthNet Compound objects''')

        remove_reactions = []
        for compound in compounds:
            remove_reactions.extend(self.NetworkCompounds[compound.SMILES].In)
            remove_reactions.extend(self.NetworkCompounds[compound.SMILES].Out)

        remove_reactions = list(set(remove_reactions))

        self.remove_reactions([self.NetworkReactions[r]
                                                    for r in remove_reactions])

        for compound in compounds:
            del self.NetworkCompounds[compound.SMILES]

    def add_reaction(self, reaction):
        '''
        Adds a reaction and its associated reactants and products into the
        Network.

        Parameters
        ----------
        reaction: NorthNet Reaction object
            reaction to be added
        '''
        if not isinstance(reaction, Classes.Reaction):
            sys.exit('''Network.add_reaction: reaction arg should be a NorthNet
            Reaction object.''')

        # check if the reaction is in NetworkReactions (avoid overwriting)
        # shouldn't be an issue if all reaction data is encapsulated properly
        # i.e. one reaction SMILES string to one set of data
        if reaction.ReactionSMILES not in self.NetworkReactions:
            smiles = reaction.ReactionSMILES

            # add the reaction into NetworkReactions
            self.NetworkReactions[smiles] = reaction

            for reactant in reaction.Reactants:
                self.add_compound(reactant)
                if smiles not in self.NetworkCompounds[reactant].Out:
                    self.NetworkCompounds[reactant].Out.append(smiles)

            for product in reaction.Products:
                self.add_compound(product)
                if smiles not in self.NetworkCompounds[product].In:
                    self.NetworkCompounds[product].In.append(smiles)

    def add_reactions(self,reactions):
        '''
        Use the standardised strings information in the Reaction objects
        to build them into the Network.

        Parameters
        ----------
        reactions: list of NorthNet Reaction objects
            reactions to be added to the Network
        '''
        if isinstance(reactions, list):
            check_reactions = [isinstance(c, Classes.Reaction) for c in reactions]
            if not all(check_reactions):
                sys.exit('''Network.add_reactions():
                reactions arg should be a list of NorthNet Reaction objects''')
        else:
            sys.exit('''Network.add_reactions():
            reactions arg should be a list of NorthNet Reaction objects''')

        for reaction in reactions:
            self.add_reaction(reaction)

    def remove_reactions(self, remove_reactions):
        '''
        Remove a list of reactions from the Network.

        Parameters
        ----------
        remove_reactions: list of NorthNet Reaction objects or
                          valid reaction SMILES
            reactions to be removed.
        '''

        if isinstance(remove_reactions, list):
            check_reactions = [isinstance(c, Classes.Reaction) for c in remove_reactions]
            check_for_strings = [isinstance(r, str) for r in remove_reactions]

            if not all(check_reactions) and not all(check_for_strings):
                sys.exit('''Network.remove_reactions():
                remove_reactions arg should be a list of NorthNet Reaction objects''')

        else:
            sys.exit('''Network.remove_reactions():
            reactions arg should be a list of NorthNet Reaction objects''')

        for reaction in remove_reactions:
            if isinstance(reaction, str):
                r_key = reaction
            else:
                r_key = reaction.ReactionSMILES

            for reactant in self.NetworkReactions[r_key].Reactants:
                self.NetworkCompounds[reactant].Out.remove(r_key)
            for product in self.NetworkReactions[r_key].Products:
                self.NetworkCompounds[product].In.remove(r_key)

            del self.NetworkReactions[r_key]

    def add_input(self, input_addition):
        '''
        Add a NetworkInput to the Network

        Parameters
        ----------
        input: NorthNet ReactionInput object
            Input to be added
        '''

        if not isinstance(input_addition, Classes.ReactionInput):
            sys.exit('''Network.add_input(): input arg must be a NorthNet
            NetworkInput object''')

        if input_addition.InputID in self.NetworkInputs:
            pass
        else:
            self.NetworkReactions[input_addition.ReactionSMILES] = input
            if input_addition.CompoundInput in self.NetworkCompounds:
                pass
            else:
                self.NetworkCompounds[input_addition.CompoundInput] = input

            self.NetworkCompounds[input_addition.CompoundInput].In.append(
                                                input_addition.ReactionSMILES
                                                )

            if input_addition.InputID in self.NetworkInputs:
                pass
            else:
                self.NetworkInputs[input_addition.InputID] = input_addition

            self.NetworkInputs[input_addition.InputID].Out.append(
                                                input_addition.ReactionSMILES
                                                )

    def add_inputs(self, inputs):
        '''
        For adding NetworkInput to the network

        Parameters
        ----------
        inputs: list of NortNet NetworkInput objects
            Inputs to be added to the network
        '''
        if isinstance(inputs, list):
            check_inputs = [isinstance(i, Classes.NetworkInput)
                                                    for i in inputs]
            if not all(check_inputs):
                sys.exit('''Network.add_inputs(): input arg must be a list of NorthNet
                NetworkInput object''')
        else:
            sys.exit('''Network.add_inputs(): input arg must be a list of NorthNet
            NetworkInput object''')

        for i in inputs:
            self.add_input(i)

    def add_output(self, output):
        '''
        Add a NetworkOutput to the Network

        Parameters
        ----------
        output NetworkOutput object
            Output to be added
        '''
        if not isinstance(input, Classes.NetworkOutput):
            sys.exit('''Network.add_output(): output arg must be a NorthNet
            NetworkOutput object''')

        if output.OutputID in self.NetworkOutputs:
            pass
        else:
            if output.CompoundOutput in self.NetworkCompounds:
                self.NetworkReactions[output.ReactionSMILES] = output
                self.NetworkCompounds[output.CompoundOutput].Out.append(
                                                        output.ReactionSMILES)
            else:
                # The compound is not currently in the network, so cannot
                # be an output
                pass

            if output.OutputID in self.NetworkOutputs:
                pass
            else:
                self.NetworkOutputs[output.OutputID] = output

            self.NetworkOutputs[output.OutputID].In.append(
                                                    output.ReactionSMILES)

    def add_outputs(self, outputs):
        '''
        For adding NetworkOutput to the network

        Parameters
        ----------
        outputs: list of NortNet NetworkOutput objects
            Outputs to be added to the network
        '''
        if isinstance(outputs, list):
            check_inputs = [isinstance(i, Classes.NetworkOutput)
                                                    for i in outputs]
            if not all(check_inputs):
                sys.exit('''Network.add_outputs(): outputs arg must be a list of NorthNet
                NetworkOutput object''')
        else:
            sys.exit('''Network.add_outputs(): outputs arg must be a list of NorthNet
            NetworkOutput object''')

        for out in outputs:
            self.add_output(out)

    def get_reaction(self, reaction):
        '''
        Convenience class for getting a reaction using a key

        Parameters
        ----------
        reaction: str
            Key in self.NetworkReactions

        Returns
        -------
        NorthNet Reaction object
        or
        None
        '''

        if not isinstance(reaction, str):
            sys.exit('''Network.get_reaction(): reaction arg must be a valid
            SMILES string.''')

        if reaction in self.NetworkReactions:
            return self.NetworkReactions[reaction]

        print('Reaction not found in Network')
        return None


    def get_reactants(self, reaction):
        '''
        Conveniently get the reactants of a reaction
        Parameters
        ----------
        reaction: str
            Key in self.NetworkReactions
        '''
        if not isinstance(reaction, str):
            sys.exit('''Network.get_reaction(): reaction arg must be a valid
            SMILES string.''')


        reaction_entry = self.get_reaction(reaction)
        if reaction_entry is None:
            print('Reaction not found in Network')
            return None

        return reaction_entry.Reactants

    def get_products(self, reaction):
        '''
        Conveniently get the products of a reaction
        Parameters
        ----------
        reaction: str
            Key in self.NetworkReactions
        '''
        if not isinstance(reaction, str):
            sys.exit('''Network.get_reaction(): reaction arg must be a valid
            SMILES string.''')

        reaction_entry = self.get_reaction(reaction)
        if reaction_entry is None:
            print('Reaction not found in Network')
            return None

        return reaction_entry.Products

    def get_reaction_template(self, reaction):
        '''
        Conveniently get the ReactionTemplate of a reaction
        Parameters
        ----------
        reaction: str
            Key in self.NetworkReactions
        '''
        if not isinstance(reaction, str):
            sys.exit('''Network.get_reaction(): reaction arg must be a valid
            SMILES string.''')

        reaction_entry = self.get_reaction(reaction)
        if reaction_entry is None:
            print('Reaction not found in Network')
            return None

        return reaction_entry.ReactionTemplate

    def get_reaction_SMARTS(self, reaction):
        '''
        Conveniently get the Reaction SMARTS of a reaction
        Parameters
        ----------
        reaction: str
            Key in self.NetworkReactions
        '''
        if not isinstance(reaction, str):
            sys.exit('''Network.get_reaction(): reaction arg must be a valid
            SMILES string.''')

        reaction_entry = self.get_reaction(reaction)

        if reaction_entry is None:
            print('Reaction not found in Network')
            return None

        if reaction_entry.ReactionTemplate is not None:
            return reaction_entry.ReactionTemplate.ReactionSMARTS

        return None

    def get_reaction_name(self, reaction):
        '''
        Conveniently get the Name of a reaction
        Parameters
        ----------
        reaction: str
            Key in self.NetworkReactions
        '''
        if not isinstance(reaction, str):
            sys.exit('''Network.get_reaction(): reaction arg must be a valid
            SMILES string.''')

        reaction_entry = self.get_reaction(reaction)

        if reaction_entry is None:
            print('Reaction not found in Network')
            return None

        if reaction_entry.ReactionTemplate is not None:
            return reaction_entry.ReactionTemplate.Name

        return None

    def convert_to_networkx(self):
        '''
        Converts NorthNet network object to networkx object.

        Returns
        -------
        G: networkx DiGraph object
            Networkx version of the NorthNet network.
        '''
        import networkx as nx

        Graph = nx.DiGraph()

        for node in self.NetworkCompounds:
            Graph.add_node(node)

        for reaction in self.NetworkReactions:
            for reactant in self.NetworkReactions[reaction].Reactants:
                for product in self.NetworkReactions[reaction].Products:
                    Graph.add_edge(reactant,reaction)
                    Graph.add_edge(reaction,product)

        return Graph

