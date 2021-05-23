from rdkit.Chem import AllChem
from rdkit import Chem
from NorthNet import Classes
from NorthNet import info_params
import numpy as np

class Substructure:
    '''
    Class to store substructures.
    '''
    def __init__(self,SMARTS):
        '''
        Parameters
        ----------
        SMARTS: str
            SMARTS corresponding to substructure.
        '''
        self.Mol = Chem.MolFromSmarts(SMARTS)

        if Chem.MolToSmarts(self.Mol) == None:
            self.SMARTS = SMARTS
        else:
            self.SMARTS = Chem.MolToSmarts(self.Mol)

        self.In = []
        self.Out = []

class Compound:
    '''
    Class to store compound information.
    '''
    def __init__(self,SMILES):
        '''
        Parameters
        ----------
        mol: str
            SMILES corresponding to compound.
        '''
        self.Mol = Chem.MolFromSmiles(SMILES)
        if self.Mol == None:
            self.SMILES = SMILES
        else:
            self.SMILES = Chem.MolToSmiles(self.Mol)

        self.In = []
        self.Out = []

        self.ReactiveSubstructures = None

class NetworkInput:
    '''
    Class to store inputs into reaction network
    '''
    def __init__(self,id):
        '''
        Designed to behave like a compound object
        self.SMILES does not actually hold a valid SMILES string,
        it stores the input ID code.

        Parameters
        ----------
        id: str
            token for reaction input
            should follow the convention
            SMILES_#0
        '''
        self.Mol = Chem.MolFromSmiles(id.split('_')[0])

        # Note that the self.SMILES does not
        # actually hold a valid SMILES string,
        # it stores the input ID code
        self.SMILES = id

        self.In = None
        self.Out = []

        self.ReactiveSubstructures = None

class NetworkOutput:
    '''
    Class to store inputs into reaction network
    '''
    def __init__(self,id):
        '''
        Designed to behave like a compound object
        self.SMILES does not actually hold a valid SMILES string,
        it stores the input ID code.

        Parameters
        ----------
        id: str
            token for reaction input
            should follow the convention
            SMILES_#0
        '''
        self.Mol = None

        # Note that the self.SMILES does not
        # actually hold a valid SMILES string,
        # it stores the input ID code
        self.SMILES = id

        self.In = []
        self.Out = None

        self.ReactiveSubstructures = None

class Reaction_Template:
    '''
    Class for reaction templates.
    '''
    def __init__(self,name,reaction_SMARTS, reactant_substructs, product_substructs):
        '''
        Parameters
        ----------
        name: str
            name for reaction.
        reaction_SMARTS: list
            List of reaction SMARTS strings.
        '''

        self.Name = name
        self.Reaction = AllChem.ReactionFromSmarts(reaction_SMARTS)
        self.ReactionSMARTS = AllChem.ReactionToSmarts(self.Reaction)
        self.ReactantSubstructures = reactant_substructs
        self.ProductSubstructures  = product_substructs

class Generated_Reaction:
    '''
    An object designed to store generated reactions and their characteristics.
    '''
    def __init__(self,rdkit_reaction,reaction_template):
        '''
        Parameters
        ----------
        rdkit_reaction: rdkit Reaction object
            Reaction object for the reaction.
        reaction_template: NorthNet Reaction_Template object
            Reaction template which created the reaction.
        '''

        self.Reaction = rdkit_reaction
        self.ReactionSMILES = AllChem.ReactionToSmiles(rdkit_reaction)
        self.Reaction_Template = reaction_template

class Reaction_Database_Entry:
        '''
        Class for storing information on a single reaction from a database.
        '''
        def __init__(self, header, line):
            '''
            Parameters
            ----------
            header: list
                Header for field entries.
            line: list
                List of field entries from database.
            '''

            self.Info = {k:v for k,v in zip(header,line)}
            self.ReactionSMILES = line[header.index('Reaction')].strip(',')

        def recompile(self):
            '''Creates a modified database entry by modifiying the original with the
            self.Info.'''

            newline = []
            for k in [*self.Info]:
                newline.append(self.Info[k])

            return newline

        def get_header(self):
            '''
            For getting the field entries for the database entry.
            '''

            newline = []
            for k in [*self.Info]:
                newline.append(k)

            return newline

class Reaction:
    '''
    A class which unites Generated_Reaction and Reaction_Database_Entry classes
    with the same reaction SMILES so they are combined in the view of a Network.

    Could use multiple inheritance, but this seems simpler.
    '''
    def __init__(self, reaction_object):
        '''
        Parameters
        ----------
        reaction_object: NorthNet Generated_Reaction or Reaction_Database_Entry objects
            Prototypical reaction to initialise the object.

        Attributes
        ----------
        self.Database_Entries: list
            List of Reaction_Database_Entry objects
        self.Generation_Details: list
            list of Generated_Reaction objects
        '''

        if reaction_object.__class__.__name__ == "Reaction_Database_Entry":
            self.Database_Entries = [reaction_object]
            self.Generation_Details = []

            spl = reaction_object.ReactionSMILES.split('>>')

            self.Reactants = [r for r in spl[0].split('.') if Chem.MolFromSmiles(r) != None and r != ""]
            self.Products  = [p for p in spl[1].split('.') if Chem.MolFromSmiles(p) != None and p != ""]

            self.ReactionTemplate = None
            self.MappedReaction = None

            self.Classification = 'Reaction_Database_Entry'
            self.update_reaction()

        if reaction_object.__class__.__name__ == "Generated_Reaction":
            self.Database_Entries = []
            self.Generation_Details = [reaction_object]
            self.Reactants = [Chem.MolToSmiles(r) for r in reaction_object.Reaction.GetReactants()]
            self.Products  = [Chem.MolToSmiles(p) for p in reaction_object.Reaction.GetProducts()]

            self.ReactionTemplate = reaction_object.Reaction_Template
            self.MappedReaction = None # If mapping can be introduced inot the reaction generation process. If it is, this can be changed accordingly.
            self.Classification = reaction_object.Reaction_Template.Name
            self.Classification = 'generated_reaction'
            self.update_reaction()

        if reaction_object.__class__.__name__ == "str":
            self.Database_Entries = []
            self.Generation_Details = []
            self.Reactants, self.Products = self.reactants_products_from_string(reaction_object)
            self.ReactionTemplate = None
            self.MappedReaction = None # If mapping can be introduced inot the reaction generation process. If it is, this can be changed accordingly.
            self.Classification = 'reaction'
            self.update_reaction()

    def add_reaction_entry(self, new_entry):
        if new_entry.__class__.__name__ == "Reaction_Database_Entry":
            self.Database_Entries.append(new_entry)
        if new_entry.__class__.__name__ == "Generated_Reaction":
            self.Generation_Details.append(new_entry)

    def reactants_products_from_string(self, reaction_smiles):
        split_rxn_smiles = reaction_smiles.split('>>')
        reactants = [x for x in split_rxn_smiles[0].split('.') if x != '']
        products = [x for x in split_rxn_smiles[1].split('.') if x != '']
        return reactants, products

    def update_reaction(self):
        '''
        Updates ReactionSMILES and Species attributes according to self.Reactants
        and self.Products.
        '''
        rxn = AllChem.ChemicalReaction() # Create an empty chemical reaction
        [rxn.AddReactantTemplate(Chem.MolFromSmiles(r)) for r in self.Reactants]
        [rxn.AddProductTemplate(Chem.MolFromSmiles(p)) for p in self.Products]
        self.Reaction = rxn
        self.ReactionSMILES = AllChem.ReactionToSmiles(rxn)

        for d in self.Database_Entries:
            d.ReactionSMILES = self.ReactionSMILES
            d.Info["Reaction"] = self.ReactionSMILES

class ReactionInput:
    '''
    Class to store reaction inputs as into reaction network
    '''
    def __init__(self, reaction_input_string):
        '''
        Designed to behave like a Reaction object
        self.SMILES does not actually hold a valid SMILES string,

        Parameters
        ----------
        reaction_input_string: str
            token for reaction input
            should follow the convention
            SMILES_#0>>SMILES
        '''
        self.Reaction = None
        self.ReactionSMILES = reaction_input_string
        self.Reactants, self.Products = self.reactants_products_from_string(reaction_input_string)

        self.CompoundInput = self.Products[0]
        self.InputID = self.Reactants[0]

        self.ReactionTemplate = None

        self.MappedReaction = None
        self.Database_Entries = []
        self.Generation_Details = []
        self.Classification = 'reaction_input'
        self.Info = None

    def reactants_products_from_string(self, reaction_smiles):
        split_rxn_smiles = reaction_smiles.split('>>')
        reactants = [x for x in split_rxn_smiles[0].split('.') if x != '']
        products = [x for x in split_rxn_smiles[1].split('.') if x != '']
        return reactants, products

class ReactionOutput:
    '''
    Class to store reaction inputs as into reaction network
    '''
    def __init__(self, reaction_input_string):
        '''
        Designed to behave like a Reaction object
        self.SMILES does not actually hold a valid SMILES string,

        Parameters
        ----------
        reaction_input_string: str
            token for reaction input
            should follow the convention
            SMILES_#0>>SMILES
        '''
        self.Reaction = None
        self.ReactionSMILES = reaction_input_string
        self.Reactants, self.Products = self.reactants_products_from_string(reaction_input_string)

        self.OutputID = self.Products[0]
        self.CompoundOutput = self.Reactants[0]

        self.ReactionTemplate = None

        self.MappedReaction = None
        self.Database_Entries = []
        self.Generation_Details = []
        self.Classification = 'reaction_input'
        self.Info = None

    def reactants_products_from_string(self, reaction_smiles):
        split_rxn_smiles = reaction_smiles.split('>>')
        reactants = [x for x in split_rxn_smiles[0].split('.') if x != '']
        products = [x for x in split_rxn_smiles[1].split('.') if x != '']
        return reactants, products


class Network:
    def __init__(self, reactions, name, description):
        '''
        A network objects consisting of Edges (reactions) and nodes (species).
        The Network object is initialised with a list of Reaction objects. If
        the list is empty, then the network is initialised as an empty network.
        '''
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

        '''4. Create dictionary of n twork outputs'''
        self.NetworkOutputs = {}

        if len(reactions) == 0:
            pass
        else:
            self.add_reactions(reactions)

    def add_reactions(self,reactions):
        for r in reactions:

            if r.ReactionSMILES in self.NetworkReactions:
                self.NetworkReactions[r.ReactionSMILES].add_reaction_entry(r)

            else:
                self.NetworkReactions[r.ReactionSMILES] = r

                for a in r.Reactants:
                    if a in self.NetworkCompounds:
                        if r.ReactionSMILES in self.NetworkCompounds[a].Out:
                            pass
                        else:
                            self.NetworkCompounds[a].Out.append(r.ReactionSMILES)
                    else:
                        self.NetworkCompounds[a] = Classes.Compound(a)
                        self.NetworkCompounds[a].Out.append(r.ReactionSMILES)

                for b in r.Products:
                    if b in self.NetworkCompounds:
                        if r.ReactionSMILES in self.NetworkCompounds[b].In:
                            pass
                        else:
                            self.NetworkCompounds[b].In.append(r.ReactionSMILES)
                    else:
                        self.NetworkCompounds[b] = Classes.Compound(b)
                        self.NetworkCompounds[b].In.append(r.ReactionSMILES)

    def add_inputs(self, inputs):
        for i in inputs:
            if i in self.NetworkInputs:
                pass
            else:
                self.NetworkReactions[i.ReactionSMILES] = i

                if i.CompoundInput in self.NetworkCompounds:
                    self.NetworkCompounds[i.CompoundInput].In.append(i.ReactionSMILES)
                else:
                    self.NetworkCompounds[i.CompoundInput] = Classes.NetworkInput(i.NetworkInput)
                    self.NetworkCompounds[a].In.append(i.ReactionSMILES)

                if i.InputID in self.NetworkInputs:
                    self.NetworkInputs[i.InputID].Out.append(i.ReactionSMILES)
                else:
                    self.NetworkInputs[i.InputID] = Classes.NetworkInput(i.InputID)
                    self.NetworkInputs[i.InputID].Out.append(i.ReactionSMILES)

    def add_outputs(self, outputs):
        for i in outputs:
            if i in self.NetworkOutputs:
                pass
            else:
                if i.CompoundOutput in self.NetworkCompounds:
                    self.NetworkReactions[i.ReactionSMILES] = i
                    self.NetworkCompounds[i.CompoundOutput].Out.append(i.ReactionSMILES)
                else:
                    # The compound is not currently in the network, so cannot
                    # be an output
                    pass
                if i.OutputID in self.NetworkOutputs:
                    self.NetworkOutputs[i.OutputID].In.append(i.ReactionSMILES)
                else:
                    self.NetworkOutputs[i.OutputID] = Classes.NetworkOutput(i.OutputID)
                    self.NetworkOutputs[i.OutputID].In.append(i.ReactionSMILES)

    def add_compounds(self,compounds):
        for n in compounds:
            if n.SMILES in self.NetworkCompounds:
                pass
            else:
                self.NetworkCompounds[n.SMILES] = n

    def remove_compounds(self, compounds):

        remove_reactions = []
        for c in compounds:
            remove_reactions.extend(self.NetworkCompounds[c].In)
            remove_reactions.extend(self.NetworkCompounds[c].Out)

        remove_reactions = list(set(remove_reactions))

        self.remove_reactions(remove_reactions)

        for c in compounds:
            del self.NetworkCompounds[c]

    def remove_reactions(self, remove_reactions):
        for r in remove_reactions:
            for rc in self.NetworkReactions[r].Reactants:
                self.NetworkCompounds[rc].Out.remove(r)
            for p in self.NetworkReactions[r].Products:
                self.NetworkCompounds[p].In.remove(r)

            del self.NetworkReactions[r]


class Substructure_Network:
    '''
    A network designed to show the relationship between functional
    group transformations.
    '''
    def __init__(self,reactions,name,description):
        '''
        reactions: list
            List of reaction objects with extracted functional group transformations.
        name: str
            A name for the network.
        description: str
            A description of the network.
        '''

        '''
        Basic idea:
            Nodes will be individual reaction centres and functional group transformations
            Edges will connect the reaction centres to functional group transformations.
        '''

        self.Name = name
        self.Description = description

        self.SNetworkSubstructs = {}

        self.SNetworkTemplates = {}

        self.SNetworkCompounds = {}

        if len(reactions) == 0:
            pass
        else:
            self.add_reactions(reactions)

    def add_reactions(self, reactions):

        for r in reactions:

            r_key = r.ReactionTemplate.ReactionSMARTS

            if r_key in self.SNetworkTemplates:
                pass
            else:
                self.SNetworkTemplates[r_key] = r

            for r_substruct in r.ReactionTemplate.ReactantSubstructures:
                # connect to reaction
                if r_substruct in self.SNetworkSubstructs:
                    self.SNetworkSubstructs[r_substruct].Out.append(r_key)
                else:
                    self.SNetworkSubstructs[r_substruct] = Classes.Substructure(r_substruct)
                    self.SNetworkSubstructs[r_substruct].Out.append(r_key)

                # connect to compound
                for reac in r.Reactants:
                    if reac in self.SNetworkCompounds:
                        if self.SNetworkCompounds[reac].Mol.HasSubstructMatch(self.SNetworkSubstructs[r_substruct].Mol):
                            self.SNetworkCompounds[reac].Out.append(r_substruct)
                            self.SNetworkSubstructs[r_substruct].In.append(reac)
                        else:
                            pass
                    else:
                        self.SNetworkCompounds[reac] = Classes.Compound(reac)

                        if self.SNetworkCompounds[reac].Mol.HasSubstructMatch(self.SNetworkSubstructs[r_substruct].Mol):
                            self.SNetworkCompounds[reac].Out.append(r_substruct)
                            self.SNetworkSubstructs[r_substruct].In.append(reac)
                        else:
                            pass

            for p_substruct in r.ReactionTemplate.ProductSubstructures:
                # connect to reaction
                if p_substruct in self.SNetworkSubstructs:
                    self.SNetworkSubstructs[p_substruct].In.append(r_key)
                else:
                    self.SNetworkSubstructs[p_substruct] = Classes.Substructure(p_substruct)
                    self.SNetworkSubstructs[p_substruct].In.append(r_key)

                # connect to compound
                for prod in r.Products:
                    if p_substruct == "[Ch:3][O:4]" and prod in self.SNetworkCompounds:
                        pass
                        #print(prod,p_substruct,self.SNetworkCompounds[prod].Mol.HasSubstructMatch(self.SNetworkSubstructs[p_substruct].Mol))
                    if prod in self.SNetworkCompounds:
                        if self.SNetworkCompounds[prod].Mol.HasSubstructMatch(self.SNetworkSubstructs[p_substruct].Mol):
                            self.SNetworkCompounds[prod].In.append(p_substruct)
                            self.SNetworkSubstructs[p_substruct].Out.append(prod)
                        else:
                            pass
                    else:
                        self.SNetworkCompounds[prod] = Classes.Compound(prod)
                        if self.SNetworkCompounds[prod].Mol.HasSubstructMatch(self.SNetworkSubstructs[p_substruct].Mol):
                            self.SNetworkCompounds[prod].In.append(p_substruct)
                            self.SNetworkSubstructs[p_substruct].Out.append(prod)
                        else:
                            pass

class ModelWriter:
    def __init__(self, network = None, experiment = None,
                       input_token = '_#0',
                       output_token = 'Sample',
                       flowrate_time_conversion = 3600,
                       time_limit = False):
        '''
        network: NorthNet Network
        experiment: NorthNet DataReport,
        input_token: str
        output_token: str
        flowrate_time_conversion: float
            Conversion for the time component:
            CAUTION: this class currently expects flow rates to be given
            in units of uL/h and reactor volumes to be given in uL,
            so conversion errors may result if the input DataReport's
            attributes fall out of this pattern.
        time_limt: bool or float
            How far in time the flow profile will be considered
            in generating the model.
        '''

        self.network = network
        self.input_token = input_token
        self.output_token = output_token
        self.flowrate_time_conversion = flowrate_time_conversion
        self.time = np.inf
        self.time_limit = time_limit
        self.observed_compounds = []
        self.reactor_volume = 1.0
        self.flow_profile_time = []
        self.flow_profiles = {}
        self.sigma_flow = []
        self.name = ''
        self.species = {}
        self.rate_constants = {}
        self.inputs = {}
        self.inflows = {}
        self.outflows = {}

        if network == None:
            pass
        else:
            self.get_network_tokens()

        if experiment == None:
            pass
        else:
            self.load_experiment_details(experiment)

    def get_network_tokens(self):
        '''

        '''
        network = self.network
        compounds = [network.NetworkCompounds[x] for x in network.NetworkCompounds]
        network_inputs = [*network.NetworkInputs]
        inflows = [r for r in network.NetworkReactions if self.input_token in r]
        outflows = [r for r in network.NetworkReactions if self.output_token in r]
        reactions = [r for r in network.NetworkReactions]

        for i in inflows:
            reactions.remove(i)
        for o in outflows:
            reactions.remove(o)

        species = {s.SMILES:"S[{}]".format(c) for c,s in enumerate(compounds) if s != ''}
        rate_consts = {k:"k[{}]".format(c) for c,k in enumerate(reactions)}
        inputs = {i:0.0 for c,i in enumerate(network_inputs)}
        flow_ins = {i:'I[{}]'.format(c) for c,i in enumerate(inflows)}
        flow_outs = {o:'sigma_flow' for o in outflows}

        self.name = network.Name
        self.species = species
        self.rate_constants = rate_consts
        self.inputs = inputs
        self.inflows = flow_ins
        self.outflows = flow_outs

    def write_equation_text(self):

        '''
        Parameters
        ----------

        Returns
        -------
        eq_text: str
            Rate equations in text form.
        '''

        network = self.network
        compounds = [x for x in network.NetworkCompounds]
        reactions = [*network.NetworkReactions]

        eq_text = ""

        for count,c in enumerate(compounds):
            eq_text += "P[{}] = ".format(count)
            for i in network.NetworkCompounds[c].In:
                if '_#0' in i:
                    in_compound = network.NetworkReactions[i].InputID
                    ki = '+{}*{}'.format(self.inflows[i], self.inputs[in_compound])
                    eq_text += ki
                else:
                    reactants = network.NetworkReactions[i].Reactants
                    # remove water from reactants
                    reactants = [x for x in reactants if x != 'O']

                    ki = "+{}*".format(self.rate_constants[i])

                    if len(reactants) == 0:
                        specs = ''#inflows[i]
                    else:
                        specs = "*".join([self.species[x] for x in reactants])

                    eq_text += "{}{}".format(ki,specs)

            for o in network.NetworkCompounds[c].Out:
                if 'Sample' in o:
                    out_compound = network.NetworkReactions[o].CompoundOutput
                    ki = '-{}*{}'.format(self.outflows[o], self.species[out_compound])
                    eq_text += ki
                else:
                    ki = "-{}*".format(self.rate_constants[o])
                    specs = "*".join([self.species[x] for x in network.NetworkReactions[o].Reactants])
                    eq_text += "{}{}".format(ki,specs)

            eq_text += "\n"

        return eq_text

    def load_experiment_details(self, experiment):
        if experiment.series_unit == 'time/ s':
            self.time = experiment.series_values

        for d in experiment.data:
            self.observed_compounds.append(d.split('/')[0].split(' ')[0])

        for c in experiment.conditions:
            if 'reactor_volume' in c:
                self.reactor_volume =experiment.conditions[c]
            elif ' M' in c:
                standardised_key = c.strip('M').strip(' ').strip('/')
                clef = info_params.canonical_SMILES[standardised_key]
                for f in self.inputs:
                    if clef in f:
                        self.inputs[f] = experiment.conditions[c]
            elif 'time' in c and 'flow' in c:
                if self.time_limit:
                    t_lim = min(np.amax(self.time), self.time_limit)
                else:
                    t_lim = np.amax(self.time)

                idx = np.where(experiment.conditions[c] < t_lim)[0]
                self.flow_profile_time = experiment.conditions[c][idx]

            elif 'flow' in c and not 'time' in c:
                standardised_key = c.split(' ')
                clef = info_params.canonical_SMILES[standardised_key[0].split('_')[0]]
                self.flow_profiles[clef] = experiment.conditions[c]

        self.sigma_flow = np.zeros(len(self.flow_profile_time))
        for f in self.flow_profiles:
            self.flow_profiles[f] = self.flow_profiles[f][idx]/self.reactor_volume
            self.flow_profiles[f] /= self.flowrate_time_conversion
            self.sigma_flow += self.flow_profiles[f]

    def write_flow_profile_text(self):
        collection_array = np.zeros((len(self.flow_profiles)+2,
                                     len(self.flow_profile_time)))

        collection_array[0] = self.flow_profile_time
        for c,f in enumerate(self.inputs,1):
            collection_array[c] = self.flow_profiles[f.split('_')[0]]

        collection_array[-1] = self.sigma_flow

        text = 'F = np.array('
        text += np.array2string(collection_array,
                                     formatter={'float_kind':lambda x: "%.9f" % x},
                                     separator=',',threshold=np.inf)
        text += ')'

        return text

    def write_model_equation_text(self):
        '''
        Returns
        -------
        eq_text: str
            Rate equations in text form.
        '''
        network = self.network

        compounds = [x for x in network.NetworkCompounds]
        reactions = [*network.NetworkReactions]


        eq_text = ""

        for count,c in enumerate(compounds):
            eq_text += "P[{}] = ".format(count)
            for i in network.NetworkCompounds[c].In:
                if '_#0' in i:
                    in_compound = network.NetworkReactions[i].InputID
                    ki = '+{}*{}'.format(self.inflows[i], self.inputs[in_compound])
                    eq_text += ki
                else:
                    reactants = network.NetworkReactions[i].Reactants
                    # remove water from reactants
                    reactants = [x for x in reactants if x != 'O']

                    ki = "+{}*".format(self.rate_constants[i])

                    if len(reactants) == 0:
                        specs = ''#inflows[i]
                    else:
                        specs = "*".join([self.species[x] for x in reactants])

                    eq_text += "{}{}".format(ki,specs)

            for o in network.NetworkCompounds[c].Out:
                if 'Sample' in o:
                    out_compound = network.NetworkReactions[o].CompoundOutput
                    ki = '-{}*{}'.format(self.outflows[o], self.species[out_compound])
                    eq_text += ki
                else:
                    ki = "-{}*".format(self.rate_constants[o])
                    specs = "*".join([self.species[x] for x in network.NetworkReactions[o].Reactants])
                    eq_text += "{}{}".format(ki,specs)

            eq_text += "\n"

        return eq_text

    def write_to_module_text_A(self, numba_decoration = False):
        get_index = lambda x: int(x[x.find("[")+1:x.find("]")])

        flow_profile_text = self.write_flow_profile_text()
        model_text = self.write_model_equation_text().split('\n')

        lines = ["import numpy as np\n"]
        if numba_decoration:
            lines.append("import numba\n")
            lines.append("\n")
            lines.append("@numba.jit(numba.float64[:](numba.float64,numba.float64[:],numba.float64[:]),\n"
                       "\tlocals={'P': numba.float64[:],'F': numba.float64[:,:],'I':numba.float64[:]},nopython=True)\n")
        lines.append("def model_function(time, S, k):\n")
        lines.append("\n")
        lines.append("\tP = np.zeros(len(S))\n")
        lines.append("\n")
        lines.append("\t")
        lines.append(flow_profile_text)
        lines.append("\n")
        lines.append("\n")
        lines.append("\tidx = np.abs(F[0] - time).argmin()\n")
        lines.append("\n")
        lines.append("\tI = F[1:-1,idx]\n")
        lines.append("\n")
        lines.append('\tsigma_flow = F[-1,idx]\n')
        lines.append("\n")
        lines.append("\n")

        for m in model_text:
            lines.append("\t{}\n".format(m))

        lines.append("\treturn P\n")
        lines.append("\n")
        lines.append("def wrapper_function(time, S, k):\n")
        lines.append("\treturn model_function(time, S, k)\n")
        lines.append("\n")

        lines.append("\n")
        lines.append("species = {")
        for k in self.species:
            idx = get_index(self.species[k])
            lines.append("'{}':{},".format(k,idx))
        lines.append("}\n")

        lines.append("\n")
        lines.append("reactions = {")
        for k in self.rate_constants:
            idx = get_index(self.rate_constants[k])
            lines.append("'{}':{},".format(k,idx))
        lines.append("}\n")

        lines.append("\n")
        lines.append("inputs = {")
        for k in self.inputs:
            idx = self.inputs[k]
            lines.append("'{}':{},".format(k,idx))

        lines.append("}\n")
        lines.append("\n")

        lines.append("k = np.zeros(max(reactions.values())+1) # rate constants\n")
        lines.append("\n")
        lines.append("S = np.zeros(len(species)) # initial concentrations\n")
        lines.append("\n")

        lines.append("C = np.zeros(len(inputs)) # input concentrations\n")
        lines.append("\n")


        text = ''
        for l in lines:
            text += l

        return text
