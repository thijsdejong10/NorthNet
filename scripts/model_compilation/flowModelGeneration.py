import numpy as np
from NorthNet.Classes import DataReport
from NorthNet.Classes import ReactionInput
from NorthNet.Classes import ReactionOutput
from NorthNet.Classes import Network
from NorthNet.Classes import ModelWriter
from NorthNet.Loading import load_network_from_reaction_list

data = DataReport(file="./scripts/model_compilation/exampleData.csv")

reaction_file = "./scripts/model_compilation/exampleReactionList.txt"

with open(reaction_file, "r") as file:
    text = file.read()

reaction_list = [x for x in text.split("\n") if x != ""]

network = load_network_from_reaction_list(reaction_list)

model = ModelWriter(network=network, experiment=data)

# Trim the flow profiles down to limit the amount of compiling needed
min_time = data.series_values[0] - 1000
max_time = data.series_values[-1]
inds = np.where(
    (model.flow_profile_time > min_time) & (model.flow_profile_time < max_time)
)[0]

model.flow_profile_time = model.flow_profile_time[inds]
model.flow_profile_time -= model.flow_profile_time[0]
model.sigma_flow = model.sigma_flow[inds]
for fl in model.flow_profiles:
    model.flow_profiles[fl] = model.flow_profiles[fl][inds]

model_text = model.write_to_module_text(numba_decoration="compile")

with open("./scripts/model_compilation/exampleModel.py", "w") as file:
    file.write(model_text)
