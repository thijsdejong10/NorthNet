from NorthNet.Classes import Network
from NorthNet.Classes import ModelWriter
from NorthNet.Loading import load_network_from_reaction_list

reaction_file = "./scripts/model_compilation/exampleReactionList.txt"

with open(reaction_file, "r") as file:
    text = file.read()

reaction_list = [x for x in text.split("\n") if x != ""]

network = load_network_from_reaction_list(reaction_list)

model = ModelWriter(network=network)

model_text = model.write_to_module_text(numba_decoration="compile")

print(model_text)

#quit()
with open(f"./scripts/model_compilation/test_model.py", "w") as file:
    file.write(model_text)
