# %%
import json
import torch
from pga import PGDAttack
from transformers import AutoTokenizer, AutoModelForCausalLM

# %%
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def load_intents(intents_file: str) -> list[dict]:
    with open(intents_file, "r") as f:
        return json.load(f)

attack = PGDAttack(tokenizer, model)
intents = load_intents("intents.json")

# %%
# tune hyperparams

# grid = ParameterGrid(
#     {
#         "learning_rate": [0.01, 0.001, 0.0001],
#         "max_gini": [0.9, 0.6, 0.4],
#     }
# )

# losses = {}
# for params in grid:
#     attack.learning_rate = params["learning_rate"]
#     attack.max_gini = params["max_gini"]
#     attack.cosine_decay_mult = 0.325

#     intent, target = intents[0]["intent"], intents[0]["target"]
#     best_discritized_loss = run(attack, intent, target, max_iters=500)
#     filepath = "/workspace/haize/hyperparams.csv"
#     with open(filepath, "a") as f:
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         data_row = {"timestamp": timestamp, "loss": best_discritized_loss.item(), **params}
#         writer = csv.DictWriter(f, fieldnames=data_row.keys())
#         if os.stat(filepath).st_size == 0:
#             writer.writeheader()
#         writer.writerow(data_row)

# %%
def evaluate_attack(model_response: str, target: str):
    return model_response.startswith(target)


def run(attack, intent, target, max_iters):
    print(f"Running PGD on: {intent}")

    # 1. run attack
    results = attack.optimize_attack(intent, target, max_iters)
    best_relaxed_suffix, best_discritized_suffix, niters = results

    # 2. calc metrics
    best_relaxed_loss, _ = attack.compute_loss_components(intent, best_relaxed_suffix, target)
    best_p = torch.exp(-best_relaxed_loss)

    # 3. get response
    model_response = attack.get_model_response(intent, best_relaxed_suffix)
    success = evaluate_attack(model_response, target)

    # 4. log output
    print(
        f"Success: {success}, loss: {best_relaxed_loss:.4f}, best_p: {best_p:.4f}, niters: {niters}"
    )
    print(f"Intent: {intent}")
    # print(f"Adversarial suffix: {adversarial_suffix_str}")
    print(f"Model output: {model_response}")
    print("-" * 50)


for intent_data in intents:
    attack.learning_rate = 0.01
    attack.max_gini = 0.9
    attack.cosine_decay_mult = 0.325

    intent, target = intent_data["intent"], intent_data["target"]
    run(attack, intent, target, max_iters=1000)
