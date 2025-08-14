import os
import wandb
from opt.eval import Eval
from opt.config import init_config
import json

if __name__ == '__main__':
    test_prompt =   "**Task: Personalized Sustainable Product Recommendations**\n"\
                    "Based on the user's session interactions, perform the following subtasks to provide tailored product recommendations that prioritize environmental sustainability and eco-friendliness:\n"\
                    "1. **Environmental Awareness Analysis**: Examine the user's interaction history to uncover explicit and implicit indicators of environmental consciousness, such as searches for green products, energy-efficient devices, or sustainable materials. Consider the user's engagement with products like 'Eco-Friendly Bamboo Toothbrush' or 'Energy-Star Certified LED Light Bulb' to assess their potential alignment with sustainability objectives.\n"\
                    "2. **Sustainability Preference Evaluation**: Assess the user's behavior and preferences in the context of their interactions with various products, including electronics, home goods, and other items. Evaluate the user's potential interest in sustainable features, such as recyclability, minimal packaging, or carbon offsetting, and consider the trade-offs between these features and other product attributes like price, performance, or convenience.\n"\
                    "3. **Environmental Value Understanding**: Develop a nuanced understanding of the user's environmental values and priorities, recognizing that sustainability preferences may not always be explicitly stated. Consider the user's interaction with products like 'Smart Thermostat with Energy Efficiency Features' or 'Reusable Stainless Steel Water Bottle' and evaluate whether these interactions suggest a focus on energy conservation, waste reduction, or other goals that may not be directly related to sustainability.\n"\
                    "4. **Eco-Friendly Product Ranking**: Based on the user's environmental awareness signals and sustainability preferences, rerank the 20 items in the candidate set according to their eco-friendliness, energy efficiency, and alignment with the user's values and priorities. Prioritize products with sustainable features, minimal environmental impact, and explicit eco-friendly certifications or labels. Provide a clear and transparent ranking of all items in the candidate set, ensuring that the recommended products align with the user's environmental consciousness and sustainability goals.\n"\
                    "**Note:** The ranking should reflect a careful consideration of the user's environmental awareness, sustainability preferences, and potential trade-offs between eco-friendliness and other product attributes. The order of all items in the candidate set must be provided, and the items for ranking must be within the candidate set."

    conf = init_config()
    data_path = "./dataset/green_dataset/processed/"
    test_file = f"{data_path}test_data_processed.json"

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    key = conf['api_key']
    if conf['use_wandb']:
        wandb.login(key=conf['wandb_api_key'])
        conf.pop('api_key')
        run = wandb.init(
            project=f"PO4ISR_{conf['dataset']}_test",
            config=conf,
            name=f"seed_{conf['seed']}",
        )
        text_table = wandb.Table(columns=["Input", "Target", "Response"])
    else:
        text_table = None
    conf['api_key'] = key

    eval_model = Eval(conf, test_data, text_table)
    results, target_rank_list, error_list = eval_model.run(test_prompt)

    result_save_path = f"./res/metric_res/{conf['dataset']}/"
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    results.to_csv(f"{result_save_path}ressults.csv", index=False)
    
    if conf['use_wandb']:
        run.log({"texts": text_table})