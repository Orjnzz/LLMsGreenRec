import math
import random
import re
import json
from opt.utils import ndcg, extract_edit_prompt, extract_item_list
from agno.agent import Agent
from concurrent.futures import ThreadPoolExecutor

class EvalAgent(Agent):
    def __init__(self,model,name='EvalAgent'):
        super().__init__(
            model=model,
            description="Using initial prompt to rerank candidate set based on user session interactions, with a focus on environmentally friendly product recommendations.",
            name=name
        )
    def evaluate(self,prompt,session_data):
        input_str = session_data['input']
        full_prompt = f"{prompt}\n{input_str}"
        # print("- Full prompt:\n",full_prompt)
        response = self.run(full_prompt).content
        return response

class DetectErrrorAgent:
    def detect_error(self, response, target):
        result_list = extract_item_list(response, target)
        # print("Detect error processing: ")
        # print("- Response: ", response)
        # print("- Target: ", target)
        # print("- Result: ", result_list)
        if not result_list:
            return True
        threshold = 10
        rank = int(result_list[-1])
        # print("- Rank: ", rank)
        return rank >= threshold

class InferReasonAgent(Agent):
    def __init__(self,model,name='InferReasonAgent'):
        super().__init__(
            model=model,
            description="Analyzes failures in eco-friendly product recommendation prompts and identifies gaps in understanding user environmental preferences.",
            name=name
        )
    def infer(self,error_input,prompt,num_feedbacks):
        reason_prompt = (
            "I'm developing a zero-shot recommender prompt for an eco-friendly product recommendation system.\n"
            "My current prompt is \"$prompt$\"\n"
            "However, this prompt failed to properly recommend sustainable products for this user interaction: $error_case$ "
            "Provide $num_feedbacks$ reasons why the prompt may have failed to capture the user's environmental consciousness or sustainability preferences.\n"
            # "Consider factors such as: misunderstanding eco-friendly signals, overlooking sustainability indicators, or misaligning with environmental values.\n"
            "Wrap each reason with <START> and <END>"
        ).replace("$prompt$", prompt).replace("$error_case$", error_input).replace("$num_feedbacks$", str(num_feedbacks))
        # reasons = self.run(reason_prompt).content
        for attempt in range(3):  # Thử lại tối đa 3 lần
            response = self.run(reason_prompt)
            if response is not None and hasattr(response, 'content') and response.content is not None:
                reasons = response.content
                # print("Reasons: ",reasons)
                break
            print(f"Attempt {attempt + 1}: Failed to get reasons, retrying...")
        extract_reasons = extract_edit_prompt(reasons)
        # print("Len extract reasons: ",len(extract_reasons))
        if len(extract_reasons) == 0:
            reasons = reasons
        else:
            reasons = extract_reasons
        # print("- Reasons: ", reasons)
        return reasons

class RefinePromptAgent(Agent):
    def __init__(self,model,name='RefinePromptAgent'):
        super().__init__(
            model=model,
            description="Refines recommendation prompts to better capture user environmental preferences and improve sustainable product suggestions.",
            name=name
        )
    def refine(self,prompt,error_input,reasons):
        refine_prompt = (
            "I'm developing a zero-shot recommender prompt for an eco-friendly product recommendation system.\n"
            "My current prompt is \"$prompt$\"\n"
            "This prompt failed to properly recommend sustainable products for: $error_case$\n"
            "Analysis shows the issues are: $reasons$.\n"
            "Create an improved prompt that better understands user environmental consciousness, captures sustainability preferences more accurately, and prioritizes eco-friendly products effectively.\n"
            "Focus on enhancing the system's ability to identify environmental awareness signals and align recommendations with sustainability goals.\n"
            "Wrap the improved prompt with <START> and <END>.\n"
            "The enhanced prompt is:"
        ).replace("$prompt$", prompt).replace("$error_case$", error_input).replace("$reasons$", '\n'.join(reasons))
        refined_prompt = self.run(refine_prompt).content
        extract_refined_prompts = extract_edit_prompt(refined_prompt)
        if extract_refined_prompts:
            refined_prompt = extract_refined_prompts[0]  
            # print("- Refined prompt 1 : ", refined_prompt)
            return refined_prompt
        else:
            # print("- Refined prompt 2 : ", refined_prompt)
            return refined_prompt

class AugmentAgent(Agent):
    def __init__(self,model,name='AugmentAgent'): 
        super().__init__(
            model=model,
            description="Generates variations of refined prompts while preserving semantics for sustainable recommendation tasks.",
            name=name
        )
    def augment(self,refined_prompt,additional_sample):
        augment_prompt = (
            "Generate a variation of the following instruction for sustainable product recommendations while keeping the semantic meaning.\n"
            "Input: $refined_prompt$\n"
            "Output:"
        ).replace("$refined_prompt$", refined_prompt)
        augmented_prompts = [self.run(augment_prompt).content for _ in range(additional_sample)]
        # print("- augmented_prompts: ", augmented_prompts)
        # print("Len augmented prompts: ", len(augmented_prompts))
        return augmented_prompts

class SelectionAgent(Agent):
    def __init__(self, model, name='SelectionAgent'):
        super().__init__(
            model=model,
            description='Selects the best prompts for sustainable product recommendations using an Upper Confidence Bound bandit algorithm to balance exploration and exploitation.',
            name=name
        )

    def select(self, prompt_list, train_data, time_steps=16, explore_param=2, sample_num=32, beam_width=5):
        if not prompt_list:
            return []
        selections = [0] * len(prompt_list)
        rewards = [0] * len(prompt_list)
        for t in range(1, time_steps + 1):
            sample_data = random.sample(train_data, min(sample_num, len(train_data)))
            ucb_values = [
                (rewards[i] / selections[i] + explore_param * math.sqrt(math.log(t) / selections[i])) if selections[i] > 0 else float('inf')
                for i in range(len(prompt_list))
            ]
            selected_idx = ucb_values.index(max(ucb_values))
            selected_prompt = prompt_list[selected_idx]
            with ThreadPoolExecutor(max_workers=100) as executor:
                predictions = list(executor.map(lambda data: self.run(f"{selected_prompt}\n{data['input']}").content, sample_data))
            reward = 0
            for prediction, data in zip(predictions, sample_data):
                result_list = extract_item_list(prediction, data['target'])
                if result_list:
                    target_idx = int(result_list[-1])
                    reward += ndcg(target_idx)
            selections[selected_idx] += len(sample_data)
            rewards[selected_idx] += reward
        prompt_reward_pairs = list(zip(rewards, prompt_list))
        prompt_reward_pairs.sort(reverse=True, key=lambda x: x[0])
        return [pair[1] for pair in prompt_reward_pairs[:beam_width]]