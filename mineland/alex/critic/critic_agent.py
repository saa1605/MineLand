'''
Critic Agent
'''
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import AzureChatOpenAI

from ..prompt_template import load_prompt


class CriticInfo(BaseModel):
    reasoning: str = Field(description="reasoning")
    success: bool = Field(description="success")
    critique: str = Field(description="critique")

class CriticAgent():
    '''
    Critic Agent
    Generate a critique for the last short-term plan.
    There are two modes: "auto" for LLM/VLM critique generation and "manual" for manual critique generation.
    Return the critique to the brain.
    '''
    def __init__(self, 
                 FAILED_TIMES_LIMIT = 2, 
                 mode = 'auto',
                 deployment_name = 'gpt-4o-v2',
                 azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT"),
                 api_version = "2024-08-01-preview",
                 max_tokens = 256,
                 temperature = 0,
                 save_path = "./save",
                 vision = True,):
        print(f"\n{'='*50}")
        print("Initializing Critic Agent...")
        print(f"Model: {deployment_name}")
        print(f"Endpoint: {azure_endpoint}")
        print(f"API Version: {api_version}")
        print(f"Mode: {mode}")
        
        self.FAILED_TIMES_LIMIT = FAILED_TIMES_LIMIT
        self.plan_failed_count = 0
        self.mode = mode
        self.vision = vision
        self.save_path = save_path
        
        assert self.mode in ['auto', 'manual'], f"Invalid mode: {mode}. Must be 'auto' or 'manual'"

        try:
            model = AzureChatOpenAI(
                deployment_name=deployment_name,
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            parser = JsonOutputParser(pydantic_object=CriticInfo)
            self.chain = model | parser
            print("✅ Successfully initialized Critic Agent")
        except Exception as e:
            print(f"❌ Failed to initialize Critic Agent: {str(e)}")
            raise

        print(f"{'='*50}\n")

    def human_check_task_success(self):
        confirmed = False
        success = False
        critique = ""
        while not confirmed:
            success = input("Success? (y/n)")
            success = success.lower() == "y"
            critique = input("Enter your critique:")
            print(f"Success: {success}\nCritique: {critique}")
            confirmed = input("Confirm? (y/n)") in ["y", ""]
        return success, critique

    def render_system_message(self):
        prompt = load_prompt("critic")
        return SystemMessage(content=prompt)

    def render_human_message(self, short_term_plan, obs):
        observation = []
        short_term_plan = short_term_plan["short_term_plan"]
        observation.append({"type": "text", "text": short_term_plan})
        observation.append({"type": "text", "text": str(obs)})
        try:
            image_base64 = obs["rgb_base64"]
            if image_base64 != "":
                observation.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "auto",
                            },
                        })
        except:
            print("No image in observation")
            pass
        
        human_message = HumanMessage(content=observation)
        return human_message

    def ai_check_task_success(self, messages, max_retries=5, verbose=False):
        """
        Use AI to check if the task was successful.
        
        Args:
            messages: Messages to send to the AI
            max_retries: Maximum number of retries
            verbose: Whether to print detailed logs
            
        Returns:
            success: Whether the task was successful
            critique: Critique message
        """
        if max_retries == 0:
            print("❌ Maximum retries reached for parsing critic response")
            return False, "Failed to parse critic response after maximum retries"

        if messages[1] is None:
            return False, ""

        try:
            critic_info = self.chain.invoke(messages)
            if verbose:
                print(f"\033[31m****Critic Agent****\n{critic_info}\033[0m")
                with open(f"{self.save_path}/log.txt", "a+") as f:
                    f.write(f"****Critic Agent****\n{critic_info}\n")
            
            # Validate response
            assert critic_info["success"] in [True, False], "Invalid success value"
            assert critic_info["critique"] != "", "Empty critique"
            
            return critic_info["success"], critic_info["critique"]
            
        except Exception as e:
            print(f"⚠️ Error parsing critic response: {e} Retrying...")
            return self.ai_check_task_success(
                messages=messages,
                max_retries=max_retries - 1,
                verbose=verbose
            )

    def critic(self, short_term_plan, obs, max_retries=5, verbose=False):
        '''
        Critique the short term plan.
        
        Args:
            short_term_plan: The plan to critique
            obs: Current observation
            max_retries: Maximum number of retries for parsing
            verbose: Whether to print detailed logs
            
        Returns:
            next_step: Next step to take ('brain' or 'action')
            success: Whether the plan was successful
            critique: Critique message
        '''
        print("\nCritiquing short-term plan...")
        
        # Check if we have a plan to critique
        if short_term_plan is None:
            print("❌ No short-term plan provided")
            return "brain", False, "need short-term plan"
        
        try:
            # Generate messages
            human_message = self.render_human_message(short_term_plan, obs)
            messages = [self.render_system_message(), human_message]

            # Get critique based on mode
            if self.mode == 'manual':
                success, critique = self.human_check_task_success()
            elif self.mode == "auto":
                success, critique = self.ai_check_task_success(
                    messages=messages, max_retries=max_retries, verbose=verbose
                )
            
            # Handle results
            if success:
                print("✅ Plan critique successful")
                next_step = "brain"
                self.plan_failed_count = 0
            else:
                print("⚠️ Plan critique failed")
                next_step = "action"
                self.plan_failed_count += 1
                if self.plan_failed_count >= self.FAILED_TIMES_LIMIT:
                    print(f"❌ Plan failed {self.plan_failed_count} times, switching to brain")
                    next_step = "brain"
                    critique = "failed"
                    self.plan_failed_count = 0
            
            if verbose:
                print(f"Next step: {next_step}")
                print(f"Success: {success}")
                print(f"Critique: {critique}")
                
            return next_step, success, critique

        except Exception as e:
            print(f"❌ Failed to critique plan: {str(e)}")
            raise
