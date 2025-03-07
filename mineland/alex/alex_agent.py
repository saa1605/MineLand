import os
import time

from .. import Action
from .action.action_agent import *
from .brain.associative_memory import *
from .brain.memory_library import *
from .critic.critic_agent import *
from .self_check.self_check_agent import *


class Alex:
    def __init__(self,
                deployment_name = "gpt-4o-v2",
                azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_version = "2024-08-01-preview",
                max_tokens = 512,
                temperature = 0,
                embeddings_api_key = os.environ.get("AZURE_OPENAI_API_KEY_EMB"),
                embeddings_deployment_name = "text-embedding-3-large",
                embeddings_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT_EMB"),
                embeddings_api_version = "2023-05-15",
                chat_api_key = os.environ.get("AZURE_OPENAI_API_KEY"),
                save_path = "./storage",
                load_path = "./load",
                FAILED_TIMES_LIMIT = 3,
                bot_name = "Alex",
                personality = "None",
                vision = True,):
        
        print(f"\n{'='*50}")
        print("Initializing Alex Agent...")
        print(f"Model: {deployment_name}")
        print(f"Endpoint: {azure_endpoint}")
        print(f"API Version: {api_version}")
        print(f"Bot Name: {bot_name}")
        print(f"Personality: {personality}")
        
        # Store configuration
        self.personality = personality
        self.deployment_name = deployment_name
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version
        self.chat_api_key = chat_api_key
        self.embeddings_api_key = embeddings_api_key
        self.embeddings_deployment_name = embeddings_deployment_name
        self.embeddings_endpoint = embeddings_endpoint
        self.embeddings_api_version = embeddings_api_version
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.save_path = save_path + "/" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        self.load_path = load_path
        self.vision = vision
        self.bot_name = bot_name
        self.FAILED_TIMES_LIMIT = FAILED_TIMES_LIMIT

        print(f"Save path: {self.save_path}")

        try:
            # Initialize self-check agent
            self.self_check_agent = SelfCheckAgent(
                FAILED_TIMES_LIMIT=self.FAILED_TIMES_LIMIT,
                save_path=self.save_path,
            )
            print("✅ Successfully initialized self-check agent")
        except Exception as e:
            print(f"❌ Failed to initialize self-check agent: {str(e)}")
            raise

        try:
            # Initialize critic agent
            self.critic_agent = CriticAgent(
                FAILED_TIMES_LIMIT=self.FAILED_TIMES_LIMIT,
                mode="auto",
                deployment_name=self.deployment_name,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                save_path=self.save_path,
                vision=self.vision,
            )
            print("✅ Successfully initialized critic agent")
        except Exception as e:
            print(f"❌ Failed to initialize critic agent: {str(e)}")
            raise

        try:
            # Initialize memory library
            self.memory_library = MemoryLibrary(
                deployment_name=self.deployment_name,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                embeddings_api_key=self.embeddings_api_key,
                embeddings_deployment_name=self.embeddings_deployment_name,
                embeddings_endpoint=self.embeddings_endpoint,
                embeddings_api_version=self.embeddings_api_version,
                max_tokens=self.max_tokens,
                save_path=self.save_path,
                load_path=self.load_path,
                personality=self.personality,
                bot_name=self.bot_name,
                vision=self.vision,
            )
            print("✅ Successfully initialized memory library")
        except Exception as e:
            print(f"❌ Failed to initialize memory library: {str(e)}")
            raise

        try:
            # Initialize associative memory
            self.associative_memory = AssociativeMemory(
                deployment_name=self.deployment_name,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                save_path=self.save_path,
                personality=self.personality,
                vision=self.vision,
            )
            print("✅ Successfully initialized associative memory")
        except Exception as e:
            print(f"❌ Failed to initialize associative memory: {str(e)}")
            raise

        try:
            # Initialize action agent
            self.action_agent = ActionAgent(
                deployment_name=self.deployment_name,
                azure_endpoint=self.azure_endpoint,
                api_version=self.api_version,
                max_tokens=self.max_tokens * 3,
                temperature=self.temperature,
                save_path=self.save_path,
            )
            print("✅ Successfully initialized action agent")
        except Exception as e:
            print(f"❌ Failed to initialize action agent: {str(e)}")
            raise

        print(f"Alex Agent Initialization Complete")
        print(f"{'='*50}\n")

    def self_check(self, obs, code_info = None, done = None, task_info = None):
        return self.self_check_agent.self_check(obs, code_info, done, task_info, associative_memory=self.associative_memory)

    def critic(self, obs, verbose = False):
        short_term_plan = self.memory_library.retrieve_latest_short_term_plan()
        return self.critic_agent.critic(short_term_plan, obs, verbose=verbose)

    def perceive(self, obs, plan_is_success, critic_info = None, code_info = None, vision = False, verbose = False):
        self.memory_library.perceive(obs, plan_is_success, critic_info, code_info, vision=vision, verbose=verbose)

    def retrieve(self, obs, verbose = False):
        retrieved = self.memory_library.retrieve(obs, verbose)
        return retrieved

    def plan(self, obs, task_info, retrieved, verbose = False):
        short_term_plan = self.associative_memory.plan(obs, task_info, retrieved, verbose=verbose)
        self.memory_library.add_short_term_plan(short_term_plan, verbose=verbose)
        return short_term_plan

    def execute(self, obs, description, code_info = None, critic_info = None, verbose = False):
        if description == "Code Unfinished":
            # return { "type": Action.RESUME, "code": ''}
            return Action(type=Action.RESUME, code='')
        short_term_plan = self.memory_library.retrieve_latest_short_term_plan()
        if description == "Code Failed" or description == "Code Error":
            return self.action_agent.retry(obs, short_term_plan, code_info, verbose=verbose)
        if description == "redo":
            return self.action_agent.redo(obs, short_term_plan, critic_info, verbose=verbose)
        return self.action_agent.execute(obs, short_term_plan, verbose=verbose)

    def run(self, obs, code_info = None, done = None, task_info = None, verbose = False):
        print("\nStarting Alex agent run cycle...")
        try:
            # 1. Self check
            if verbose:
                print("==========self check==========")

            next_step, description = self.self_check(obs, code_info, done, task_info)

            # If task is done
            if next_step is None:
                print(f"Task complete: {description}")
                return None

            if verbose:
                print(f"Next step after self check: {next_step}")
                print(f"Description: {description}")
                print("==============================\n")
                if next_step != "action":
                    with open(f"{self.save_path}/log.txt", "a+") as f:
                        f.write("==========self check==========\n")
                        f.write(f"Next step after self check: {next_step}\n")
                        f.write(f"Description: {description}\n")
                        f.write("==============================\n")
            
            # 2. Critic
            plan_is_success = False
            critic_info = None
            if next_step == "critic":
                if verbose:
                    print("==========critic==========")
                    with open(f"{self.save_path}/log.txt", "a+") as f:
                        f.write("==========critic==========\n")

                next_step, plan_is_success, critic_info = self.critic(obs, verbose=verbose)

                if next_step == "action":
                    description = "redo"

                if verbose:
                    print(f"Next step after critic: {next_step}")
                    print(f"Critic info: {critic_info}")
                    print("==========================\n")
                    with open(f"{self.save_path}/log.txt", "a+") as f:
                        f.write(f"Next step after critic: {next_step}\n")
                        f.write(f"Critic info: {critic_info}\n")
                        f.write("==========================\n")

            # 3. Brain
            if next_step == "action":
                self.perceive(obs, plan_is_success, critic_info=None, code_info=None, vision=False, verbose=False)

            if next_step == "brain":
                if verbose:
                    print("==========brain==========")
                    with open(f"{self.save_path}/log.txt", "a+") as f:
                        f.write("==========brain==========\n")
                
                if description == "Code Failed":
                    critic_info = "action failed, maybe the plan is too difficult. please change to a easy plan."

                self.perceive(obs, plan_is_success, critic_info, code_info, vision=True, verbose=verbose)
                self.memory_library.generate_long_term_plan(obs, task_info)
                retrieved = self.retrieve(obs, verbose=verbose)
                self.plan(obs, task_info, retrieved, verbose=verbose)

                next_step = "action"
                description = "execute the plan"

                if verbose:
                    print(f"Next step after brain: {next_step}")
                    print(f"Description: {description}")
                    print("========================\n")
                    with open(f"{self.save_path}/log.txt", "a+") as f:
                        f.write(f"Next step after brain: {next_step}\n")
                        f.write(f"Description: {description}\n")
                        f.write("========================\n")

            # 4. Action
            if next_step == "action":
                if verbose:
                    print("==========action==========")
                    if description != "Code Unfinished":
                        with open(f"{self.save_path}/log.txt", "a+") as f:
                            f.write("==========action==========\n")

                act = self.execute(obs, 
                                description, 
                                code_info=code_info, 
                                critic_info=critic_info, 
                                verbose=verbose)

                if verbose:
                    print("==========================\n")
                    if description != "Code Unfinished":
                        with open(f"{self.save_path}/log.txt", "a+") as f:
                            f.write("==========================\n\n\n")
                
                return act

            print("Run cycle completed successfully")
            return None

        except Exception as e:
            print(f"❌ Error in run cycle: {str(e)}")
            raise