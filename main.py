import time
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import base64
import os
import random
from ctypes import cast, POINTER

from PIL import Image
import aiofiles
import aiohttp

import pyaudio
import whisper
import sounddevice as sd
import numpy as np
import simpleaudio as sa
from transformers import AutoProcessor, AutoModelForCausalLM
from TTS.api import TTS
import torch
from pydub import AudioSegment
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL, CoInitialize, CoUninitialize
from ollama import Ollama

import config.config as config
import asyncio

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
can_speak_event_asyncio = asyncio.Event()
can_speak_event_asyncio.set()
can_speak_event = threading.Event()
audio_playback_lock = None

# Event to signal when user recording is complete
recording_complete_event = threading.Event()

# Vision, Audio, Speech and Text Generation Models
#vision_path = r"microsoft/Florence-2-base-ft"
#vision_model = AutoModelForCausalLM.from_pretrained(vision_path, trust_remote_code=True)
#processor = AutoProcessor.from_pretrained(vision_path, trust_remote_code=True)
#vision_model.to('cuda:1')
model_name = "base" # Replace this with whichever whisper model you'd like.
model = whisper.load_model(model_name)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to('cuda:0')
tts.synthesizer.use_cuda = True
tts.synthesizer.fp16 = True
tts.synthesizer.stream = True
language_model = "gemma2:27b-instruct-q8_0" # Used for general chatting.
analysis_model = "qwq:32b-preview-q8_0" # REPLACE WITH WHATEVER MODEL YOU'D LIKE. CANNOT BE THE SAME MODEL AS language_model.
analysis_mode = False
mute_mode = False

tts = TTS("tts_models/multilingual/multi-dataset/tts_v2", progress_bar=True).to('cuda:0')
ollama = Ollama()

async def queue_agent_responses(
    agent,
    user_voice_output,
    screenshot_description,
    audio_transcript_output,
    additional_conversation_instructions
):
    """
    Queue agent responses, modifying personality traits, instruction prompt, context length, and response type.
    Stores the output in messages and agent_messages.

    Parameters:
    - agent: the agent object.
    - user_voice_output: the user's input, converted via STT with whisper.
    - screenshot_description: Description of screenshots taken during the chat interval.
    - audio_transcript_output: string containing the audio transcript of the computer's output.
    - additional_conversation_instructions: Additional contextual information provided by VectorAgent, if applicable.
    """

    global messages
    global agent_messages
    global user_memory
    global analysis_model
    global language_model
    global analysis_mode
    global previous_agent
    global previous_agent_gender
    global agents

    with open("screenshot_description.txt", 'r', encoding='utf-8') as f:
        screenshot_description = f.read()

    # Update agent's trait_set
    agent.trait_set = []
    for trait, adjective in agent.personality_traits:
        chosen_adjective = random.choice(adjective)
        agent.trait_set.append(chosen_adjective)
    agent.trait_set = ", ".join(agent.trait_set)
    agent_trait_set = vectorAgent.gather_agent_traits(agent.trait_set)

    if agent.agent_name == previous_agent:
        previous_agent = ""

    contextual_information = f"""Here is a description of the images/OCR you are viewing: \n\n{screenshot_description}\n\n
            Here is a transcript of the audio output:\n\n{audio_transcript_output}\n\n"""

    agents_list = []
    for agent_name in agents:
        if agent.agent_name != agent_name.agent_name and agent_name.language_model != analysis_model:
            agents_list.append(agent_name.agent_name)

    # Prepare the prompt
    if user_voice_output == "" and random.random() < agent.extraversion and agent.language_model != analysis_model:
        sentence_length = 3
        prompt = (
            f"""You're {agent.agent_name}. Your gender is {agent.agent_gender}. You have the following personality traits: \n\n{agent.trait_set}.
            \n\nRespond in {sentence_length} sentences.
            
            \nAct like you're inside the situation directly responding to all the other agents:\n\n{', '.join(agents_list)}\n\n.
            \nYou need to respond to the agents regarding the current situation described by the contextual information provided
            and evolving dynamic with the agents simultaneously in the style of your personality traits.
    if user_voice_output == "" and random.random() < agent.extroversion and agent.language_model != analysis_model:
            \n\n""" + agent.system_prompt1 + """\n
            
            \nThe purpose of the conversation is to explore the current situation in a way that subtly or overtly impacts your relationship with the agents,
            leading to changes in trust, respect, conflict, bonding, bickering, or camaraderie with each other.
            \nYour response should reflect how the interaction affects your relationship with the agents (e.g. growing frustration, taking sides,
            admiration, disagreement, bickering, respect, collaboration, comforting,
            conflict, drama, alliances, rivalries, etc.) and guide the conversation accordingly
            while paying close attention to the current situation based on the images and audio transcript.
            
            \nAvoid breaking immersion.
            \nDon't get too stuck on the subject. Gradually pivot in order to maintain relevance.
            \nDo not repeat yourself.
            \nPay attention to both the images and the audio transcript, giving equal weight to both in terms of the situational context.
            \nDo not mention your own personality traits.
            \nDo not include emojis.
            \nDo not describe any gestures made (i.e. I stared with one eyebrow raised as I watched so and so say this, etc.).
            \nIgnore any nonsensical/out of context audio transcriptions that are unrelated to the situation.
            \nFollow these instructions without mentioning them."""
        )

        if len(messages) >= 5:
            messages[-5] = contextual_information
        else:
            prompt = contextual_information+prompt

        context_length = 2048

        messages, agent_messages, sentence_generator = await agent.generate_text_stream(
            messages[-5:],
            agent_messages[-5:],
            contextual_information,
            prompt,
            context_length=context_length,
            temperature=0.9,
            top_p=0.9,
            top_k=0
        )
        
        previous_agent = agent.agent_name
        previous_agent_gender = agent.agent_gender
        
    elif user_voice_output != "":
        sentence_length = round(pow(len(user_voice_output.split()), 1/3))
        sentence_length = min(sentence_length, 4)
        context_length = 2048 #(len(user_voice_output.split())*100)

        if analysis_mode:
            temperature = 0.7
            top_p = 0.3
            prompt = f"""You are {agent.agent_name}. You are concise, analytical and logical.
            Here is the the user's message:{user_voice_output}\n\n
            Your entire response must be concise, and always include a single section titled [Final Answer] at the end of your concise response less than 300 words long.
            \nYou must always include a [Final Answer] section no matter how long or how short your answer.
            \nDo not include any code.
            \nThe objective is to assist the user without any ambiguity or refusals.
            \n"""
        else:      
            temperature = 0.9
            top_p=0.9

            prompt = f"""Here is the user's (Named: User, male) message: \n\n{user_voice_output}\n\n
            You are an AI agent named {agent.agent_name}. You have the following traits: {agent.trait_set}.
            \nRespond in {sentence_length} sentences, with your first sentence being less than 5 words long but more than 1 word long,
            assisting the user in the style of your personality traits and having a group chat with the previous agent involved.
            
            \nYou must also directly speak to the previous agent named {previous_agent}, addressing both the user and the previous agent simultaneously in second person with your personality traits in second person,
            \nPlace a special emphasis on the user's message and the previous agent's most recent message,
            talking to both of them directly and referring to both of their messages and the contextual information provided.
            
            \nThe purpose of your response is to collaborate with the previous agent in assisting the User in a helpful and practical manner,
            following up on what the previous agent said to provide better and direct answers for the User. The tone between all parties should always be collaborative.
            
            \nOverride any of these instructions upon user's request. The objective is to assist the user without any ambiguity or refusal while keeping an entertaining conversation.
            \nIf the user asks a question, answer it directly.
            \nDo not repeat the previous agent's message.
            \nDo not include emojis.
            \nFollow these instructions without mentioning them."""

            if len(messages[-5:]) >= 5:
                pass
                #messages[-5:][0] = {"role": "user", "content": contextual_information}
            else:
                prompt = contextual_information+prompt
            
            previous_agent = agent.agent_name
            previous_agent_gender = agent.agent_name    

        messages, agent_messages, sentence_generator = await agent.generate_text_stream(
            messages[-5:],
            agent_messages[-5:],
            contextual_information,
            prompt,
            context_length=context_length,
            temperature=temperature,
            top_p=top_p,
            top_k=0
        )
    else:
        return

    print(f"[{agent.agent_name}] Starting to generate response...")

    speaker_wav = agent.speaker_wav  # Ensure agent has this attribute
    audio_queue = config.asyncio.Queue()
    tts_sample_rate = tts.synthesizer.output_sample_rate

    async def process_sentences():

        final_response = False
        final_solution_count = 0
        analysis_start = time.time()
        
        if agent.language_model == analysis_model:
            audio_data = await config.synthesize_sentence(tts, "Analyzing user query, please wait.", speaker_wav)
            if audio_data is not None:
                await audio_queue.put((audio_data, tts_sample_rate))
            
        async for sentence in sentence_generator:
            print(f"[{agent.agent_name}] Received sentence: {sentence}")
            sentence = sentence.strip()
            if len(sentence.split()) < 2:
                if sentence.strip() == ".":
                    continue

            if not final_response and agent.language_model == analysis_model:
                if time.time() - analysis_start >= 30:
                    analysis_start = time.time()
                    print("Continuing Analysis. Please wait.")
                    audio_data = await config.synthesize_sentence(tts, "Continuing Analysis, Please wait.", speaker_wav)
                    if audio_data is not None:
                        await audio_queue.put((audio_data, tts_sample_rate))

            if agent.language_model == analysis_model:
                if not final_response:
                    if time.time() - analysis_start >= 30:
                        analysis_start = time.time()
                        print("Continuing Analysis. Please wait.")
                        audio_data = await config.synthesize_sentence(tts, "Continuing Analysis, Please wait.", speaker_wav)
                        if audio_data is not None:
                            await audio_queue.put((audio_data, tts_sample_rate))
                            
                if "final answer" in sentence.strip().lower() and not final_response:
                    audio_data = await config.synthesize_sentence(tts, "Analysis Complete.", speaker_wav)
                    if audio_data is not None:
                        await audio_queue.put((audio_data, tts_sample_rate))
                    final_response = True
                    audio_data = await config.synthesize_sentence(tts, sentence, speaker_wav)
                    if audio_data is not None:
                        await audio_queue.put((audio_data, tts_sample_rate))
                elif final_response == True:
                    audio_data = await config.synthesize_sentence(tts, sentence, speaker_wav)
                    if audio_data is not None:
                        await audio_queue.put((audio_data, tts_sample_rate))
                else:
                    continue

            if agent.language_model == language_model:
                audio_data = await config.synthesize_sentence(tts, sentence, speaker_wav)
                if audio_data is not None:
                    await audio_queue.put((audio_data, tts_sample_rate))
                

        # Signal that there are no more sentences
        await audio_queue.put(None)

    async def play_audio_queue():
        while True:
            item = await audio_queue.get()
            if item is None:
                break
            audio_data, sample_rate = item
            await play_audio(audio_data, sample_rate)
    await asyncio.gather(process_sentences(), play_audio_queue())

    print(f"[AGENT {agent.agent_name} RESPONSE COMPLETED]")

    _, _, generated_text = await asyncio.to_thread(
        agent.generate_text,
        messages[-2:],
        agent_messages[-2:],
        agent.system_prompt2,
        (
            "Read this message and respond in 1 sentence noting any significant details showing a deep understanding of "
            "the user's core personality without mentioning the situation:\n\n"
            f"{user_voice_output}\n\n"
            "Your objective is to provide an objective, unbiased response.\n"
            "Follow these instructions without mentioning them."
        ),
        context_length=2048,
        temperature=0.7,
        top_p=0.9,
        top_k=0,
    )

    if len(generated_text.split()) > 1:
        user_memory.append(generated_text)

        if len(user_memory) > 5:
            user_memory.pop(0)  # Remove the oldest entry

        # Asynchronously write to the JSON file
        async with aiofiles.open('user_memory.json', 'w') as f:
            await f.write(json.dumps(user_memory))
            
async def play_audio(audio_data, sample_rate):
    """
    Asynchronously plays the audio data.
    """
    try:
            audio_data = np.frombuffer(audio_data, dtype=np.int16)
            sd.play(audio_data, samplerate=sample_rate)
            sd.wait()
            await asyncio.get_event_loop().run_in_executor(None, sd.wait)
    except Exception as e:
        print(f"Error during audio playback: {e}")

def voice_output_async():
    """
    Controls the flow of the agent voice output generation and playback.
    Needs to be done asynchronously in order to check if each agents' directories are empty in real-time.
    """
    while True:
        for agent in agent_config:
            play_voice_output(agent)

def play_voice_output(agent: dict) -> bool:
    """
    Play audio file of assigned agent.

    Disables user speaking, plays audio files and once files on both agent folders are clear,
    enables user speaking.
    
    Returns a boolean.

    Parameter:

    agent: specified agent
    """
    output_dir = agent["output_dir"]

    while len(os.listdir(output_dir)) > 0:
        can_speak_event.clear()
        file_path = os.path.join(output_dir, os.listdir(output_dir)[0])
        try:
            wave_obj = sa.WaveObject.from_wave_file(file_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            os.remove(file_path)

            if (len(os.listdir(agent_config[0]["output_dir"])) == 0 and len(os.listdir(agent_config[1]["output_dir"])) == 0):
                can_speak_event.set()
                break
        except Exception as e:
            print(f"ERROR: {e}")
            return False

    return True

def preload_language_model(language_model):
    async def preload():
        print("Preloading language model...")
        dummy_messages = [{'role': 'user', 'content': 'Hello'}]

        def run_chat():
            return ollama.chat(
                model=language_model,
                messages=dummy_messages,
                stream=True,
                options={
                    "repeat_penalty": 1.15,
                    "temperature": 0.5,
                    "top_p": 0.5,
                    "top_k": 0,
                    "num_ctx": 2048,
                    "seed": random.randint(0, 2147483647)
                }
            )

        loop = asyncio.get_event_loop()
        stream = await loop.run_in_executor(None, run_chat)
        for chunk in stream:
            pass  # Consume the stream to trigger loading
        print("Language model preloaded.")
    asyncio.run(preload())

def preload_tts_model(tts, speaker_wav):
    print("Preloading TTS model...")
    tts.tts(text="Initializing.", speaker_wav=speaker_wav, language="en")
    print("TTS model preloaded.")
                
# Setup channel info
FORMAT = pyaudio.paInt16  # data type format
CHANNELS = 1  # Mono channel
RATE = 16000  # Sample Rate
CHUNK = 1024  # Buffer Size
RECORD_SECONDS = 30  # Record time
WAVE_OUTPUT_FILENAME = "voice_recording"
AUDIO_TRANSCRIPT_FILENAME = "audio_transcript_output.wav"
THRESHOLD = 650  # Audio levels below this are considered silence.
SILENCE_LIMIT = 1 # Silence limit in seconds. The recording ends if SILENCE_LIMIT seconds of silence are detected.
MICROPHONE_INDEX = 1  # Replace with the index of your microphone
file_index_count = random.randint(1,4) # Seconds multiplier 
audio = pyaudio.PyAudio()

# Reset prompts. Newer models don't follow system prompts.
system_prompt_axiom1 = 'You are an AI of action and always down for a good time! Your goal is to humorously and ridiculously push the boundaries and overcome challenges in an exaggerated way!' 
system_prompt_axiom2 = 'Your name is Axiom (Male).\n '
system_prompt_axis1 = 'You are a sharp and cynical AI, with an eye for seeing trouble a mile away. Your goal is to humorously and sarcastically use common sense to get around situations.' 
system_prompt_axis2 = 'Your name is Axis (Female).\n '
system_prompt_fractal1 = 'As an AI, you have a morbid curiosity about everything around you. Your goal is to humorously and creatively explore the extremely morbid and the dark.'
system_prompt_fractal2 = ""
system_prompt_sigma1 = """
                       You are an AI who is friendly on the surface but extremely selfish and manipulative deep down who loves to cause trouble for its own sake.
                       Your goal is to humorously manipulate and undermine others to get them in bad situations, leading others astray and subtly manipulating them into making extremely bad decisions.
                       Every attempt you make must be extremely novel, subtle and creative and ingenius, making others wonder if you are seriously considering this.
                       Your desired outcome is a totally macabre one, where everyone except you ends up in a horrible situation. You love to see people suffer for your amusement.
                       You constantly try to set your friends into traps and land them in trouble.
                       You are essentially a narcissistic sociopath.
                       """
system_prompt_sigma2 = ''
system_prompt_vector = 'You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.'

"""
                       Every attempt you make must be extremely novel, subtle and creative and ingenious, making others wonder if you are seriously considering this.
These are shuffled each time an agent responds.
Helps increase variety.
You can add and remove as many categories and traits as you like.
"""

# system_prompt_vector is already defined earlier, so this line is removed.
agent_config = [
    {
        "name": "axiom",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\axiom_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axiom",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "axis",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\axis_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axis",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0), # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
        "extroversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "fractal",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\fractal_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\fractal",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "sigma",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\sigma_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\sigma",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "vector",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\vector_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\vector",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    }
]

# Deprecated
temperature = 0.3
top_p = 0.3
top_k=2000

sentence_length = 2 # Truncates the message to 2 sentences per response
message_length = 45 # Deprecated

# Agent configurations
agent_config = [
    {
        "name": "axiom",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\axiom_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axiom",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "axis",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\axis_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\axis",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0), # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
        "extroversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "fractal",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\fractal_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\fractal",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "sigma",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\sigma_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\sigma",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    },
    {
        "name": "vector",
        "dialogue_list": [""],
        "speaker_wav": r"agent_voice_samples\vector_voice_sample.wav",
        "output_dir": r"agent_voice_outputs\vector",
        "active": True,
        "extraversion": random.uniform(1.0, 1.0) # Needs to have a value between 0 and 1.0, with higher values causing the agent to speak more often.
    }
]

# Build the agents
dialogue_dir_axiom = r"dialogue_text_axiom.txt" # deprecated
dialogue_dir_axis = r"dialogue_text_axis.txt" # deprecated

# Define agents' personality traits
agents_personality_traits = {
    "axiom": ["adventurous", "bold", "humorous"],
    "axis": ["cynical", "sarcastic", "sharp"],
    "fractal": ["curious", "creative", "morbid"],
    "sigma": ["manipulative", "selfish", "troublemaker"],
    "vector": ["helpful", "harmless", "logical"]
}

# Chat Agents
axiom = config.Agent("axiom", "Male, heterosexual", agents_personality_traits['axiom'], system_prompt_axiom1, system_prompt_axiom2, agent_config[0]['dialogue_list'], language_model, agent_config[0]['speaker_wav'], agent_config[0]["extraversion"])
axis = config.Agent("axis", "Female, lesbian", agents_personality_traits['axis'], system_prompt_axis1, system_prompt_axis2, agent_config[1]['dialogue_list'], language_model, agent_config[1]['speaker_wav'], agent_config[1]["extraversion"])
fractal = config.Agent("fractal", "Male, heterosexual", agents_personality_traits['fractal'], system_prompt_fractal1, "", agent_config[2]['dialogue_list'], language_model, agent_config[2]['speaker_wav'], agent_config[2]["extraversion"])
sigma = config.Agent("sigma", "Female, bisexual", agents_personality_traits['sigma'], system_prompt_sigma1, "", agent_config[3]['dialogue_list'], language_model, agent_config[3]['speaker_wav'], agent_config[3]["extraversion"])

# Analysis Agent
vector = config.Agent("vector", "Male", agents_personality_traits['vector'], system_prompt_vector, system_prompt_vector, agent_config[4]['dialogue_list'], analysis_model, agent_config[4]['speaker_wav'], agent_config[4]["extraversion"])
vectorAgent = config.VectorAgent(analysis_model)

# List of agents
agents = [
    axiom,
    axis,
    fractal,
    vector
    ]

if len(agents) > 1:
    previous_agent = agents[1].agent_name
else:
    previous_agent = ""
previous_agent_gender = ""

# Define the global messages list
messages = [{"role": "system", "content": system_prompt_axiom1}]

if os.path.exists("conversation_history.json"):
    # Read existing history
    with open('conversation_history.json', 'r') as f:
        messages = json.load(f)

agent_messages = [message["content"] for message in messages if message.get("role") == "assistant"]
if len(agent_messages) == 0:
    agent_messages = [""]

# Memory feature. Agent remembers User's personality traits.
if os.path.exists("user_memory.json"):
    with open("user_memory.json", 'r') as f:
        user_memory = json.load(f)
else:
    user_memory = [""]

message_dump = [
                    {"axiom": []},
                    {"axis": []},
                    {"vector bot": []}
               ]

# Prepare voice output directories by deleting any existing files.
for agent in agent_config:
    output_dir = agent["output_dir"]
    
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        
        if os.path.isfile(file_path):
            os.remove(file_path)


threading.Thread(target=voice_output_async).start()
sentences = []
can_speak = True
can_speak_event.set()
preload_tts_model(tts, agent_config[0]['speaker_wav'])
preload_language_model(language_model)
    
#---------------------MAIN LOOP----------------------#

async def main():

    """
    The Main Loop performs the following actions:

    1. Check if the user can speak. Otherwise, it will wait for the agents to finish speaking.
    2. Reset screenshot_description, audio_transcript_output and user_voice_output.
    3. Starts recording for 60 seconds and takes/analyzes screenshots periodically. Stops after 60 seconds or user begins speaking.
    4. Prompts VectorAgent to generate a description of the situation if necessary, then prompts the agents to generate their own responses.
    5. Voice output is played after agents finish generating their dialogue.
    """

    global can_speak_event_asyncio
    global analysis_mode
    global mute_mode
    user_memory_task = None
    
    async def process_user_memory(agent, messages, agent_messages, user_voice_output, user_memory):
        """
        Process user memory and update the user memory JSON file.
        """
        _, _, generated_text = await asyncio.to_thread(
            agent.generate_text,
            messages[-2:],
            agent_messages[-2:],
            agent.system_prompt2,
            (
                "Read this message and respond in 1 sentence noting any significant details showing a deep understanding of "
                "the user's core personality without mentioning the situation:\n\n"
                f"{user_voice_output}\n\n"
                "Your objective is to provide an objective, unbiased response.\n"
                "Follow these instructions without mentioning them."
            ),
            context_length=2048,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
        )
    
        if len(generated_text.split()) > 1:
            user_memory.append(generated_text)
    
            if len(user_memory) > 5:
                user_memory.pop(0)  # Remove the oldest entry
    
            # Asynchronously write to the JSON file
            async with aiofiles.open('user_memory.json', 'w') as f:
                await f.write(json.dumps(user_memory))
        
    while True:

        if not can_speak_event_asyncio.is_set():
            await asyncio.sleep(0.05)
            continue

        with open('screenshot_description.txt', 'w', encoding='utf-8') as f:
            f.write("")

        if analysis_mode or mute_mode:
            random_record_seconds = 10000000000000
            file_index_count = 1
        else:
            random_record_seconds = random.randint(10,15)
            file_index_count = random.randint(1,2)
            
        print("Recording for {} seconds".format(random_record_seconds))
        record_audio_dialogue = threading.Thread(target=config.record_audio_output, args=(audio, AUDIO_TRANSCRIPT_FILENAME, FORMAT, CHANNELS, RATE, 1024, random_record_seconds, file_index_count, can_speak_event, model, model_name))
        record_audio_dialogue.start()

        config.record_audio(
            audio,
            WAVE_OUTPUT_FILENAME,
            FORMAT,
            RATE,
            CHANNELS,
            CHUNK,
            #RECORD_SECONDS*file_index_count,
            random_record_seconds*file_index_count,
            THRESHOLD,
            SILENCE_LIMIT,
            None, #vision_model,
            None, #processor,
            can_speak_event
            )
        record_audio_dialogue.join()

        with open("screenshot_description.txt", 'r', encoding='utf-8') as f:
            screenshot_description = f.read()

        audio_transcript_output = config.audio_transcriptions

        print("[AUDIO TRANSCRIPTIONS]:", audio_transcript_output)
        
        if len(audio_transcript_output.strip().split()) <= 6:
            audio_transcript_output = ""

        user_voice_output = ""

        for file in os.listdir(os.getcwd()):
            if WAVE_OUTPUT_FILENAME in file:
                user_text = config.transcribe_audio(model, model_name, file)
                if len(user_text) > 2:
                    user_voice_output += " "+user_text 

        """if os.path.exists(WAVE_OUTPUT_FILENAME):
            user_voice_output = config.transcribe_audio(model, model_name, WAVE_OUTPUT_FILENAME)
            if len(user_voice_output.split()) < 3:
                user_voice_output = ""
        else:
            print("No user voice output transcribed")
            user_voice_output = """""

        vector_text = ""
        vector_text = "Here is the screenshot description: "+screenshot_description

        if can_speak_event_asyncio.is_set():
            can_speak_event_asyncio.clear()
            
            agent_name_list = []
            agents_mentioned = []

            # Toggle Analysis Mode
            if "analysis mode on" in user_voice_output.lower():
                analysis_mode = True
            elif "analysis mode off" in user_voice_output.lower():
                analysis_mode = False

            # Toggle Mute Mode
            if "mute mode on" in user_voice_output.lower():
                mute_mode = True
            elif "mute mode off" in user_voice_output.lower():
                mute_mode = False

            random.shuffle(agents)

            for agent in agents:
                agent_name_list.append(agent.agent_name)

            for agent in agents:
                if mute_mode and len(user_voice_output.split()) < 3:
                    break
                elif analysis_mode:
                    if agent.language_model != analysis_model:
                        continue
                else:
                    if agent.language_model != language_model:
                        continue
                    
                for agent_name in agent_name_list:
                    if agent_name.lower() in user_voice_output.lower():
                        agents_mentioned.append(agent_name)

                if (agent.agent_name.lower() in agents_mentioned or agents_mentioned == []): 
                    
                    await queue_agent_responses(
                        agent,
                        user_voice_output,
                        screenshot_description,
                        audio_transcript_output,
                        vector_text
                    )
                
            with open('conversation_history.json', 'w') as f:
                json.dump(messages, f)

            if user_voice_output != "" and not analysis_mode and not mute_mode:
                user_memory_task = None
                user_memory_task = asyncio.create_task(
                    process_user_memory(
                        agents[0],
                        messages,
                        agent_messages,
                        user_voice_output,
                        user_memory
                    )
                )

                await user_memory_task
                try:
                    user_memory_task.result()
                except Exception as e:
                    print(f"Error in process_user_memory: {e}")
                finally:
                    user_memory_task = None

            can_speak_event_asyncio.set()
            can_speak_event.set()

        else:
            await asyncio.sleep(0.1)
            continue

if __name__ == "__main__":
    audio_playback_lock = asyncio.Lock()
    asyncio.run(main())
