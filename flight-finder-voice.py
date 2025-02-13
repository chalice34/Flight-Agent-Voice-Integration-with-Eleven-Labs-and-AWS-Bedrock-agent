from dotenv import load_dotenv,dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os
import boto3
import uuid
load_dotenv()
envvals=dotenv_values()

os.environ['PATH'] += os.pathsep + r'path\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared\bin'





ELEVENLABS_API_KEY=os.getenv("ELEVENLABS_API_KEY")



region_name = "ap-south-1"
bedrock_agent_runtime = boto3.client(service_name='bedrock-agent-runtime', region_name=region_name,aws_access_key_id=envvals["aws_access_key_id"],aws_secret_access_key=envvals["aws_secret_access_key"])

def invoke_agent(agent_id, agent_alias_id, session_id, prompt):
    """
    Sends a prompt for the agent to process and respond to.

    :param agent_id: The unique identifier of the agent to use.
    :param agent_alias_id: The alias of the agent to use.
    :param session_id: The unique identifier of the session. Use the same value across requests
                        to continue the same conversation.
    :param prompt: The prompt that you want Claude to complete.
    :return: Inference response from the model.
    """
    completion = ""
    try:
        # Note: The execution time depends on the foundation model, complexity of the agent,
        # and the length of the prompt. In some cases, it can take up to a minute or more to
        # generate a response.
        response = bedrock_agent_runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            sessionId=session_id,
            inputText=prompt,
        )

        for event in response.get("completion"):
            chunk = event["chunk"]
            completion = completion + chunk["bytes"].decode()

    except Exception as e:
        print(f"Couldn't invoke agent. {e}")

    return completion


session_id = str(uuid.uuid4())
print(envvals)
x = invoke_agent(envvals["agent_id"],envvals["agent_alias"], session_id, prompt="find flights from vtz to blr on 2025-02-20")
response = f"{x}"

print(len(response))

client = ElevenLabs(    api_key=ELEVENLABS_API_KEY,
)
if response!=None:

    audio = client.text_to_speech.convert(
        text=response,
        voice_id="wJqPPQ618aTW29mptyoc",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    # play(audio)
    
    save_file_path = f"{uuid.uuid4()}.mp3"
    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)
    print(f"{save_file_path}: A new audio file was saved successfully!")
    # Return the path of the saved audio file
