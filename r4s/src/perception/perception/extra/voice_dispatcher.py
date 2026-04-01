#!/usr/bin/env python3
"""
Voice-controlled ROS2 perception dispatcher.
Pipeline: Microphone → ElevenLabs STT → Gemini Intent → ROS2 Action

Usage:
  Live mic:             python3 voice_dispatcher.py
  Windows relay mic:    python3 voice_dispatcher.py --relay
  Recorded file:        python3 voice_dispatcher.py --audio /path/to/recorded.wav
"""

import sys
import os
import json
import threading
import argparse

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from my_robot_interfaces.action import RunVision

from elevenlabs.client import ElevenLabs
import google.generativeai as genai

# ============================================================
# CONFIG
# ============================================================
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "your_elevenlabs_key_here")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY",     "your_gemini_key_here")

# ============================================================
# GEMINI SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = """
You are a robot task dispatcher that controls a robotic perception system.
Your ONLY job is to map a voice command to exactly one task and return valid JSON.

AVAILABLE TASKS:
1. table_height        — Estimate the height of the table
2. detect_screws       — Detect screws on the table (no class needed)
3. car_objects         — Detect car parts. Optionally specify object_class.
   Valid object_class values: assembly, enclosure, motor, motor_grip, speaker, unit, wire_plug
   Leave object_class empty if the specific part is not mentioned.
4. subdoor_pose        — Estimate the pose of the sub-door
5. detect_screwdriver  — Detect the screwdriver
6. place_object        — Find a safe spot on the table to place an object.
   If the user mentions a size or radius, extract it in metres (e.g. "5 centimeters" → 0.05). Default to 0.07 if not mentioned.

OUTPUT FORMAT (return ONLY this JSON, no explanation, no markdown):
{
  "task": "<task_name>",
  "object_class": "<class or empty string>",
  "radius": <float, always 0.0 except for place_object>,
  "confidence": <float 0.0-1.0>,
  "understood_as": "<brief description of what you understood>"
}

RULES:
- If the command mentions screws, bolts, fasteners → detect_screws, object_class: ""
- If the command mentions a car part, map it to the closest valid class:
    "motor"                              → motor
    "motor grip", "gripper"              → motor_grip
    "speaker"                            → speaker
    "enclosure", "casing", "housing"     → enclosure
    "assembly", "assembled part"         → assembly
    "unit"                               → unit
    "wire plug", "plug", "wire"          → wire_plug
    unspecified or unclear car part      → object_class: ""
- If the command mentions placing, dropping, putting down an object → place_object
    Extract radius if mentioned: "5 centimeters" → 0.05, "7cm" → 0.07, "10mm" → 0.01
    Default radius to 0.07 if not mentioned.
- If the command mentions a screwdriver, tool → detect_screwdriver
- If the command mentions table height, how high the table is → table_height
- If the command mentions the sub-door, subdoor pose, body → subdoor_pose
- For all tasks except place_object, radius must always be 0.0
- If confidence is below 0.5, still return your best guess but set confidence accordingly.
- NEVER return anything except the JSON object.
"""

VALID_TASKS = [
    'table_height',
    'detect_screws',
    'car_objects',
    'subdoor_pose',
    'detect_screwdriver',
    'place_object',
]


# ============================================================
# INTENT EXTRACTOR (Gemini)
# ============================================================
class IntentExtractor:
    def __init__(self):
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT
        )
        print("[Gemini] Intent extractor ready.")

    def extract(self, transcript: str) -> dict | None:
        try:
            response = self.model.generate_content(transcript)
            raw = response.text.strip()

            # Strip markdown fences if Gemini adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            intent = json.loads(raw)

            if intent.get('task') not in VALID_TASKS:
                print(f"[Gemini] Returned unknown task: {intent.get('task')}")
                return None

            return intent

        except json.JSONDecodeError as e:
            print(f"[Gemini] Failed to parse JSON: {e}")
            print(f"[Gemini] Raw response: {response.text}")
            return None
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return None


# ============================================================
# SPEECH-TO-TEXT (ElevenLabs)
# ============================================================
class SpeechListener:
    def __init__(self):
        self.client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        print("[ElevenLabs] Speech listener ready.")

    def transcribe_file(self, filepath: str) -> str | None:
        """Transcribe a pre-recorded audio file."""
        if not os.path.exists(filepath):
            print(f"[STT] File not found: {filepath}")
            return None

        print(f"[STT] Transcribing file: {filepath}")
        try:
            with open(filepath, 'rb') as f:
                filename = os.path.basename(filepath)
                ext = os.path.splitext(filename)[1].lower()
                mime_map = {
                    '.wav':  'audio/wav',
                    '.mp3':  'audio/mpeg',
                    '.mp4':  'audio/mp4',
                    '.m4a':  'audio/mp4',
                    '.webm': 'audio/webm',
                    '.ogg':  'audio/ogg',
                    '.flac': 'audio/flac',
                }
                mime_type = mime_map.get(ext, 'audio/wav')

                result = self.client.speech_to_text.convert(
                    file=(filename, f, mime_type),
                    model_id="scribe_v1",
                    language_code="en"
                )
            transcript = result.text.strip()
            print(f"[STT] Heard: \"{transcript}\"")
            return transcript if transcript else None
        except Exception as e:
            print(f"[STT] Error transcribing file: {e}")
            return None

    def listen_once(self, relay: bool = False) -> str | None:
        """
        Record audio and transcribe.
        relay=True  → read from /tmp/mic_pipe (Windows mic streamed over Tailscale)
        relay=False → record from local sounddevice mic (default)
        """

        if relay:
            return self._listen_from_pipe()
        else:
            return self._listen_from_mic()

    # ----------------------------------------------------------
    # RELAY MODE: read raw audio from /tmp/mic_pipe
    # ----------------------------------------------------------
    def _listen_from_pipe(self) -> str | None:
        """Read audio from Windows mic relay via /tmp/mic_pipe."""
        import numpy as np
        from io import BytesIO
        import soundfile as sf

        SAMPLE_RATE = 16000
        PIPE_PATH = "/tmp/mic_pipe"
        recorded_data = []

        if not os.path.exists(PIPE_PATH):
            print(f"[STT] Pipe not found: {PIPE_PATH}")
            print("[STT] Make sure the ffmpeg relay loop is running on Linux")
            print("[STT] and mic_stream.py is running on Windows.")
            return None

        input("\n▶️  Press Enter to START recording...")
        print("🛑 Recording... Press Enter to STOP.")

        stop_event = threading.Event()

        def read_pipe():
            try:
                with open(PIPE_PATH, 'rb') as f:
                    while not stop_event.is_set():
                        chunk = f.read(1024)
                        if chunk:
                            recorded_data.append(np.frombuffer(chunk, dtype=np.int16))
            except Exception as e:
                print(f"[STT] Pipe read error: {e}")

        t = threading.Thread(target=read_pipe, daemon=True)
        t.start()
        input()
        stop_event.set()
        t.join(timeout=1)

        if not recorded_data:
            print("[STT] No audio captured from pipe.")
            return None

        audio_data = np.concatenate(recorded_data).astype(np.float32) / 32768.0
        buf = BytesIO()
        sf.write(buf, audio_data, SAMPLE_RATE, format='WAV')
        buf.seek(0)

        print("[STT] Transcribing...")
        try:
            result = self.client.speech_to_text.convert(
                file=("audio.wav", buf, "audio/wav"),
                model_id="scribe_v1",
                language_code="en"
            )
            transcript = result.text.strip()
            print(f"[STT] Heard: \"{transcript}\"")
            return transcript if transcript else None
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            return None

    # ----------------------------------------------------------
    # LOCAL MIC MODE: record from sounddevice (original behaviour)
    # ----------------------------------------------------------
    def _listen_from_mic(self) -> str | None:
        """Record from local microphone: Press Enter to start, Press Enter to stop."""
        try:
            import sounddevice as sd
            import numpy as np
            from io import BytesIO
            import soundfile as sf
        except ImportError:
            print("[STT] Missing packages for mic input.")
            return None

        # Display microphone name
        try:
            default_input_id = sd.default.device[0]
            device_info = sd.query_devices(default_input_id)
            print(f"🎤 Using Microphone: {device_info.get('name', 'Default')}")
        except:
            pass

        SAMPLE_RATE = 16000
        recorded_data = []

        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            recorded_data.append(indata.copy())

        input("\n▶️  Press Enter to START recording...")

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype='float32', callback=callback):
            input("🛑 Recording... Press Enter to STOP.")

        if not recorded_data:
            print("[STT] No audio captured.")
            return None

        audio_data = np.concatenate(recorded_data, axis=0)
        buf = BytesIO()
        sf.write(buf, audio_data, SAMPLE_RATE, format='WAV')
        buf.seek(0)

        print("[STT] Transcribing...")
        try:
            result = self.client.speech_to_text.convert(
                file=("audio.wav", buf, "audio/wav"),
                model_id="scribe_v1",
                language_code="en"
            )
            transcript = result.text.strip()
            print(f"[STT] Heard: \"{transcript}\"")
            return transcript if transcript else None
        except Exception as e:
            print(f"[STT] Transcription error: {e}")
            return None


# ============================================================
# ROS2 BRAIN CLIENT
# ============================================================
class VoiceBrainClient(Node):
    def __init__(self):
        super().__init__('voice_brain_client_node')
        self._action_client = ActionClient(self, RunVision, 'run_perception_pipeline')
        self._result_event = threading.Event()
        self._last_result = None
        self.get_logger().info("Voice Brain Client ROS2 node ready.")

    def dispatch(self, task: str, object_class: str = '', radius: float = 0.0) -> dict:
        self._result_event.clear()
        self._last_result = None

        if not self._action_client.wait_for_server(timeout_sec=10.0):
            return {'success': False, 'message': "Dispatcher server not available."}

        goal_msg = RunVision.Goal()
        goal_msg.task_name    = task
        goal_msg.object_class = object_class
        goal_msg.radius       = float(radius)

        self.get_logger().info(
            f"Dispatching: task='{task}' class='{object_class}' radius={radius:.4f}m"
        )

        future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_cb
        )
        future.add_done_callback(self._goal_response_cb)

        self._result_event.wait(timeout=30.0)
        return self._last_result or {'success': False, 'message': "Timed out waiting for result."}

    def _feedback_cb(self, feedback_msg):
        print(f"  📡 [{feedback_msg.feedback.current_phase}]")

    def _goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self._last_result = {'success': False, 'message': "Goal rejected."}
            self._result_event.set()
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._result_cb)

    def _result_cb(self, future):
        try:
            action_result  = future.result()
            result         = action_result.result
            success        = action_result.status == 4
            self._last_result = {'success': success, 'message': result.message}
        except Exception as e:
            self._last_result = {'success': False, 'message': str(e)}
        finally:
            self._result_event.set()


# ============================================================
# SHARED HELPERS
# ============================================================
def confirm_intent(intent: dict) -> bool:
    print("\n" + "="*50)
    print(f"🤖 Understood: {intent['understood_as']}")
    print(f"   Task:        {intent['task']}")
    if intent.get('object_class'):
        print(f"   Class:       {intent['object_class']}")
    if intent.get('radius', 0.0) > 0.0:
        print(f"   Radius:      {intent['radius']}m")
    print(f"   Confidence:  {intent['confidence']:.0%}")
    print("="*50)
    if intent['confidence'] < 0.6:
        print("⚠️  Low confidence.")
    answer = input("Proceed? [Y/n]: ").strip().lower()
    return answer in ('', 'y', 'yes')


def run_once(transcript: str, llm: IntentExtractor, ros_node: VoiceBrainClient):
    """Shared logic: transcript → intent → confirm → dispatch."""
    print("[Gemini] Extracting intent...")
    intent = llm.extract(transcript)
    if not intent:
        print("❌ Could not extract a valid task. Try rephrasing.")
        return

    if not confirm_intent(intent):
        print("↩️  Cancelled.")
        return

    print(f"\n🚀 Dispatching task: {intent['task']}...")
    result = ros_node.dispatch(
        task=intent['task'],
        object_class=intent.get('object_class', ''),
        radius=float(intent.get('radius', 0.0))
    )

    if result['success']:
        print(f"\n✅ SUCCESS: {result['message']}")
    else:
        print(f"\n❌ FAILED:  {result['message']}")


# ============================================================
# ENTRY POINT
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Voice-controlled ROS2 perception dispatcher",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Live local mic (loop):       python3 voice_dispatcher.py\n"
            "  Windows relay mic (loop):    python3 voice_dispatcher.py --relay\n"
            "  Single audio file:           python3 voice_dispatcher.py --audio recording.wav\n"
            "  MP3 file:                    python3 voice_dispatcher.py --audio command.mp3\n"
        )
    )
    parser.add_argument(
        '--audio',
        type=str,
        default=None,
        metavar='FILE',
        help=(
            "Path to a pre-recorded audio file to transcribe instead of using the microphone.\n"
            "Supported formats: .wav .mp3 .mp4 .m4a .webm .ogg .flac\n"
            "When set, the script runs once and exits instead of looping."
        )
    )
    parser.add_argument(
        '--relay',
        action='store_true',
        default=False,
        help=(
            "Read audio from Windows mic streamed over Tailscale via /tmp/mic_pipe.\n"
            "Requires: ffmpeg relay loop running on Linux + mic_stream.py running on Windows.\n"
            "When not set, uses the local PC microphone (default behaviour)."
        )
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- ROS2 ---
    rclpy.init()
    ros_node = VoiceBrainClient()
    threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True).start()

    # --- STT + LLM ---
    stt = SpeechListener()
    llm = IntentExtractor()

    print("\n" + "="*50)
    print("🤖 Voice Perception Controller")
    if args.audio:
        print(f"   Mode: FILE      →  {args.audio}")
    elif args.relay:
        print("   Mode: RELAY MIC →  /tmp/mic_pipe (Windows via Tailscale)")
    else:
        print("   Mode: LOCAL MIC →  sounddevice default")
    print("   Press Ctrl+C to exit.")
    print("="*50)

    try:
        if args.audio:
            # ── FILE MODE: transcribe once and exit ──
            transcript = stt.transcribe_file(args.audio)
            if transcript:
                run_once(transcript, llm, ros_node)
            else:
                print("❌ Could not transcribe audio file.")

        else:
            # ── MIC MODE (local or relay): loop until Ctrl+C ──
            while True:
                input("\n⏎  Press Enter to start listening (or Ctrl+C to quit)...")
                transcript = stt.listen_once(relay=args.relay)
                if transcript:
                    run_once(transcript, llm, ros_node)
                else:
                    print("❌ Could not understand audio. Try again.")

    except KeyboardInterrupt:
        print("\n\nShutting down voice controller...")
    finally:
        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()


## Relay Command
# ffmpeg -y -f s16le -ar 16000 -ac 1 -i udp://0.0.0.0:4444 -f s16le -ar 16000 -ac 1 /tmp/mic_pipe
