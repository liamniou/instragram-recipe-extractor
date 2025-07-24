import shutil
import sys
import os
import google.generativeai as genai
import time
import yt_dlp
import telebot
from urlextract import URLExtract
from dataclasses import dataclass, field


@dataclass
class Video:
    url: str
    chat_id: int
    filepath: str = field(init=False)
    width: int = field(init=False, default=None)
    height: int = field(init=False, default=None)
    duration: int = field(init=False, default=None)
    description: str = field(init=False, default=None)
    format: str = field(init=False, default="mp4")

    def __post_init__(self):
        self.filepath = f"video_{self.chat_id}_{int(time.time())}.mp4"


@dataclass
class Audio:
    url: str
    chat_id: int
    filepath: str = field(init=False)

    def __post_init__(self):
        self.filepath = f"audio_{self.chat_id}_{int(time.time())}.mp3"


def refine_with_gemini(text_to_refine):
    """
    Uses Gemini 2.5 Flash to refine the extracted text into a structured recipe.
    """
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = (
            "You are a helpful assistant that specializes in formatting recipes. "
            "The following text was extracted from an Instagram video description. "
            "Please format it into a clear, easy-to-read recipe with a title, "
            "a list of ingredients, and numbered instructions. If the text does not "
            "appear to be a recipe, just return the original text."
            "\n\n---\n\n"
            f"{text_to_refine}"
        )

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"An error occurred while communicating with the Gemini API: {e}", file=sys.stderr)
        return None


def convert_recipe_to_metric(recipe_text):
    """
    Uses Gemini 2.5 Flash to convert imperial measurements in a recipe to grams.
    """
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "API key not found, cannot convert to metric."

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = (
            "You are a helpful assistant that specializes in converting recipe measurements. "
            "The following recipe uses imperial units. Please convert all volume and weight "
            "measurements (e.g., oz, lb, cups, tbsp, tsp) to grams. "
            "Keep the original structure and instructions. If a measurement cannot be "
            "converted to grams (e.g., '1 large egg', 'a pinch of salt'), leave it as is. "
            "Return only the converted recipe."
            "\n\n---\n\n"
            f"{recipe_text}"
        )

        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"An error occurred during metric conversion with the Gemini API: {e}", file=sys.stderr)
        return None


def analyze_audio_with_gemini(audio_file_path):
    """
    Uploads an audio file and uses Gemini 2.5 Flash to extract spoken instructions.
    """
    print("🗣️ Analyzing audio for spoken instructions...")
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "API key not found, cannot analyze audio."

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        print(f"Uploading {audio_file_path} to Gemini...")
        audio_file = genai.upload_file(path=audio_file_path)

        while audio_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(5)
            audio_file = genai.get_file(audio_file.name)
        
        print("\nUpload complete.")

        if audio_file.state.name == "FAILED":
            print(f"Audio file processing failed: {audio_file.state}", file=sys.stderr)
            return None

        prompt = (
            "Listen to the audio from this cooking video. Transcribe any spoken instructions, tips, "
            "or steps that are relevant to the recipe. Ignore background music or irrelevant chatter. "
            "If there are no spoken instructions, please state that clearly. "
            "Present the instructions as a clear list."
        )

        response = model.generate_content([prompt, audio_file])
        return response.text

    except Exception as e:
        print(f"An error occurred during audio analysis with the Gemini API: {e}", file=sys.stderr)
        return None


def combine_recipe_and_audio(metric_recipe, audio_notes):
    """
    Combines the text-based recipe with insights from the audio using Gemini.
    """
    print("📝 Combining text recipe with audio notes...")
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return "API key not found, cannot combine recipe."

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = (
            "You are a master recipe editor. Below is a recipe generated from a video's description text, "
            "and a set of notes transcribed from the video's audio. Your task is to create one final, "
            "comprehensive recipe by intelligently merging the audio notes into the recipe instructions."
            "The final recipe should be logical, easy to follow, and complete. Remove unneccesary information (i.e. comment calls, etc.)."
            "\n\n--- RECIPE FROM TEXT ---\n"
            f"{metric_recipe}"
            "\n\n--- NOTES FROM AUDIO ---\n"
            f"{audio_notes}"
            "\n\n--- FINAL COMPREHENSIVE RECIPE ---"
        )
        
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"An error occurred during the final recipe combination with Gemini: {e}", file=sys.stderr)
        return None


def download_video(video: Video):
    """Downloads a video and populates its metadata."""
    ydl_opts = {
        'outtmpl': video.filepath,
        'format': video.format,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_info = ydl.extract_info(video.url, download=True)
        video.width = video_info.get("width")
        video.height = video_info.get("height")
        video.duration = video_info.get("duration")
        video.description = video_info.get("description")
        return {"status": "success"}
    except Exception as e:
        print(f"Failed to download video from {video.url}. Error: {e}")
        return {"status": "failed"}


def download_audio(audio: Audio):
    """Downloads audio from a URL."""
    ydl_opts = {
        'outtmpl': audio.filepath,
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([audio.url])
        return {"status": "success"} if os.path.exists(audio.filepath) else {"status": "failed"}
    except Exception as e:
        print(f"Failed to download audio from {audio.url}. Error: {e}")
        return {"status": "failed"}


def process_url(url, message, bot):
    """
    The core logic to process a URL, sending the final recipe and video when done.
    """
    video = Video(url=url, chat_id=message.chat.id)
    audio = Audio(url=url, chat_id=message.chat.id)

    try:
        # 1. Download video and get info
        bot.reply_to(message, "🔎 Fetching video info and downloading...")
        download_status = download_video(video)
        
        if download_status["status"] == "failed":
            bot.reply_to(message, "❌ Could not download the video.")
            return
            
        if not video.description:
            bot.reply_to(message, "❌ Could not fetch a description from the link.")
            return

        # 2. Refine, convert, and analyze
        bot.reply_to(message, "✨ Refining, converting, and analyzing audio...")
        refined_recipe = refine_with_gemini(video.description)
        if not refined_recipe:
            bot.reply_to(message, "❌ An error occurred while refining the recipe.")
            return

        metric_recipe = convert_recipe_to_metric(refined_recipe) or refined_recipe

        audio_download_status = download_audio(audio)
        final_recipe_text = metric_recipe

        if audio_download_status["status"] == "success":
            audio_notes = analyze_audio_with_gemini(audio.filepath)
            if audio_notes:
                combined_recipe = combine_recipe_and_audio(metric_recipe, audio_notes)
                if combined_recipe:
                    final_recipe_text = combined_recipe
        
        # 3. Send the final results
        if video.filepath and os.path.exists(video.filepath):
            # Prepare caption, truncating if it exceeds Telegram's limit
            caption_text = final_recipe_text
            max_len = 1024
            overhead = 7  # for ```\n...\n```
            if len(caption_text) > max_len:
                truncate_at = max_len - overhead - len("\n...(truncated)")
                caption_text = caption_text[:truncate_at] + "\n...(truncated)"
            
            final_caption = f"```{caption_text}```"

            with open(video.filepath, 'rb') as video_file:
                try:
                    bot.send_video(
                        message.chat.id, video_file,
                        caption=final_caption,
                        parse_mode='MarkdownV2',
                        width=video.width,
                        height=video.height,
                        duration=video.duration
                    )
                except Exception as e:
                    print(f"Failed to send video with formatted caption: {e}. Sending with plain caption.")
                    # Rewind file to read again
                    video_file.seek(0)
                    bot.send_video(
                        message.chat.id, video_file,
                        caption=final_recipe_text, # Send plain text if formatted fails
                        width=video.width,
                        height=video.height,
                        duration=video.duration
                    )
    except Exception as e:
        bot.reply_to(message, f"An unexpected error occurred: {e}")
    finally:
        # Cleanup
        if video.filepath and os.path.exists(video.filepath):
            os.remove(video.filepath)
        if audio.filepath and os.path.exists(audio.filepath):
            os.remove(audio.filepath)


bot = telebot.TeleBot(os.getenv("TELEGRAM_BOT_TOKEN"), threaded=True)
extractor = URLExtract()


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Welcome! Send me a link to an Instagram video, and I'll extract the recipe.")


@bot.message_handler(func=lambda m: True)
def handle_message(message):
    urls = extractor.find_urls(message.text)
    if not urls:
        bot.reply_to(message, "Please send a message with a valid URL.")
        return

    # Process the first valid URL
    process_url(urls[0], message, bot)


def main():
    """Runs the Telebot."""
    # Dependency checks
    if not shutil.which("yt-dlp") or not shutil.which("ffmpeg"):
        print("Error: yt-dlp and ffmpeg must be installed and in your PATH.", file=sys.stderr)
        sys.exit(1)
    if not os.getenv("TELEGRAM_BOT_TOKEN") or not os.getenv("GOOGLE_API_KEY"):
        print("Error: Please set TELEGRAM_BOT_TOKEN and GOOGLE_API_KEY environment variables.", file=sys.stderr)
        sys.exit(1)

    print("Bot is running...")
    bot.polling()


if __name__ == "__main__":
    main()
