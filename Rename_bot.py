#!/usr/bin/env python3
from pyrogram import idle
import os
import asyncio
import logging
import time
import re
import shutil
from pyrogram import filters
from pyrogram.client import Client
from pyrogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from pathlib import Path
import json
import subprocess
from collections import deque
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from config01 import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('professional_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global storage for user settings
user_data = {}

# File sorting system
class FileSorter:
    def __init__(self):
        self.user_files = {}  # user_id: {episode: [files], resolution: [files]}
        self.sort_queue = {}  # user_id: [file_info]
        
    def add_file_for_sorting(self, user_id, file_info):
        """Add file to sorting queue"""
        if user_id not in self.sort_queue:
            self.sort_queue[user_id] = []
        self.sort_queue[user_id].append(file_info)
    
    def get_resolution_priority(self, resolution):
        """Get sorting priority for resolution (lower number = higher priority)"""
        resolution = resolution.lower()
        priority_map = {
            '480p': 1,
            '720p': 2, 
            '1080p': 3,
            'hdrip': 4,
            '4k': 5,
            '2160p': 6,
            'uhd': 5,  # Same as 4K
            'fhd': 3,  # Same as 1080p
            'hd': 2,   # Same as 720p
            'sd': 1    # Same as 480p
        }
        return priority_map.get(resolution, 999)  # Unknown resolutions go last
    
    def should_trigger_sort(self, user_id):
        """Check if we should trigger sorting (when no more files are being processed)"""
        if user_id not in self.sort_queue or not self.sort_queue[user_id]:
            return False
        
        # Check if user has any files in processing queue
        queue_count = processor.get_queue_position(user_id)
        processing_count = processor.processing.get(user_id, 0)
        
        # Trigger sort only when all processing is complete
        return queue_count == 0 and processing_count <= 1
    
    def group_files_for_sorting(self, user_id):
        """Group files by episode and resolution for smart sorting"""
        if user_id not in self.sort_queue:
            return []
        
        files = self.sort_queue[user_id]
        grouped = {}
        
        for file_info in files:
            episode = file_info.get('episode', 'unknown')
            resolution = file_info.get('resolution', 'unknown')
            
            # Group by episode first
            if episode not in grouped:
                grouped[episode] = []
            grouped[episode].append(file_info)
        
        return grouped
    
    def sort_files(self, user_id, sort_method='resolution'):
        """Sort files based on method and return sorted list"""
        if user_id not in self.sort_queue or not self.sort_queue[user_id]:
            return []
        
        files = self.sort_queue[user_id].copy()
        
        if sort_method == 'resolution':
            # Sort by resolution priority (480p -> 720p -> 1080p -> HDRip -> 4K -> 2160p)
            files.sort(key=lambda f: (
                f.get('episode', 'Z99'),  # Episode first
                self.get_resolution_priority(f.get('resolution', 'unknown'))  # Then resolution
            ))
        elif sort_method == 'episodes':
            # Sort by episode number
            files.sort(key=lambda f: (
                f.get('resolution', 'Z999'),  # Resolution first
                f.get('episode', 'Z99')       # Then episode
            ))
        
        # Clear the queue after sorting
        self.sort_queue[user_id] = []
        return files
    
    def clear_user_queue(self, user_id):
        """Clear sorting queue for user"""
        if user_id in self.sort_queue:
            del self.sort_queue[user_id]

# Initialize file sorter
file_sorter = FileSorter()

# Simple concurrent processing system
class SimpleProcessor:
    def __init__(self, max_concurrent=2):
        self.processing = {}  # user_id: count of files being processed
        self.queued_messages = {}  # user_id: [{'message': Message, 'queue_msg': Message}]
        self.queue_counters = {}  # user_id: next queue number
        self.max_concurrent = max_concurrent
        self.lock = asyncio.Lock()
        self.active_tasks = {}  # user_id: [task objects]
        self.cancelled_users = set()  # Track cancelled users
    
    async def delete_message_after_delay(self, message, delay_seconds):
        """Delete a message after specified delay"""
        try:
            await asyncio.sleep(delay_seconds)
            await message.delete()
            logger.debug(f"Auto-deleted message after {delay_seconds} seconds")
        except Exception as e:
            logger.debug(f"Could not auto-delete message: {e}")
    
    async def can_process(self, user_id: int) -> bool:
        async with self.lock:
            current_count = self.processing.get(user_id, 0)
            return current_count < self.max_concurrent
    
    async def start_processing(self, user_id: int):
        async with self.lock:
            self.processing[user_id] = self.processing.get(user_id, 0) + 1
    
    async def finish_processing(self, user_id: int):
        async with self.lock:
            if user_id in self.processing:
                self.processing[user_id] -= 1
                if self.processing[user_id] <= 0:
                    del self.processing[user_id]
    
    async def add_to_queue(self, user_id: int, message: Message, queue_message: Message):
        async with self.lock:
            if user_id not in self.queued_messages:
                self.queued_messages[user_id] = []
            if user_id not in self.queue_counters:
                self.queue_counters[user_id] = 1
            
            queue_number = self.queue_counters[user_id]
            self.queue_counters[user_id] += 1
            
            self.queued_messages[user_id].append({
                'message': message,
                'queue_msg': queue_message,
                'queue_number': queue_number
            })
    
    async def get_next_queued(self, user_id: int) -> Optional[dict]:
        async with self.lock:
            if user_id in self.queued_messages and self.queued_messages[user_id]:
                return self.queued_messages[user_id].pop(0)
            return None
    
    def get_queue_position(self, user_id: int) -> int:
        return len(self.queued_messages.get(user_id, []))
    
    def get_next_queue_number(self, user_id: int) -> int:
        """Get the next queue number for a user"""
        if user_id not in self.queue_counters:
            self.queue_counters[user_id] = 1
        return self.queue_counters[user_id]
    
    async def reset_queue_counter(self, user_id: int):
        """Reset queue counter when all files are processed"""
        async with self.lock:
            if user_id in self.queue_counters:
                self.queue_counters[user_id] = 1
    
    async def cancel_user_tasks(self, user_id: int):
        """Cancel all tasks for a user (only used by /cancelall command)"""
        async with self.lock:
            self.cancelled_users.add(user_id)
            # Clear queue
            if user_id in self.queued_messages:
                for item in self.queued_messages[user_id]:
                    try:
                        await item['queue_msg'].edit_text("❌ **Cancelled by user**")
                    except:
                        pass
                del self.queued_messages[user_id]
            
            # Cancel active tasks
            if user_id in self.active_tasks:
                for task in self.active_tasks[user_id]:
                    if not task.done():
                        task.cancel()
                del self.active_tasks[user_id]
            
            # Reset processing count
            if user_id in self.processing:
                del self.processing[user_id]
            
            # Clean up any temp files for this user
            try:
                import glob
                temp_files = glob.glob(f"temp/*{user_id}*") + glob.glob(f"temp/downloaded_*") + glob.glob(f"temp/processed_*")
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            logger.info(f"Cancelled: Deleted {temp_file}")
                    except:
                        pass
            except Exception as e:
                logger.error(f"Error cleaning up cancelled files: {e}")
    
    async def cancel_current_file(self, user_id: int):
        """Cancel only the current file being processed"""
        async with self.lock:
            self.cancelled_users.add(user_id)
            # Don't clear queue or other files, just set cancel flag
    
    def is_cancelled(self, user_id: int) -> bool:
        """Check if user has cancelled operations"""
        return user_id in self.cancelled_users
    
    def clear_cancel_flag(self, user_id: int):
        """Clear cancel flag for user"""
        if user_id in self.cancelled_users:
            self.cancelled_users.remove(user_id)

# Initialize simple processor
processor = SimpleProcessor(max_concurrent=2)

async def delete_message_after_delay(message, delay_seconds):
    """Delete a message after specified delay"""
    try:
        await asyncio.sleep(delay_seconds)
        await message.delete()
        logger.debug(f"Auto-deleted message after {delay_seconds} seconds")
    except Exception as e:
        logger.debug(f"Could not auto-delete message: {e}")

# --- Enhanced User Settings Management ---
class UserSettings:
    """Enhanced user settings management with persistent storage."""
    
    def __init__(self):
        self.settings_file = USER_SETTINGS_FILE
        self.settings = self._load_settings()

    def _load_settings(self):
        """Load user settings from JSON file."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error("Invalid JSON in settings file, creating new one")
                return {}
        return {}

    def _save_settings(self):
        """Save user settings to JSON file."""
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

    def get_user_settings(self, user_id):
        """Get user settings with defaults if not exist."""
        user_id = str(user_id)
        if user_id not in self.settings:
            self.settings[user_id] = DEFAULT_USER_PREFERENCES.copy()
            self._save_settings()
        return self.settings[user_id]

    def update_setting(self, user_id, key, value):
        """Update a specific user setting."""
        user_settings = self.get_user_settings(user_id)
        if key == 'metadata' and isinstance(value, dict):
            user_settings['metadata'].update(value)
        else:
            user_settings[key] = value
        self.settings[str(user_id)] = user_settings
        self._save_settings()

    def get_send_mode(self, user_id):
        """Get user's preferred send mode."""
        return self.get_user_settings(user_id).get('send_as', 'video')

    def set_send_mode(self, user_id, mode):
        """Set user's send mode (video/file)."""
        if mode.lower() in ['video', 'file']:
            self.update_setting(user_id, 'send_as', mode.lower())
            return True
        return False

    def get_metadata(self, user_id):
        """Get user's metadata settings."""
        return self.get_user_settings(user_id).get('metadata', DEFAULT_USER_PREFERENCES['metadata'])

    def set_metadata_title(self, user_id, title):
        """Set user's default metadata title."""
        self.update_setting(user_id, 'metadata', {'title': title})

    def set_metadata_language(self, user_id, lang_type, language):
        """Set audio or subtitle language."""
        if lang_type in ['audio_language', 'subtitle_language']:
            self.update_setting(user_id, 'metadata', {lang_type: language})

    def set_audio_channel(self, user_id, channel):
        """Set user's audio channel branding."""
        self.update_setting(user_id, 'metadata', {'audio_channel': channel})

    def get_audio_channel(self, user_id):
        """Get user's audio channel branding."""
        return self.get_metadata(user_id).get('audio_channel', '@Animaxclan')

    def get_filename_format(self, user_id):
        """Get user's filename format."""
        return self.get_user_settings(user_id).get('filename_format', DEFAULT_USER_PREFERENCES['filename_format'])

    def set_filename_format(self, user_id, format_string):
        """Set user's filename format."""
        self.update_setting(user_id, 'filename_format', format_string)

    def get_watermark(self, user_id):
        """Get user's watermark settings."""
        return self.get_user_settings(user_id).get('watermark', DEFAULT_USER_PREFERENCES['watermark'])

    def set_watermark(self, user_id, text, enabled=True):
        """Set user's watermark text and status."""
        self.update_setting(user_id, 'watermark', {'text': text, 'enabled': enabled})

    def toggle_watermark(self, user_id):
        """Toggle watermark enabled/disabled status."""
        current = self.get_watermark(user_id)
        new_status = not current.get('enabled', False)
        self.update_setting(user_id, 'watermark', {'text': current.get('text', 'Encoded by @Animaxclan'), 'enabled': new_status})
        return new_status
    
    def get_sort_settings(self, user_id):
        """Get user's sorting settings."""
        settings = self.get_user_settings(user_id)
        return settings.get('sorting', {'enabled': False, 'method': 'resolution'})
    
    def set_sort_method(self, user_id, method):
        """Set sorting method (resolution/episodes)."""
        if method.lower() in ['resolution', 'episodes']:
            current_sort = self.get_sort_settings(user_id)
            current_sort['method'] = method.lower()
            self.update_setting(user_id, 'sorting', current_sort)
            return True
        return False
    
    def toggle_sorting(self, user_id, enabled=None):
        """Toggle sorting on/off or set specific state."""
        current_sort = self.get_sort_settings(user_id)
        if enabled is None:
            current_sort['enabled'] = not current_sort.get('enabled', False)
        else:
            current_sort['enabled'] = enabled
        self.update_setting(user_id, 'sorting', current_sort)
        return current_sort['enabled']
    
    def get_dumb_channel(self, user_id):
        """Get user's dumb channel ID."""
        return self.get_user_settings(user_id).get('dumb_channel')
    
    def set_dumb_channel(self, user_id, channel_id):
        """Set user's dumb channel ID."""
        try:
            # Store channel ID as provided for better compatibility
            if isinstance(channel_id, str):
                if channel_id.startswith('@'):
                    # Keep @ symbol for username format
                    stored_id = channel_id
                elif channel_id.startswith('-100'):
                    # Convert to int for numeric channel ID
                    stored_id = int(channel_id)
                else:
                    # Try to convert to int, otherwise keep as string
                    try:
                        stored_id = int(channel_id)
                    except ValueError:
                        stored_id = channel_id
            else:
                stored_id = channel_id
            
            self.update_setting(user_id, 'dumb_channel', stored_id)
            return True
        except (ValueError, TypeError) as e:
            logger.error(f"Error setting dump channel {channel_id}: {e}")
            return False

# Initialize User Settings Manager
user_settings = UserSettings()


# --- Progress Tracking System ---
class ProgressTracker:
    """Track download and processing progress with real-time updates."""
    
    def __init__(self):
        self.active_progress = {}  # user_id: progress_data
        self.user_statistics = {}  # user_id: processing stats
    
    def create_progress_message(self, file_name, send_mode, percentage=0, 
                              downloaded=0, total_size=0, speed=0, eta="calculating"):
        """Create formatted progress message."""
        # Create progress bar
        filled_blocks = int((percentage / 100) * PROGRESS_BAR_LENGTH)
        empty_blocks = PROGRESS_BAR_LENGTH - filled_blocks
        progress_bar = PROGRESS_FILLED * filled_blocks + PROGRESS_EMPTY * empty_blocks
        
        # Format sizes
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        speed_mb = speed / (1024 * 1024)
        
        message = f"""┌─╼**ᴅᴏᴡɴʟᴏᴀᴅɪɴɢ:**{file_name} [{send_mode}]
**├─╼ᴛɪᴍᴇ:** {self._format_time(time.time())}
**├─╼ᴩʀᴏɢʀᴇꜱꜱ:** {progress_bar} {percentage:.1f}%
**├─╼ꜱɪᴢᴇ:** {downloaded_mb:.1f} MB of {total_mb:.1f} MB
**├─╼ꜱᴩᴇᴇᴅ:** {speed_mb:.1f} MB/s
**└─╼ᴇᴛᴀ:** {eta}"""
        
        return message
    
    def create_cancel_keyboard(self, user_id):
        """Create cancel button keyboard"""
        from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel", callback_data=f"cancel_{user_id}")]
        ])
    
    def _format_time(self, timestamp):
        """Format timestamp to readable time."""
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
    
    def init_user_stats(self, user_id):
        """Initialize processing statistics for user."""
        if user_id not in self.user_statistics:
            self.user_statistics[user_id] = {
                'total_files': 0,
                'total_time': 0,
                'file_times': [],
                'total_size': 0,
                'session_start': time.time()
            }
    
    def add_file_stats(self, user_id, processing_time, file_size):
        """Add file processing statistics."""
        self.init_user_stats(user_id)
        stats = self.user_statistics[user_id]
        stats['total_files'] += 1
        stats['total_time'] += processing_time
        stats['file_times'].append(processing_time)
        stats['total_size'] += file_size
    
    def get_stats_summary(self, user_id):
        """Get processing statistics summary."""
        if user_id not in self.user_statistics:
            return None
        
        stats = self.user_statistics[user_id]
        if stats['total_files'] == 0:
            return None
        
        avg_time = stats['total_time'] / stats['total_files']
        avg_speed = stats['total_size'] / stats['total_time'] if stats['total_time'] > 0 else 0
        avg_speed_mb = avg_speed / (1024 * 1024)
        
        return {
            'total_files': stats['total_files'],
            'total_time': stats['total_time'],
            'avg_time_per_file': avg_time,
            'avg_speed_mb': avg_speed_mb,
            'total_size_mb': stats['total_size'] / (1024 * 1024)
        }
    
    def reset_user_stats(self, user_id):
        """Reset user statistics."""
        if user_id in self.user_statistics:
            del self.user_statistics[user_id]

# Initialize Progress Tracker
progress_tracker = ProgressTracker()

# Create connection semaphore optimized for large files
connection_semaphore = asyncio.Semaphore(2)  # Further reduced to 2 for 2GB file stability

async def check_memory_available():
    """Check if enough memory is available for large file processing"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        return available_gb > 1.0  # Require at least 1GB free memory
    except:
        return True  # If psutil not available, assume OK

class FFmpegProcessor:
    """Advanced FFmpeg processor for comprehensive media operations"""

    @staticmethod
    async def check_ffmpeg():
        """Check if FFmpeg is available"""
        try:
            result = subprocess.run([FFMPEG_PATH, '-version'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    @staticmethod
    async def get_media_info(file_path):
        """Get detailed media information using ffprobe"""
        try:
            cmd = [
                FFPROBE_PATH, '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', file_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                logger.error(f"FFprobe error: {result.stderr}")
                return None
        except Exception as e:
            logger.error(f"Error getting media info: {e}")
            return None

    @staticmethod
    async def extract_thumbnail(video_path, output_path, timestamp="00:00:01"):
        """Extract thumbnail from video"""
        try:
            cmd = [
                FFMPEG_PATH, '-y', '-i', video_path, '-ss', timestamp,
                '-vframes', '1', '-vf', 'scale=320:320:force_original_aspect_ratio=decrease,pad=320:320:(ow-iw)/2:(oh-ih)/2',
                '-q:v', '2', output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Thumbnail extraction error: {e}")
            return False

    @staticmethod
    async def replace_audio_stream(video_path, audio_path, output_path, audio_title="Custom Audio"):
        """Replace audio stream in video with metadata"""
        try:
            cmd = [
                FFMPEG_PATH, '-y', '-i', video_path, '-i', audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
                '-map', '0:v:0', '-map', '1:a:0',
                '-metadata:s:a:0', f'title={audio_title}',
                '-metadata:s:a:0', 'language=eng',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Audio replaced successfully: {output_path}")
                return True
            else:
                logger.error(f"Audio replacement error: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Audio replacement exception: {e}")
            return False

    @staticmethod
    async def add_subtitles(video_path, subtitle_path, output_path, subtitle_title="Subtitles"):
        """Add subtitle stream to video"""
        try:
            cmd = [
                FFMPEG_PATH, '-y', '-i', video_path, '-i', subtitle_path,
                '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'srt',
                '-metadata:s:s:0', f'title={subtitle_title}',
                '-metadata:s:s:0', 'language=eng',
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Subtitles added successfully: {output_path}")
                return True
            else:
                logger.error(f"Subtitle addition error: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Subtitle addition exception: {e}")
            return False

    @staticmethod
    async def edit_metadata(file_path, metadata_dict, output_path):
        """Edit video metadata while preserving ALL original audio and subtitle tracks"""
        try:
            # Get detailed media information to preserve all tracks
            media_info = await FFmpegProcessor.get_media_info(file_path)
            
            if not media_info or 'streams' not in media_info:
                logger.error("Could not analyze media file")
                return False

            cmd = [FFMPEG_PATH, '-y', '-i', file_path]

            # Add global metadata parameters
            for key, value in metadata_dict.items():
                cmd.extend(['-metadata', f'{key}={value}'])

            title = metadata_dict.get('title', 'Alpha Zenin')
            audio_channel = metadata_dict.get('audio_channel', '@Animaxclan')
            
            # Map ALL streams to preserve exact structure
            stream_maps = []
            video_index = 0
            audio_index = 0
            subtitle_index = 0
            
            # First pass: map all streams to preserve them
            for i, stream in enumerate(media_info['streams']):
                stream_type = stream.get('codec_type', '')
                cmd.extend(['-map', f'0:{i}'])  # Map each stream explicitly
                
                if stream_type == 'video':
                    # Set video stream metadata
                    cmd.extend([f'-metadata:s:v:{video_index}', f'title={title}'])
                    video_index += 1
                    
                elif stream_type == 'audio':
                    # Detect original language for each audio track
                    original_language = "Japanese"  # Default
                    tags = stream.get('tags', {})
                    
                    if 'language' in tags:
                        lang_code = tags['language'].lower()
                        lang_map = {
                            'jpn': 'Japanese', 'ja': 'Japanese',
                            'eng': 'English', 'en': 'English', 
                            'hin': 'Hindi', 'hi': 'Hindi',
                            'kor': 'Korean', 'ko': 'Korean',
                            'chi': 'Chinese', 'zh': 'Chinese',
                            'spa': 'Spanish', 'es': 'Spanish',
                            'fre': 'French', 'fr': 'French',
                            'ger': 'German', 'de': 'German'
                        }
                        original_language = lang_map.get(lang_code, lang_code.capitalize())
                    elif 'title' in tags:
                        title_lower = tags['title'].lower()
                        if 'english' in title_lower or 'eng' in title_lower:
                            original_language = 'English'
                        elif 'hindi' in title_lower or 'hin' in title_lower:
                            original_language = 'Hindi'
                        elif 'korean' in title_lower or 'kor' in title_lower:
                            original_language = 'Korean'
                        elif 'chinese' in title_lower or 'chi' in title_lower:
                            original_language = 'Chinese'
                    
                    # Set audio stream metadata preserving original language
                    cmd.extend([f'-metadata:s:a:{audio_index}', f'title=By {audio_channel} - {original_language}'])
                    
                    # Preserve original language code if it exists
                    if 'language' in tags:
                        cmd.extend([f'-metadata:s:a:{audio_index}', f'language={tags["language"]}'])
                    
                    audio_index += 1
                    
                elif stream_type == 'subtitle':
                    # Detect original language for each subtitle track
                    original_language = "Japanese"  # Default
                    tags = stream.get('tags', {})
                    
                    if 'language' in tags:
                        lang_code = tags['language'].lower()
                        lang_map = {
                            'jpn': 'Japanese', 'ja': 'Japanese',
                            'eng': 'English', 'en': 'English', 
                            'hin': 'Hindi', 'hi': 'Hindi',
                            'kor': 'Korean', 'ko': 'Korean',
                            'chi': 'Chinese', 'zh': 'Chinese',
                            'spa': 'Spanish', 'es': 'Spanish',
                            'fre': 'French', 'fr': 'French',
                            'ger': 'German', 'de': 'German'
                        }
                        original_language = lang_map.get(lang_code, lang_code.capitalize())
                    elif 'title' in tags:
                        title_lower = tags['title'].lower()
                        if 'english' in title_lower or 'eng' in title_lower:
                            original_language = 'English'
                        elif 'hindi' in title_lower or 'hin' in title_lower:
                            original_language = 'Hindi'
                        elif 'korean' in title_lower or 'kor' in title_lower:
                            original_language = 'Korean'
                        elif 'chinese' in title_lower or 'chi' in title_lower:
                            original_language = 'Chinese'
                    
                    # Set subtitle stream metadata preserving original language
                    cmd.extend([f'-metadata:s:s:{subtitle_index}', f'title=By {audio_channel} - {original_language}'])
                    
                    # Preserve original language code if it exists
                    if 'language' in tags:
                        cmd.extend([f'-metadata:s:s:{subtitle_index}', f'language={tags["language"]}'])
                    
                    subtitle_index += 1

            # Copy all streams without re-encoding to preserve quality and all tracks
            cmd.extend(['-c', 'copy', output_path])

            logger.info(f"FFmpeg command: {' '.join(cmd[:15])}... (preserving {audio_index} audio tracks, {subtitle_index} subtitle tracks)")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Metadata edited successfully preserving all {audio_index} audio tracks and {subtitle_index} subtitle tracks")
                return True
            else:
                logger.error(f"Metadata editing error: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Metadata editing exception: {e}")
            return False

    @staticmethod
    async def replace_anime_unity_metadata(file_path, output_path, title="Alpha Zenin", audio_channel="@Animaxclan"):
        """Change video, audio, subtitle stream titles and main media title with custom channel branding"""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', file_path,
                '-metadata', f'title={title}',
                '-metadata:s:v:0', f'title={title}',
                '-metadata:s:a:0', f'title=By {audio_channel}',
                '-metadata:s:s:0', f'title=By {audio_channel}',
                '-c', 'copy', output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Successfully set media title to {title} and audio to 'By {audio_channel}'")
                return True
            else:
                logger.error(f"Media title update error: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Media title update exception: {e}")
            return False

    @staticmethod
    async def add_watermark(file_path, output_path, watermark_text="Encoded by @Animaxclan"):
        """Add text watermark to video for 3 seconds at bottom position with bold text"""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', file_path,
                '-vf', f"drawtext=text='{watermark_text}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=h-th-20:enable='between(t,0,3)':fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                '-codec:a', 'copy', output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"Successfully added 3-second bottom watermark: {watermark_text}")
                return True
            else:
                logger.error(f"Watermark addition error: {result.stderr}")
                # Fallback without bold font if font file not found
                cmd_fallback = [
                    'ffmpeg', '-y', '-i', file_path,
                    '-vf', f"drawtext=text='{watermark_text}':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=h-th-20:enable='between(t,0,3)'",
                    '-codec:a', 'copy', output_path
                ]
                result_fallback = subprocess.run(cmd_fallback, capture_output=True, text=True)
                if result_fallback.returncode == 0:
                    logger.info(f"Successfully added 3-second bottom watermark (fallback): {watermark_text}")
                    return True
                else:
                    logger.error(f"Watermark fallback error: {result_fallback.stderr}")
                    return False
        except Exception as e:
            logger.error(f"Watermark addition exception: {e}")
            return False

    @staticmethod
    async def merge_intro_with_video(intro_path, video_path, output_path):
        """Merge intro video with main video using filter_complex"""
        try:
            # Use filter_complex for more reliable concatenation
            cmd = [
                FFMPEG_PATH, '-y', 
                '-i', intro_path, 
                '-i', video_path,
                '-filter_complex', '[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[outv][outa]',
                '-map', '[outv]', 
                '-map', '[outa]',
                '-c:v', 'libx264', 
                '-c:a', 'aac',
                '-preset', 'ultrafast',
                '-crf', '18',
                output_path
            ]

            logger.info(f"Merging intro: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info(f"Intro merged successfully: {output_path}")
                return True
            else:
                logger.error(f"Intro merge error: {result.stderr}")
                # Try alternative method with concat demuxer
                return await FFmpegProcessor._merge_intro_concat_fallback(intro_path, video_path, output_path)
        except Exception as e:
            logger.error(f"Intro merge exception: {e}")
            return False

    @staticmethod
    async def _merge_intro_concat_fallback(intro_path, video_path, output_path):
        """Fallback method using concat demuxer"""
        try:
            # Create concat file
            concat_file = f"temp/concat_{int(time.time())}.txt"
            os.makedirs("temp", exist_ok=True)

            with open(concat_file, 'w') as f:
                f.write(f"file '{os.path.abspath(intro_path)}'\n")
                f.write(f"file '{os.path.abspath(video_path)}'\n")

            cmd = [
                FFMPEG_PATH, '-y', '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-c', 'copy', output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            # Clean up concat file
            if os.path.exists(concat_file):
                os.remove(concat_file)

            if result.returncode == 0:
                logger.info(f"Intro merged with fallback method: {output_path}")
                return True
            else:
                logger.error(f"Fallback intro merge error: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Fallback intro merge exception: {e}")
            return False

class FileProcessor:
    @staticmethod
    def extract_metadata(filename, caption=""):
        """Extract season, episode, resolution, audio from filename and caption"""
        text = f"{filename} {caption}".lower()

        metadata = {
            'season': 'S01',
            'episode': 'E01', 
            'resolution': '720p',
            'audio': 'Dual'
        }

        # Enhanced season extraction
        season_patterns = [
            r'(?:season|s)[\s._-]*0*(\d+)',
            r'(?:^|\W)s(\d+)(?:\W|$)',
            r'(?:^|\W)(\d+)(?:st|nd|rd|th)?\s*season',
        ]
        
        for pattern in season_patterns:
            match = re.search(pattern, text)
            if match:
                season_num = int(match.group(1))
                metadata['season'] = f"S{season_num:02d}"
                break

        # Enhanced episode extraction - returns just the number
        episode_patterns = [
            r'(?:episode|ep|e)[\s._-]*0*(\d+)',
            r'(?:^|\W)e(\d+)(?:\W|$)',
            r'(?:^|\W)(\d+)(?:st|nd|rd|th)?\s*episode',
            r'\b(\d{1,3})(?:\W|$)',  # Generic number at word boundary
        ]
        
        for pattern in episode_patterns:
            match = re.search(pattern, text)
            if match:
                episode_num = int(match.group(1))
                if 1 <= episode_num <= 999:  # Reasonable episode range
                    metadata['episode'] = f"{episode_num:02d}"  # Just the number, not "E01"
                    break

        # Resolution extraction
        resolution_patterns = [
            r'(\d{3,4}p)',
            r'(\d{3,4}x\d{3,4})',
            r'(4k|uhd|2160p)',
            r'(fhd|1080p)',
            r'(hd|720p)',
            r'(480p|sd)',
        ]
        
        for pattern in resolution_patterns:
            match = re.search(pattern, text)
            if match:
                resolution = match.group(1)
                if 'fhd' in resolution or '1080' in resolution:
                    metadata['resolution'] = '1080p'
                elif 'hd' in resolution or '720' in resolution:
                    metadata['resolution'] = '720p'
                elif '4k' in resolution or 'uhd' in resolution or '2160' in resolution:
                    metadata['resolution'] = '4K'
                elif '480' in resolution or 'sd' in resolution:
                    metadata['resolution'] = '480p'
                else:
                    metadata['resolution'] = resolution
                break

        # Audio format extraction
        audio_patterns = [
            r'(dual[\s._-]*audio|dual)',
            r'(hindi[\s._-]*dubbed|hindi)',
            r'(english[\s._-]*dubbed|english)',
            r'(multi[\s._-]*audio|multi)',
            r'(japanese|jap)',
            r'(tamil|telugu|kannada)',
        ]
        
        for pattern in audio_patterns:
            match = re.search(pattern, text)
            if match:
                audio_type = match.group(1).lower()
                if 'dual' in audio_type:
                    metadata['audio'] = 'Dual Audio'
                elif 'hindi' in audio_type:
                    metadata['audio'] = 'Hindi'
                elif 'english' in audio_type:
                    metadata['audio'] = 'English'
                elif 'multi' in audio_type:
                    metadata['audio'] = 'Multi Audio'
                elif 'japanese' in audio_type or 'jap' in audio_type:
                    metadata['audio'] = 'Japanese'
                else:
                    metadata['audio'] = audio_type.title()
                break

        return metadata

    @staticmethod
    def apply_template(template, metadata):
        """Apply template with metadata placeholders"""
        result = template
        for key, value in metadata.items():
            placeholder = f"{{{key}}}"
            result = result.replace(placeholder, str(value))
        return result

    @staticmethod
    def is_supported_format(filename):
        """Check if file format is supported"""
        file_ext = Path(filename).suffix.lower()
        return file_ext in SUPPORTED_VIDEO_FORMATS

# Initialize Pyrogram client with download optimizations
app = Client(
    "professional_rename_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN,
    workers=4,  # Optimized for large file downloads
    workdir=".",
    sleep_threshold=60,  # Better connection handling
    max_concurrent_transmissions=2  # Limit concurrent downloads for stability
)

# --- Bot Handlers ---

@app.on_message(filters.command("start"))
async def start_command(client, message):
    """Start command handler"""
    if not message.from_user:
        logger.warning("Received start command without from_user attribute")
        return
    
    user_id = message.from_user.id
    user_prefs = user_settings.get_user_settings(user_id)
    
    welcome_text = f"""
🎬 **Professional Rename Bot** 🎬

Welcome {message.from_user.first_name}! 

I'm your advanced file renaming assistant with powerful media processing capabilities.

**Features:**
• 🔄 Auto rename with smart templates
• 🎞️ Season/Episode detection
• 🖼️ Thumbnail management  
• 🎵 Audio & subtitle language support
• 📊 Real-time progress tracking
• 📦 Queue system for multiple files
• 💾 Persistent user preferences

**Current Settings:**
• Send Mode: {user_prefs['send_as'].title()}
• Metadata: {user_prefs['metadata']['title']} - {user_prefs['metadata']['audio_language']}

**Key Features:**
• **Custom Branding**: Replaces @any_channel_name with your custom name in metadata
• **Smart Captions**: File caption matches renamed filename in bold
• **Processing Stats**: Shows total files, time taken, and average speed

**Commands:**
/mode video|file - Set sending mode
/setmeta <title> - Set default metadata (Default: Alpha Zenin)
/file <format> - Set filename format
/watermark <text> - Set watermark text (Default: Encoded by @Animaxclan)
/stats - View current processing statistics
/settings - Configure preferences

Send me video files to start renaming with custom metadata!
"""
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("⚙️ Settings", callback_data="settings")],
        [InlineKeyboardButton("📊 Status", callback_data="status")],
        [InlineKeyboardButton("📖 Help", callback_data="help")]
    ])
    
    await message.reply(welcome_text, reply_markup=keyboard)

@app.on_message(filters.command("mode"))
async def mode_command(client, message):
    """Handle /mode command for setting send mode"""
    if not message.from_user:
        logger.warning("Received mode command without from_user attribute")
        return
    
    user_id = message.from_user.id
    
    if len(message.command) > 1:
        mode = message.command[1].lower()
        if user_settings.set_send_mode(user_id, mode):
            await message.reply(f"✅ Send mode updated to: **{mode.title()}**")
        else:
            await message.reply("❌ Invalid mode. Use: `/mode video` or `/mode file`")
    else:
        current_mode = user_settings.get_send_mode(user_id)
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📹 Video", callback_data="mode_video")],
            [InlineKeyboardButton("📄 File", callback_data="mode_file")]
        ])
        await message.reply(f"🔧 **Current Mode:** {current_mode.title()}\n\nChoose your preferred send mode:", reply_markup=keyboard)

@app.on_message(filters.command("setmeta"))
async def setmeta_command(client, message):
    """Handle /setmeta command for setting default metadata"""
    if not message.from_user:
        logger.warning("Received setmeta command without from_user attribute")
        return
    
    user_id = message.from_user.id
    
    if len(message.command) > 1:
        title = " ".join(message.command[1:])
        user_settings.set_metadata_title(user_id, title)
        await message.reply(f"✅ Default metadata title set to: **{title}**")
    else:
        current_meta = user_settings.get_metadata(user_id)
        await message.reply(f"🏷️ **Current Metadata:**\n• Title: {current_meta['title']}\n• Audio: {current_meta['audio_language']}\n• Subtitles: {current_meta['subtitle_language']}\n\n**Usage:** `/setmeta Your Title Here`")

@app.on_message(filters.command("file"))
async def file_format_command(client, message):
    """Handle /file command for setting filename format"""
    if not message.from_user:
        return
    
    user_id = message.from_user.id
    
    if len(message.command) > 1:
        format_string = " ".join(message.command[1:])
        user_settings.set_filename_format(user_id, format_string)
        await message.reply(f"✅ Filename format updated to:\n`{format_string}`")
    else:
        current_format = user_settings.get_filename_format(user_id)
        await message.reply(f"🎯 **Current Format:**\n`{current_format}`\n\n**Placeholders:**\n• `{{season}}` - Season number\n• `{{episode}}` - Episode number\n• `{{resolution}}` - Video quality\n• `{{audio}}` - Audio type\n\n**Usage:** `/file [S-{{season}}-Ep-{{episode}}] Title [{{resolution}}][{{audio}}]@Channel.mkv`")

@app.on_message(filters.command("stats"))
async def stats_command(client, message):
    """Show current processing statistics"""
    if not message.from_user:
        return
    
    user_id = message.from_user.id
    stats = progress_tracker.get_stats_summary(user_id)
    
    if stats and stats['total_files'] > 0:
        stats_message = f"""
📊 **Current Session Statistics** 📊

**Files Processed:** {stats['total_files']}
**Total Processing Time:** {int(stats['total_time'])}s ({int(stats['total_time'] // 60)}m {int(stats['total_time'] % 60)}s)
**Average Time per File:** {stats['avg_time_per_file']:.1f}s
**Average Processing Speed:** {stats['avg_speed_mb']:.2f} MB/s
**Total Size Processed:** {stats['total_size_mb']:.1f} MB

**Metadata Branding:** All files processed with **Custom** metadata
**Caption Format:** Filenames displayed in **bold** format

Use `/reset_stats` to reset these statistics.
"""
        await message.reply(stats_message)
    else:
        await message.reply("📊 **No processing statistics available yet.**\n\nSend some files to start tracking statistics!")

@app.on_message(filters.command("reset_stats"))
async def reset_stats_command(client, message):
    """Reset processing statistics"""
    user_id = message.from_user.id
    progress_tracker.reset_user_stats(user_id)
    await message.reply("✅ **Processing statistics have been reset.**\n\nNew statistics will be tracked from your next file processing session.")

@app.on_message(filters.command("watermark"))
async def watermark_command(client, message):
    """Handle /watermark command for setting watermark text"""
    user_id = message.from_user.id
    
    if len(message.command) > 1:
        watermark_text = " ".join(message.command[1:])
        user_settings.set_watermark(user_id, watermark_text, enabled=True)
        await message.reply(f"✅ **Watermark updated:**\n`{watermark_text}`\n\n🎯 Watermark is now **enabled** and will appear for 3 seconds at the bottom of videos.")
    else:
        current_watermark = user_settings.get_watermark(user_id)
        status = "Enabled" if current_watermark.get('enabled') else "Disabled"
        await message.reply(f"🎨 **Current Watermark:**\n• Text: `{current_watermark.get('text', 'Encoded by @Animaxclan')}`\n• Status: **{status}**\n\n**Usage:** `/watermark Your Custom Text Here`\n\nUse settings menu to enable/disable watermark.")

@app.on_message(filters.command("settings"))
async def settings_command(client, message):
    """Enhanced settings command handler"""
    user_id = message.from_user.id
    prefs = user_settings.get_user_settings(user_id)
    
    watermark_info = prefs.get('watermark', {'enabled': False, 'text': 'Encoded by @Animaxclan'})
    watermark_status = '✅ Enabled' if watermark_info.get('enabled') else '❌ Disabled'
    watermark_text_display = f"\n• Text: `{watermark_info.get('text', 'Encoded by @Animaxclan')}`" if watermark_info.get('enabled') else ''
    
    settings_text = f"""
⚙️ **Your Current Settings** ⚙️

**Send Mode:** {prefs['send_as'].title()}
**Filename Format:** 
`{prefs['filename_format']}`

**Metadata:**
• Title: {prefs['metadata']['title']}
• Audio Language: {prefs['metadata']['audio_language']}
• Subtitle Language: {prefs['metadata']['subtitle_language']}

**Watermark:** {watermark_status}{watermark_text_display}

**Thumbnail:** {'✅ Set' if prefs.get('thumbnail') else '❌ Not Set'}
**Intro:** {'✅ Set' if prefs.get('intro') else '❌ Not Set'}
"""
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔧 Send Mode", callback_data="set_mode"),
         InlineKeyboardButton("🎯 Format", callback_data="set_format")],
        [InlineKeyboardButton("🏷️ Metadata", callback_data="metadata_menu"),
         InlineKeyboardButton("🎨 Watermark", callback_data="watermark_menu")],
        [InlineKeyboardButton("🖼️ Thumbnail", callback_data="set_thumbnail"),
         InlineKeyboardButton("🎬 Intro", callback_data="set_intro")]
    ])
    
    await message.reply(settings_text, reply_markup=keyboard)

@app.on_message(filters.command("template"))
async def template_command(client, message):
    """Template setting command"""
    user_id = message.from_user.id
    
    if len(message.command) > 1:
        template = " ".join(message.command[1:])
        user_settings.update_setting(user_id, 'template', template)
        await message.reply(f"✅ Template updated to: `{template}`")
    else:
        current_template = user_settings.get_user_settings(user_id)['template']
        template_help = f"""
📝 **Template System** 📝

Current template: `{current_template}`

**Available placeholders:**
• `{{season}}` - Season number (01, 02, etc.)
• `{{episode}}` - Episode number (01, 02...) 
• `{{resolution}}` - Video resolution (480p, 720p, 1080p, etc.)
• `{{audio}}` - Audio type (Dual, Hindi, English, etc.)

**Examples:**
• `S-{{season}}-Ep-{{episode}}` → S-01-Ep-05
• `{{season}}{{episode}} [{{resolution}}]` → S01E05 [1080p]
• `Episode {{episode}} {{audio}}` → Episode E05 Dual

**Usage:** `/template S-{{season}}-Ep-{{episode}}`
"""
        await message.reply(template_help)

@app.on_message(filters.command("help"))
async def help_command(client, message):
    """Help command handler"""
    help_text = """
📖 **Help & Commands** 📖

**Basic Commands:**
/start - Start the bot
/settings - Open settings panel
/mode video|file - Set sending mode
/file <format> - Set filename format
/stats - View processing statistics
/cleanup - Clean temporary files
/cancelall - Cancel all ongoing processes
/help - Show this help

**Metadata Commands:**
/setmeta <title> - Set custom media title (Default: Alpha Zenin)
/setchannel <channel> - Set audio channel branding (Default: @Animaxclan)

**File Processing:**
• Send any video file to rename it
• Caption will be analyzed for metadata
• Bot supports: MP4, MKV, AVI, MOV, WMV, FLV, WEBM
• **Cancel Button:** Click cancel during processing to stop and delete file
• Files show as "By @YourChannel" in metadata (clean, no duplicates)

**Template Placeholders:**
• `{season}` - Season (S01, S02...)
• `{episode}` - Episode (01, 02...) 
• `{resolution}` - Quality (720p, 1080p...)
• `{audio}` - Audio (Dual, Hindi, English...)

**Example Format:**
`/file [S-{season}-Ep-{episode}] Title [{resolution}][{audio}]@Channel.mkv`

🎨 **Watermark System**
- `/watermark <text>` - Set custom watermark text
- Shows for 3 seconds at video start (bottom center)
- Toggle watermark on/off in settings
- Default: "Encoded by @Animaxclan"

🔀 **Smart Sorting System**
- `/sort` - Choose sorting method (resolution/episodes)
- `/sort on` - Enable auto-sorting
- `/sort off` - Disable auto-sorting
- **Resolution Sort:** 480p → 720p → 1080p → HDRip → 4K → 2160p
- **Episode Sort:** Sequential episode ordering (Ep-01 → Ep-02...)
- Sorting triggers automatically after all files are processed

📤 **Dump Channel System**
- `/set_dumb <channel_id>` - Set channel for auto-forwarding
- Bot must be admin in that channel
- Files sent directly (no "forwarded from" tag)
- Supports @channelname or -100xxxxxxxx format
- Sends files in sorted order if sorting is enabled

**Queue & Cancel System:**
• Process up to 2 files simultaneously  
• Sequential numbering (01, 02, 03...)
• Additional files are queued automatically
• **Cancel Button:** Stops current file only, deletes partial file instantly
• **`/cancelall`:** Cancels all queue + ongoing + deletes all temp files
• Smart cleanup prevents stuck downloads

**Processing Features:**
• Large file support (up to 2GB)
• Real-time progress tracking with ETA
• Custom thumbnail support (send photo to set)
• Intro video merging (send video with "intro" caption)
• Processing statistics and speed tracking
• Bold filename captions for clean display

**Tips:**
• Use descriptive captions for better detection
• Set thumbnail once, applies to all videos
• Processing stats reset after each session
• All preferences are saved permanently
• Cancelled files are automatically deleted

Need more help? Contact @Roxy_bot_support
"""
    await message.reply(help_text)

@app.on_message(filters.document | filters.video)
async def handle_media_file(client, message):
    """Enhanced media file handler with optimized queue system"""
    # Check if message has from_user attribute
    if not message.from_user:
        logger.warning("Received message without from_user attribute, ignoring")
        return
    
    user_id = message.from_user.id
    
    # Check if user can process immediately or needs to queue
    if not await processor.can_process(user_id):
        queue_number = processor.get_next_queue_number(user_id)
        queue_msg = await message.reply(f"**ᴀᴅᴅᴇᴅ ᴛᴏ qᴜᴇᴜᴇ {queue_number:02d}**")
        await processor.add_to_queue(user_id, message, queue_msg)
        return
    
    # Start processing
    await processor.start_processing(user_id)
    
    try:
        await process_file_enhanced(client, message)
    finally:
        await processor.finish_processing(user_id)
        
        # Check if there are queued files to process
        next_queued = await processor.get_next_queued(user_id)
        if next_queued and await processor.can_process(user_id):
            # Delete the queue message for cleaner chat
            try:
                await next_queued['queue_msg'].delete()
            except:
                pass  # Message might already be deleted
            
            # Process the next file
            await handle_media_file(client, next_queued['message'])
            
        # Check if no more files in queue to reset counter
        if processor.get_queue_position(user_id) == 0:
            await processor.reset_queue_counter(user_id)

async def update_upload_progress(status_msg, current, total, filename):
    """Update upload progress with speed optimization"""
    try:
        # Only update every 20MB or on completion to prevent stuck uploads
        if current % (1024 * 1024 * 20) == 0 or current == total:
            percentage = (current / total) * 100 if total > 0 else 0
            current_mb = current / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            
            progress_bar = "▬" * int(percentage / 10) + "▭" * (10 - int(percentage / 10))
            
            upload_msg = f"""**┌─╼Uploading: {filename}**
├─╼{progress_bar} {percentage:.1f}%
├─╼{current_mb:.1f} MB of {total_mb:.1f} MB
└─╼ETA: {progress_tracker.get_eta(current, total)}"""
            
            await status_msg.edit(upload_msg)
    except Exception as e:
        logger.debug(f"Upload progress update failed: {e}")
        pass  # Ignore rate limits

async def emergency_cleanup():
    """Enhanced emergency cleanup for large file processing"""
    try:
        import time
        import gc
        temp_dir = Path("temp")
        current_time = time.time()
        
        # Check temp directory size
        total_size = 0
        old_files = []
        
        for file_path in temp_dir.glob("*"):
            if file_path.is_file():
                file_size = file_path.stat().st_size
                file_age = current_time - file_path.stat().st_mtime
                total_size += file_size
                old_files.append((file_path, file_size, file_age))
        
        # Aggressive cleanup for large file processing
        cleanup_threshold = 500 * 1024 * 1024  # 500MB threshold for large files
        target_size = 200 * 1024 * 1024  # Target 200MB after cleanup
        
        if total_size > cleanup_threshold:
            logger.warning(f"Emergency cleanup: temp directory size {total_size/1024/1024/1024:.1f}GB")
            
            # Sort by age (oldest first) and size (largest first)
            old_files.sort(key=lambda x: (x[2], -x[1]))
            
            # Remove files until under target size
            for file_path, file_size, file_age in old_files:
                try:
                    if file_age > 300:  # Files older than 5 minutes
                        file_path.unlink()
                        total_size -= file_size
                        logger.info(f"Emergency cleanup removed: {file_path} ({file_size/1024/1024:.1f}MB)")
                        
                        if total_size < target_size:
                            break
                except Exception as cleanup_error:
                    logger.debug(f"Could not remove {file_path}: {cleanup_error}")
                    pass
        
        # Force garbage collection for memory management
        gc.collect()
        logger.debug(f"Post-cleanup temp directory size: {total_size/1024/1024:.1f}MB")
                    
    except Exception as e:
        logger.error(f"Emergency cleanup error: {e}")

async def process_file_enhanced(client, message):
    """Enhanced file processing with progress tracking and disk management"""
    user_id = message.from_user.id
    start_time = time.time()
    
    # Ensure temp directory exists
    os.makedirs("temp", exist_ok=True)
    
    # Emergency cleanup before processing
    await emergency_cleanup()
    
    # Get file information first
    if message.document:
        file_info = message.document
        file_name = file_info.file_name
    elif message.video:
        file_info = message.video
        file_name = getattr(file_info, 'file_name', 'video.mp4')
    else:
        await message.reply("❌ Unsupported file type")
        return
    
    # Sanitize file name for safe filesystem operations
    import re
    safe_file_name = re.sub(r'[\\/*?:"<>|]', "_", file_name)
    if len(safe_file_name) > 200:  # Limit length for very long filenames
        name, ext = os.path.splitext(safe_file_name)
        safe_file_name = name[:190] + ext
    
    # Enhanced space and memory checks for large files
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        required_space = file_info.file_size * 2.5  # Optimized space calculation
        free_gb = free / (1024**3)
        required_gb = required_space / (1024**3)
        
        if free < required_space:
            await message.reply(f"❌ **Insufficient disk space for large file**\n**Required:** {required_gb:.1f}GB\n**Available:** {free_gb:.1f}GB\n**File Size:** {file_info.file_size/(1024**3):.1f}GB")
            return
            
        # Check memory for large files
        if file_info.file_size > LARGE_FILE_THRESHOLD:
            if not await check_memory_available():
                await message.reply(f"❌ **Insufficient memory for large file processing**\n**File Size:** {file_info.file_size/(1024**3):.1f}GB\n**Please try again later or use a smaller file.**")
                return
                
        logger.info(f"Space check passed: {free_gb:.1f}GB available, {required_gb:.1f}GB required")
    except Exception as e:
        logger.warning(f"Could not check system resources: {e}")
    
    
    
    # Check file size
    if file_info.file_size > MAX_FILE_SIZE:
        await message.reply(f"❌ File too large. Max size: {MAX_FILE_SIZE // (1024*1024*1024)}GB")
        return
    
    # Check if supported format
    if not FileProcessor.is_supported_format(file_name):
        await message.reply("❌ Unsupported file format")
        return
    
    # Get user settings
    user_prefs = user_settings.get_user_settings(user_id)
    send_mode = user_prefs['send_as']
    
    # Initial progress message
    progress_msg = progress_tracker.create_progress_message(
        file_name, send_mode.title(), 0, 0, file_info.file_size
    )
    cancel_keyboard = progress_tracker.create_cancel_keyboard(user_id)
    status_msg = await message.reply(progress_msg, reply_markup=cancel_keyboard)
    
    try:
        # Single-attempt reliable download system for large files (up to 2GB)
        download_start = time.time()
        downloaded_bytes = 0
        
        # Dynamic timeout based on file size (minimum 45 minutes, up to 180 minutes for 2GB files)
        base_timeout = 2700  # 45 minutes
        size_based_timeout = min(base_timeout + (file_info.file_size / (1024*1024)) * 5, 10800)  # Max 3 hours
        download_timeout = int(size_based_timeout)
        
        # Optimized progress tracking for large files
        last_update = 0
        update_interval = 8 if file_info.file_size > 500 * 1024 * 1024 else 5  # Longer intervals for stability
        
        file_size_mb = file_info.file_size / (1024*1024)
        await status_msg.edit(f"📥 **Downloading File**\n**File:** {file_name}\n**Size:** {file_size_mb:.1f} MB\n**Timeout:** {download_timeout//60}m\n**Single-attempt optimized download**")
        
        # Create unique temp filename to avoid conflicts
        timestamp = int(time.time())
        temp_filename = f"temp/downloading_{timestamp}_{safe_file_name}"
        
        # Enhanced progress callback with stability optimizations
        async def stable_progress_callback(current, total):
            nonlocal downloaded_bytes, last_update
            
            # Check for cancellation frequently
            if processor.is_cancelled(user_id):
                logger.info(f"Download cancelled by user {user_id}")
                raise asyncio.CancelledError("User cancelled operation")
                
            downloaded_bytes = current
            current_time = time.time()
            
            # Conservative update intervals to prevent connection issues
            if current_time - last_update >= update_interval or current == total:
                last_update = current_time
                percentage = (current / total) * 100 if total > 0 else 0
                elapsed = current_time - download_start
                speed = current / elapsed if elapsed > 0 else 0
                eta = ((total - current) / speed) if speed > 0 else 0
                eta_str = f"{int(eta // 60)}m {int(eta % 60)}s" if eta < 3600 else f"{int(eta // 3600)}h {int((eta % 3600) // 60)}m"
                
                progress_msg = progress_tracker.create_progress_message(
                    file_name, send_mode.title(), percentage, current, total, speed, eta_str
                )
                cancel_keyboard = progress_tracker.create_cancel_keyboard(user_id)
                try:
                    await status_msg.edit(progress_msg, reply_markup=cancel_keyboard)
                except:
                    pass  # Ignore edit rate limits
        
        # Single download attempt with enhanced stability
        try:
            # Download with optimized settings for reliability
            async with connection_semaphore:
                # Use direct download without temp extension to avoid .temp file issues
                downloaded_file = await asyncio.wait_for(
                    client.download_media(
                        message, 
                        file_name=temp_filename,
                        progress=stable_progress_callback
                    ),
                    timeout=download_timeout
                )
            
            # Enhanced download verification with better error handling
            if not downloaded_file or not os.path.exists(downloaded_file):
                error_msg = await status_msg.edit("❌ **Download failed - file not created**\nPlease try sending the file again.")
                logger.error(f"Download failed. File not found at {downloaded_file}")
                # Auto-delete error message after 5 seconds
                asyncio.create_task(delete_message_after_delay(error_msg, 5))
                return
            
            # Comprehensive download verification with increased tolerance for large files
            actual_size = os.path.getsize(downloaded_file)
            expected_size = file_info.file_size
            size_tolerance = expected_size * 0.1  # 10% leeway for Telegram metadata inconsistencies
            
            # Check if download is reasonably complete
            if abs(actual_size - expected_size) <= size_tolerance:
                logger.info(f"Download successful: {downloaded_file} ({actual_size}/{expected_size} bytes)")
                # Rename to expected filename using sanitized name
                final_path = f"temp/processed_{timestamp}_{safe_file_name}"
                if os.path.exists(final_path):
                    os.remove(final_path)
                os.rename(downloaded_file, final_path)
                downloaded_file = final_path
            else:
                # File size mismatch beyond tolerance
                completion_percentage = (actual_size / expected_size) * 100
                logger.error(f"Size mismatch beyond tolerance: {completion_percentage:.1f}% ({actual_size}/{expected_size} bytes)")
                try:
                    os.remove(downloaded_file)
                except:
                    pass
                error_msg = await status_msg.edit(f"❌ **Download incomplete: {completion_percentage:.1f}%**\n**Expected:** {expected_size} bytes\n**Received:** {actual_size} bytes\n**File size difference too large. Please try again.**")
                # Auto-delete error message after 5 seconds
                asyncio.create_task(delete_message_after_delay(error_msg, 5))
                return
                        
        except asyncio.CancelledError:
            logger.info(f"Download cancelled by user {user_id}")
            cancel_msg = await status_msg.edit("❌ **Download cancelled by user**")
            # Auto-delete cancelled message after 5 seconds
            asyncio.create_task(delete_message_after_delay(cancel_msg, 5))
            # Clean up partial download and any temp files
            try:
                # Clean up the main download file
                if 'temp_filename' in locals() and temp_filename and os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    logger.info(f"Cleaned up cancelled download: {temp_filename}")
                
                # Clean up any .temp files in the temp directory
                temp_dir = Path("temp")
                for temp_file in temp_dir.glob("*.temp"):
                    try:
                        temp_file.unlink()
                        logger.info(f"Cleaned up temp file: {temp_file}")
                    except:
                        pass
                
                # Clean up any partial files matching the pattern
                for partial_file in temp_dir.glob(f"downloading_{timestamp}_*"):
                    try:
                        partial_file.unlink()
                        logger.info(f"Cleaned up partial file: {partial_file}")
                    except:
                        pass
                        
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")
            processor.clear_cancel_flag(user_id)
            return
            
        except asyncio.TimeoutError:
            logger.error(f"Download timeout after {download_timeout//60}m")
            timeout_msg = await status_msg.edit(f"⏰ **Download timeout after {download_timeout//60} minutes**\n**File:** {file_name}\n**The file is too large or connection is slow. Please try a smaller file or check your connection.**")
            # Auto-delete timeout message after 5 seconds
            asyncio.create_task(delete_message_after_delay(timeout_msg, 5))
            # Clean up partial download and any temp files
            try:
                if 'temp_filename' in locals() and temp_filename and os.path.exists(temp_filename):
                    os.remove(temp_filename)
                temp_dir = Path("temp")
                for temp_file in temp_dir.glob("*.temp"):
                    try:
                        temp_file.unlink()
                    except:
                        pass
                for partial_file in temp_dir.glob(f"downloading_{timestamp}_*"):
                    try:
                        partial_file.unlink()
                    except:
                        pass
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")
            return
            
        except Exception as e:
            logger.error(f"Download error: {e}")
            error_msg = str(e)[:150] + "..." if len(str(e)) > 150 else str(e)
            error_message = await status_msg.edit(f"❌ **Download failed:**\n`{error_msg}`\n**Please try sending the file again.**")
            # Auto-delete error message after 5 seconds
            asyncio.create_task(delete_message_after_delay(error_message, 5))
            # Clean up partial download and any temp files
            try:
                if 'temp_filename' in locals() and temp_filename and os.path.exists(temp_filename):
                    os.remove(temp_filename)
                temp_dir = Path("temp")
                for temp_file in temp_dir.glob("*.temp"):
                    try:
                        temp_file.unlink()
                    except:
                        pass
                for partial_file in temp_dir.glob(f"downloading_{timestamp}_*"):
                    try:
                        partial_file.unlink()
                    except:
                        pass
            except Exception as cleanup_error:
                logger.warning(f"Cleanup error: {cleanup_error}")
            return
        
        # Validate final download success
        if not downloaded_file or not os.path.exists(downloaded_file):
            await status_msg.edit("❌ **Download validation failed**\n**Please try sending the file again.**")
            return
        
        # Check for cancellation before processing
        if processor.is_cancelled(user_id):
            cancel_msg = await status_msg.edit("❌ **Cancelled by user**")
            # Auto-delete cancelled message after 5 seconds
            asyncio.create_task(delete_message_after_delay(cancel_msg, 5))
            processor.clear_cancel_flag(user_id)
            return
            
        # Extract metadata and apply smart naming
        await status_msg.edit("🔍 **Analyzing file and extracting metadata...**")
        caption = message.caption or ""
        metadata = FileProcessor.extract_metadata(file_name, caption)
        
        # Use custom filename format
        filename_format = user_prefs['filename_format']
        new_filename = FileProcessor.apply_template(filename_format, metadata)
        
        # Ensure filename has proper extension
        if not new_filename.endswith(('.mkv', '.mp4', '.avi', '.mov', '.webm', '.flv', '.wmv')):
            original_ext = Path(safe_file_name).suffix
            if original_ext:
                new_filename = new_filename.rsplit('.', 1)[0] + original_ext
            else:
                new_filename = new_filename + '.mkv'
        
        output_path = f"temp/{new_filename}"
        
        # Check for cancellation before FFmpeg processing
        if processor.is_cancelled(user_id):
            await status_msg.edit("❌ **Cancelled by user**")
            processor.clear_cancel_flag(user_id)
            return
            
        # Process with FFmpeg
        await status_msg.edit("⚙️ **Processing with FFmpeg...**")
        
        # Copy original file as base
        try:
            shutil.copy2(downloaded_file, output_path)
            logger.info(f"File copied from {downloaded_file} to {output_path}")
        except Exception as e:
            logger.error(f"File copy error: {e}")
            await status_msg.edit(f"❌ **File copy failed: {str(e)}**")
            return
        
        # Add intro if configured
        if user_prefs.get('intro'):
            await status_msg.edit("🎬 **Adding intro video...**")
            intro_path = user_prefs['intro'].get('path')
            if intro_path and os.path.exists(intro_path):
                temp_output = f"temp/with_intro_{int(time.time())}.{Path(new_filename).suffix[1:]}"
                success = await FFmpegProcessor.merge_intro_with_video(
                    intro_path, output_path, temp_output
                )
                if success:
                    os.replace(temp_output, output_path)
        
        # Apply enhanced metadata with Alpha Zenin branding
        await status_msg.edit("📝 **Applying metadata...**")
        
        # Use the new method to replace @Anime_Unity with custom metadata
        temp_output = f"temp/with_metadata_{timestamp}.{Path(new_filename).suffix[1:]}"
        audio_channel = user_prefs['metadata'].get('audio_channel', '@Animaxclan')
        success = await FFmpegProcessor.replace_anime_unity_metadata(
            output_path, temp_output, user_prefs['metadata']['title'], audio_channel
        )
        if success:
            os.replace(temp_output, output_path)

        # Apply watermark if enabled
        watermark_settings = user_prefs.get('watermark', {'enabled': False})
        if watermark_settings.get('enabled', False):
            await status_msg.edit("🎨 **Adding watermark...**")
            temp_watermark = f"temp/with_watermark_{timestamp}.{Path(new_filename).suffix[1:]}"
            watermark_text = watermark_settings.get('text', 'Encoded by @Animaxclan')
            success = await FFmpegProcessor.add_watermark(
                output_path, temp_watermark, watermark_text
            )
            if success:
                os.replace(temp_watermark, output_path)
        
        # Upload processed file with speed optimization
        await status_msg.edit("📤 **Uploading processed file (enhanced speed)...**")
        
        # Get thumbnail
        thumbnail_path = None
        if user_prefs.get('thumbnail'):
            thumbnail_path = user_prefs['thumbnail'].get('path')
        
        # Calculate processing time and add to statistics
        processing_time = time.time() - start_time
        progress_tracker.add_file_stats(user_id, processing_time, file_info.file_size)
        
        # Create enhanced caption with just the filename in bold
        caption = f"**{new_filename}**"
        
        # Verify output file exists before upload
        if not os.path.exists(output_path):
            await status_msg.edit("❌ **Processed file not found. Processing failed.**")
            return
        
        # Check for sorting only if this is the last file being processed
        queue_count = processor.get_queue_position(user_id)
        processing_count = processor.processing.get(user_id, 0)
        
        sort_settings = user_settings.get_sort_settings(user_id)
        if sort_settings.get('enabled', False) and queue_count == 0 and processing_count <= 1:
            sort_method = sort_settings.get('method', 'resolution')
            await status_msg.edit(f"🗂️ **All files processed! Now sorting by {sort_method}...**")
            # Add a small delay to show the message
            await asyncio.sleep(1)
            
        # Send based on user preference with upload optimization
        await status_msg.edit("📤 **Uploading processed file (optimized speed)...**")
        
        try:
            # Simple upload progress without async tasks to prevent hanging
            upload_last_update = [0.0]
            async def upload_progress(current, total):
                current_time = time.time()
                if current_time - upload_last_update[0] >= 3.0 or current == total:
                    upload_last_update[0] = current_time
                    percentage = (current / total) * 100 if total > 0 else 0
                    
                    # Calculate progress details
                    downloaded_mb = current / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    
                    # Calculate speed
                    time_elapsed = current_time - start_time
                    speed_mb = (current / (1024 * 1024)) / time_elapsed if time_elapsed > 0 else 0
                    
                    # Calculate ETA
                    if speed_mb > 0 and current < total:
                        remaining_mb = (total - current) / (1024 * 1024)
                        eta_seconds = remaining_mb / speed_mb
                        eta = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
                    else:
                        eta = "Calculating"
                    
                    # Create progress bar
                    filled_blocks = int((percentage / 100) * 10)
                    empty_blocks = 10 - filled_blocks
                    progress_bar = "▰" * filled_blocks + "▱" * empty_blocks
                    
                    # Format time
                    formatted_time = time.strftime("%H:%M:%S", time.localtime(current_time))
                    
                    try:
                        upload_msg = f"""┌─╼**ᴜᴩʟᴏᴀᴅɪɴɢ:** {new_filename} [{send_mode}]
├─╼**ᴛɪᴍᴇ:** {formatted_time}
├─╼**ᴩʀᴏɢʀᴇꜱꜱ:** {progress_bar} {percentage:.1f}%
├─╼**ꜱɪᴢᴇ:** {downloaded_mb:.1f} MB of {total_mb:.1f} MB
├─╼**ꜱᴩᴇᴇᴅ:** {speed_mb:.1f} MB/s
└─╼**ᴇᴛᴀ:** {eta}"""
                        await status_msg.edit(upload_msg)
                    except:
                        pass
            
            if send_mode == 'video':
                sent_message = await client.send_video(
                    chat_id=message.chat.id,
                    video=output_path,
                    caption=caption,
                    thumb=thumbnail_path,
                    reply_to_message_id=message.id,
                    progress=upload_progress
                )
            else:
                sent_message = await client.send_document(
                    chat_id=message.chat.id,
                    document=output_path,
                    caption=caption,
                    thumb=thumbnail_path,
                    reply_to_message_id=message.id,
                    progress=upload_progress
                )
        except Exception as upload_error:
            logger.error(f"Upload error: {upload_error}")
            await status_msg.edit(f"❌ **Upload failed: {str(upload_error)}**")
            return
        
        # Add file to sorting queue with message object for dump channel
        metadata = FileProcessor.extract_metadata(file_name, caption)
        file_info = {
            'message': sent_message,  # Use the sent message for forwarding
            'filename': new_filename,
            'caption': caption,
            'episode': metadata.get('episode', 'unknown'),
            'resolution': metadata.get('resolution', 'unknown'),
            'path': None,  # Don't use file path for dump channel
            'thumbnail': thumbnail_path,
            'send_mode': send_mode,
            'user_id': user_id
        }
        file_sorter.add_file_for_sorting(user_id, file_info)
        
        # Clean up after successful processing
        await status_msg.delete()
        
        # Enhanced cleanup of temporary files immediately
        cleanup_files = [downloaded_file, output_path]
        
        # Add any temp watermark files and other patterns
        temp_dir = Path("temp")
        for pattern in ["with_watermark_*", "with_metadata_*", "with_intro_*", "downloading_*", "processed_*"]:
            cleanup_files.extend(temp_dir.glob(pattern))
        
        for temp_file in cleanup_files:
            try:
                temp_path = Path(temp_file) if isinstance(temp_file, str) else temp_file
                if temp_path.exists():
                    temp_path.unlink()
                    logger.debug(f"Cleaned up: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {temp_file}: {e}")
        
        # Clean up any .temp files that might be left
        for temp_file in temp_dir.glob("*.temp"):
            try:
                temp_file.unlink()
                logger.debug(f"Cleaned up temp file: {temp_file}")
            except:
                pass
        
        # Automatic cleanup - remove old temporary files after each successful processing
        async def cleanup_temp_files():
            """Remove old temporary files from the temp directory."""
            try:
                temp_dir = Path("temp")
                current_time = time.time()
                for temp_file in temp_dir.glob("*"):
                    if temp_file.is_file():
                        file_age = current_time - temp_file.stat().st_mtime
                        if file_age > 3600:  # Remove files older than 1 hour
                            try:
                                temp_file.unlink()
                                logger.debug(f"Cleaned up old temp file: {temp_file}")
                            except Exception as e:
                                logger.warning(f"Failed to remove temp file {temp_file}: {e}")
            except Exception as e:
                logger.error(f"Error during temp file cleanup: {e}")

        await cleanup_temp_files()
        
        # Check for sorting and dump channel forwarding after all files are processed
        await handle_sorting_and_forwarding(client, message, user_id)
        
        # Check if this was the last file in queue and send statistics
        await check_and_send_statistics(client, message, user_id)
                
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        await status_msg.edit(f"❌ **Error processing file:**\n`{str(e)}`")
        
        # Emergency cleanup on error
        try:
            cleanup_files = []
            # Use locals() to safely check for variables
            local_vars = locals()
            if 'downloaded_file' in local_vars and downloaded_file:
                cleanup_files.append(downloaded_file)
            if 'output_path' in local_vars and output_path:
                cleanup_files.append(output_path)
            temp_dir = Path("temp")
            cleanup_files.extend(temp_dir.glob("with_watermark_*.mkv"))
            cleanup_files.extend(temp_dir.glob("with_metadata_*.mkv"))
            cleanup_files.extend(temp_dir.glob("with_intro_*.mkv"))
            
            for temp_file in cleanup_files:
                try:
                    temp_path = Path(temp_file) if isinstance(temp_file, str) else temp_file
                    if temp_path.exists():
                        temp_path.unlink()
                        logger.debug(f"Emergency cleanup: {temp_path}")
                except:
                    pass
        except:
            pass

async def handle_sorting_and_forwarding(client, message, user_id):
    """Handle file sorting and dump channel forwarding when all files are processed"""
    # Check if sorting should be triggered
    if not file_sorter.should_trigger_sort(user_id):
        return
    
    sort_settings = user_settings.get_sort_settings(user_id)
    dumb_channel = user_settings.get_dumb_channel(user_id)
    
    # Get files to sort
    if user_id not in file_sorter.sort_queue or not file_sorter.sort_queue[user_id]:
        return
    
    files_to_sort = file_sorter.sort_queue[user_id].copy()
    
    # Show sorting message if enabled
    if sort_settings.get('enabled', False) and len(files_to_sort) > 1:
        sort_method = sort_settings.get('method', 'resolution')
        sort_msg = await message.reply(f"🗂️ **Files are being sorted by {sort_method}...**")
        
        # Sort the files
        sorted_files = file_sorter.sort_files(user_id, sort_method)
        
        await asyncio.sleep(1)  # Brief delay to show sorting message
        await sort_msg.delete()
        
        # Show sorted order
        sorted_list = "\n".join([f"• {f['filename']}" for f in sorted_files])
        sort_complete_msg = await message.reply(f"✅ **Files sorted by {sort_method}!**\n\n**Order:**\n{sorted_list}")
        # Auto-delete after 8 seconds
        asyncio.create_task(delete_message_after_delay(sort_complete_msg, 8))
    else:
        sorted_files = files_to_sort
    
    # Send to dump channel if configured
    if dumb_channel and sorted_files:
        try:
            forward_msg = await message.reply("📤 **Files are being sent to dump channel...**")
            
            success_count = 0
            for file_info in sorted_files:
                try:
                    # Copy message to dump channel (no forward tag)
                    sent_message = file_info['message']
                    if sent_message:
                        await sent_message.copy(dumb_channel)
                        success_count += 1
                        await asyncio.sleep(0.5)  # Small delay between sends
                except Exception as send_error:
                    logger.error(f"Failed to copy file to dump channel: {send_error}")
            
            await forward_msg.delete()
            
            if success_count > 0:
                success_msg = await message.reply(f"✅ **{success_count} file(s) sent to dump channel successfully!**\n📤 Channel: `{dumb_channel}`")
                # Auto-delete success message after 10 seconds
                asyncio.create_task(delete_message_after_delay(success_msg, 10))
            
        except Exception as e:
            logger.error(f"Failed to send to dump channel {dumb_channel}: {e}")
            error_msg = str(e)
            if "Peer id invalid" in error_msg or "PEER_ID_INVALID" in error_msg:
                error_message = await message.reply(f"❌ **Invalid channel ID or bot not added to channel**\n\n**Channel ID:** `{dumb_channel}`\n\n**Please ensure:**\n• Bot is added to the channel\n• Bot has admin permissions\n• Channel ID is correct")
            elif "CHAT_ADMIN_REQUIRED" in error_msg:
                error_message = await message.reply(f"❌ **Bot needs admin permissions in channel:** `{dumb_channel}`\n\n**Steps:**\n1. Go to channel settings\n2. Add bot as administrator\n3. Enable 'Post Messages' permission")
            else:
                error_message = await message.reply(f"❌ **Failed to send to dump channel:**\n`{error_msg}`\n\n**Channel:** `{dumb_channel}`")
            # Auto-delete error message after 8 seconds
            asyncio.create_task(delete_message_after_delay(error_message, 8))
    
    # Clear the sort queue after processing
    file_sorter.clear_user_queue(user_id)

async def check_and_send_statistics(client, message, user_id):
    """Check if all files are processed and send statistics summary."""
    # Check if user has any files in queue or being processed
    queue_count = processor.get_queue_position(user_id)
    processing_count = processor.processing.get(user_id, 0)
    
    # If no files in queue and processing is done, send statistics
    if queue_count == 0 and processing_count <= 1:  # 1 because current file is still counted
        stats = progress_tracker.get_stats_summary(user_id)
        if stats and stats['total_files'] > 0:
            stats_message = f"""
📊 **Processing Complete!** 📊

**Summary:**
• **Total Files:** {stats['total_files']}
• **Total Time:** {int(stats['total_time'])}s ({int(stats['total_time'] // 60)}m {int(stats['total_time'] % 60)}s)
• **Average Time per File:** {stats['avg_time_per_file']:.1f}s
• **Average Speed:** {stats['avg_speed_mb']:.2f} MB/s
• **Total Size Processed:** {stats['total_size_mb']:.1f} MB

All files have been successfully renamed with **Alpha Zenin** metadata! ✨
"""
            await message.reply(stats_message)
            # Reset statistics for next session
            progress_tracker.reset_user_stats(user_id)

@app.on_callback_query()
async def handle_callbacks(client, callback_query):
    """Enhanced callback handler for interactive features"""
    if not callback_query.from_user:
        logger.warning("Received callback query without from_user attribute")
        return
    
    data = callback_query.data
    user_id = callback_query.from_user.id
    
    if data == "settings":
        await settings_command(client, callback_query.message)
        
    elif data == "status":
        await status_command(client, callback_query.message)
        
    elif data == "help":
        await help_command(client, callback_query.message)
        
    # Send mode callbacks
    elif data == "mode_video":
        user_settings.set_send_mode(user_id, 'video')
        await callback_query.answer("✅ Send mode set to Video")
        await callback_query.message.edit_text("✅ **Send mode updated to: Video**\n\nFiles will be sent as video messages.")
        
    elif data == "mode_file":
        user_settings.set_send_mode(user_id, 'file')
        await callback_query.answer("✅ Send mode set to File")
        await callback_query.message.edit_text("✅ **Send mode updated to: File**\n\nFiles will be sent as document attachments.")
        
    elif data == "set_mode":
        current_mode = user_settings.get_send_mode(user_id)
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📹 Video", callback_data="mode_video")],
            [InlineKeyboardButton("📄 File", callback_data="mode_file")]
        ])
        await callback_query.message.edit_text(f"🔧 **Current Mode:** {current_mode.title()}\n\nChoose your preferred send mode:", reply_markup=keyboard)
        
    elif data == "set_format":
        current_format = user_settings.get_filename_format(user_id)
        await callback_query.answer("Use /file command to set format")
        await callback_query.message.reply(f"🎯 **Current Format:**\n`{current_format}`\n\n**Usage:** `/file [S-{{season}}-Ep-{{episode}}] Title [{{resolution}}][{{audio}}]@Channel.mkv`")
        
    elif data == "change_title":
        current_meta = user_settings.get_metadata(user_id)
        await callback_query.answer("Use /setmeta command")
        await callback_query.message.reply(f"🏷️ **Current Title:** {current_meta['title']}\n\n**Usage:** `/setmeta Your Custom Title`")
        
    elif data == "change_audio_lang":
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🇯🇵 Japanese", callback_data="audio_japanese"),
             InlineKeyboardButton("🇺🇸 English", callback_data="audio_english")],
            [InlineKeyboardButton("🇮🇳 Hindi", callback_data="audio_hindi"),
             InlineKeyboardButton("🌐 Multi", callback_data="audio_multi")]
        ])
        await callback_query.message.edit_text("🎵 **Choose Audio Language:**", reply_markup=keyboard)
        
    elif data == "change_sub_lang":
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🇯🇵 Japanese", callback_data="sub_japanese"),
             InlineKeyboardButton("🇺🇸 English", callback_data="sub_english")],
            [InlineKeyboardButton("🇮🇳 Hindi", callback_data="sub_hindi"),
             InlineKeyboardButton("❌ None", callback_data="sub_none")]
        ])
        await callback_query.message.edit_text("📝 **Choose Subtitle Language:**", reply_markup=keyboard)
        
    # Audio language callbacks
    elif data.startswith("audio_"):
        lang = data.split("_")[1].title()
        user_settings.set_metadata_language(user_id, 'audio_language', lang)
        await callback_query.answer(f"✅ Audio language set to {lang}")
        await callback_query.message.edit_text(f"✅ **Audio language updated to: {lang}**")
        
    # Subtitle language callbacks
    elif data.startswith("sub_"):
        lang = data.split("_")[1].title()
        if lang == "None":
            lang = "None"
        user_settings.set_metadata_language(user_id, 'subtitle_language', lang)
        await callback_query.answer(f"✅ Subtitle language set to {lang}")
        await callback_query.message.edit_text(f"✅ **Subtitle language updated to: {lang}**")
        
    # Metadata menu
    elif data == "metadata_menu":
        current_meta = user_settings.get_metadata(user_id)
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("🏷️ Change Title", callback_data="change_title"),
             InlineKeyboardButton("🎵 Audio Lang", callback_data="change_audio_lang")],
            [InlineKeyboardButton("📝 Subtitle Lang", callback_data="change_sub_lang")]
        ])
        await callback_query.message.edit_text(f"🏷️ **Metadata Settings**\n\n**Current Metadata:**\n• Title: {current_meta['title']}\n• Audio: {current_meta['audio_language']}\n• Subtitles: {current_meta['subtitle_language']}\n\nChoose what to modify:", reply_markup=keyboard)
        
    # Watermark menu
    elif data == "watermark_menu":
        current_watermark = user_settings.get_watermark(user_id)
        status = "✅ Enabled" if current_watermark.get('enabled') else "❌ Disabled"
        toggle_text = "Disable" if current_watermark.get('enabled') else "Enable"
        
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(f"🔄 {toggle_text}", callback_data="toggle_watermark"),
             InlineKeyboardButton("✏️ Edit Text", callback_data="edit_watermark")]
        ])
        
        await callback_query.message.edit_text(f"🎨 **Watermark Settings**\n\n**Status:** {status}\n**Text:** `{current_watermark.get('text', 'Encoded by @Animaxclan')}`\n**Duration:** 3 seconds at video start\n**Position:** Bottom center\n\nChoose action:", reply_markup=keyboard)
        
    # Watermark callbacks
    elif data == "toggle_watermark":
        new_status = user_settings.toggle_watermark(user_id)
        status_text = "enabled" if new_status else "disabled"
        await callback_query.answer(f"✅ Watermark {status_text}")
        await callback_query.message.edit_text(f"✅ **Watermark {status_text}!**\n\nUse /watermark command to set custom text.")
        
    elif data == "edit_watermark":
        current_watermark = user_settings.get_watermark(user_id)
        await callback_query.answer("Use /watermark command")
        await callback_query.message.reply(f"🎨 **Current Watermark Text:**\n`{current_watermark.get('text', 'Encoded by @Animaxclan')}`\n\n**Usage:** `/watermark Your Custom Text Here`")
    
    # Sort callbacks
    elif data == "sort_resolution":
        user_settings.set_sort_method(user_id, 'resolution')
        await callback_query.answer("✅ Sort method set to Resolution")
        await callback_query.message.edit_text("✅ **Sorting method updated to Resolution!**\n\nFiles will be sorted by quality (480p, 720p, 1080p, 4K)")
        
    elif data == "sort_episodes":
        user_settings.set_sort_method(user_id, 'episodes')
        await callback_query.answer("✅ Sort method set to Episodes")
        await callback_query.message.edit_text("✅ **Sorting method updated to Episodes!**\n\nFiles will be sorted by episode number")
        
    elif data == "sort_toggle":
        new_status = user_settings.toggle_sorting(user_id)
        status_text = "enabled" if new_status else "disabled"
        await callback_query.answer(f"✅ Auto-sorting {status_text}")
        await callback_query.message.edit_text(f"✅ **Auto-sorting {status_text}!**\n\nUse /sort command to configure sorting options.")
        
    elif data == "set_thumbnail":
        await callback_query.answer("Send a photo to set as thumbnail")
        await callback_query.message.reply("🖼️ **Send a photo** to set it as your default thumbnail for all processed videos.")
        
    elif data == "set_intro":
        await callback_query.answer("Send a video to set as intro")
        await callback_query.message.reply("🎬 **Send a video** to set it as your intro. This will be added to the beginning of all processed videos.")
    
    # Cancel callback - only cancel single file, not all files
    elif data.startswith("cancel_"):
        user_id_to_cancel = int(data.split("_")[1])
        if callback_query.from_user.id == user_id_to_cancel:
            # Set cancel flag for current operation only
            processor.cancelled_users.add(user_id_to_cancel)
            await callback_query.answer("❌ Current file cancelled!")
            await callback_query.message.edit_text("❌ **Current file cancelled by user**")
        else:
            await callback_query.answer("❌ You can only cancel your own processes!")
    
    await callback_query.answer()

@app.on_message(filters.photo)
async def handle_thumbnail(client, message):
    """Enhanced thumbnail handler with new settings system"""
    if not message.from_user:
        logger.warning("Received photo without from_user attribute")
        return
    
    user_id = message.from_user.id
    
    try:
        # Download thumbnail
        thumbnail_path = f"thumbnail/thumb_{user_id}_{int(time.time())}.jpg"
        downloaded = await client.download_media(message, file_name=thumbnail_path)
        
        # Save to user settings
        user_settings.update_setting(user_id, 'thumbnail', {
            'file_id': message.photo.file_id,
            'path': downloaded
        })
        
        await message.reply("✅ **Thumbnail saved successfully!**\n\nThis thumbnail will be used for all your processed videos.")
        
    except Exception as e:
        logger.error(f"Error saving thumbnail: {e}")
        await message.reply(f"❌ **Error saving thumbnail:**\n`{str(e)}`")

@app.on_message(filters.video & ~filters.document)
async def handle_intro_video(client, message):
    """Handle intro video setting when user sends a video outside of file processing"""
    user_id = message.from_user.id
    
    # Check if this is meant to be an intro video (not for processing)
    if message.caption and ('intro' in message.caption.lower() or 'opening' in message.caption.lower()):
        try:
            # Download intro video
            intro_path = f"intro/intro_{user_id}_{int(time.time())}.mp4"
            downloaded = await client.download_media(message, file_name=intro_path)
            
            # Save to user settings
            user_settings.update_setting(user_id, 'intro', {
                'file_id': message.video.file_id,
                'path': downloaded
            })
            
            await message.reply("✅ **Intro video saved successfully!**\n\nThis intro will be added to the beginning of all your processed videos.")
            
        except Exception as e:
            logger.error(f"Error saving intro: {e}")
            await message.reply(f"❌ **Error saving intro:**\n`{str(e)}`")
    else:
        # This is a regular video for processing
        await handle_media_file(client, message)

@app.on_message(filters.command("status"))
async def status_command(client, message):
    """Show bot status and queue information"""
    user_id = message.from_user.id
    
    # Check FFmpeg
    ffmpeg_status = "✅ Available" if await FFmpegProcessor.check_ffmpeg() else "❌ Not Available"
    
    # Queue info
    queue_position = processor.get_queue_position(user_id)
    processing_count = processor.processing.get(user_id, 0)
    
    status_text = f"""
📊 **Bot Status** 📊

**System:**
• FFmpeg: {ffmpeg_status}
• Max Concurrent: {processor.max_concurrent}
• Max File Size: 3GB (Enhanced)

**Your Queue:**
• Currently Processing: {processing_count}
• Queued Files: {queue_position}

**Performance Optimizations:**
• Ultra-fast encoding presets
• Optimized progress tracking
• Auto queue cleanup

**Supported Formats:**
• Video: {', '.join(SUPPORTED_VIDEO_FORMATS)}
• Audio: {', '.join(SUPPORTED_AUDIO_FORMATS)}

**Limits:**
• Max File Size: {MAX_FILE_SIZE // (1024*1024*1024)}GB
• Max Thumbnail: {MAX_THUMBNAIL_SIZE // (1024*1024)}MB
"""
    
    await message.reply(status_text)

@app.on_message(filters.command("sort"))
async def sort_command(client, message):
    """Handle sorting commands"""
    user_id = message.from_user.id
    args = message.text.split()
    
    if len(args) == 1:
        # Show sorting options
        current_sort = user_settings.get_sort_settings(user_id)
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("📺 Resolution", callback_data="sort_resolution"),
             InlineKeyboardButton("🔢 Episodes", callback_data="sort_episodes")],
            [InlineKeyboardButton("🔄 Toggle On/Off", callback_data="sort_toggle")]
        ])
        
        status = "✅ Enabled" if current_sort['enabled'] else "❌ Disabled"
        method = current_sort['method'].title()
        
        await message.reply(f"🗂️ **Auto-Sorting Settings**\n\n**Status:** {status}\n**Method:** {method}\n\nChoose how you'd like to sort your files:", reply_markup=keyboard)
        
    elif len(args) == 2:
        command = args[1].lower()
        if command == "on":
            user_settings.toggle_sorting(user_id, True)
            await message.reply("✅ **Auto-sorting enabled!**\n\nFiles will be sorted automatically after all processing is complete.")
        elif command == "off":
            user_settings.toggle_sorting(user_id, False)
            await message.reply("❌ **Auto-sorting disabled!**\n\nFiles will not be sorted automatically.")
        else:
            await message.reply("❌ **Invalid command!**\n\n**Usage:**\n• `/sort` - Choose sorting method\n• `/sort on` - Enable auto-sorting\n• `/sort off` - Disable auto-sorting")

@app.on_message(filters.command("set_dumb"))
async def set_dumb_channel(client, message):
    """Set dumb channel for file forwarding"""
    user_id = message.from_user.id
    args = message.text.split()
    
    if len(args) < 2:
        current_channel = user_settings.get_dumb_channel(user_id)
        if current_channel:
            await message.reply(f"📤 **Current Dumb Channel:** `{current_channel}`\n\n**Usage:** `/set_dumb <channel_id>`\n\n**Note:** Bot must be admin in the channel!")
        else:
            await message.reply("📤 **No dumb channel set**\n\n**Usage:** `/set_dumb <channel_id>`\n\n**Examples:**\n• `/set_dumb @mychannel`\n• `/set_dumb -1001234567890`\n\n**Note:** Bot must be admin in the channel!")
        return
    
    channel_id = args[1]
    
    try:
        # Normalize channel ID format
        if channel_id.startswith('@'):
            test_channel_id = channel_id
        elif channel_id.startswith('-100'):
            test_channel_id = int(channel_id)
        else:
            # Try as integer first
            try:
                test_channel_id = int(channel_id)
            except ValueError:
                test_channel_id = channel_id
        
        # Test if bot can send to the channel
        test_msg = await client.send_message(test_channel_id, "🤖 **Bot Test** - Channel configured successfully!")
        await asyncio.sleep(2)  # Wait 2 seconds before deleting
        await test_msg.delete()
        
        # Save channel if successful
        if user_settings.set_dumb_channel(user_id, test_channel_id):
            await message.reply(f"✅ **Dump channel set successfully!**\n\n**Channel:** `{channel_id}`\n\n📤 All renamed files will be automatically sent to this channel after processing.")
        else:
            await message.reply("❌ **Invalid channel ID format!**")
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Dump channel error for {channel_id}: {e}")
        if "Peer id invalid" in error_msg or "PEER_ID_INVALID" in error_msg:
            await message.reply(f"❌ **Bot cannot access channel: `{channel_id}`**\n\n**Required steps:**\n1. Add bot to the channel as admin\n2. Grant 'Post Messages' permission\n3. Ensure channel ID is correct\n4. For private channels, use full numeric ID (-100xxxxxxxxx)\n\n**Current Error:** Channel not accessible or bot lacks permissions")
        elif "CHAT_ADMIN_REQUIRED" in error_msg:
            await message.reply(f"❌ **Bot needs admin permissions in channel: `{channel_id}`**\n\n**Steps:**\n1. Go to channel settings\n2. Add bot as administrator\n3. Enable 'Post Messages' permission\n4. Try command again")
        else:
            await message.reply(f"❌ **Cannot access channel!**\n\n**Channel:** `{channel_id}`\n**Error:** {error_msg}\n\n**Troubleshooting:**\n• Verify bot is admin in channel\n• Check channel ID format\n• Ensure bot has post permissions")

# Admin commands
@app.on_message(filters.command("admin") & filters.user(ADMIN_ID))
async def admin_panel(client, message):
    """Admin panel for bot management"""
    admin_text = """
👑 **Admin Panel** 👑

**System Status:**
• Active Users: Loading...
• Total Files Processed: Loading...
• Queue Status: Loading...

**Commands:**
/broadcast - Send message to all users
/stats - Detailed statistics
/cleanup - Clean temporary files
"""
    
    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("📊 Stats", callback_data="admin_stats")],
        [InlineKeyboardButton("🧹 Cleanup", callback_data="admin_cleanup")],
        [InlineKeyboardButton("📢 Broadcast", callback_data="admin_broadcast")]
    ])
    
    await message.reply(admin_text, reply_markup=keyboard)

@app.on_message(filters.command("setmeta"))
async def set_metadata_command(client, message):
    """Set custom metadata title and audio channel"""
    user_id = message.from_user.id
    args = message.text.split(maxsplit=1)
    
    if len(args) < 2:
        current_metadata = user_settings.get_metadata(user_id)
        current_title = current_metadata.get('title', 'Alpha Zenin')
        current_channel = current_metadata.get('audio_channel', '@Animaxclan')
        
        await message.reply(f"""🏷️ **Current Metadata Settings**

**Title:** {current_title}
**Audio Channel:** {current_channel}

**Usage:**
• `/setmeta <title>` - Set media title
• `/setchannel <@channel>` - Set audio channel branding

**Examples:**
• `/setmeta Alpha Zenin`
• `/setchannel @YourChannel`""")
        return
    
    new_title = args[1].strip()
    user_settings.set_metadata_title(user_id, new_title)
    
    await message.reply(f"✅ **Metadata title set successfully!**\n\n**New Title:** {new_title}\n\nThis will be applied to all processed files.")

@app.on_message(filters.command("setchannel"))
async def set_channel_command(client, message):
    """Set custom audio channel branding"""
    user_id = message.from_user.id
    args = message.text.split(maxsplit=1)
    
    if len(args) < 2:
        current_channel = user_settings.get_audio_channel(user_id)
        await message.reply(f"""📢 **Current Audio Channel:** {current_channel}

**Usage:** `/setchannel <@channel_name>`

**Examples:**
• `/setchannel @YourChannel`
• `/setchannel @Animaxclan`

This will appear in audio track metadata as "By @YourChannel - Japanese".""")
        return
    
    new_channel = args[1].strip()
    # Don't automatically add @ symbol - use exactly what user provides
    user_settings.set_audio_channel(user_id, new_channel)
    
    await message.reply(f"✅ **Audio channel set successfully!**\n\n**New Channel:** {new_channel}\n\nAudio tracks will show: \"By {new_channel} - Japanese\"")
    if not message.from_user:
        return
    
    user_id = message.from_user.id
    
    # Cancel all user tasks
    await processor.cancel_user_tasks(user_id)
    
    # Clear sorting queue
    file_sorter.clear_user_queue(user_id)
    
    await message.reply("❌ **All processes cancelled!**\n\n🗑️ **Cleared:**\n• All queued files\n• Ongoing downloads\n• Active processing\n• Sorting queue\n\nYou can now send new files to process.")

# Error handler
@app.on_message(filters.text & ~filters.command(["start", "settings", "template", "help", "status", "admin", "sort", "set_dumb", "setmeta", "setchannel", "cleanup"]))
async def handle_text(client, message):
    """Handle unknown text messages"""
    await message.reply("""
❓ **Unknown Command**

Send me a video file to rename it, or use these commands:
• /start - Get started
• /settings - Configure preferences  
• /template - Set naming template
• /help - Show all commands
• /cleanup - Clean temporary files

Type /help for detailed information.
""")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("thumbnail", exist_ok=True)
    os.makedirs("intro", exist_ok=True)
    8
    # Check FFmpeg availability
    async def startup_check():
        if not await FFmpegProcessor.check_ffmpeg():
           logger.warning("FFmpeg not found! Media processing features will be limited.")
        else:
            logger.info("FFmpeg detected successfully")
    
    # Global error handler to prevent crashes
    @app.on_raw_update()
    async def handle_raw_update(client, update, users, chats):
        try:
            # Let the default handlers process the update
            pass
        except Exception as e:
            logger.error(f"Unhandled error in raw update: {e}")
    
    logger.info("Starting Professional Rename Bot...")
    try:
        app.run()
    except Exception as e:
        logger.error(f"Bot crashed with error: {e}")
        # Try to restart
        logger.info("Attempting to restart bot...")
        app.run() # This will run the bot again, hopefully with the issue resolved