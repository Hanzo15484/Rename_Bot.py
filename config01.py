import os

# Bot configuration
BOT_TOKEN = os.getenv("BOT_TOKEN", "7549459194:AAEUh5HSxS56cOqGSFnUkF6w80n820UYlGY")
API_ID = int(os.getenv("TELEGRAM_API_ID", "22471192"))
API_HASH = os.getenv("TELEGRAM_API_HASH", "135380e8d132eb94f6a3ef14b6b576e6")
ADMIN_ID = int(os.getenv("ADMIN_ID", "5373577888"))

# Database configuration
DATABASE_PATH = "bot_database.db"

# File size limits (in bytes) - Enhanced for large files
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB optimized limit
MAX_THUMBNAIL_SIZE = 10 * 1024 * 1024  # 10MB

# Optimized single-attempt download settings for large files
DOWNLOAD_CHUNK_SIZE = 1024 * 1024 * 4  # 4MB chunks for better stability
UPLOAD_CHUNK_SIZE = 1024 * 1024 * 4    # 4MB chunks for stable upload
PROGRESS_UPDATE_INTERVAL = 1024 * 1024 * 30  # Update every 30MB for better stability
DOWNLOAD_BASE_TIMEOUT = 2700  # 45 minutes base timeout for large files
DOWNLOAD_MAX_TIMEOUT = 7200  # 2 hours maximum timeout for 2GB files
SINGLE_ATTEMPT_DOWNLOAD = True  # No retry system - single reliable attempt
STABILITY_UPDATE_INTERVAL = 8  # Conservative progress updates for large files

# Large file optimization settings
LARGE_FILE_THRESHOLD = 500 * 1024 * 1024  # 500MB threshold for large file handling
CONNECTION_POOL_SIZE = 2  # Reduced for stability with large files
MEMORY_LIMIT_CHECK = True  # Enable memory checking for large files

# Supported file formats
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

# FFmpeg settings
FFMPEG_PATH = "ffmpeg"
FFPROBE_PATH = "ffprobe"

# Placeholder patterns for template system
PLACEHOLDER_PATTERNS = {
    'season': r'\{season\}',
    'episode': r'\{episode\}',
    'resolution': r'\{resolution\}',
    'audio': r'\{audio\}'
}

# Processing settings
MAX_CONCURRENT_PROCESSES = 2
TEMP_DIR = "temp"
THUMBNAIL_DIR = "thumbnail"
INTRO_DIR = "intro"

# Default user preferences
DEFAULT_USER_PREFERENCES = {
    'template': 'S-{season}-Ep-{episode}',
    'thumbnail': None,
    'intro': None,
    'mode': 'Auto Rename',
    'send_as': 'video',  # 'video' or 'file'
    'metadata': {
        'title': 'Alpha Zenin',
        'audio_language': 'Japanese',
        'subtitle_language': 'Japanese',
        'audio_channel': '@Animaxclan'  # Channel branding for audio track
    },
    'filename_format': '[S-{season}-Ep-{episode}] Chuahi Lips [{resolution}][{audio}]@Animaxclan.mkv',
    'audio_replacement': None,
    'subtitle_file': None,
    'watermark': {
        'enabled': False,
        'text': 'Encoded by @Animaxclan'
    },
    'sorting': {
        'enabled': False,
        'method': 'resolution'  # 'resolution' or 'episodes'
    },
    'dumb_channel': None
}

# Progress bar configuration
PROGRESS_BAR_LENGTH = 10
PROGRESS_FILLED = "▬"
PROGRESS_EMPTY = "▭"

# User settings file
USER_SETTINGS_FILE = "user_settings.json"

# Logging configuration
LOG_FILE = "professional_bot.log"
LOG_LEVEL = "INFO"

# Queue settings
QUEUE_TIMEOUT = 300  # 5 minutes
MAX_QUEUE_SIZE = 10  # Maximum files in queue per user

# Media processing settings
DEFAULT_THUMBNAIL_TIMESTAMP = "00:00:01"
DEFAULT_VIDEO_QUALITY = "720p"
DEFAULT_AUDIO_BITRATE = "128k"

# Metadata extraction patterns
SEASON_PATTERNS = [
    r'(?:season|s)[\s._-]*0*(\d+)',
    r'(?:^|\W)s(\d+)(?:\W|$)',
    r'(?:^|\W)(\d+)(?:st|nd|rd|th)?\s*season',
]

EPISODE_PATTERNS = [
    r'(?:episode|ep|e)[\s._-]*0*(\d+)',
    r'(?:^|\W)e(\d+)(?:\W|$)',
    r'(?:^|\W)(\d+)(?:st|nd|rd|th)?\s*episode',
    r'\b(\d{1,3})(?:\W|$)',
]

RESOLUTION_PATTERNS = [
    r'(\d{3,4}p)',
    r'(\d{3,4}x\d{3,4})',
    r'(4k|uhd|2160p)',
    r'(fhd|1080p)',
    r'(hd|720p)',
    r'(480p|sd)',
]

AUDIO_PATTERNS = [
    r'(dual[\s._-]*audio|dual)',
    r'(hindi[\s._-]*dubbed|hindi)',
    r'(english[\s._-]*dubbed|english)',
    r'(multi[\s._-]*audio|multi)',
    r'(japanese|jap)',
    r'(tamil|telugu|kannada)',
]

# ARIA2 / DC4 config (add to config01.py)
ARIA2C_PATH = "aria2c"                # path to aria2c binary
ARIA2C_EXTRA_ARGS = "-x 16 -s 32 -j 32 --file-allocation=none"  # default aria2 args

# DC4: set to True and provide your server upload URL if you have a DC4 / cloud mirror
DC4_ENABLED = False
DC4_UPLOAD_URL = "https://your-dc4.example/upload"   # HTTP endpoint that accepts file uploads and returns JSON { "url": "https://..." }
DC4_UPLOAD_FIELD = "file"     # field name the DC4 server expects (POST multipart)
DC4_API_KEY = ""              # optional API key if your DC4 needs auth — sent as header "Authorization: Bearer <key>"
