
import os
import math
import tempfile
import shutil
import streamlit as st
from pathlib import Path
from typing import List, Tuple, Optional
import subprocess
from datetime import datetime
import json
import cv2
import numpy as np

# --- Configuration ---
SUPPORTED_VIDEO_FORMATS = [
    ".mkv", ".mp4", ".mov", ".avi", ".flv", ".wmv", ".webm",
    ".MKV", ".MP4", ".MOV", ".AVI", ".FLV", ".WMV", ".WEBM"
]
# YouTube Shorts specifications
SHORTS_ASPECT_RATIO = "9:16"
SHORTS_RESOLUTION = (1080, 1920)  # width x height

# --- Helpers ---
def detect_faces_in_frame(frame):
    """Detect faces in a frame using OpenCV Haar Cascade"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces
    except Exception:
        return []


def get_smart_crop_position(video_path: str, sample_seconds: int = 10) -> str:
    """
    Analyze video to detect faces and determine best crop position
    Returns: 'center', 'left', 'right' based on face detection
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return "center"
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        sample_frames = min(sample_seconds * int(fps if fps else 0), total_frames)
        frame_interval = max(total_frames // sample_frames, 1) if sample_frames else 1
        face_positions = []
        for i in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            faces = detect_faces_in_frame(frame)
            for (x, y, w, h) in faces:
                face_center_x = x + w // 2
                face_positions.append(face_center_x)
        cap.release()
        if not face_positions:
            return "center"
        avg_face_x = sum(face_positions) / len(face_positions)
        if avg_face_x < frame_width * 0.33:
            return "left"
        elif avg_face_x > frame_width * 0.67:
            return "right"
        else:
            return "center"
    except Exception:
        return "center"


def ffprobe_duration(input_path: str) -> float:
    """Get video duration using ffprobe"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
        input_path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode().strip()
        return float(out)
    except Exception as e:
        st.error(f"Error reading video duration: {e}")
        return 0


def ffprobe_video_info(input_path: str) -> dict:
    """Get detailed video information"""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", input_path
    ]
    try:
        out = subprocess.check_output(cmd).decode()
        data = json.loads(out)
        video_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'video'), {})
        audio_stream = next((s for s in data.get('streams', []) if s['codec_type'] == 'audio'), {})
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        fps_val = video_stream.get('r_frame_rate', '0/1')
        try:
            fps = eval(fps_val) if '/' in str(fps_val) else float(fps_val)
        except Exception:
            fps = 0
        return {
            'duration': float(data.get('format', {}).get('duration', 0)),
            'size': int(data.get('format', {}).get('size', 0)),
            'bitrate': int(data.get('format', {}).get('bit_rate', 0)) if data.get('format', {}).get('bit_rate') else 0,
            'video_codec': video_stream.get('codec_name', 'N/A'),
            'width': width,
            'height': height,
            'resolution': f"{width}x{height}",
            'fps': fps,
            'audio_codec': audio_stream.get('codec_name', 'N/A'),
            'audio_sample_rate': audio_stream.get('sample_rate', 'N/A')
        }
    except Exception:
        return {}


def calculate_crop_for_shorts(source_width: int, source_height: int, crop_position: str = "center") -> dict:
    """Calculate crop parameters to convert video to 9:16 (YouTube Shorts format)"""
    target_ratio = 9 / 16
    source_ratio = source_width / source_height if source_height else 1
    if source_ratio > target_ratio:
        # Source is wider (landscape), crop width
        crop_height = source_height
        crop_width = int(crop_height * target_ratio)
        if crop_position == "left":
            origin_x = 0
        elif crop_position == "right":
            origin_x = source_width - crop_width
        else:
            origin_x = (source_width - crop_width) // 2
        origin_y = 0
    else:
        # Source is taller or already vertical, crop height
        crop_width = source_width
        crop_height = int(crop_width / target_ratio)
        if crop_position == "top":
            origin_y = 0
        elif crop_position == "bottom":
            origin_y = source_height - crop_height
        else:
            origin_y = (source_height - crop_height) // 2
        origin_x = 0
    return {
        'crop_width': crop_width,
        'crop_height': crop_height,
        'origin_x': origin_x,
        'origin_y': origin_y
    }


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes}:{secs:02d}"


def format_size(bytes_val: int) -> str:
    bytes_val = float(bytes_val)
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def check_ffmpeg_installed() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def ffmpeg_segment_crop_shorts(input_path: str, start: float, dur: float, output_path: str,
                               crop_params: dict, scale_to_1080p: bool = True,
                               quality_preset: str = "medium") -> None:
    crop_w = crop_params['crop_width']
    crop_h = crop_params['crop_height']
    origin_x = crop_params['origin_x']
    origin_y = crop_params['origin_y']
    filters = f"crop={crop_w}:{crop_h}:{origin_x}:{origin_y}"
    if scale_to_1080p:
        filters += f",scale={SHORTS_RESOLUTION[0]}:{SHORTS_RESOLUTION[1]}"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start}", "-i", input_path, "-t", f"{dur}",
        "-vf", filters,
        "-c:v", "libx264", "-preset", quality_preset, "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "1",
        output_path, "-y"
    ]
    subprocess.check_call(cmd)


def ffmpeg_segment_copy(input_path: str, start: float, dur: float, output_path: str) -> None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start}", "-i", input_path, "-t", f"{dur}",
        "-c", "copy", "-map", "0", "-avoid_negative_ts", "1",
        output_path, "-y"
    ]
    subprocess.check_call(cmd)


def save_uploaded_file(uploaded_file) -> str:
    """Saves uploaded file to a temp location and returns the path"""
    suffix = ''.join(Path(uploaded_file.name).suffixes) or ".mp4"
    tmp_dir = Path(tempfile.mkdtemp(prefix="stvid_"))
    tmp_path = tmp_dir / f"input{suffix}"
    with open(tmp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return str(tmp_path)


# --- UI ---
st.set_page_config(
    page_title="YouTube Shorts Splitter",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📱 YouTube Shorts Video Splitter (Cloud)")
st.markdown("**Split long videos into YouTube Shorts (9:16 vertical) with Smart Auto-Focus**")

# Check FFmpeg
if not check_ffmpeg_installed():
    st.error("❌ **FFmpeg is not installed!** This environment must have FFmpeg.")
    st.info("On Streamlit Cloud, add a file named `packages.txt` with a single line: `ffmpeg`. On Docker/Azure, install via apt.")
    st.stop()

# Session state
if 'input_video_path' not in st.session_state:
    st.session_state.input_video_path = ""
if 'work_dir' not in st.session_state:
    st.session_state.work_dir = tempfile.mkdtemp(prefix="shorts_out_")
if 'detected_crop_position' not in st.session_state:
    st.session_state.detected_crop_position = None

with st.sidebar:
    st.header("⚙️ Settings")
    st.subheader("🎬 Output Format")
    convert_to_shorts = st.checkbox("Convert to YouTube Shorts format (9:16)", value=True,
                                    help="Converts video to vertical 9:16 format (1080x1920)")
    opencv_available = True  # we install headless build
    if convert_to_shorts:
        st.info("📱 Videos will be converted to 1080x1920 (9:16 aspect ratio)")
        use_auto_focus = st.checkbox("🎯 Smart Auto-Focus (Face Detection)", value=True,
                                    help="Automatically detect faces and adjust crop position")
        if use_auto_focus:
            st.success("✅ Smart cropping enabled - will focus on detected faces")
            crop_position = "auto"
        else:
            crop_position = st.selectbox("Crop Position",
                                         options=["center", "left", "right", "top", "bottom"], index=0,
                                         help="Where to position the crop")
    else:
        crop_position = "center"
        use_auto_focus = False
        st.warning("⚠️ Videos will keep original aspect ratio")

    st.divider()
    st.subheader("⏱️ Clip Duration")
    duration_mode = st.radio("Duration Input", options=["Preset (Quick)", "Custom (Any Duration)"], index=0)
    if duration_mode == "Preset (Quick)":
        st.caption("📌 Popular durations for YouTube Shorts")
        preset_seconds = st.selectbox("Select duration", options=[15, 30, 45, 60, 90, 120, 180, 300], index=3,
                                      format_func=lambda x: f"{x} seconds ({x//60}:{x%60:02d})" if x < 60 else f"{x} seconds ({x//60} min {x%60} sec)",
                                      help="Quick presets")
        clip_len_seconds = preset_seconds
    else:
        st.caption("⏱️ Set any custom duration")
        col1, col2 = st.columns(2)
        with col1:
            custom_minutes = st.number_input("Minutes", min_value=0, max_value=999, value=1, step=1)
        with col2:
            custom_seconds = st.number_input("Seconds", min_value=0, max_value=59, value=0, step=5)
        clip_len_seconds = custom_minutes * 60 + custom_seconds
        if clip_len_seconds == 0:
            st.error("❌ Duration cannot be 0!")
            clip_len_seconds = 60
    st.info(f"⏱️ Clip duration: **{format_duration(clip_len_seconds)}**")
    if clip_len_seconds > 60:
        st.warning(f"⚠️ {clip_len_seconds}s clips won't be eligible for YouTube Shorts feed (max 60s)")
    else:
        st.success(f"✅ {clip_len_seconds}s clips are perfect for YouTube Shorts!")

    st.divider()
    with st.expander("🔧 Advanced Settings"):
        preview_duration = st.slider("Preview duration (seconds)", 5, 60, 10)
        show_detailed_info = st.checkbox("Show detailed video info", value=True)
        quality_preset = st.selectbox("Encoding Quality",
                                      options=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"], index=5,
                                      help="Medium = balanced quality/speed")

st.header("📁 Upload Video")
uploaded = st.file_uploader("Upload a video file", type=[ext.strip('.') for ext in SUPPORTED_VIDEO_FORMATS])

if uploaded:
    st.session_state.input_video_path = save_uploaded_file(uploaded)
    st.session_state.detected_crop_position = None
    st.success(f"Uploaded: **{uploaded.name}**")

video_path = st.session_state.input_video_path

st.divider()

# Video info display
if video_path and Path(video_path).exists():
    try:
        info = ffprobe_video_info(video_path)
        file_size = Path(video_path).stat().st_size
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 File Size", format_size(file_size))
        with col2:
            st.metric("📹 Resolution", info.get('resolution', 'N/A'))
        with col3:
            current_ratio = f"{info.get('width', 0)}:{info.get('height', 0)}"
            st.metric("📐 Current Ratio", current_ratio)
        with col4:
            if convert_to_shorts:
                st.metric("🎯 Target", "9:16 Shorts")
            else:
                st.metric("📝 Mode", "Original")
        if convert_to_shorts and use_auto_focus and st.session_state.detected_crop_position is None:
            with st.spinner("🎯 Detecting faces for smart cropping..."):
                detected_pos = get_smart_crop_position(video_path, sample_seconds=5)
                st.session_state.detected_crop_position = detected_pos
            st.success(f"✅ Smart crop position detected: **{detected_pos.upper()}**")
        elif st.session_state.detected_crop_position:
            st.info(f"🎯 Smart crop position: **{st.session_state.detected_crop_position.upper()}**")
    except Exception:
        pass

st.divider()

# Action Buttons
st.header("🚀 Actions")
col1, col2, col3 = st.columns(3)
with col1:
    preview_btn = st.button("🎥 Preview", use_container_width=True, disabled=not video_path,
                            help="Preview how Shorts will look")
with col2:
    analyze_btn = st.button("📊 Analyze Video", use_container_width=True, disabled=not video_path,
                            help="Show detailed video information")
with col3:
    generate_btn = st.button("⚡ Generate Shorts", type="primary", use_container_width=True,
                             disabled=not video_path, help="Start creating YouTube Shorts")

st.divider()

# Preview Section
if preview_btn and video_path:
    try:
        st.subheader("🎬 Video Preview")
        if convert_to_shorts:
            info = ffprobe_video_info(video_path)
            if use_auto_focus and st.session_state.detected_crop_position:
                preview_crop_pos = st.session_state.detected_crop_position
            else:
                preview_crop_pos = crop_position
            crop_params = calculate_crop_for_shorts(info['width'], info['height'], preview_crop_pos)
            st.info(f"📱 Preview in YouTube Shorts format (9:16) - Crop position: **{preview_crop_pos.upper()}**")
        else:
            crop_params = None
            st.info("📺 Preview in original format")

        with tempfile.TemporaryDirectory() as tmp:
            preview = Path(tmp) / "preview.mp4"
            if convert_to_shorts and crop_params:
                filters = f"crop={crop_params['crop_width']}:{crop_params['crop_height']}:{crop_params['origin_x']}:{crop_params['origin_y']},scale={SHORTS_RESOLUTION[0]}:{SHORTS_RESOLUTION[1]}"
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-ss", "0", "-i", video_path, "-t", str(preview_duration),
                    "-vf", filters, "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(preview), "-y"
                ]
            else:
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-ss", "0", "-i", video_path, "-t", str(preview_duration),
                    "-c", "copy", "-map", "0", str(preview), "-y"
                ]
            subprocess.check_call(cmd)
            st.video(preview.read_bytes())
            st.success(f"Preview ready")
    except Exception as e:
        st.error(f"❌ Preview failed: {e}")

# Analysis Section
if analyze_btn and video_path:
    try:
        st.subheader("📊 Video Analysis")
        with st.spinner("Analyzing video..."):
            info = ffprobe_video_info(video_path)
            if info:
                duration_sec = info['duration']
                file_size = Path(video_path).stat().st_size
                total_clips = math.ceil(duration_sec / clip_len_seconds)
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("⏱️ Duration", format_duration(duration_sec))
                with metric_col2:
                    st.metric("💾 File Size", format_size(file_size))
                with metric_col3:
                    st.metric("🎞️ Total Clips", total_clips)
                with metric_col4:
                    st.metric("📏 Clip Length", format_duration(clip_len_seconds))
                if show_detailed_info:
                    with st.expander("🔍 Detailed Information", expanded=True):
                        detail_col1, detail_col2 = st.columns(2)
                        with detail_col1:
                            st.write("**Video Details:**")
                            st.write(f"- Codec: {info.get('video_codec', 'N/A')}")
                            st.write(f"- Resolution: {info.get('resolution', 'N/A')}")
                            st.write(f"- FPS: {info.get('fps', 0):.2f}")
                            st.write(f"- Bitrate: {info.get('bitrate', 0) // 1000} kbps")
                            if convert_to_shorts:
                                used_pos = st.session_state.detected_crop_position if (use_auto_focus and st.session_state.detected_crop_position) else crop_position
                                crop_params = calculate_crop_for_shorts(info['width'], info['height'], used_pos)
                                st.write(f"\n**Crop Settings:**")
                                st.write(f"- Mode: {'Smart Auto-Focus' if use_auto_focus else 'Manual'}")
                                st.write(f"- Position: {used_pos.upper()}")
                                st.write(f"- Crop size: {crop_params['crop_width']}x{crop_params['crop_height']}")
                                st.write(f"- Output: {SHORTS_RESOLUTION[0]}x{SHORTS_RESOLUTION[1]}")
                        with detail_col2:
                            st.write("**Audio Details:**")
                            st.write(f"- Codec: {info.get('audio_codec', 'N/A')}")
                            st.write(f"- Sample Rate: {info.get('audio_sample_rate', 'N/A')}")
                            st.write(f"\n**YouTube Shorts Info:**")
                            st.write(f"- ✅ Aspect Ratio: 9:16")
                            st.write(f"- ✅ Resolution: 1080x1920")
                            st.write(f"- Recommended: Under 60 seconds")
                            st.write(f"- Your duration: {format_duration(clip_len_seconds)}")
                            if clip_len_seconds <= 60:
                                st.write(f"- ✅ Perfect for Shorts!")
                            else:
                                st.write(f"- ⚠️ Over 60s limit")
                if clip_len_seconds <= 60:
                    st.success(f"✅ Will create **{total_clips} YouTube Shorts** of **{format_duration(clip_len_seconds)}** each")
                else:
                    st.info(f"📹 Will create **{total_clips} clips** of **{format_duration(clip_len_seconds)}** each")
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")

# Generate Clips Section
if generate_btn and video_path:
    try:
        ensure_dir(st.session_state.work_dir)
        st.subheader("🔄 Processing Video")
        with st.spinner("Reading video information..."):
            info = ffprobe_video_info(video_path)
            duration_sec = info['duration']
            total_clips = math.ceil(duration_sec / clip_len_seconds)
            if convert_to_shorts:
                final_crop_pos = (st.session_state.detected_crop_position if (use_auto_focus and st.session_state.detected_crop_position)
                                  else crop_position)
                crop_params = calculate_crop_for_shorts(info['width'], info['height'], final_crop_pos)
            if total_clips == 0:
                st.error("❌ Video is too short for the selected clip length")
                st.stop()

            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                st.write("**📥 Source:**")
                st.write(f"- File: {Path(video_path).name}")
                st.write(f"- Duration: {format_duration(duration_sec)}")
                st.write(f"- Resolution: {info.get('resolution', 'N/A')}")
            with summary_col2:
                st.write("**📤 Output:**")
                st.write(f"- Total clips: {total_clips}")
                st.write(f"- Clip length: {format_duration(clip_len_seconds)}")
                if convert_to_shorts:
                    st.write(f"- Format: 9:16 (1080x1920)")
                    st.write(f"- Crop: {final_crop_pos.upper()}")
                    if use_auto_focus:
                        st.write(f"- Mode: Smart Auto-Focus ✨")
                st.write(f"- Location: {st.session_state.work_dir}")

            st.divider()
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()
            start_time = datetime.now()
            outputs = []
            base_stem = Path(video_path).stem
            out_ext = ".mp4"
            for i in range(total_clips):
                start = i * clip_len_seconds
                dur = min(clip_len_seconds, duration_sec - start)
                if dur <= 0:
                    break
                out_path = str(Path(st.session_state.work_dir) / f"{base_stem}_short_{i+1:03d}{out_ext}")
                elapsed = (datetime.now() - start_time).total_seconds()
                if i > 0:
                    eta = (elapsed / i) * (total_clips - i)
                    time_text.text(f"⏱️ Elapsed: {format_duration(elapsed)}  ETA: {format_duration(eta)}")
                status_text.text(f"🔄 Creating clip {i+1}/{total_clips}...")
                if convert_to_shorts:
                    ffmpeg_segment_crop_shorts(video_path, start, dur, out_path, crop_params,
                                               scale_to_1080p=True, quality_preset=quality_preset)
                else:
                    ffmpeg_segment_copy(video_path, start, dur, out_path)
                outputs.append(out_path)
                progress = int(((i + 1) / total_clips) * 100)
                progress_bar.progress(progress)

            progress_bar.progress(100)
            status_text.empty()
            time_text.empty()
            total_time = (datetime.now() - start_time).total_seconds()
            st.success(f"🎉 Successfully created {len(outputs)} clips in {format_duration(total_time)}!")
            total_output_size = sum(Path(p).stat().st_size for p in outputs)
            result_col1, result_col2, result_col3 = st.columns(3)
            with result_col1:
                st.metric("✅ Clips Created", len(outputs))
            with result_col2:
                st.metric("💾 Total Size", format_size(total_output_size))
            with result_col3:
                st.metric("⚡ Processing Time", format_duration(total_time))

            with st.expander("📁 Generated Clips", expanded=True):
                for idx, clip_path in enumerate(outputs, 1):
                    clip_size = Path(clip_path).stat().st_size
                    clip_name = Path(clip_path).name
                    clip_dur = min(clip_len_seconds, duration_sec - ((idx-1) * clip_len_seconds))
                    st.write(f"{idx}. ✅ **{clip_name}** ({format_size(clip_size)}) - {format_duration(clip_dur)}")

            # Prepare ZIP for download
            zip_name = f"shorts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
            zip_path = str(Path(st.session_state.work_dir) / zip_name)
            import zipfile
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for p in outputs:
                    zf.write(p, arcname=Path(p).name)
            with open(zip_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download all clips (ZIP)",
                    data=f.read(),
                    file_name=zip_name,
                    mime="application/zip",
                    use_container_width=True
                )

            st.info("""
            📱 **How to upload to YouTube Shorts:**
            1. Open YouTube app on mobile
            2. Tap the **+** button
            3. Select **Create a Short**
            4. Upload your video files
            5. Add **#Shorts** in title or description
            """)
    except subprocess.CalledProcessError as e:
        st.error(f"❌ FFmpeg error: {e}")
        st.error("Try using a different video file or check FFmpeg installation")
    except Exception as ex:
        st.error(f"❌ Unexpected error: {ex}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

st.divider()
st.caption("""
**📱 Features:** ✨ Smart Auto-Focus (Face Detection)  ⏱️ Unlimited Duration  📐 Perfect 9:16 Format  🎯 Auto Crop Position
**🔧 Requirements:** FFmpeg (required), OpenCV (headless) for auto-focus
""")
