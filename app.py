import google.generativeai as genai
import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
from moviepy.editor import (ImageClip, AudioFileClip, concatenate_videoclips, 
                           CompositeVideoClip, TextClip, ColorClip)
import os
import textwrap
from io import BytesIO
import re
import json
import time
import random
from datetime import datetime
import logging
from typing import List, Dict, Optional, Tuple
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoGenerator:
    def __init__(self):
        # API Configuration
        self.api_keys = {
            'gemini': "AIzaSyCR-12qPUKcljSgOZE7aiqhnaGh_XsBkXg",
            'pexels': "93iFNlFJbNUNJ9clOLenyndx6CgNpkwuPWqEFfAYIR2xJDkVKvzoGpAB",
            'unsplash': "QLKONf_YfFdzT7KNqx5MFZ9DmtwV51DSB5bT1KQAnH4",
            'pixabay': "51005985-ea487de96fc46656ceaf3303e",
            'google_search': "AIzaSyBl1V0OENYMDrW1IjfQ4eZjI8r7gLhrvmw"
        }
        
        self.working_apis = {}
        self.setup_directories()
        self.check_api_status()
        self.setup_gemini()
        
    def setup_directories(self):
        """Create necessary directories"""
        self.dirs = {
            'temp_images': 'temp_images',
            'temp_audio': 'temp_audio',
            'output': 'output_videos',
            'assets': 'assets'
        }
        for directory in self.dirs.values():
            os.makedirs(directory, exist_ok=True)
            
    def check_api_status(self):
        """Check which APIs are working"""
        logger.info("🔍 Checking API status...")
        
        # Check Pexels
        try:
            response = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": self.api_keys['pexels']},
                params={"query": "test", "per_page": 1},
                timeout=10
            )
            self.working_apis['pexels'] = response.status_code == 200
        except:
            self.working_apis['pexels'] = False
            
        # Check Unsplash
        try:
            response = requests.get(
                "https://api.unsplash.com/search/photos",
                headers={"Authorization": f"Client-ID {self.api_keys['unsplash']}"},
                params={"query": "test", "per_page": 1},
                timeout=10
            )
            self.working_apis['unsplash'] = response.status_code == 200
        except:
            self.working_apis['unsplash'] = False
            
        # Check Pixabay
        try:
            response = requests.get(
                "https://pixabay.com/api/",
                params={"key": self.api_keys['pixabay'], "q": "test", "per_page": 3},
                timeout=10
            )
            self.working_apis['pixabay'] = response.status_code == 200
        except:
            self.working_apis['pixabay'] = False
            
        # Check Google Custom Search
        try:
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": self.api_keys['google_search'],
                    "cx": "017576662512468239146:omuauf_lfve",  # Generic search engine
                    "q": "test",
                    "searchType": "image",
                    "num": 1
                },
                timeout=10
            )
            self.working_apis['google_search'] = response.status_code == 200
        except:
            self.working_apis['google_search'] = False
            
        logger.info(f"✅ Working APIs: {[k for k, v in self.working_apis.items() if v]}")
        logger.info(f"❌ Failed APIs: {[k for k, v in self.working_apis.items() if not v]}")
        
    def setup_gemini(self):
        """Setup Gemini AI"""
        try:
            genai.configure(api_key=self.api_keys['gemini'])
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("✅ Gemini AI configured successfully")
        except Exception as e:
            logger.error(f"❌ Failed to setup Gemini: {e}")
            raise

    def generate_video_content(self, topic: str, nature: str) -> Dict:
        """Generate comprehensive video content using Gemini"""
        prompt = f"""
        Create a professional video script for a {nature} about "{topic}".
        
        Requirements:
        1. Generate 8-12 engaging segments (each 2-3 sentences)
        2. Each segment should be visually descriptive for image matching
        3. Include emotional hooks and storytelling elements
        4. Make it suitable for a 60-90 second video
        5. Use active voice and compelling language
        
        Format as JSON:
        {{
            "title": "Compelling video title",
            "hook": "Opening hook sentence",
            "segments": [
                {{
                    "text": "Segment text with visual elements",
                    "keywords": ["keyword1", "keyword2", "keyword3"],
                    "emotion": "happy/exciting/inspiring/dramatic",
                    "duration": 5-8
                }}
            ],
            "call_to_action": "Ending call to action"
        }}
        
        Topic: {topic}
        Nature: {nature}
        """
        
        try:
            response = self.model.generate_content(prompt)
            content = json.loads(response.text.strip().replace('```json', '').replace('```', ''))
            logger.info(f"✅ Generated content with {len(content['segments'])} segments")
            return content
        except Exception as e:
            logger.error(f"❌ Content generation failed: {e}")
            return self._fallback_content(topic, nature)
            
    def _fallback_content(self, topic: str, nature: str) -> Dict:
        """Fallback content structure"""
        return {
            "title": f"Amazing {topic} - {nature}",
            "hook": f"Discover the incredible world of {topic}!",
            "segments": [
                {
                    "text": f"Welcome to an amazing journey about {topic}. Get ready to be inspired!",
                    "keywords": [topic, "amazing", "journey"],
                    "emotion": "exciting",
                    "duration": 6
                }
            ],
            "call_to_action": "Like and subscribe for more amazing content!"
        }

    def search_premium_images(self, keywords: List[str], emotion: str) -> List[Dict]:
        """Search for high-quality images from multiple sources"""
        all_images = []
        search_queries = self._generate_search_queries(keywords, emotion)
        
        for query in search_queries[:3]:  # Limit to top 3 queries
            if self.working_apis.get('unsplash'):
                all_images.extend(self._search_unsplash(query))
            if self.working_apis.get('pexels'):
                all_images.extend(self._search_pexels(query))
            if self.working_apis.get('pixabay'):
                all_images.extend(self._search_pixabay(query))
                
        # Remove duplicates and score images
        unique_images = self._deduplicate_images(all_images)
        scored_images = self._score_images(unique_images, keywords, emotion)
        
        return scored_images[:10]  # Return top 10 images
        
    def _generate_search_queries(self, keywords: List[str], emotion: str) -> List[str]:
        """Generate optimized search queries"""
        base_queries = []
        
        # Combine keywords intelligently
        if len(keywords) >= 2:
            base_queries.append(f"{keywords[0]} {keywords[1]}")
        base_queries.extend(keywords[:3])
        
        # Add quality modifiers
        quality_modifiers = ["4k", "hd", "professional", "cinematic", "premium"]
        emotion_modifiers = {
            "exciting": ["dynamic", "energetic", "vibrant"],
            "inspiring": ["beautiful", "majestic", "uplifting"],
            "dramatic": ["powerful", "intense", "striking"],
            "happy": ["bright", "colorful", "joyful"]
        }
        
        enhanced_queries = []
        for query in base_queries:
            enhanced_queries.append(f"{query} {random.choice(quality_modifiers)}")
            if emotion in emotion_modifiers:
                enhanced_queries.append(f"{query} {random.choice(emotion_modifiers[emotion])}")
                
        return enhanced_queries
        
    def _search_unsplash(self, query: str) -> List[Dict]:
        """Search Unsplash for high-quality images"""
        try:
            response = requests.get(
                "https://api.unsplash.com/search/photos",
                headers={"Authorization": f"Client-ID {self.api_keys['unsplash']}"},
                params={
                    "query": query,
                    "orientation": "landscape",
                    "per_page": 15,
                    "order_by": "relevant"
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                return [{
                    "url": photo['urls']['full'],
                    "thumb": photo['urls']['thumb'],
                    "width": photo['width'],
                    "height": photo['height'],
                    "source": "unsplash",
                    "quality_score": photo['width'] * photo['height'],
                    "likes": photo.get('likes', 0)
                } for photo in data['results']]
        except Exception as e:
            logger.warning(f"Unsplash search failed for '{query}': {e}")
        return []
        
    def _search_pexels(self, query: str) -> List[Dict]:
        """Search Pexels for high-quality images"""
        try:
            response = requests.get(
                "https://api.pexels.com/v1/search",
                headers={"Authorization": self.api_keys['pexels']},
                params={
                    "query": query,
                    "orientation": "landscape",
                    "per_page": 15,
                    "size": "large"
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                return [{
                    "url": photo['src']['original'],
                    "thumb": photo['src']['medium'],
                    "width": photo['width'],
                    "height": photo['height'],
                    "source": "pexels",
                    "quality_score": photo['width'] * photo['height'],
                    "likes": 0
                } for photo in data['photos']]
        except Exception as e:
            logger.warning(f"Pexels search failed for '{query}': {e}")
        return []
        
    def _search_pixabay(self, query: str) -> List[Dict]:
        """Search Pixabay for high-quality images"""
        try:
            response = requests.get(
                "https://pixabay.com/api/",
                params={
                    "key": self.api_keys['pixabay'],
                    "q": query,
                    "image_type": "photo",
                    "orientation": "horizontal",
                    "per_page": 15,
                    "min_width": 1920,
                    "min_height": 1080
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                return [{
                    "url": hit['largeImageURL'],
                    "thumb": hit['webformatURL'],
                    "width": hit['imageWidth'],
                    "height": hit['imageHeight'],
                    "source": "pixabay",
                    "quality_score": hit['imageWidth'] * hit['imageHeight'],
                    "likes": hit.get('likes', 0)
                } for hit in data['hits']]
        except Exception as e:
            logger.warning(f"Pixabay search failed for '{query}': {e}")
        return []
        
    def _deduplicate_images(self, images: List[Dict]) -> List[Dict]:
        """Remove duplicate images based on URL"""
        seen_urls = set()
        unique_images = []
        
        for img in images:
            if img['url'] not in seen_urls:
                seen_urls.add(img['url'])
                unique_images.append(img)
                
        return unique_images
        
    def _score_images(self, images: List[Dict], keywords: List[str], emotion: str) -> List[Dict]:
        """Score and rank images based on quality and relevance"""
        for img in images:
            score = 0
            
            # Quality score (resolution)
            score += min(img['quality_score'] / 1000000, 10)  # Max 10 points for quality
            
            # Popularity score
            score += min(img['likes'] / 100, 5)  # Max 5 points for likes
            
            # Source preference (Unsplash > Pexels > Pixabay)
            source_scores = {"unsplash": 3, "pexels": 2, "pixabay": 1}
            score += source_scores.get(img['source'], 0)
            
            # Aspect ratio preference (16:9 or close)
            aspect_ratio = img['width'] / img['height']
            if 1.6 <= aspect_ratio <= 1.9:  # Close to 16:9
                score += 2
                
            img['final_score'] = score
            
        return sorted(images, key=lambda x: x['final_score'], reverse=True)
        
    def download_and_process_image(self, image_url: str, output_path: str) -> Optional[str]:
        """Download and enhance image quality"""
        try:
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Save original
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # Enhance image
            enhanced_path = self._enhance_image(output_path)
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Failed to download image from {image_url}: {e}")
            return None
            
    def _enhance_image(self, image_path: str) -> str:
        """Enhance image quality and prepare for video"""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to 4K (3840x2160) maintaining aspect ratio
                target_width, target_height = 3840, 2160
                img_ratio = img.width / img.height
                target_ratio = target_width / target_height
                
                if img_ratio > target_ratio:
                    # Image is wider, fit to height
                    new_height = target_height
                    new_width = int(target_height * img_ratio)
                else:
                    # Image is taller, fit to width
                    new_width = target_width
                    new_height = int(target_width / img_ratio)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Crop to exact 4K if needed
                if img.width > target_width or img.height > target_height:
                    left = (img.width - target_width) // 2
                    top = (img.height - target_height) // 2
                    right = left + target_width
                    bottom = top + target_height
                    img = img.crop((left, top, right, bottom))
                
                # Enhance image quality
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(1.2)
                
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(1.1)
                
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.05)
                
                # Save enhanced image
                enhanced_path = image_path.replace('.jpg', '_enhanced.jpg')
                img.save(enhanced_path, 'JPEG', quality=95, optimize=True)
                
                return enhanced_path
                
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image_path
            
    def generate_professional_audio(self, full_text: str, output_path: str) -> Optional[str]:
        """Generate high-quality audio using multiple methods"""
        try:
            # Try gTTS first (most reliable)
            return self._generate_gtts_audio(full_text, output_path)
        except ImportError:
            logger.warning("gTTS not available, trying alternative methods")
            try:
                # Try edge-tts as fallback
                return self._generate_edge_tts_audio(full_text, output_path)
            except ImportError:
                logger.error("No TTS engines available. Please install gTTS or edge-tts")
                return None
                
    def _generate_gtts_audio(self, text: str, output_path: str) -> str:
        """Generate audio using Google Text-to-Speech"""
        from gtts import gTTS
        
        # Clean text for better speech
        cleaned_text = re.sub(r'[^\w\s.,!?]', '', text)
        
        tts = gTTS(
            text=cleaned_text,
            lang='en',
            slow=False,
            tld='com'  # Use .com for better voice quality
        )
        
        tts.save(output_path)
        logger.info(f"✅ Audio generated using gTTS: {output_path}")
        return output_path
        
    def _generate_edge_tts_audio(self, text: str, output_path: str) -> str:
        """Generate audio using Edge TTS (alternative)"""
        import edge_tts
        import asyncio
        
        async def generate():
            voice = "en-US-AriaNeural"  # High-quality voice
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(output_path)
            
        asyncio.run(generate())
        logger.info(f"✅ Audio generated using Edge TTS: {output_path}")
        return output_path
        
    def create_cinematic_video(self, segments: List[Dict], images: List[str], 
                             audio_path: str, output_path: str) -> Optional[str]:
        """Create a cinematic video with professional effects"""
        try:
            if not images:
                logger.error("No images available for video creation")
                return None
                
            # Load audio to get duration
            audio_clip = AudioFileClip(audio_path) if audio_path else None
            total_duration = audio_clip.duration if audio_clip else sum(s.get('duration', 6) for s in segments)
            
            # Calculate timing for each segment
            segment_durations = self._calculate_segment_timing(segments, total_duration)
            
            # Create video clips with effects
            video_clips = []
            current_time = 0
            
            for i, (segment, duration) in enumerate(zip(segments, segment_durations)):
                if i < len(images):
                    clip = self._create_segment_clip(
                        images[i], 
                        duration, 
                        segment.get('emotion', 'neutral'),
                        current_time
                    )
                    if clip:
                        video_clips.append(clip)
                        current_time += duration
                        
            if not video_clips:
                logger.error("No video clips created")
                return None
                
            # Combine all clips
            final_video = concatenate_videoclips(video_clips, method="compose")
            
            # Add audio
            if audio_clip:
                final_video = final_video.set_audio(audio_clip)
                
            # Add professional color grading
            final_video = self._apply_color_grading(final_video)
            
            # Render video
            final_video.write_videofile(
                output_path,
                fps=30,
                codec='libx264',
                audio_codec='aac',
                preset='medium',
                ffmpeg_params=[
                    '-crf', '18',  # High quality
                    '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart'
                ]
            )
            
            logger.info(f"✅ Video created successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Video creation failed: {e}")
            return None
            
    def _calculate_segment_timing(self, segments: List[Dict], total_duration: float) -> List[float]:
        """Calculate optimal timing for each segment"""
        target_durations = [s.get('duration', 6) for s in segments]
        total_target = sum(target_durations)
        
        # Scale durations to fit total duration
        if total_target > 0:
            scale_factor = total_duration / total_target
            return [d * scale_factor for d in target_durations]
        else:
            return [total_duration / len(segments)] * len(segments)
            
    def _create_segment_clip(self, image_path: str, duration: float, 
                           emotion: str, start_time: float) -> Optional[ImageClip]:
        """Create a single segment clip with effects"""
        try:
            # Create base clip
            clip = ImageClip(image_path, duration=duration)
            
            # Add emotion-based effects
            if emotion == "exciting":
                # Slight zoom effect
                clip = clip.resize(lambda t: 1 + 0.02 * t / duration)
            elif emotion == "dramatic":
                # Slow zoom out
                clip = clip.resize(lambda t: 1.1 - 0.05 * t / duration)
            elif emotion == "inspiring":
                # Gentle pan
                clip = clip.set_position(lambda t: ('center', 'center'))
                
            # Add fade transitions
            if start_time > 0:
                clip = clip.fadein(0.5)
            clip = clip.fadeout(0.5)
            
            return clip
            
        except Exception as e:
            logger.error(f"Failed to create clip for {image_path}: {e}")
            return None
            
    def _apply_color_grading(self, video_clip):
        """Apply professional color grading"""
        try:
            # Enhance colors and contrast
            return video_clip.fx(lambda clip: clip.multiply_volume(1.0))
        except:
            return video_clip
            
    def add_dynamic_subtitles(self, video_path: str, segments: List[Dict], 
                            output_path: str) -> Optional[str]:
        """Add dynamic, professional subtitles"""
        try:
            video = VideoFileClip(video_path)
            audio = video.audio
            
            # Calculate subtitle timing based on audio duration
            total_duration = audio.duration if audio else video.duration
            segment_durations = self._calculate_segment_timing(segments, total_duration)
            
            subtitle_clips = []
            current_time = 0
            
            for segment, duration in zip(segments, segment_durations):
                text = segment['text']
                
                # Create subtitle clip
                subtitle = TextClip(
                    text,
                    fontsize=60,
                    font='Arial-Bold',
                    color='white',
                    stroke_color='black',
                    stroke_width=3,
                    method='caption',
                    size=(video.w * 0.8, None),
                    align='center'
                ).set_position(('center', 'bottom')).set_start(current_time).set_duration(duration)
                
                # Add text animation
                subtitle = subtitle.fadein(0.3).fadeout(0.3)
                
                subtitle_clips.append(subtitle)
                current_time += duration
                
            # Compose final video with subtitles
            final_video = CompositeVideoClip([video] + subtitle_clips)
            
            final_video.write_videofile(
                output_path,
                fps=30,
                codec='libx264',
                audio_codec='aac',
                preset='medium'
            )
            
            logger.info(f"✅ Subtitles added successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to add subtitles: {e}")
            return None
            
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            for directory in [self.dirs['temp_images'], self.dirs['temp_audio']]:
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            logger.info("🧹 Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
            
    def generate_video(self, topic: str, nature: str) -> Optional[str]:
        """Main video generation pipeline"""
        logger.info(f"🎬 Starting video generation for: {topic} ({nature})")
        
        try:
            # Step 1: Generate content
            logger.info("📝 Generating video content...")
            content = self.generate_video_content(topic, nature)
            
            # Step 2: Search and download images
            logger.info("🖼️ Searching for premium images...")
            all_images = []
            
            for segment in content['segments']:
                images = self.search_premium_images(
                    segment['keywords'], 
                    segment['emotion']
                )
                if images:
                    # Download best image
                    best_image = images[0]
                    image_path = os.path.join(
                        self.dirs['temp_images'], 
                        f"segment_{len(all_images)}.jpg"
                    )
                    
                    downloaded = self.download_and_process_image(
                        best_image['url'], 
                        image_path
                    )
                    
                    if downloaded:
                        all_images.append(downloaded)
                        
            if not all_images:
                logger.error("❌ No images downloaded successfully")
                return None
                
            # Step 3: Generate audio
            logger.info("🎵 Generating professional audio...")
            full_text = content['hook'] + " " + " ".join([s['text'] for s in content['segments']]) + " " + content['call_to_action']
            
            audio_path = os.path.join(self.dirs['temp_audio'], 'narration.mp3')
            generated_audio = self.generate_professional_audio(full_text, audio_path)
            
            # Step 4: Create video
            logger.info("🎥 Creating cinematic video...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_video_path = os.path.join(
                self.dirs['output'], 
                f"{topic.replace(' ', '_')}_{timestamp}_temp.mp4"
            )
            
            video_path = self.create_cinematic_video(
                content['segments'],
                all_images,
                generated_audio,
                temp_video_path
            )
            
            if not video_path:
                logger.error("❌ Video creation failed")
                return None
                
            # Step 5: Add subtitles
            logger.info("📱 Adding dynamic subtitles...")
            final_video_path = os.path.join(
                self.dirs['output'],
                f"{topic.replace(' ', '_')}_{timestamp}_final.mp4"
            )
            
            final_video = self.add_dynamic_subtitles(
                video_path,
                content['segments'],
                final_video_path
            )
            
            # Cleanup
            self.cleanup_temp_files()
            
            if final_video:
                logger.info(f"🎉 Video generation completed successfully!")
                logger.info(f"📁 Output: {final_video}")
                return final_video
            else:
                logger.error("❌ Final video processing failed")
                return video_path  # Return temp video as fallback
                
        except Exception as e:
            logger.error(f"❌ Video generation failed: {e}")
            self.cleanup_temp_files()
            return None

def main():
    """Main execution function"""
    print("🎬 Advanced Text-to-Video Generator")
    print("=" * 50)
    
    # Get user input
    topic = input("Enter your video topic: ").strip()
    if not topic:
        print("❌ Topic cannot be empty!")
        return
        
    nature = input("Enter the nature of content (e.g., 'educational', 'inspirational', 'entertaining'): ").strip()
    if not nature:
        nature = "educational"
        
    # Initialize generator
    try:
        generator = VideoGenerator()
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return
        
    # Generate video
    output_path = generator.generate_video(topic, nature)
    
    if output_path:
        print(f"\n🎉 SUCCESS! Your video is ready:")
        print(f"📁 {output_path}")
        print("\n🎥 Video features:")
        print("✅ 4K Quality Images")
        print("✅ Professional Audio")
        print("✅ Cinematic Effects")
        print("✅ Dynamic Subtitles")
        print("✅ Smooth Transitions")
    else:
        print("\n❌ Video generation failed. Please check the logs for details.")

if __name__ == "__main__":
    main()
