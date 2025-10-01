"""
Dataset download and extraction utilities.
"""

import os
import requests
import zipfile
import rarfile
from typing import Dict, Any
import logging
from tqdm import tqdm
import shutil


class UCF101Downloader:
    """Class for downloading and extracting the UCF101 dataset."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the UCF101Downloader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # UCF101 dataset URLs
        self.dataset_urls = {
            'ucf101': 'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar',
            'annotations': 'https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip'
        }
        
        self.raw_data_dir = config['paths']['raw_data_dir']
        self.processed_data_dir = config['paths']['processed_data_dir']
    
    def download_file(self, url: str, destination: str) -> bool:
        """
        Download a file from URL with progress bar.
        
        Args:
            url: URL to download from
            destination: Destination file path
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as file:
                with tqdm(
                    desc=f"Downloading {os.path.basename(destination)}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            
            self.logger.info(f"Downloaded: {destination}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {str(e)}")
            return False
    
    def extract_rar(self, rar_path: str, extract_to: str) -> bool:
        """
        Extract RAR file.
        
        Args:
            rar_path: Path to RAR file
            extract_to: Directory to extract to
            
        Returns:
            True if extraction successful, False otherwise
        """
        try:
            with rarfile.RarFile(rar_path) as rf:
                rf.extractall(extract_to)
            self.logger.info(f"Extracted RAR: {rar_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error extracting RAR {rar_path}: {str(e)}")
            return False
    
    def extract_zip(self, zip_path: str, extract_to: str) -> bool:
        """
        Extract ZIP file.
        
        Args:
            zip_path: Path to ZIP file
            extract_to: Directory to extract to
            
        Returns:
            True if extraction successful, False otherwise
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_to)
            self.logger.info(f"Extracted ZIP: {zip_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error extracting ZIP {zip_path}: {str(e)}")
            return False
    
    def organize_dataset(self, source_dir: str, target_dir: str) -> None:
        """
        Organize dataset into proper structure.
        
        Args:
            source_dir: Source directory containing extracted files
            target_dir: Target directory for organized dataset
        """
        self.logger.info("Organizing dataset structure...")
        
        # Find UCF101 directory
        ucf101_dir = None
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            if os.path.isdir(item_path) and 'UCF-101' in item:
                ucf101_dir = item_path
                break
        
        if not ucf101_dir:
            self.logger.error("UCF-101 directory not found")
            return
        
        # Get selected classes
        selected_classes = self.config['dataset']['selected_classes']
        
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy selected classes
        for class_name in selected_classes:
            source_class_dir = os.path.join(ucf101_dir, class_name)
            target_class_dir = os.path.join(target_dir, class_name)
            
            if os.path.exists(source_class_dir):
                shutil.copytree(source_class_dir, target_class_dir, dirs_exist_ok=True)
                video_count = len([f for f in os.listdir(target_class_dir) if f.endswith(('.mp4', '.avi'))])
                self.logger.info(f"Copied {class_name}: {video_count} videos")
            else:
                self.logger.warning(f"Class directory not found: {source_class_dir}")
    
    def download_dataset(self) -> bool:
        """
        Download the complete UCF101 dataset.
        
        Returns:
            True if download and extraction successful, False otherwise
        """
        # Create directories
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        success = True
        
        # Download UCF101 dataset
        ucf101_rar = os.path.join(self.raw_data_dir, 'UCF101.rar')
        if not os.path.exists(ucf101_rar):
            self.logger.info("Downloading UCF101 dataset...")
            if not self.download_file(self.dataset_urls['ucf101'], ucf101_rar):
                success = False
        else:
            self.logger.info("UCF101 dataset already downloaded")
        
        # Download annotations
        annotations_zip = os.path.join(self.raw_data_dir, 'UCF101TrainTestSplits.zip')
        if not os.path.exists(annotations_zip):
            self.logger.info("Downloading UCF101 annotations...")
            if not self.download_file(self.dataset_urls['annotations'], annotations_zip):
                success = False
        else:
            self.logger.info("UCF101 annotations already downloaded")
        
        if not success:
            return False
        
        # Extract files
        extract_dir = os.path.join(self.raw_data_dir, 'extracted')
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract UCF101 dataset
        if os.path.exists(ucf101_rar):
            self.logger.info("Extracting UCF101 dataset...")
            if not self.extract_rar(ucf101_rar, extract_dir):
                success = False
        
        # Extract annotations
        if os.path.exists(annotations_zip):
            self.logger.info("Extracting UCF101 annotations...")
            if not self.extract_zip(annotations_zip, extract_dir):
                success = False
        
        if success:
            # Organize dataset
            organized_dir = os.path.join(self.processed_data_dir, 'videos')
            self.organize_dataset(extract_dir, organized_dir)
            
            self.logger.info("Dataset download and organization completed successfully!")
        
        return success
    
    def download_sample_dataset(self) -> bool:
        """
        Download a smaller sample dataset for testing.
        This downloads a few sample videos for each class for quick testing.
        
        Returns:
            True if successful, False otherwise
        """
        # Alternative: Create sample dataset from YouTube videos
        # This is a placeholder implementation
        self.logger.info("Creating sample dataset...")
        
        # Create sample data directory
        sample_dir = os.path.join(self.processed_data_dir, 'sample_videos')
        os.makedirs(sample_dir, exist_ok=True)
        
        # For demonstration, create directories for selected classes
        selected_classes = self.config['dataset']['selected_classes'][:5]  # First 5 classes
        
        for class_name in selected_classes:
            class_dir = os.path.join(sample_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            self.logger.info(f"Created directory for class: {class_name}")
        
        self.logger.info("Sample dataset structure created.")
        self.logger.warning("Note: You need to manually add video files to each class directory.")
        
        return True
    
    def verify_dataset(self) -> Dict[str, Any]:
        """
        Verify the downloaded dataset.
        
        Returns:
            Dictionary with verification results
        """
        video_dir = os.path.join(self.processed_data_dir, 'videos')
        
        if not os.path.exists(video_dir):
            return {
                'status': 'error',
                'message': 'Dataset directory not found',
                'classes_found': 0,
                'total_videos': 0
            }
        
        selected_classes = self.config['dataset']['selected_classes']
        classes_found = 0
        total_videos = 0
        class_info = {}
        
        for class_name in selected_classes:
            class_dir = os.path.join(video_dir, class_name)
            if os.path.exists(class_dir):
                video_files = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
                video_count = len(video_files)
                
                if video_count > 0:
                    classes_found += 1
                    total_videos += video_count
                    class_info[class_name] = video_count
        
        return {
            'status': 'success' if classes_found > 0 else 'warning',
            'message': f'Found {classes_found}/{len(selected_classes)} classes',
            'classes_found': classes_found,
            'total_videos': total_videos,
            'class_info': class_info
        }


def main():
    """Main function to download dataset."""
    import sys
    import argparse
    from utils.helpers import load_config, setup_logging
    
    parser = argparse.ArgumentParser(description='Download UCF101 dataset')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    parser.add_argument('--sample', action='store_true', help='Download sample dataset only')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config.get('logging', {}).get('level', 'INFO'))
    
    # Create downloader
    downloader = UCF101Downloader(config)
    
    # Download dataset
    if args.sample:
        success = downloader.download_sample_dataset()
    else:
        success = downloader.download_dataset()
    
    if success:
        # Verify dataset
        verification = downloader.verify_dataset()
        print(f"Dataset verification: {verification}")
    else:
        print("Dataset download failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()