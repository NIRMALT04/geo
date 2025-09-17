"""
ISRO EO Data Processing Pipeline
Handles downloading, preprocessing, and annotation of satellite imagery from ISRO/Bhuvan
"""

import asyncio
import aiohttp
import aiofiles
from pathlib import Path
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.enums import Resampling
import geopandas as gpd
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from datetime import datetime, timedelta
import hashlib
from dataclasses import dataclass, asdict
import yaml
from concurrent.futures import ThreadPoolExecutor
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import zipfile
import tarfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EOTileMetadata:
    """Metadata for EO satellite tiles"""
    tile_id: str
    satellite: str
    sensor: str
    acquisition_date: str
    path: int
    row: int
    cloud_cover: float
    sun_elevation: float
    sun_azimuth: float
    processing_level: str
    spatial_resolution: float
    spectral_bands: List[str]
    bbox: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    epsg_code: int
    file_size: int
    download_url: str
    thumbnail_url: Optional[str] = None
    metadata_url: Optional[str] = None

@dataclass
class ProcessingConfig:
    """Configuration for data processing pipeline"""
    # Data sources
    isro_base_url: str = "https://bhuvan-app1.nrsc.gov.in/data"
    bhuvan_api_key: Optional[str] = None
    
    # Processing parameters
    target_crs: str = "EPSG:4326"
    target_resolution: Tuple[int, int] = (512, 512)
    resampling_method: str = "bilinear"
    compression: str = "lzw"
    
    # Filtering criteria
    max_cloud_cover: float = 30.0
    min_date: str = "2020-01-01"
    max_date: str = "2024-12-31"
    
    # Spatial bounds (India)
    spatial_bounds: Dict[str, float] = None
    
    # Output paths
    raw_data_path: str = "./data/raw"
    processed_data_path: str = "./data/processed"
    metadata_path: str = "./data/metadata"
    annotations_path: str = "./data/annotations"
    
    # Processing options
    parallel_downloads: int = 4
    chunk_size: int = 8192
    retry_attempts: int = 3
    timeout: int = 300

    def __post_init__(self):
        if self.spatial_bounds is None:
            # Default to India bounds
            self.spatial_bounds = {
                "min_lon": 68.0,
                "min_lat": 6.0,
                "max_lon": 98.0,
                "max_lat": 38.0
            }

class ISRODataSource:
    """Interface for ISRO/Bhuvan data access"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.retry_attempts,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    async def search_tiles(
        self,
        start_date: str,
        end_date: str,
        bbox: List[float],
        satellite: Optional[str] = None,
        max_cloud_cover: Optional[float] = None
    ) -> List[EOTileMetadata]:
        """Search for available EO tiles"""
        
        # This is a simplified implementation
        # In practice, you'd need to implement the actual ISRO API calls
        logger.info(f"Searching for tiles from {start_date} to {end_date}")
        
        # Mock data for demonstration
        tiles = []
        
        # Generate sample tile metadata
        for i in range(10):
            tile = EOTileMetadata(
                tile_id=f"ISRO_L1C_{start_date.replace('-', '')}_{i:03d}",
                satellite="ResourceSat-2A",
                sensor="LISS-IV",
                acquisition_date=start_date,
                path=100 + i,
                row=50 + i,
                cloud_cover=np.random.uniform(0, max_cloud_cover or 30),
                sun_elevation=np.random.uniform(30, 60),
                sun_azimuth=np.random.uniform(120, 180),
                processing_level="L1C",
                spatial_resolution=5.8,  # meters
                spectral_bands=["B1", "B2", "B3", "B4"],
                bbox=[
                    bbox[0] + i * 0.1,
                    bbox[1] + i * 0.1,
                    bbox[0] + (i + 1) * 0.1,
                    bbox[1] + (i + 1) * 0.1
                ],
                epsg_code=4326,
                file_size=1024 * 1024 * 50,  # 50MB
                download_url=f"{self.config.isro_base_url}/tile_{i}.zip",
                thumbnail_url=f"{self.config.isro_base_url}/thumb_{i}.jpg"
            )
            tiles.append(tile)
        
        return tiles
    
    async def download_tile(
        self,
        tile: EOTileMetadata,
        output_path: Path
    ) -> Path:
        """Download a single EO tile"""
        
        output_file = output_path / f"{tile.tile_id}.zip"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.exists():
            logger.info(f"Tile {tile.tile_id} already exists, skipping download")
            return output_file
        
        logger.info(f"Downloading tile {tile.tile_id}")
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    tile.download_url,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    response.raise_for_status()
                    
                    async with aiofiles.open(output_file, 'wb') as f:
                        async for chunk in response.content.iter_chunked(self.config.chunk_size):
                            await f.write(chunk)
                
                logger.info(f"Successfully downloaded {tile.tile_id}")
                return output_file
                
            except Exception as e:
                logger.error(f"Failed to download {tile.tile_id}: {e}")
                if output_file.exists():
                    output_file.unlink()
                raise

class EODataProcessor:
    """Process raw EO satellite data"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def extract_archive(self, archive_path: Path, extract_to: Path) -> List[Path]:
        """Extract downloaded archive files"""
        extract_to.mkdir(parents=True, exist_ok=True)
        extracted_files = []
        
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                extracted_files = [extract_to / name for name in zip_ref.namelist()]
                
        elif archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
                extracted_files = [extract_to / name for name in tar_ref.getnames()]
        
        return extracted_files
    
    def preprocess_raster(
        self,
        input_path: Path,
        output_path: Path,
        tile_metadata: EOTileMetadata
    ) -> Dict[str, Any]:
        """Preprocess raster data"""
        
        logger.info(f"Processing raster: {input_path}")
        
        with rasterio.open(input_path) as src:
            # Get source info
            src_crs = src.crs
            src_transform = src.transform
            src_width = src.width
            src_height = src.height
            
            # Calculate target transform
            dst_crs = self.config.target_crs
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src_crs, dst_crs, src_width, src_height, *src.bounds
            )
            
            # Update metadata
            dst_profile = src.profile.copy()
            dst_profile.update({
                'crs': dst_crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'compress': self.config.compression
            })
            
            # Reproject and resample
            with rasterio.open(output_path, 'w', **dst_profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=dst_crs,
                        resampling=getattr(Resampling, self.config.resampling_method)
                    )
        
        # Generate statistics
        with rasterio.open(output_path) as processed:
            stats = {}
            for i in range(1, processed.count + 1):
                band_data = processed.read(i)
                stats[f'band_{i}'] = {
                    'min': float(np.min(band_data)),
                    'max': float(np.max(band_data)),
                    'mean': float(np.mean(band_data)),
                    'std': float(np.std(band_data)),
                    'nodata_pixels': int(np.sum(band_data == processed.nodata))
                }
        
        return {
            'processed_file': str(output_path),
            'original_crs': str(src_crs),
            'target_crs': dst_crs,
            'original_size': [src_width, src_height],
            'processed_size': [dst_width, dst_height],
            'band_statistics': stats,
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def generate_thumbnail(
        self,
        raster_path: Path,
        thumbnail_path: Path,
        size: Tuple[int, int] = (256, 256)
    ) -> Path:
        """Generate thumbnail for visualization"""
        
        with rasterio.open(raster_path) as src:
            # Read RGB bands (assuming bands 3,2,1 for natural color)
            try:
                rgb_bands = [3, 2, 1] if src.count >= 3 else [1, 1, 1]
                rgb_data = src.read(rgb_bands)
                
                # Normalize to 0-255
                rgb_normalized = np.zeros_like(rgb_data, dtype=np.uint8)
                for i, band in enumerate(rgb_data):
                    band_min, band_max = np.percentile(band[band > 0], [2, 98])
                    band_normalized = np.clip((band - band_min) / (band_max - band_min) * 255, 0, 255)
                    rgb_normalized[i] = band_normalized.astype(np.uint8)
                
                # Create thumbnail using PIL
                from PIL import Image
                
                # Transpose to HWC format
                rgb_image = np.transpose(rgb_normalized, (1, 2, 0))
                pil_image = Image.fromarray(rgb_image)
                pil_image.thumbnail(size, Image.Resampling.LANCZOS)
                
                thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
                pil_image.save(thumbnail_path, 'JPEG', quality=85)
                
                return thumbnail_path
                
            except Exception as e:
                logger.error(f"Failed to generate thumbnail for {raster_path}: {e}")
                raise

class AnnotationGenerator:
    """Generate annotations for EO imagery"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def generate_land_cover_labels(
        self,
        raster_path: Path,
        tile_metadata: EOTileMetadata
    ) -> Dict[str, Any]:
        """Generate land cover classification labels"""
        
        # This is a simplified implementation
        # In practice, you'd use ML models or existing land cover datasets
        
        with rasterio.open(raster_path) as src:
            data = src.read()
            height, width = data.shape[1], data.shape[2]
            
            # Mock land cover classification
            land_cover_classes = [
                'water', 'forest', 'agriculture', 'urban', 'barren', 'grassland'
            ]
            
            # Generate random land cover percentages
            percentages = np.random.dirichlet(np.ones(len(land_cover_classes)))
            
            land_cover = {
                'classes': {
                    cls: float(pct) for cls, pct in zip(land_cover_classes, percentages)
                },
                'dominant_class': land_cover_classes[np.argmax(percentages)],
                'confidence': float(np.max(percentages)),
                'spatial_resolution': tile_metadata.spatial_resolution,
                'total_area_km2': (height * width * tile_metadata.spatial_resolution ** 2) / 1e6
            }
        
        return land_cover
    
    def generate_image_caption(
        self,
        raster_path: Path,
        tile_metadata: EOTileMetadata,
        land_cover: Dict[str, Any]
    ) -> str:
        """Generate natural language caption for the image"""
        
        dominant_class = land_cover['dominant_class']
        confidence = land_cover['confidence']
        
        # Get location info (simplified)
        center_lat = (tile_metadata.bbox[1] + tile_metadata.bbox[3]) / 2
        center_lon = (tile_metadata.bbox[0] + tile_metadata.bbox[2]) / 2
        
        # Generate caption based on dominant land cover
        caption_templates = {
            'water': f"Satellite image showing water bodies captured by {tile_metadata.satellite} on {tile_metadata.acquisition_date}. The image covers an area around {center_lat:.2f}°N, {center_lon:.2f}°E with {confidence*100:.1f}% water coverage.",
            
            'forest': f"Satellite imagery of forested area taken by {tile_metadata.satellite} on {tile_metadata.acquisition_date}. The scene shows dense forest coverage around {center_lat:.2f}°N, {center_lon:.2f}°E with {confidence*100:.1f}% forest area.",
            
            'agriculture': f"Agricultural landscape captured by {tile_metadata.satellite} on {tile_metadata.acquisition_date}. The image shows farmland and agricultural fields around {center_lat:.2f}°N, {center_lon:.2f}°E with {confidence*100:.1f}% agricultural land use.",
            
            'urban': f"Urban area satellite image from {tile_metadata.satellite} acquired on {tile_metadata.acquisition_date}. The scene shows built-up areas and infrastructure around {center_lat:.2f}°N, {center_lon:.2f}°E with {confidence*100:.1f}% urban coverage.",
            
            'barren': f"Barren land satellite imagery captured by {tile_metadata.satellite} on {tile_metadata.acquisition_date}. The image shows sparse vegetation and bare ground around {center_lat:.2f}°N, {center_lon:.2f}°E with {confidence*100:.1f}% barren land.",
            
            'grassland': f"Grassland satellite image from {tile_metadata.satellite} taken on {tile_metadata.acquisition_date}. The scene shows grassland and pasture areas around {center_lat:.2f}°N, {center_lon:.2f}°E with {confidence*100:.1f}% grassland coverage."
        }
        
        return caption_templates.get(dominant_class, f"Satellite image captured by {tile_metadata.satellite} on {tile_metadata.acquisition_date} around {center_lat:.2f}°N, {center_lon:.2f}°E.")

class EODataPipeline:
    """Main pipeline for EO data processing"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        self.config = ProcessingConfig(**config_dict)
        self.data_source = ISRODataSource(self.config)
        self.processor = EODataProcessor(self.config)
        self.annotator = AnnotationGenerator(self.config)
        
        # Create output directories
        for path in [self.config.raw_data_path, self.config.processed_data_path, 
                    self.config.metadata_path, self.config.annotations_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    async def run_pipeline(
        self,
        start_date: str,
        end_date: str,
        spatial_bounds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Run the complete data processing pipeline"""
        
        logger.info("Starting EO data processing pipeline")
        
        # Use config bounds if not specified
        bounds = spatial_bounds or self.config.spatial_bounds
        bbox = [bounds['min_lon'], bounds['min_lat'], bounds['max_lon'], bounds['max_lat']]
        
        # Step 1: Search for available tiles
        logger.info("Searching for available tiles...")
        tiles = await self.data_source.search_tiles(
            start_date=start_date,
            end_date=end_date,
            bbox=bbox,
            max_cloud_cover=self.config.max_cloud_cover
        )
        
        logger.info(f"Found {len(tiles)} tiles matching criteria")
        
        # Step 2: Download tiles
        logger.info("Downloading tiles...")
        download_tasks = []
        for tile in tiles:
            task = self.data_source.download_tile(
                tile, 
                Path(self.config.raw_data_path)
            )
            download_tasks.append(task)
        
        # Limit concurrent downloads
        semaphore = asyncio.Semaphore(self.config.parallel_downloads)
        
        async def download_with_semaphore(task):
            async with semaphore:
                return await task
        
        downloaded_files = await asyncio.gather(
            *[download_with_semaphore(task) for task in download_tasks],
            return_exceptions=True
        )
        
        # Step 3: Process downloaded data
        logger.info("Processing downloaded data...")
        processed_data = []
        
        for i, (tile, download_result) in enumerate(zip(tiles, downloaded_files)):
            if isinstance(download_result, Exception):
                logger.error(f"Skipping tile {tile.tile_id} due to download error: {download_result}")
                continue
            
            try:
                # Extract archive
                extract_path = Path(self.config.raw_data_path) / f"extracted_{tile.tile_id}"
                extracted_files = self.processor.extract_archive(download_result, extract_path)
                
                # Find raster file
                raster_files = [f for f in extracted_files if f.suffix.lower() in ['.tif', '.tiff', '.img']]
                if not raster_files:
                    logger.warning(f"No raster files found in {tile.tile_id}")
                    continue
                
                raster_file = raster_files[0]  # Use first raster file
                
                # Process raster
                output_path = Path(self.config.processed_data_path) / f"{tile.tile_id}_processed.tif"
                processing_info = self.processor.preprocess_raster(
                    raster_file, output_path, tile
                )
                
                # Generate thumbnail
                thumbnail_path = Path(self.config.processed_data_path) / f"{tile.tile_id}_thumb.jpg"
                self.processor.generate_thumbnail(output_path, thumbnail_path)
                
                # Generate annotations
                land_cover = self.annotator.generate_land_cover_labels(output_path, tile)
                caption = self.annotator.generate_image_caption(output_path, tile, land_cover)
                
                # Compile metadata
                complete_metadata = {
                    'tile_metadata': asdict(tile),
                    'processing_info': processing_info,
                    'land_cover': land_cover,
                    'caption': caption,
                    'files': {
                        'processed_raster': str(output_path),
                        'thumbnail': str(thumbnail_path),
                        'raw_archive': str(download_result)
                    }
                }
                
                # Save metadata
                metadata_file = Path(self.config.metadata_path) / f"{tile.tile_id}_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(complete_metadata, f, indent=2, default=str)
                
                processed_data.append(complete_metadata)
                logger.info(f"Successfully processed tile {tile.tile_id}")
                
            except Exception as e:
                logger.error(f"Failed to process tile {tile.tile_id}: {e}")
                continue
        
        # Step 4: Generate dataset manifest
        manifest = {
            'pipeline_config': asdict(self.config),
            'processing_timestamp': datetime.now().isoformat(),
            'query_parameters': {
                'start_date': start_date,
                'end_date': end_date,
                'spatial_bounds': bounds
            },
            'statistics': {
                'total_tiles_found': len(tiles),
                'successfully_downloaded': len([r for r in downloaded_files if not isinstance(r, Exception)]),
                'successfully_processed': len(processed_data)
            },
            'processed_tiles': processed_data
        }
        
        manifest_file = Path(self.config.metadata_path) / f"dataset_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        logger.info(f"Pipeline completed. Processed {len(processed_data)} tiles.")
        logger.info(f"Dataset manifest saved to {manifest_file}")
        
        return manifest

async def main():
    """Main function for running the pipeline"""
    config_path = "configs/isro_pipeline_config.yaml"
    
    pipeline = EODataPipeline(config_path)
    
    # Run pipeline for a date range
    result = await pipeline.run_pipeline(
        start_date="2024-01-01",
        end_date="2024-01-31"
    )
    
    print(f"Pipeline completed successfully. Processed {result['statistics']['successfully_processed']} tiles.")

if __name__ == "__main__":
    asyncio.run(main())
