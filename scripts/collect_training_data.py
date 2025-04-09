#!/usr/bin/env python
import os
import argparse
from data.preprocessing.data_collector import ContentDataCollector

def main():
    parser = argparse.ArgumentParser(description='Collect training data for fusion model')
    parser.add_argument('--api-key', type=str, help='Nielsen API key')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date for data collection')
    parser.add_argument('--end-date', type=str, default='2023-12-31', help='End date for data collection')
    parser.add_argument('--max-samples', type=int, default=1000, help='Maximum number of samples to collect')
    parser.add_argument('--output-dir', type=str, default='./data/datasets', help='Output directory')
    parser.add_argument('--cache-dir', type=str, default='./data/cache', help='Cache directory')
    args = parser.parse_args()
    
    # Initialize collector
    collector = ContentDataCollector(data_dir=args.output_dir, cache_dir=args.cache_dir)
    
    # Collect data
    content_data = collector.collect_nielsen_content(
        api_key=args.api_key,
        start_date=args.start_date,
        end_date=args.end_date,
        max_samples=args.max_samples
    )
    
    # Download thumbnails
    content_data = collector.download_thumbnails(content_data)
    
    # Prepare training data
    data_splits = collector.prepare_training_data(content_data)
    
    # Save splits
    output_paths = collector.save_dataset_splits(data_splits)
    
    print("Data collection complete!")
    print(f"Train data: {output_paths['train']}")
    print(f"Validation data: {output_paths['val']}")
    print(f"Test data: {output_paths['test']}")

if __name__ == '__main__':
    main() 